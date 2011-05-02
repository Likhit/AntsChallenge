#!/usr/bin/python
import os
import shlex
import signal
import subprocess
import sys
import time
from optparse import OptionParser
from Queue import Queue, Empty
from threading import Thread

try:
    from server_info import server_info
    _SECURE_DEFAULT = True
except ImportError:
    _SECURE_DEFAULT = False

class JailError(StandardError):
    pass

class _Jail(object):
    def __init__(self):
        self.locked = False
        jail_base = "/srv/chroot"
        all_jails = os.listdir(jail_base)
        all_jails = [j for j in all_jails if j.startswith("jailuser")]
        for jail in all_jails:
            lock_dir = os.path.join(jail_base, jail, "locked")
            try:
                os.mkdir(lock_dir)
            except OSError:
                # if the directory could not be created, that should mean the
                # jail is already locked and in use
                continue
            with open(os.path.join(lock_dir, "lock.pid"), "w") as pid_file:
                pid_file.write(str(os.getpid()))
            self.locked = True
            self.name = jail
            break
        else:
            raise JailError("Could not find an unlocked jail")
        self.jchown = os.path.join(server_info["root_path"],
                server_info["repo_path"], "worker/jail_own")
        self.base_dir = os.path.join(jail_base, jail)
        self.number = int(jail[len("jailuser"):])
        self.chroot_cmd = "sudo -u {0} schroot -u {0} -c {0} -d {1} ".format(
                self.name, "/home/jailuser")

    def __del__(self):
        if self.locked:
            raise JailError("Jail object for %s freed without being released"
                    % (self.name))

    def release(self):
        if not self.locked:
            raise JailError("Attempt to release jail that is already unlocked")
        lock_dir = os.path.join(self.base_dir, "locked")
        pid_filename = os.path.join(lock_dir, "lock.pid")
        with open(pid_filename, 'r') as pid_file:
            lock_pid = int(pid_file.read())
            if lock_pid != os.getpid():
                # if we ever get here something has gone seriously wrong
                # most likely the jail locking mechanism has failed
                raise JailError("Jail released by different pid, name %s, lock_pid %d, release_pid %d"
                        % (self.name, lock_pid, os.getpid()))
        os.unlink(pid_filename)
        os.rmdir(lock_dir)
        self.locked = False

    def prepare_with(self, command_dir):
        if os.system("%s c %d" % (self.jchown, self.number)) != 0:
            raise JailError("Error returned from jail_own c %d in prepare"
                    % (self.number,))
        scratch_dir = os.path.join(self.base_dir, "scratch")
        if os.system("rm -rf %s" % (scratch_dir,)) != 0:
            raise JailError("Could not remove old scratch area from jail %d"
                    % (self.number,))
        home_dir = os.path.join(scratch_dir, "home/jailuser")
        os.makedirs(home_dir)
        if os.system("cp -r %s %s" % (command_dir, home_dir)) != 0:
            raise JailError("Error copying working directory '%s' to jail %d"
                    % (command_dir, self.number))
        if os.system("%s j %d" % (self.jchown, self.number)) != 0:
            raise JailError("Error returned from jail_own j %d in prepare"
                    % (self.number,))

    def signal(self, signal):
        if not self.locked:
            raise JailError("Attempt to send %s to unlocked jail" % (signal,))
        if os.system("sudo -u {0} killall -{1} -u {0}".format(
                self.name, signal)) != 0:
            raise JailError("Error returned from jail kill for %s"
                    % (self.name,))

    def kill(self):
        self.signal("KILL")

    def pause(self):
        self.signal("STOP")

    def resume(self):
        self.signal("CONT")


def _monitor_input_channel(sandbox):
    while sandbox.is_alive:
        try:
            line = sandbox.command_process.stdout.readline()
        except:
            print >> sys.stderr, sys.exc_info()
            sandbox.kill()
            break
        if not line:
            sandbox.kill()
            break
        sandbox.stdout_queue.put(line.strip())

class Sandbox:
    """Provide a sandbox to run arbitrary commands in.

    The sandbox class is used to invoke arbitrary shell commands. Its main
    feature is that it has the option to launch the shell command inside a
    jail, in order to totally isolate the command.

    """

    def __init__(self, working_directory, shell_command, stderr=None,
            secure=_SECURE_DEFAULT):
        """Initialize a new sandbox and invoke the given shell command inside.

        working_directory: the directory in which the shell command should
                           be launched. If security is enabled, files from
                           this directory are copied into the VM before the
                           shell command is executed.
        shell_command: the shell command to launch inside the sandbox.
        stderr: where the bot's stderr output should be written out to
                defaults to keeping the current stderr for the child process
        secure: really use a jail or just run the command directly
                defaults to True when a server_info module is found, False
                otherwise

        """
        self.is_alive = False
        self.command_process = None
        self.stdout_queue = Queue()

        if secure:
            self.jail = _Jail()
            self.jail.prepare_with(working_directory)
            shell_command = self.jail.chroot_cmd + shell_command
            working_directory = None
        else:
            self.jail = None

        shell_command = shlex.split(shell_command.replace('\\','/'))
        self.command_process = subprocess.Popen(shell_command,
                                                stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE,
                                                stderr=stderr,
                                                cwd=working_directory)
        self.is_alive = not self.command_process is None
        stdout_monitor = Thread(target=_monitor_input_channel, args=(self,))
        stdout_monitor.start()

    def release(self):
        """Release the sandbox for further use

        If running in a jail unlocks and releases the jail for reuse by others.
        Must be called exactly once after Sandbox.kill has been called.

        """
        if self.is_alive:
            raise JailError("Jail released while still alive")
        if self.jail:
            self.jail.release()

    def kill(self):
        """Shuts down the sandbox.

        Shuts down the sandbox, cleaning up any spawned processes, threads, and
        other resources. The shell command running inside the sandbox may be
        suddenly terminated.

        """
        if self.is_alive:
            if self.jail:
                self.jail.kill()
            else:
                try:
                    self.command_process.kill()
                    self.command_process.wait()
                except OSError:
                    pass
            self.is_alive = False

    def pause(self):
        """Pause the process by sending a SIGSTOP to the child

        This method is a no-op on Windows
        """
        if self.jail:
            self.jail.pause()
        else:
            try:
                self.command_process.send_signal(signal.SIGSTOP)
            except (ValueError, AttributeError):
                pass

    def resume(self):
        """Resume the process by sending a SIGCONT to the child

        This method is a no-op on Windows
        """
        if self.jail:
            self.jail.resume()
        else:
            try:
                self.command_process.send_signal(signal.SIGCONT)
            except (ValueError, AttributeError):
                pass

    def write(self, str):
        """Write str to stdin of the process being run"""
        if not self.is_alive:
            return False
        try:
            self.command_process.stdin.write(str)
            self.command_process.stdin.flush()
        except (OSError, IOError):
            self.kill()
            return False
        return True

    def write_line(self, line):
        """Write line to stdin of the process being run

        A newline is appended to line and written to stdin of the child process

        """
        if not self.is_alive:
            return False
        try:
            self.command_process.stdin.write(line + "\n")
            self.command_process.stdin.flush()
        except (OSError, IOError):
            self.kill()
            return False
        return True

    def read_line(self):
        """Read line from child process

        Returns a line of the child process' stdout, if one isn't available
        returns None.

        """
        try:
            return self.stdout_queue.get(block=False)
        except Empty:
            return None

def main():
    parser = OptionParser(usage="usage: %prog [options] <command to run>")
    parser.add_option("-d", "--directory", action="store", dest="working_dir",
            default=".",
            help="Working directory to run command in (copied in secure mode)")
    parser.add_option("-l", action="append", dest="send_lines", default=list(),
            help="String to send as a line on commands stdin")
    parser.add_option("-s", "--send-delay", action="store", dest="send_delay",
            type="float", default=0.0,
            help="Time in seconds to sleep after sending a line")
    parser.add_option("-r", "--receive-delay", action="store",
            dest="resp_delay", type="float", default=0.01,
            help="Time in seconds to sleep before checking for a response line")
    parser.add_option("-j", "--jail", action="store_true", dest="secure",
            default=_SECURE_DEFAULT,
            help="Run in a secure jail")
    parser.add_option("-o", "--open", action="store_false", dest="secure",
            help="Run without using a secure jail")
    options, args = parser.parse_args()
    if len(args) == 0:
        parser.error("Must include a command to run.\
                \nRun with --help for more information.")

    sandbox = Sandbox(options.working_dir, " ".join(args),
            secure=options.secure)
    for line in options.send_lines:
        if not sandbox.write_line(line):
            print >> sys.stderr, "Could not send line '%s'" % (line,)
            sandbox.kill()
            sys.exit(1)
        print "sent:", line
        time.sleep(options.send_delay)
    while True:
        time.sleep(options.resp_delay)
        response = sandbox.read_line()
        if response is None:
            print "No more responses. Terminating."
            break
        print "response: " + response
    sandbox.kill()

if __name__ == "__main__":
    main()
