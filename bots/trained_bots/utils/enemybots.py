import abc
import os
import sys
import time

SAMPLE_BOTS_MODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../sample_bots/python'
)
sys.path.append(SAMPLE_BOTS_MODULE)
try:
    import sample_game as sample_bot_game
    import GreedyBot as greedy_bot
    import HunterBot as hunter_bot
    import LeftyBot as lefty_bot
    import RandomBot as random_bot
    import TestBot as test_bot
except ImportError as e:
    print(f'Module not found in {SAMPLE_BOTS_MODULE}')
    raise e

SANDBOX_MODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../worker'
)
sys.path.append(SANDBOX_MODULE)
try:
    from sandbox import get_sandbox
except ImportError as e:
    print(f'Module not found in {SANDBOX_MODULE}')
    raise e

class Bot(abc.ABC):
    """
    Interface to implement for an enemy bot used in a gym
    environment for the Ants game.

    Args:
        - name (str): The name of the bot that is displayed in the visualization.
    """
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def setup(self, map_data):
        """
        Called during initialization/reset. Feeds the bots the
        initial state and game parameters.
        """
        pass

    @abc.abstractmethod
    def update_map(self, map_data):
        """
        Called before each turn to update the state space visible
        to the bot.
        """
        pass

    @abc.abstractmethod
    def get_moves(self):
        """
        Called after update_map. Should return the set of moves to
        be made by the bot. The moves should be a list of string of
        the form 'o {row} {col} {direction}' where direction is one
        of n, e, s, or w.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Called when the gym environment is reset.
        """
        pass

class SampleBots(Bot):
    """
    Implementation of the Bot interface for the sample bots.

    Args:
        - name (str): The name of the bot.
        - bot: Instance of the sample bot.
    """
    def __init__(self, name, bot):
        super().__init__(name)
        self.ants = sample_bot_game.Game()
        self.bot = bot

    def setup(self, map_data):
        self.ants.setup(map_data)
        return self

    def update_map(self, map_data):
        self.ants.update(map_data)
        return self

    def get_moves(self):
        return self.bot.do_turn(self.ants, False)

    def reset(self):
        self.ants = sample_bot_game.Game()
        return self

    @staticmethod
    def random_bot():
        """
        Creates a RandomBot enemy.
        """
        return SampleBots('RandomBot', random_bot.RandomBot())

    @staticmethod
    def greedy_bot():
        """
        Creates a GreedyBot enemy.
        """
        return SampleBots('GreedyBot', greedy_bot.GreedyBot())

    @staticmethod
    def hunter_bot():
        """
        Creates a HunterBot enemy.
        """
        return SampleBots('HunterBot', hunter_bot.HunterBot())

    @staticmethod
    def lefty_bot():
        """
        Creates a LeftyBot enemy.
        """
        return SampleBots('LeftyBot', lefty_bot.LeftyBot())

    @staticmethod
    def test_bot():
        """
        Creates a TestBot enemy.
        """
        return SampleBots('TestBot', test_bot.TestBot())


class CmdBot(Bot):
    """
    A bot that is played throught the command line interface (i.e.
    the original input/output form factor of the game).

    Args:
        - name (str): The name of the bot.
        - cmd (str): The shell command to execute to start the bot.
    """
    def __init__(self, name, cmd):
        super().__init__(name)
        self.wd, self.cmd = CmdBot._get_cmd_wd(cmd)
        self.reset()

    def setup(self, map_data):
        map_data += 'ready\n'
        self.sandbox.write(map_data)
        self.get_moves() # Wait for go from bot (moves will be empty)
        self.turn += 1
        return self

    def update_map(self, map_data):
        map_data = f'turn {self.turn}\n' + map_data + 'go\n'
        self.sandbox.write(map_data)
        self.turn += 1
        return self

    def get_moves(self):
        moves = []
        self.sandbox.resume()
        finished = False
        while not finished:
            time.sleep(0.01)
            if not self.sandbox.is_alive:
                finished = True
                break
            # Read 100 lines at a time.
            for i in range(100):
                move = self.sandbox.read_line()
                if move is None:
                    break
                move = move.strip()
                if move.lower() == 'go':
                    finished = True
                    break
                moves.append(move)
        self.sandbox.pause()
        return moves

    def reset(self):
        if hasattr(self, 'sandbox'):
            self.sandbox.kill()
            self.sandbox.release()
        self.turn = 0
        self.sandbox = get_sandbox(self.wd, False)
        self.sandbox.start(self.cmd)
        if not self.sandbox.is_alive:
            raise Exception('The bot crashed at start. Check the commands.')
        self.sandbox.pause()
        return self

    @staticmethod
    def _get_cmd_wd(cmd):
        ''' get the proper working directory from a command line '''
        new_cmd = []
        wd = None
        for i, part in reversed(list(enumerate(cmd.split()))):
            if wd == None and os.path.exists(part):
                wd = os.path.dirname(os.path.realpath(part))
                basename = os.path.basename(part)
                if i == 0 or not basename:
                    new_cmd.insert(0, part)
                else:
                    new_cmd.insert(0, basename)
            else:
                new_cmd.insert(0, part)
        return wd, ' '.join(new_cmd)
