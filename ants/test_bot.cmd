@echo off
python "%~dp0playgame.py" --engine_seed 42 --player_seed 42 --end_wait=0.25 --verbose --log_dir game_logs --turns 30 --map_file "%~dp0maps\example\tutorial1.map" %* "python ""%~dp0..\bots\sample_bots\python\TestBot.py""" "python ""%~dp0..\bots\sample_bots\python\TestBot.py""" -e --strict --capture_errors
