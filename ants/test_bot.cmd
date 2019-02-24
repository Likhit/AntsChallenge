@echo off
python playgame.py --engine_seed 42 --player_seed 42 --food none --end_wait=0.25 --verbose --log_dir game_logs --turns 30 --map_file "./maps/example/tutorial1.map" "python submission_test/TestBot.py" "python submission_test/TestBot.py" -e --strict --capture_errors
