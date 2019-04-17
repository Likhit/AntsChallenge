"""
This script plays x * 5^n games on the specifed maps and stores
the results to a file. 'n' is the number of players in the map,
while 'x' is the number of runs to play each player combination.
The final outpu is stored in a pickle file with the name
<map_name>_<bot1>_<bot2>_...<botn>_<run_number>.pickle.
"""
import argparse
import glob
import itertools as it
import os
import pickle
import re
import sys
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

GYM_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../'
)
sys.path.append(GYM_MODULE_PATH)
try:
    import trained_bots.utils as utils
except ImportError as e:
    print(f'Module not found in {GYM_MODULE_PATH}')
    raise e

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WINNER_BOT_DIR = os.path.join(
    CURRENT_DIR,
    '../../../winner_bots/first_place/'
)
WINNER_BOT_CMD = f'java -cp {WINNER_BOT_DIR} MyBot'

def get_players_combinations(num_players):
    bot_gen = [
        lambda: utils.enemybots.CmdBot('xanthis', WINNER_BOT_CMD),
        lambda: utils.enemybots.SampleBots.random_bot(),
        lambda: utils.enemybots.SampleBots.lefty_bot(),
        lambda: utils.enemybots.SampleBots.hunter_bot(),
        lambda: utils.enemybots.SampleBots.greedy_bot(),
    ]

    result = []
    ind_list = list(range(len(bot_gen)))
    for selection in it.product(*([ind_list] * num_players)):
        result.append([bot_gen[i]() for i in selection])
    return result

def get_slim_info(info):
    return SimpleNamespace(
        food_distr = info.FoodFunc.food_distr,
        raze_loc = info.ScoreFunc.scores
    )

def play_game(map_file, players):
    game_opts = utils.antsgym.AntsEnvOptions()  \
        .set_map_file(map_file)                 \
        .set_turns(2000)                        \
        .set_scenario(True)
    reward_func = utils.reward.CompositeFunc([
        (utils.reward.FoodFunc(), 0),
        (utils.reward.ScoreFunc(), 1)
    ])
    agent_names = [player.name for player in players]
    env = utils.antsgym.AntsEnv(game_opts, [], reward_func, agent_names)
    agents = [utils.antsgym.SampleAgent(player, env, i) for i, player in enumerate(players)]
    for agent in agents:
        agent.setup()

    result = []
    state = env.reset()
    while True:
        actions = []
        for agent in agents:
            agent.update_map(state)
            actions.append(agent.get_moves())
        actions = np.concatenate(actions, axis=0)
        n_state, reward, done, info = env.step(actions)
        result.append(SimpleNamespace(
            state=state, next_state=n_state,
            reward=reward, info=get_slim_info(info),
            action=actions
        ))
        state = n_state
        if done:
            result = SimpleNamespace(
                history=result, result=env.get_game_result()
            )
            break
    env.reset()
    return result

def evaluate_step_values(episode_data, discount_factor):
    e_instance = episode_data.history[0]
    prev_reward = np.zeros(e_instance.reward.shape)
    prev_food_distr = np.zeros(e_instance.info.food_distr.shape)
    for i, step in enumerate(reversed(episode_data.history)):
        step.reward += (discount_factor * prev_reward)
        prev_reward = step.reward

        step.info.food_distr += (discount_factor * prev_food_distr)
        prev_food_distr = step.info.food_distr

def run_episode_data(map_file, trial, discount_factor, players):
    map_name = os.path.basename(map_file)
    map_name = os.path.splitext(map_name)[0]
    save_file_name = os.path.join(
        CURRENT_DIR,
        f'episodes/{map_name}_{"_".join([p.name for p in players])}_{trial}.pickle'
    )
    print(f'Started {save_file_name}')
    data = play_game(map_file, players)
    evaluate_step_values(data, discount_factor)
    with open(save_file_name, 'wb') as file_handle:
        pickle.dump(data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Finished {save_file_name}')

NUM_PROCESSES = 2

def main(args):
    for map_file in glob.glob(args.map_glob):
        num_players = re.search('p(\d+)', os.path.basename(map_file))
        try:
            if not num_players:
                raise Exception(f'Unable to find the number of player in map {map_file}')
            num_players = int(num_players.group(1))
        except Exception as e:
            print(e)
            continue

        for trial in range(args.num_trials):
            player_combs = get_players_combinations(num_players)
            func = partial(run_episode_data, map_file, trial, args.discount_factor)
            with Pool(NUM_PROCESSES) as p:
                p.map(func, player_combs)
            print(f'Finished trial {trial}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run games on a map and store gameplay to file.')
    parser.add_argument('--map_glob', help='Glob identifing all map files to play on')
    parser.add_argument('--num_trials', type=int, help='Number of times to run each map-player combination.')
    parser.add_argument('--discount_factor', type=float, help='The discount factor for the rewards.')
    args = parser.parse_args()
    main(args)
