import json
import os
import sys

import gym
import numpy as np

from .enemybots import SampleBots
from .reward import RewardInputs, FoodScoreFunc

ANTS_MODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../ants'
)
sys.path.append(ANTS_MODULE)
try:
    from ants import Ants
    import visualizer.visualize_locally as visualizer
except ImportError as e:
    print(f'Ants module not found in {ANTS_MODULE}')
    raise e

class AntsEnvOptions(object):
    """
    Options required for starting an Ants game.
    """
    def __init__(self):
        # Initialize all options with None or default
        self.map_file = None
        self.turns = 1000

        self.view_radius_sq = 77
        self.attack_radius_sq = 5
        self.consumption_radius_sq = 1
        self.attack_mode = "focus"

        self.food_rate = 5
        self.food_turn = 20
        self.food_start = 75
        self.food_visible = 3
        self.food_spawn_mode = "symmetric"

        self.scenario = False

    def _read_map(self):
        with open(self.map_file, 'r') as map_file:
            return map_file.read()

    def as_dict(self):
        return {
            'map': self._read_map(),
            'turns': self.turns,
            'loadtime': 3000,
            'turntime': 1000,
            'viewradius2': self.view_radius_sq,
            'attackradius2': self.attack_radius_sq,
            'spawnradius2': self.consumption_radius_sq,
            'food_rate': self.food_rate,
            'food_turn': self.food_turn,
            'food_start': self.food_start,
            'food_visible': self.food_visible,
            'attack': self.attack_mode,
            'food': self.food_spawn_mode,
            'scenario': self.scenario
        }

    def set_map_file(self, path):
        """
        Path to the file containing the map.
        """
        self.map_file = path
        return self

    def set_turns(self, turns):
        """
        The number of turns to run the game for. Default: 1000.
        """
        self.turns = turns
        return self

    def set_view_radius_sq(self, radius):
        """
        The squared vision radius of the ants (in # of cells).
        Default: 77.
        """
        self.view_radius_sq = radius
        return self

    def set_attack_radius_sq(self, radius):
        """
        The squared attack radius of an ant (in # of cells).
        Default: 5
        """
        self.attack_radius_sq = radius
        return self

    def set_consumption_radius_sq(self, radius):
        """
        The squared consumption radius (in # of cells). Default: 1.
        An ant will consume a food block within the consumption
        radius.
        """
        self.consumption_radius_sq = radius
        return self

    def set_attack_mode(self, mode):
        """
        Attack method to use. One of the following:
            "closest", "focus", "support", or "damage"
        Default: "focus"
        """
        self.attack_mode = mode
        return self

    def set_food_rate(self, rate):
        """
        Determines the amount of food that should exist on the
        map at all times. The total food on the map is equal to
        food_rate / food_turn * num_players.
        Use this function to set the food_rate. Default: 5
        """
        self.food_rate = rate
        return self

    def set_food_turn(self, turn):
        """
        Determines the amount of food that should exist on the
        map at all times. The total food on the map is equal to
        food_rate / food_turn * num_players.
        Use this function to set the food_turn. Default: 20
        """
        self.food_turn = turn
        return self

    def set_food_start(self, start):
        """
        One over percentage of land area filled with food at start.
        Default: 75
        """
        self.food_start = start
        return self

    def set_food_visible(self, visible):
        """
        Amount of food guaranteed to be visible to starting ants.
        Default: 3
        """
        self.food_visible = visible
        return self

    def set_food_spawn_mode(self, mode):
        """
        Food spawning method. One of:
            "none", "random", "sections", or "symmetric"
        Default: "symmetric"
        """
        self.food_spawn_mode = mode
        return self

    def set_scenario(self, scenario):
        """
        Set to True if the map file should be followed exactly.
        If False, all ants on the map file will be ignored and
        only a single ant at is created per hill.
        Default: False
        """
        self.scenario = scenario
        return self


class AntsEnv(gym.Env):
    """
    Open AI gym environment for the Ants game.

    Arguments:
        - game_opts (AntsEnvOptions): Options to customize
            the game.
        - enemies ([Bot]): A list of enemy bots.
        - reward_func (RewardFunc): A RewardFunc object to
            calculate the reward at each step.

    Properties:
        - observation_space: Box of shape (4 + num_players,
            game_hight, game_width) of type np.uint8. Most cell
            values are 0, or 1 to represent True, or False
            respectively. There are (4 + num_player) channels for
            the following:
                - Channel 0: If the cell is visible to the agent.
                - Channel 1: If the cell is land (else water).
                - Channel 2: If the cell has food.
                - Channel 3: If the cell has the agent's ant.
                - Channel 4: 1 if the cell has the agent's ant
                    hill. 2 if the cell has an enemy hill. Else 0.
                - Channel 5 and greater: If the cell has an enemy
                    ant (one channel per enemy player).

        - action_space: Box of shape (game_height, game_width) of
            with 0 <= cell value <= 4. The value represents the
            move to make with the ant at each cell position:
                - 0: If the cell position has no player ants then
                    don't move.
                - 1: Move 1 step north.
                - 2: Move 1 step east.
                - 3: Move 1 step south.
                - 4: Move 1 step west.
    """
    AGENT_PLAYER_NUM = 0    # NOTE: DO NOT CHANGE!!

    # Observation channels
    IS_VISIBLE_CHANNEL = 0
    IS_LAND_CHANNEL = 1
    HAS_FOOD_CHANNEL = 2
    ANT_HILL_CHANNEL = 3
    AGENT_ANTS_CHANNEL = 4
    ENEMY_ANTS_CHANNEL_START = 5

    # Action types
    NUM_ACTIONS = 5
    DONT_MOVE = 0
    MOVE_NORTH = 1
    MOVE_EAST = 2
    MOVE_SOUTH = 3
    MOVE_WEST = 4

    # Step info keys
    INFO_GAME_RESULT = 'game_result'
    INFO_AGENT_IGNORED_MOVES = 'agent_ignored_moves'
    INFO_AGENT_INVALID_MOVES = 'agent_invalid_moves'

    def __init__(self, game_opts, enemies, reward_func):
        self.game_opts = game_opts
        self.enemies = enemies
        self.reward_func = reward_func
        self.game = Ants(self.game_opts.as_dict())
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4 + self.game.num_players, self.game.height, self.game.width), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=0, high=4, shape=(self.game.height, self.game.width), dtype=np.uint8)
        self.reset(self.game)

    def step(self, action):
        info = {}
        if self.is_done:
            raise Exception("Can't make moves on a game that has finished. Reset board to continue.")
        # Play moves for each bot in a random order
        old_state = self._current_state
        self.game.start_turn()
        player_order = list(range(self.game.num_players))
        np.random.shuffle(player_order)
        for index in player_order:
            if index == 0:
                player_num = AntsEnv.AGENT_PLAYER_NUM
                moves = self._action_to_moves(action)
            else:
                player_num = index
                enemy_ind = index - 1
                if not self.game.is_alive(player_num): continue
                moves = self.enemies[enemy_ind].get_moves()

            valid, ignored, invalid = self.game.do_moves(player_num, moves)
            if index == 0:
                info[AntsEnv.INFO_AGENT_IGNORED_MOVES] = ignored
                info[AntsEnv.INFO_AGENT_INVALID_MOVES] = invalid
        self.game.finish_turn()

        if self.game.game_over() or self.game.turn > self.game.turns:
            self.is_done = True
            self.game.finish_game()
            info[AntsEnv.INFO_GAME_RESULT] = self.get_game_result()

        for i, enemy in enumerate(self.enemies):
            player_num = i + 1
            if self.game.is_alive(player_num):
                enemy.update_map(self.game.get_player_state(player_num))
        self._current_state, dead_ants = self._make_observation(self.game.get_player_state(AntsEnv.AGENT_PLAYER_NUM))

        # Calculate reward
        reward_inputs = RewardInputs()
        reward_inputs.old_state = old_state
        reward_inputs.new_state = self._current_state
        reward_inputs.ignored_moves = info[AntsEnv.INFO_AGENT_IGNORED_MOVES]
        reward_inputs.invalid_moves = info[AntsEnv.INFO_AGENT_INVALID_MOVES]
        reward_inputs.dead_ants = dead_ants
        reward_inputs.food_consumed = self.game.hive_food[AntsEnv.AGENT_PLAYER_NUM]
        reward_inputs.score = self.game.score[AntsEnv.AGENT_PLAYER_NUM]
        reward = self.reward_func(reward_inputs)

        return self._current_state, reward, self.is_done, info

    def reset(self, init_game=None):
        self.reward_func.reset()
        self.game = init_game or Ants(self.game_opts.as_dict())
        self.is_done = False
        self._current_state = None
        self.game.start_game()
        for i, enemy in enumerate(self.enemies):
            enemy.setup(self.game.get_player_start(i + 1))
        obs, _ = self._make_observation(self.game.get_player_state(AntsEnv.AGENT_PLAYER_NUM))
        return obs

    def visualize(self, game_result=None):
        """
        Visualize a game till the current state.
        Opens the game in a new browser, and will start
        visualization from the start, so is not the same as
        render.

        Arguments:
            - game_result: The game result to visualize. Default: None. Uses current game_result if None.
        """
        game_result = game_result or self.get_game_result()
        visualizer.launch(game_result_json=json.dumps(game_result, sort_keys=True))

    def get_game_result(self):
        """
        Get the result of the game.
        """
        game_result = {
            'challenge': self.game.__class__.__name__.lower(),
            'score': self.game.get_scores(),
            'replayformat': 'json',
            'replaydata': self.game.get_replay(),
            'playernames': ['Agent'] + [enemy.name for enemy in self.enemies]
        }
        return game_result

    def _make_observation(self, player_state_str):
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        obs[AntsEnv.IS_LAND_CHANNEL] = 1    # Set to all land.
        dead_ants = [[] for i in range(self.game.num_players)]
        for line in player_state_str.strip().split('\n'):
            line = line.strip().lower()
            tokens = line.split()
            key, (row, col) = tokens[0], map(int, tokens[1:3])
            owner = None if len(tokens) <= 3 else int(tokens[3])
            obs[AntsEnv.IS_VISIBLE_CHANNEL, row, col] = 1
            if key == 'w':
                obs[AntsEnv.IS_LAND_CHANNEL, row, col] = 0
            elif key == 'f':
                obs[AntsEnv.HAS_FOOD_CHANNEL, row, col] = 1
            elif owner == AntsEnv.AGENT_PLAYER_NUM:
                if key == 'h':
                    obs[AntsEnv.ANT_HILL_CHANNEL, row, col] = 1
                elif key == 'a':
                    obs[AntsEnv.AGENT_ANTS_CHANNEL, row, col] = 1
                elif key == 'd':
                    dead_ants[owner].append((row, col))
            elif owner is not None:
                if key == 'h':
                    obs[AntsEnv.ANT_HILL_CHANNEL, row, col] = 2
                elif key == 'a':
                    channel = AntsEnv.ENEMY_ANTS_CHANNEL_START + owner - 1
                    obs[channel, row, col] = 1
                elif key == 'd':
                    dead_ants[owner].append((row, col))
        return obs, dead_ants

    def _action_to_moves(self, action):
        moves = []
        for row in range(action.shape[0]):
            for col in range(action.shape[1]):
                if action[row, col] == AntsEnv.DONT_MOVE:
                    pass
                elif action[row, col] == AntsEnv.MOVE_NORTH:
                    moves.append(f'o {row} {col} n')
                elif action[row, col] == AntsEnv.MOVE_EAST:
                    moves.append(f'o {row} {col} e')
                elif action[row, col] == AntsEnv.MOVE_SOUTH:
                    moves.append(f'o {row} {col} s')
                elif action[row, col] == AntsEnv.MOVE_WEST:
                    moves.append(f'o {row} {col} w')
                else:
                    raise ValueError(f'action[{row}, {col}] = {action[row, col]} is not a valid move.')
        return moves

def test():
    map_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../../ants/maps/cell_maze/cell_maze_p06_04.map'
    )
    #map_file = '../../../ants/maps/example/tutorial1.map'
    opts = AntsEnvOptions()                     \
        .set_map_file(map_file)                 \
        .set_turns(500)                         \
        .set_view_radius_sq(81)                 \
        .set_attack_radius_sq(7)                \
        .set_consumption_radius_sq(3)           \
        .set_attack_mode('closest')             \
        .set_food_rate(10)                      \
        .set_food_turn(30)                      \
        .set_food_start(80)                     \
        .set_food_visible(10)                   \
        .set_food_spawn_mode('random')          \
        .set_scenario(True)

    agent = SampleBots.random_bot()
    enemies = [
        SampleBots.random_bot(), SampleBots.greedy_bot(),
        SampleBots.hunter_bot(), SampleBots.lefty_bot(),
        SampleBots.test_bot()
    ]
    reward_func = FoodScoreFunc()
    env = AntsEnv(opts, enemies, reward_func)

    state = env.reset()
    agent.setup(env.game.get_player_start(AntsEnv.AGENT_PLAYER_NUM))
    turns = 0
    while True:
        moves = agent.update_map(env.game.get_player_state(AntsEnv.AGENT_PLAYER_NUM)).get_moves()
        action = np.zeros((env.game.height, env.game.width))
        for move in moves:
            items = move.strip().split(' ')
            row, col, direction = int(items[1]), int(items[2]), items[3]
            if direction == 'n':
                direction = AntsEnv.MOVE_NORTH
            elif direction == 'e':
                direction = AntsEnv.MOVE_EAST
            elif direction == 's':
                direction = AntsEnv.MOVE_SOUTH
            elif direction == 'w':
                direction = AntsEnv.MOVE_WEST
            action[row, col] = direction
        state, reward, done, info = env.step(action)
        turns += 1
        if turns % 10 == 0 or reward != 0:
            print(f'Ran for {turns} turns. Last reward: {reward}')
        if done:
            print(info)
            break
    return env

if __name__ == '__main__':
    env = test()
    env.visualize()
