import json
import os
import sys

import gym
import numpy as np

from .enemybots import Bot, SampleBots, CmdBot
from .reward import RewardInputs, FoodFunc

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

    Args:
        - game_opts (AntsEnvOptions): Options to customize
            the game.
        - enemies ([Bot]): A list of enemy bots.
        - reward_func (RewardFunc): A RewardFunc object to
            calculate the reward at each step.
        - agent_names ([str]): A list of names for each agent.

    Properties:
        - observation_space: Box of shape (n, 5 + num_players,
            game_hight, game_width) of type np.uint8 where n is
            the number of agents. Most cell values are 0, or 1
            to represent True, or False respectively. There are
            (4 + num_player) channels for the following:
                - Channel 0: If the cell is visible to the agent.
                - Channel 1: If the cell is land (else water).
                - Channel 2: If the cell has food.
                - Channel 3: If the cell has the agent's ant.
                - Channel 4: If the cell has an ant hill,
                    then the player num (starting from 1) of the
                    hill owner. Else 0.
                - Channel 5: If the cell has an ant which died the
                    previous turn, then the player num (starting
                    from 1) of the dead ant. Else 0.
                - Channel 5 and greater: If the cell has an enemy
                    ant (one channel per enemy player).

        - action_space: Box of shape (n, game_height, game_width)
            with 0 <= cell value <= 4. 'n' is the number of agents. The cell value represents the move to make
            with the ant at each cell position for each agent:
                - 0: If the cell position has no player ants then
                    don't move.
                - 1: Move 1 step north.
                - 2: Move 1 step east.
                - 3: Move 1 step south.
                - 4: Move 1 step west.
    """
    # Observation channels
    CHANNEL_IS_VISIBLE = 0
    CHANNEL_IS_LAND = 1
    CHANNEL_HAS_FOOD = 2
    CHANNEL_ANT_HILL = 3
    CHANNEL_AGENT_ANT = 4
    CHANNEL_DEAD_ANTS = 5
    CHANNEL_ENEMY_ANT_START = 6

    # Action types
    NUM_ACTIONS = 5
    ACTION_DONT_MOVE = 0
    ACTION_MOVE_NORTH = 1
    ACTION_MOVE_EAST = 2
    ACTION_MOVE_SOUTH = 3
    ACTION_MOVE_WEST = 4

    def __init__(self, game_opts, enemies, reward_func, agent_names=None):
        self.game_opts = game_opts
        self.enemies = enemies
        self.reward_func = reward_func
        self.game = Ants(self.game_opts.as_dict())

        if len(self.enemies) >= self.game.num_players:
            raise ValueError('The number of enemies should be strictly less than the number of players. Otherwise there will be no agent to play against.')
        self.num_agents = self.game.num_players - len(self.enemies)
        if not agent_names:
            self.agent_names = [f'Agent {i}' for i in range(self.num_agents)]
        elif len(agent_names) != self.num_agents:
            raise ValueError(f'len(agent_names) == {len(agent_names)} should be 0 or equal to the number of agents == {self.num_agents}.')
        else:
            self.agent_names = agent_names

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_agents, 5 + self.game.num_players, self.game.height, self.game.width), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=0, high=4, shape=(self.num_agents, self.game.height, self.game.width), dtype=np.uint8)
        self.reset(self.game)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation,
        reward, done, info).

        Args:
            - action (object): An action provided by the
                environment. If one of the agents is no longer
                alive, but the game is still going on, then that
                player's action is ignored.

        Returns:
            observation (object): Agent's observation of the
                current environment.
            reward ([float]) : Amount of reward returned after
                previous action for each agent.
            done (boolean): Whether the episode has ended, in
                which case further step() calls will throw
                an exception.
            info ([object]): The metadata with wich the reward
                was calculated per agent.
        """
        if action.shape != self.action_space.shape:
            raise Exception(f'Shape of action {action.shape} should be same as shape of action space {self.action_space.shape}')
        if self.is_done:
            raise Exception("Can't make moves on a game that has finished. Reset board to continue.")

        reward_inputs = RewardInputs(self.num_agents) \
            .set_game(self.game)
        reward_inputs.set_old_state(np.copy(self._current_state))
        # Play moves for each bot in a random order

        self.game.start_turn()
        player_order = list(range(self.game.num_players))
        np.random.shuffle(player_order)
        for player_num in player_order:
            if not self.game.is_alive(player_num): continue
            if 0 <= player_num < self.num_agents:
                moves = self._action_to_moves(action[player_num])
            else:
                enemy_ind = player_num - self.num_agents
                moves = self.enemies[enemy_ind].get_moves()

            valid, ignored, invalid = self.game.do_moves(player_num, moves)
            if 0 <= player_num < self.num_agents:
                reward_inputs.set_ignored_moves(player_num, self._parse_bad_moves(ignored))
                reward_inputs.set_invalid_moves(player_num, self._parse_bad_moves(invalid))
        self.game.finish_turn()

        if self.game.game_over() or self.game.turn > self.game.turns:
            self.is_done = True
            self.game.finish_game()

        for i, enemy in enumerate(self.enemies):
            player_num = i + self.num_agents
            if self.game.is_alive(player_num):
                enemy.update_map(self.game.get_player_state(player_num))
        obs = self._get_observations()
        # Calculate reward
        reward_inputs.set_new_state(obs)
        reward, info = self.reward_func(reward_inputs)

        return obs, reward, self.is_done, info

    def reset(self, init_game=None):
        self.reward_func.reset()
        self.game = init_game or Ants(self.game_opts.as_dict())
        self.is_done = False
        self.game.start_game()
        for i, enemy in enumerate(self.enemies):
            enemy.reset()
            enemy.setup(self.game.get_player_start(i + self.num_agents))
            enemy.update_map(self.game.get_player_state(i + self.num_agents))
        obs = self._get_observations(reset=True)
        return obs

    def visualize(self, game_result=None):
        """
        Visualize a game till the current state.
        Opens the game in a new browser, and will start
        visualization from the start, so is not the same as
        render.

        Args:
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
            'playernames': [name for name in self.agent_names] + [enemy.name for enemy in self.enemies]
        }
        return game_result

    def _get_observations(self, reset=False):
        if reset:
            self._current_state = np.zeros(
                self.observation_space.shape,
                dtype=self.observation_space.dtype
            )
            # Set all cells to land.
            self._current_state[:, AntsEnv.CHANNEL_IS_LAND] = 1
        for i in range(self.num_agents):
            if not self.game.is_alive(i): continue
            state = self.game.get_player_state(i)
            self._update_observation(i, state)
        return np.copy(self._current_state)

    def _update_observation(self, agent_num, player_state_str):
        obs = self._current_state[agent_num]

        # clear all transient entities.
        obs[AntsEnv.CHANNEL_IS_VISIBLE] = 0
        obs[AntsEnv.CHANNEL_AGENT_ANT] = 0
        obs[AntsEnv.CHANNEL_DEAD_ANTS] = 0
        obs[AntsEnv.CHANNEL_ENEMY_ANT_START:] = 0
        obs[AntsEnv.CHANNEL_ANT_HILL] = 0
        obs[AntsEnv.CHANNEL_HAS_FOOD] = 0

        # update map
        for line in player_state_str.strip().split('\n'):
            line = line.strip().lower()
            tokens = line.split()
            key, (row, col) = tokens[0], map(int, tokens[1:3])
            owner = None if len(tokens) <= 3 else int(tokens[3])
            obs[AntsEnv.CHANNEL_IS_VISIBLE, row, col] = 1
            if key == 'w':
                obs[AntsEnv.CHANNEL_IS_LAND, row, col] = 0
            elif key == 'f':
                obs[AntsEnv.CHANNEL_HAS_FOOD, row, col] = 1
            elif 0 <= owner < self.num_agents:
                player_num = owner + 1
                if key == 'h':
                    obs[AntsEnv.CHANNEL_ANT_HILL, row, col] = player_num
                elif key == 'a':
                    obs[AntsEnv.CHANNEL_AGENT_ANT, row, col] = 1
                    self._update_visibility(obs, row, col)
                elif key == 'd':
                    obs[AntsEnv.CHANNEL_DEAD_ANTS, row, col] = player_num
            elif owner is not None:
                player_num = owner + 1
                if key == 'h':
                    obs[AntsEnv.CHANNEL_ANT_HILL, row, col] = owner
                elif key == 'a':
                    channel = AntsEnv.CHANNEL_ENEMY_ANT_START + owner - self.num_agents
                    obs[channel, row, col] = 1
                elif key == 'd':
                    obs[AntsEnv.CHANNEL_DEAD_ANTS, row, col] = player_num
        return obs

    def _action_to_moves(self, action):
        moves = []
        for row in range(action.shape[0]):
            for col in range(action.shape[1]):
                if action[row, col] == AntsEnv.ACTION_DONT_MOVE:
                    pass
                elif action[row, col] == AntsEnv.ACTION_MOVE_NORTH:
                    moves.append(f'o {row} {col} n')
                elif action[row, col] == AntsEnv.ACTION_MOVE_EAST:
                    moves.append(f'o {row} {col} e')
                elif action[row, col] == AntsEnv.ACTION_MOVE_SOUTH:
                    moves.append(f'o {row} {col} s')
                elif action[row, col] == AntsEnv.ACTION_MOVE_WEST:
                    moves.append(f'o {row} {col} w')
                else:
                    raise ValueError(f'action[{row}, {col}] = {action[row, col]} is not a valid move.')
        return moves

    def _update_visibility(self, obs, row, col):
        """
        Updates the IS_VISIBLE channel given the location of
        an ant.
        """
        h, w = obs.shape[1:]
        view_radius_sq = self.game_opts.view_radius_sq
        view_radius = int(view_radius_sq ** 0.5)
        offsets = []
        for off_row in range(-view_radius, view_radius + 1):
            for off_col in range(-view_radius, view_radius + 1):
                dist = off_row ** 2 + off_col ** 2
                if dist <= view_radius_sq:
                    obs[
                        AntsEnv.CHANNEL_IS_VISIBLE,
                        (row + off_row) % h,
                        (col + off_col) % w
                    ] = 1

    def _parse_bad_moves(self, lines):
        store = np.zeros(self.action_space.shape[1:], dtype=self.action_space.dtype)
        for line in lines:
            tokens = line.strip().split()
            row, col = int(tokens[1]), int(tokens[2])
            d = tokens[3]
            if d == 'n':
                d = 1
            elif d == 'e':
                d = 2
            elif d == 's':
                d = 3
            elif d == 'w':
                d = 4
            else:
                raise ValueError(f'Unexpected entry {line} in moves.')
            store[row, col] = d
        return store


class SampleAgent(Bot):
    """
    Use one of the sample bots as an agent.
    The expected input to setup(), and update_map() are now
    instances of AntsGym.observation_space, and the output of
    get_moves() is an instance of AntsGym.action_space.

    Args:
        - bot (SampleBots): Instance of the sample bot.
        - env (AntsGym): The environment from which the state
            will be obtained.
        - player_num (int): The player nuber of the agent in the env.
    """
    def __init__(self, bot, env, player_num):
        self.bot = bot
        self.env = env
        self.player_num = player_num
        name = self.env.agent_names[player_num]
        super().__init__(name)

    def setup(self):
        start = self.env.game.get_player_start(self.player_num)
        self.bot.setup(start)

    def update_map(self, state):
        # Discard the state for now.
        moves = self.env.game.get_player_state(self.player_num)
        self.bot.update_map(moves)

    def get_moves(self):
        moves = self.bot.get_moves()
        return self._moves_to_action(moves)

    def reset(self):
        self.bot.reset()
        return self

    def _moves_to_action(self, moves):
        action = np.zeros((1, *self.env.action_space.shape[1:]), dtype=self.env.action_space.dtype)
        for move in moves:
            items = move.lower().strip().split(' ')
            row, col, direction = int(items[1]), int(items[2]), items[3]
            if direction == 'n':
                direction = AntsEnv.ACTION_MOVE_NORTH
            elif direction == 'e':
                direction = AntsEnv.ACTION_MOVE_EAST
            elif direction == 's':
                direction = AntsEnv.ACTION_MOVE_SOUTH
            elif direction == 'w':
                direction = AntsEnv.ACTION_MOVE_WEST
            action[0, row, col] = direction
        return action


def test(map_file=None, turns=500, num_enemies=5, num_agents=1):
    map_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../../ants/maps/cell_maze/cell_maze_p06_04.map'
    )
    #map_file = '../../../ants/maps/example/tutorial_p2_1.map'
    opts = AntsEnvOptions()                     \
        .set_map_file(map_file)                 \
        .set_turns(turns)                       \
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

    winner_bot_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../winner_bots/first_place/'
    )
    winner_bot_cmd = f'java -cp {winner_bot_dir} MyBot'
    enemies = [
        CmdBot('xanthis', winner_bot_cmd),
        SampleBots.random_bot(), SampleBots.lefty_bot(),
        SampleBots.hunter_bot(), SampleBots.greedy_bot()
    ][:num_enemies]
    reward_func = FoodFunc()
    agent_names = [f'RandomAgent {i}' for i in range(num_agents)]
    env = AntsEnv(opts, enemies, reward_func, agent_names)

    agents = [SampleAgent(SampleBots.random_bot(), env, i) for i in range(num_agents)]
    for agent in agents:
        agent.setup()

    state = env.reset()
    turns = 0
    while True:
        actions = []
        for agent in agents:
            agent.update_map(state)
            actions.append(agent.get_moves())
        state, reward, done, info = env.step(np.concatenate(actions, axis=0))
        turns += 1
        if turns % 10 == 0 or any(reward != 0):
            print(f'Ran for {turns} turns. Last reward: {reward}')
        if done:
            break
    return env

if __name__ == '__main__':
    env = test()
    env.visualize()
