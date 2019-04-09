import abc
import os
import sys

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

class Bot(abc.ABC):
    """
    Interface to implement for an enemy bot used in a gym
    environment for the Ants game.

    Attributes:
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

class SampleBots(Bot):
    """
    Implementation of the Bot interface for the sample bots.

    Attributes:
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
