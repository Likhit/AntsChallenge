import abc
from types import SimpleNamespace

import numpy as np

class RewardInputs(object):
    """
    Struct to hold inputs to the reward function.
    """
    def __init__(self, num_agents):
        self.game = None
        self.num_agents = num_agents
        self.old_state = None
        self.new_state = None
        self.ignored_moves = [None] * num_agents
        self.invalid_moves = [None] * num_agents

    def set_old_state(self, state):
        """Sets the old state (state before the current move)."""
        self.old_state = state
        return self

    def set_new_state(self, state):
        """Sets the new state (state after the current move)."""
        self.new_state = state
        return self

    def set_ignored_moves(self, agent_num, moves):
        """
        The moves ignored by the game in the current step.
        Moves are ignored because they are blocked.
        """
        self.ignored_moves[agent_num] = moves
        return self

    def set_invalid_moves(self, agent_num, moves):
        """
        The moves that were invalid in the current step.
        Moves are invalide if there is no ant in the move location.
        """
        self.invalid_moves[agent_num] = moves
        return self

    def set_game(self, game):
        """
        Sets the game object associated with the states.
        """
        self.game = game
        return self


class RewardFunc(abc.ABC):
    """
    Implement this function to create a reward function which
    calculates the reward per turn.
    """
    @abc.abstractmethod
    def _calculate_reward(self, reward_inputs):
        """
        Calculates reward with custom logic based on the input.

        Args:
            - reward_inputs (RewardInputs): A named tuple of
                various properties of the game and agent at
                the current step to calculate the reward.

        Returns:
            reward (float): The net reward of this step.
            reward_info (dict): Meta-data used to calculate the reward.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Called when the environment is reset.
        """
        pass

    def __call__(self, reward_inputs):
        """
        Calculates the net reward against the input using
        _calculate_reward.
        """
        return self._calculate_reward(reward_inputs)


class ScoreFunc(RewardFunc):
    """
    Reward is the difference between the score in the current step
    and the previous step. RewardInfo is the locations of the
    hill which was razed (if reward was positive).
    """
    def __init__(self):
        self.reset()

    def reset(self):
         # Scores start at 1, so 0 means uninitialized.
        self._old_score = None

    def _calculate_reward(self, reward_inputs):
        if self._old_score is None:
            # Initialize old_score to number of hills owned by
            # player.
            self._old_score = np.asarray([len(reward_inputs.game.player_hills(i)) for i in range(reward_inputs.num_agents)])
        score = np.asarray([reward_inputs.game.score[i] for i in range(reward_inputs.num_agents)])
        diff = score - self._old_score
        self._old_score = score
        meta = []
        for i in range(reward_inputs.num_agents):
            meta.append(None)
            if diff[0] > 0:
                for hill in reward_inputs.game.hills.values():
                    if hill.end_turn == reward_inputs.game.turn and hill.killed_by == i:
                        meta[i] = hill.loc
        return diff, SimpleNamespace(inputs=reward_inputs, scores=meta)

class FoodFunc(RewardFunc):
    """
    Reward is the food collected at this timestep. RewardInfo
    returned is the food collected by each ant.

    Args:
        consume_weight (float): The reward to give for food that was sucessfully consumed.
        lost_weight (float): The reward/penalty to give for food that was sucessfully consumed by an enemy.
        denied_weight (float): The reward/penalty to give for food that was lost because of ant conflict.
    """
    def __init__(self, consume_weight=1, lost_weight=-1, denied_weight=0):
        self.consume_weight = consume_weight
        self.lost_weight = lost_weight
        self.denied_weight = denied_weight

    def reset(self):
        pass

    def _calculate_reward(self, reward_inputs):
        game = reward_inputs.game
        num_agents = reward_inputs.num_agents
        old_state = reward_inputs.old_state
        new_state = reward_inputs.new_state

        meta = np.zeros((
            num_agents, game.num_players + 1,
            game.height, game.width
        ))
        for agent, row, col in FoodFunc._get_consumed_food(old_state, new_state):
            for food in game.all_food:
                if food.loc == (row, col) and food.end_turn == game.turn:
                    r, w = food.loc
                    owner = -1 if food.owner is None else food.owner
                    meta[agent, owner, r, w] = 1
                    break

        meta[np.arange(num_agents), np.arange(num_agents)] *= self.consume_weight
        meta[:, -1] *= self.denied_weight
        for agent in range(num_agents):
            meta[agent][~np.isin(np.arange(game.num_players + 1), [agent, game.num_players - 1])] *= self.lost_weight

        info = SimpleNamespace(
            inputs=reward_inputs, food_distr=meta
        )
        return meta.sum(axis=tuple(range(1, len(meta.shape)))), info

    @staticmethod
    def _get_consumed_food(old_state, new_state):
        c = 2 # Food channel
        summation = old_state[:, c] + new_state[:, c]
        consumed = np.where((summation == 1))
        for agent, row, col in zip(*consumed):
            yield agent, row, col


class CompositeFunc(RewardFunc):
    """
    Reward is a linear combination of multiple reward fucntions.
    The returned info object contains the info of all individual
    scorers.

    Args:
        funcs ([(reard_func, float)]): A list of tuples for each
            reward function to use, along with the weight to give
            it.
    """
    def __init__(self, funcs):
        self.funcs = funcs

    def reset(self):
        for func, weight in self.funcs:
            func.reset()

    def _calculate_reward(self, reward_inputs):
        total = 0
        info = SimpleNamespace()
        for func, weight in self.funcs:
            reward, reward_info = func(reward_inputs)
            total += (weight * reward)
            setattr(info, type(func).__name__, reward_info)
        return total, info
