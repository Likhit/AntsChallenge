import abc

class RewardInputs(object):
    """
    Struct to hold inputs to the reward function.
    """
    def __init__(self):
        self._old_state = None
        self._new_state = None
        self._ignored_moves = []
        self._invalid_moves = []
        self._dead_ants = []
        self._food_consumed = 0
        self._score = 0

    @property
    def old_state(self):
        """The old state before the current move."""
        return self._old_state

    @old_state.setter
    def old_state(self, value):
        self._old_state = value

    @property
    def new_state(self):
        """The new state after the current move."""
        return self._new_state

    @new_state.setter
    def new_state(self, value):
        self._new_state = value

    @property
    def ignored_moves(self):
        """The moves ignored by the game in the current step."""
        return self._ignored_moves

    @ignored_moves.setter
    def ignored_moves(self, value):
        self._ignored_moves = value

    @property
    def invalid_moves(self):
        """The moves that were invalid in the current step."""
        return self._invalid_moves

    @invalid_moves.setter
    def invalid_moves(self, value):
        self._invalid_moves = value

    @property
    def dead_ants(self):
        """List of location of ants that died this step (within the agent's view)."""
        return self._dead_ants

    @dead_ants.setter
    def dead_ants(self, value):
        self._dead_ants = value

    @property
    def food_consumed(self):
        """Number of food morsels consumed in this step."""
        return self._food_consumed

    @food_consumed.setter
    def food_consumed(self, value):
        self._food_consumed = value

    @property
    def score(self):
        """The agent's score at the end of the step."""
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

class RewardFunc(abc.ABC):
    """
    Implement this function to create a reward function which
    calculates the reward per turn.
    """
    @abc.abstractmethod
    def _calculate_reward(self, reward_inputs):
        """
        Calculates reward with custom logic based on the input.

        Attributes:
            - reward_inputs (RewardInputs): A named tuple of
                various properties of the game and agent at
                the current step to calculate the reward.

        Returns:
            reward (float): The net reward of this step.
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
    and the previous step.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._old_score = 1 # Score starts at 1 (# of hills).

    def _calculate_reward(self, reward_inputs):
        diff = reward_inputs.score - self._old_score
        self._old_score = reward_inputs.score
        return diff

class FoodScoreFunc(RewardFunc):
    """
    Reward is the difference between the score in the current step
    and the previous step.
    """
    def __init__(self, food_weight=1, score_weight=1):
        self.score_func = ScoreFunc()
        self.food_weight = food_weight
        self.score_weight = score_weight

    def reset(self):
        self.score_func.reset()

    def _calculate_reward(self, reward_inputs):
        diff = self.score_func(reward_inputs)
        return self.food_weight * reward_inputs.food_consumed \
            + self.score_weight * diff
