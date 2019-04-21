import abc
import torch
#from AntsChallenge.ants import ants

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

class AllScoreFunc(RewardFunc):
    """
    Reward is the difference between the score in the current step
    and the previous step.
    """
    def __init__(self, food_weight=1, score_weight=1, attack_weight=1, defense_weight=-1,
                 kill_weight=1, die_weight=-1, collision_weight=-1, obstacle_weight=-1, exploration_weight=1):
        self.score_func = ScoreFunc()
        #self.food_func = FoodScoreFunc()
        self.score_weight = score_weight
        self.food_weight = food_weight
        self.attack_weight = attack_weight
        self.defense_weight = defense_weight
        self.kill_weight = kill_weight
        self.die_weight = die_weight
        self.collision_weight = collision_weight
        self.obstacle_weight = obstacle_weight
        self.exploration_weight = exploration_weight
        self.ant_count = 0

    def reset(self):
        self.score_func.reset()

    def distance(self, a_loc, b_loc, width, height):
        """ Returns distance between x and y squared """
        d_row = abs(a_loc[0] - b_loc[0])
        d_row = min(d_row, height - d_row)
        d_col = abs(a_loc[1] - b_loc[1])
        d_col = min(d_col, width - d_col)
        return d_row**2 + d_col**2

    def _calculate_reward_die(self, reward_inputs):
        reward = torch.zeros([1, reward_inputs.new_state.shape[1], reward_inputs.new_state.shape[2]],
                             dtype=torch.float32)
        for loc in reward_inputs.dead_ants[0]:
            reward[0][loc[0]][loc[1]] = 1
        return reward #len(reward_inputs.dead_ants[0])/self.ant_count

    def _calculate_reward_collision(self, reward_inputs):
        dead_ants = reward_inputs.dead_ants[0]
        reward = torch.zeros([1, reward_inputs.new_state.shape[1], reward_inputs.new_state.shape[2]],
                             dtype=torch.float32)
        loc_list = set([x for x in dead_ants if dead_ants.count(x) > 1])
        for loc in loc_list:
            reward[0][loc[0]][loc[1]] = 1
        return reward #(len(reward_inputs.dead_ants[0])-len(set(reward_inputs.dead_ants[0]))) / self.ant_count

    def _calculate_reward_obstacle(self, reward_inputs):
        reward = torch.zeros([1, reward_inputs.new_state.shape[1], reward_inputs.new_state.shape[2]],
                             dtype=torch.float32)
        water_locations = []
        min_dis = []
        ind_rewards = []
        distance_reward = 1
        #count_ants = 0
        #old_state = reward_inputs.old_state
        new_state = reward_inputs.new_state
        # agent_ants_locations_old = old_state[3]

        """ distance between water and agent ants """
        for row in range(new_state.shape[1]):
            for col in range(new_state.shape[2]):
                if new_state[1][row][col] == 1:
                    water_locations.append((row, col))
        for row in range(new_state.shape[1]):
            for col in range(new_state.shape[2]):
                if new_state[3][row][col] == 1:
                    #count_ants += 1
                    temp_min = max([new_state.shape[2], new_state.shape[1]])
                    for f_loc in water_locations:
                        curr_dis = self.distance(f_loc, (row, col), new_state.shape[1], new_state.shape[2])
                        if (curr_dis < temp_min):
                            temp_min = curr_dis
                    reward[0][row][col] = distance_reward/temp_min
                    #min_dis.append(temp_min)
        #for each in min_dis:
            #ind_rewards.append(distance_reward / each)
        #food_reward = sum(ind_rewards)
        #""" give reward to move towards closest food location (use old_state, new_state ) """
        #if self.ant_count==0: return 0
        return reward #(reward_inputs.food_consumed *0.75 + food_reward * 0.25)/self.ant_count


    def _calculate_reward_food(self, reward_inputs):
        reward = torch.zeros([1, reward_inputs.new_state.shape[1], reward_inputs.new_state.shape[2]],
                             dtype=torch.float32)
        food_locations = []
        min_dis = []
        ind_rewards = []
        distance_reward = 1
        count_ants = 0
        old_state = reward_inputs.old_state
        new_state = reward_inputs.new_state
        #agent_ants_locations_old = old_state[3]

        """ distance between food and agent ants """
        for row in range(new_state.shape[1]):
            for col in range(new_state.shape[2]):
                if new_state[2][row][col]==1:
                    food_locations.append((row, col))
        for row in range(new_state.shape[1]):
            for col in range(new_state.shape[2]):
                if new_state[3][row][col]==1:
                    count_ants+=1
                    temp_min = 999
                    for f_loc in food_locations:
                        curr_dis = self.distance(f_loc,(row,col), new_state.shape[1],new_state.shape[2])
                        if(curr_dis<temp_min):
                            temp_min = curr_dis
                    reward[0][row][col] = distance_reward/temp_min
                    #min_dis.append(temp_min)

        #for each in min_dis:
            #if(each!=0):
                #ind_rewards.append(distance_reward/each)
            #else:
            #    ind_rewards.append(distance_reward)
        #food_reward = sum(ind_rewards)
        #""" give reward to move towards closest food location (use old_state, new_state ) """ #TODO

        #self.ant_count = count_ants
        #if self.ant_count==0: return 0
        return reward #(reward_inputs.food_consumed *0.75 + food_reward * 0.25)/self.ant_count

    def __calculate_reward_exploration(self, reward_inputs):
        return #TODO

    def _calculate_reward(self, reward_inputs):
        print("in reward : l_161 : ", reward_inputs.dead_ants[0])
        food_reward = self._calculate_reward_food(reward_inputs)
        if self.ant_count==0: return -1
        diff = self.score_func(reward_inputs)
        die_reward = self._calculate_reward_die(reward_inputs)
        collision_reward = self._calculate_reward_collision(reward_inputs)

        total_reward = self.food_weight * food_reward \
            + self.score_weight * diff \
            + self.die_weight * die_reward \
            + self.collision_weight * collision_reward
        print("in reward : l_220 : ", total_reward)
        return total_reward


    """
    food_weight: 
    score_weight:
    attack_weight:
        - more agent ants wrt enemy ants
        - agent ants near enemy ant hill
    defense_weight:
        - negative reward if enemy ants surrounding are more
    kill_weight:
        - kill enemy ant
    die_weight:
        - negative for when agent dies because of enemy/water
    collision_weight:
        - negative, when it collides with itself 
    obstacle_weight:
        - cannot remain in same position for a long time
    
    """
