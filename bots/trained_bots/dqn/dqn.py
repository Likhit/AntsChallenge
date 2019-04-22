import sys
sys.path.append("..")

from math import floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

import AntsChallenge.bots.trained_bots.utils as utils

def conv_output_shape(shape, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(shape) is not tuple:
        shape = (shape, shape)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (shape[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (shape[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w

def convtransp_output_shape(shape, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(shape) is not tuple:
        shape = (shape, shape)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (shape[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (shape[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w

class Net1(nn.Module):
    def __init__(self, env, hidden=None):
        super().__init__()
        self.env = env
        self.hidden = hidden
        self._define_net()

    def _define_net(self):
        obs_shape = self.env.observation_space.shape    #(C,W,H)
        act_shape = self.env.action_space.shape     #(5,W,H)
        view_radius = int(self.env.game_opts.view_radius_sq ** 0.5)
        #view_radius = 7
        #print("in dqn l_69 : ", obs_shape, act_shape, view_radius)
        self.zero_pad = nn.ZeroPad2d(view_radius)
        shape = (obs_shape[1] + 2 * view_radius, obs_shape[2] + 2 * view_radius)
        self.conv1 = nn.Conv2d(obs_shape[0], 64, view_radius)
        shape = conv_output_shape(shape, view_radius)
        self.pool1 = nn.MaxPool2d(4, stride=1)
        shape = conv_output_shape(shape, 4)
        self.conv2 = nn.Conv2d(64, 32, 5)
        shape = conv_output_shape(shape, 5)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        shape = conv_output_shape(shape, 2)
        self.conv3 = nn.Conv2d(32, 5, 3)
        #self.fc3 = nn.Linear(8*8, )
        #print("dqn: 1_82", self.pool2)
        # self.conv3 = nn.Conv2d(32, 16, 3)
        # shape = conv_output_shape(shape, 3)
        #
        # self.hidden_in = 16 * shape[0] * shape[1]
        # out = act_shape[0] * act_shape[1]
        # if self.hidden is None:
        #     self.hidden = [2 * out, int(out * 1.5)]

        # fc = []
        # inp = self.hidden_in
        # for h in (self.hidden + [self.env.NUM_ACTIONS * out]):
        #     fc.append(nn.ReLU())
        #     fc.append(nn.Linear(inp, h))
        #     inp = h
        # self.fc = nn.Sequential(*fc)
        self.out_shape = (self.env.NUM_ACTIONS, *act_shape)
        #print("in dqn l_97 : ", self.out_shape)

    def forward(self, x):
        x = self._symmetric_pad(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #print("in dqn : l_106 : ", x.shape)
        x = self.conv3(x)

        # x = x.view(-1, self.hidden_in)
        # x = self.fc(x)
        # x = x.view(-1, *self.out_shape)
        return F.softmax(x, dim=1)

    def backward(self):
        pass

    def train_batch(self, state_batch):
        q_values = []
        for state in state_batch:
            q_values.append(self.forward(state))
        return q_values


    def _symmetric_pad(self, tensor):
        padded = self.zero_pad(tensor)
        l_p, r_p, t_p, b_p = self.zero_pad.padding

        left_pad = tensor[:, :, :, -l_p:]
        padded[:, :, t_p:-b_p, :l_p] += left_pad

        right_pad = tensor[:, :, :, :r_p]
        padded[:, :, t_p:-b_p, -r_p:] += right_pad

        top_pad = tensor[:, :, -t_p:, :]
        padded[:, :, :t_p, l_p:-r_p] += top_pad

        bottom_pad = tensor[:, :, :b_p, :]
        padded[:, :, -b_p:, l_p:-r_p] += bottom_pad
        return padded

class Trainer(object):
    def __init__(self, map_file, device):
        self.map_file = map_file
        self.device = device
        self.env = self._get_env()
        self.net = Net1(self.env).to(device)
        self.replay_memory = []

    def _get_env(self):
        opts = utils.antsgym.AntsEnvOptions()       \
            .set_map_file(self.map_file)            \
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

        enemies = [utils.enemybots.SampleBots.random_bot()]
        reward_func = utils.reward.AllScoreFunc(score_weight=250, food_weight=400, obstacle_weight=100, collision_weight=200)
        env = utils.antsgym.AntsEnv(opts, enemies, reward_func)
        return env

    def train(self, num_episodes, vis_every=1, lr=1e-5):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
        batch_size = 6
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        episode_results = []
        for episode in range(num_episodes):
            state, turn, is_done = self.env.reset(), 0, False
            self.optim.zero_grad()
            while not is_done:
                #print("dqn l_168", state.shape)
                t_state = torch.tensor([state], dtype=torch.float32, device=self.device)
                q_values = self.net(t_state)
                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                math.exp(-1. * turn/ EPS_DECAY)
                turn += 1
                #print("q l_178", q_values.shape)
                if sample > eps_threshold:
                    #exploration
                    action = torch.argmax(q_values.permute(0, 2, 3, 1), dim=3)[0]
                    #print("action for explore l_181", action[0].shape, action.shape)
                else:
                    #exploitation
                    action = torch.distributions.Categorical(
                        probs=q_values.permute(0, 2, 3, 1)
                    ).sample()[0]

                #print("in dqn l_189! : ", action, q_values)
                #indices = torch.LongTensor(action)
                #indices_ = torch.unsqueeze(indices, 0)
                #one_hot = torch.zeros(5, action.shape[0], action.shape[1])
                #print("in dqn l_192! : ", indices_.shape, indices_)
                #one_hot = one_hot.scatter_(0, indices_, 1)
                #torch.set_default_tensor_type('torch.DoubleTensor')
                indices = np.array(action)
                one_hot = np.eye(5)[indices]
                one_hot = one_hot.astype(np.float32)
                one_hot = torch.from_numpy(one_hot).permute(2, 0, 1)
                #one_hot = one_hot.to(dtype=torch.DoubleTensor)
                #one_hot = one_hot.astype(np.float32)
                #q_values = q_values.to(dtype=torch.DoubleTensor)
                #print("in dqn l_193! : ", one_hot.shape, one_hot.type, (q_values*one_hot).shape)
                q_values = torch.sum((q_values*one_hot), dim=1)
                #print("l_200 in DQN ", q_values.shape)
                n_state, reward, is_done, info = self.env.step(action)
                next_state = torch.tensor([n_state], dtype=torch.float32, device=self.device)
                #print("in dqn l_178 : ", next_state.shape, t_state.shape)
                n_q_values = self.net(next_state)
                max_n_q_values = torch.max(n_q_values, 1)
                print("in dqn l_167 : ", reward, q_values.shape, action.shape, state.shape, max_n_q_values[0].shape)
                target = self._get_target(max_n_q_values[0], reward, state, 0.99, info)
                #print("in dqn l_169 : ", target.shape, q_values.shape, action.shape, state.shape)
                loss = self._get_loss(q_values, target)
                loss.backward(retain_graph=True)
                if turn % 10 == 0 or is_done:
                    self.optim.step()
                    self.optim.zero_grad()
                    print(f'Episode {episode}, turn {turn}:  Reward: {reward}, Loss: {loss}')


                #experience replay - add into list
                self.replay_memory.append((t_state, next_state, reward, action, is_done))

                #sample from experiences
                batch_size = min(batch_size, len(self.replay_memory))
                samples = random.sample(self.replay_memory, batch_size)
                states_batch, next_states_batch, reward_batch, action_batch, done_batch = zip(*samples)
                #use these experiences to pass through our model
                n_q_values = self.net.train_batch(next_states_batch)
                targets = []
                for i in range(len(n_q_values)):
                    max_n_q_values = torch.max(n_q_values[i], 1)
                    target = self._get_target(max_n_q_values[0], reward_batch[i], next_states_batch[i], 0.9, info)
                    targets.append(target)
                    #update the loss after calculating target from experience replay
                    loss = self._get_loss(q_values, target)
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()

                turn += 1
                state = n_state
            game_result = self.env.get_game_result()
            episode_results.append(game_result)
            if episode % vis_every == 0:
                self.env.visualize(game_result)
        return episode_results

    def _get_loss(self, q_values, target):
        lossF = nn.SmoothL1Loss()
        return lossF.forward(q_values, target)
        #prod = torch.log(q_values) * target
        #return -prod.sum(1).mean(0).sum()

    def _get_target(self, max_n_q_values, reward, state, gamma, info):
        #target = torch.zeros(*(1, self.env.NUM_ACTIONS, *action.shape), dtype=torch.float32, device=self.device)
        target = reward + gamma * max_n_q_values
        return target

    '''
    def _get_target(self, action, state, reward, info):   #shape is (1, 5, H, W)
        target = torch.zeros(*(1, self.env.NUM_ACTIONS, *action.shape), dtype=torch.float32, device=self.device)
        #actions = q_values[0]
        
        for row in range(q_values.shape[0]):
            for col in range(q_values.shape[1]):
                if state[self.env.AGENT_ANTS_CHANNEL, row, col] != 1:
                    target[0, self.env.DONT_MOVE, row, col] = 1
                else:
                    num_agent_ants = np.count_nonzero(state[self.env.AGENT_ANTS_CHANNEL] == 1)
                    target[0, action_taken[row, col], row, col] = reward / num_agent_ants
        for line in info.get(self.env.INFO_AGENT_IGNORED_MOVES, []):
            tokens = line.strip().split()
            row, col = int(tokens[1]), int(tokens[2])
            target[0, action_taken[row, col], row, col] = -1
        return F.softmax(target, dim=1)
        
        for row in range(q_values.shape[2]):
            for col in range(q_values[3]):
                target[:,]
    '''
def test(map_file, num_episodes, lr):
    device = torch.device('cpu')
    trainer = Trainer(map_file, device)
    return trainer.train(num_episodes, lr)

if __name__ == '__main__':
    sys.stdout.write("in dqn l_267 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    env = test("C:/Users/apoor/PyCharmProjects/DL/AntsChallenge/ants/maps/cell_maze/cell_maze_p02_04.map", 2000, 0.3)
    print("dqn l_211 : ", type(env))
    #env.visualize()




