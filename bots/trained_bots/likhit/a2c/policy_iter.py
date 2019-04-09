from math import floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import utils

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
        obs_shape = self.env.observation_space.shape
        act_shape = self.env.action_space.shape
        view_radius = int(self.env.game_opts.view_radius_sq ** 0.5)

        self.zero_pad = nn.ZeroPad2d(view_radius)
        shape = (obs_shape[2] + 2 * view_radius, obs_shape[3] + 2 * view_radius)
        self.conv1 = nn.Conv2d(obs_shape[1], 64, view_radius)
        shape = conv_output_shape(shape, view_radius)
        self.pool1 = nn.MaxPool2d(4, stride=1)
        shape = conv_output_shape(shape, 4)
        self.conv2 = nn.Conv2d(64, 32, 5)
        shape = conv_output_shape(shape, 5)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        shape = conv_output_shape(shape, 2)
        self.conv3 = nn.Conv2d(32, 5, 3)
        # self.conv3 = nn.Conv2d(32, 16, 3)
        # shape = conv_output_shape(shape, 3)
        #
        # self.hidden_in = 16 * shape[0] * shape[1]
        # out = act_shape[1] * act_shape[2]
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

    def forward(self, x):
        x = self._symmetric_pad(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.conv3(x)
        # x = x.view(-1, self.hidden_in)
        # x = self.fc(x)
        # x = x.view(-1, *self.out_shape)
        return F.softmax(x, dim=1)

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
        reward_func = utils.reward.FoodFunc()
        env = utils.antsgym.AntsEnv(opts, enemies, reward_func)
        return env

    def train(self, num_episodes, vis_every=1, lr=1e-3):
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        episode_results = []
        for episode in range(num_episodes):
            state, turn, is_done = self.env.reset(), 0, False
            self.optim.zero_grad()
            while not is_done:
                t_state = torch.tensor(state, dtype=torch.float32, device=self.device)
                action_probs = self.net(t_state)
                action = torch.distributions.Categorical(
                    probs=action_probs.permute(0, 2, 3, 1)
                ).sample()

                n_state, reward, is_done, info = self.env.step(action)

                target = self._get_target(action, state, reward, info)
                loss = self._get_loss(action_probs, target)
                loss.backward()

                if turn % 10 == 0 or is_done:
                    self.optim.step()
                    self.optim.zero_grad()
                    print(f'Episode {episode}, turn {turn}:  Reward: {reward}, Loss: {loss}')
                turn += 1
                state = n_state
            game_result = self.env.get_game_result()
            episode_results.append(game_result)
            if episode % vis_every == 0:
                self.env.visualize(game_result)
        return episode_results

    def _get_loss(self, action_probs, target):
        prod = torch.log(action_probs) * target
        return -prod.sum(1).mean(0).sum()

    def _get_target(self, action_taken, state, reward, info):
        target = torch.zeros(*(1, self.env.NUM_ACTIONS, *action_taken.shape[1:]), dtype=torch.float32, device=self.device)
        for row in range(action_taken.shape[1]):
            for col in range(action_taken.shape[2]):
                if info['reward_inputs'].ignored_moves[0][row, col] != 0:
                    target[0, action_taken[0, row, col], row, col] = -1
                if state[0, self.env.CHANNEL_AGENT_ANT, row, col] != 1:
                    target[0, self.env.ACTION_DONT_MOVE, row, col] = 1
                else:
                    target[0, action_taken[0, row, col], row, col] = info['food_distr'][0, 0, row, col]
        return F.softmax(target, dim=1)

def test(map_file, num_episodes, lr):
    device = torch.device('cuda')
    trainer = Trainer(map_file, device, lr)
    return trainer.train(num_episodes)
