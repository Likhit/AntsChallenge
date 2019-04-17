import os
import shelve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

from ignite.engine.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from .. import nets
from ..data.traingen import ShelveDataset
from ... import utils

class PolicyDataset(ShelveDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return SimpleNamespace(
            state=item.state, action=item.action,
            players=item.players
        )


class PolicyNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.featurizer = nets.nets.ConvNet(input_shape)
        self.feat_out = self.featurizer.output_size
        self.hidden_size = self.feat_out // 8
        self.linear1 = nn.Linear(self.feat_out, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, 5)

    def forward(self, x):
        x = self.featurizer(x)
        x = x.view(-1, self.feat_out)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class PolicyNet2(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = 1
        for i in input_shape:
            self.input_shape *= i
        self.hidden_size1 = self.input_shape // 8
        self.hidden_size2 = self.hidden_size1 // 2
        self.linear1 = nn.Linear(self.input_shape, self.hidden_size1)
        self.linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 32)
        self.linear6 = nn.Linear(32, 5)

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        x = torch.tanh(self.linear5(x))
        return self.linear6(x)


class IterBox(object):
    def __init__(self, iter_gen):
        self.iter_gen = iter_gen

    def __iter__(self):
        return self.iter_gen()


class Trainer(object):
    def __init__(self, ds_path, view_radius, device):
        self.ds_path = ds_path
        self.device = device
        self.view_radius = view_radius
        self.net = PolicyNet((7, 2*view_radius, 2*view_radius)).to(device)
        self.optim = torch.optim.Adam(self.net.parameters())

        self.trainer = Engine(self._train_update_func)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda e: self._val_loss(e))

        self.evaluator = Engine(self._val_update_func)
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, self._log_val)

        self.saver = ModelCheckpoint(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'policy'
        ), 'model', save_interval=1, n_saved=10, require_empty=False)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.saver, {'net': self.net})

    def train(self, max_epochs, batch_size=32):
        self.net.train()
        with PolicyDataset(os.path.join(self.ds_path, 'train')) as ds:
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=nets.netutils.collate_namespace)
            loader = IterBox(lambda: (rebatch for batch in dl for rebatch in self.rebatch(batch, self.view_radius)))
            self.trainer.run(loader, max_epochs=max_epochs)

    def rebatch(self, batch, view_radius):
        """
        For each state in the batch, return a new batch with one
        element for every ant in each state.
        """
        result = SimpleNamespace(state=[], action=[])
        padder = nets.netutils.WrapPadTransform(view_radius)
        for state, action, players in zip(batch.state, batch.action, batch.players):
            padded_state = padder(state)
            for i in range(state.shape[0]):
                if players[i] != 'xanthis':
                    continue
                # Search for player ants in non-padded area.
                locs = np.where(padded_state[i, utils.antsgym.AntsEnv.CHANNEL_AGENT_ANT] == 1)
                for row, col in zip(*locs):
                    if row < view_radius or col < view_radius:
                        continue
                    elif row >= (state.shape[2] + view_radius) or col >= (state.shape[3] + view_radius):
                        continue
                    view = padded_state[i, :, (row - view_radius):(row + view_radius), (col - view_radius):(col + view_radius)]
                    result.state.append(view)
                    result.action.append(action[i, row - view_radius, col - view_radius])
        batch_size = len(batch.state)
        for i in range(0, len(result.state), batch_size):
            state = torch.from_numpy(np.stack(result.state[i:(i + batch_size)])).float().to(self.device)
            #state[state == 0] = -1
            action = torch.from_numpy(np.stack(result.action[i:(i + batch_size)])).long().to(self.device)
            yield SimpleNamespace(state=state, action=action)

    def _train_update_func(self, engine, batch):
        self.optim.zero_grad()
        out = self.net(batch.state)
        loss = self._loss(out, batch.action)
        loss.backward()
        self.optim.step()
        return loss

    def _loss(self, output, target):
        return F.cross_entropy(output, target)

    def _log(self, engine):
        if engine.state.iteration % 100 == 0:
            print(f'Epoch {engine.state.epoch}, iter {engine.state.iteration}. Loss: {engine.state.output}')

    def _val_update_func(self, engine, batch):
        out = self.net(batch.state)
        if engine.state.iteration == 1:
            engine.state.val_loss = 0
            engine.state.val_loss_n = 0
        engine.state.val_loss += self._loss(out, batch.action).item()
        engine.state.val_loss_n += 1
        return engine.state.val_loss

    def _log_val(self, engine):
        print(f'Val loss: {engine.state.val_loss / engine.state.val_loss_n}')
        engine.state.val_loss = 0
        engine.state.val_loss = 0

    def _val_loss(self, engine):
        self.net.eval()
        with PolicyDataset(os.path.join(self.ds_path, 'val')) as ds:
            dl = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=nets.netutils.collate_namespace)
            loader = IterBox(lambda: (rebatch for batch in dl for rebatch in self.rebatch(batch, self.view_radius)))
            self.evaluator.run(loader, max_epochs=1)

    def test(self):
        map_file = map_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../../../ants/maps/example/tutorial_p2_1.map'
            # '../../../../ants/maps/cell_maze/cell_maze_p02_04.map'
        )
        opts = utils.antsgym.AntsEnvOptions()   \
            .set_map_file(map_file)
        enemies = [utils.enemybots.SampleBots.greedy_bot()]
        reward_func = utils.reward.ScoreFunc()
        env = utils.antsgym.AntsEnv(opts, enemies, reward_func, ['policy'])
        self.env = env

        turn = 0
        state = env.reset()
        padder = nets.netutils.WrapPadTransform(self.view_radius)
        while True:
            batch_locs = []
            batch = []
            padded_state = padder(state)
            locs = np.where(padded_state[0, utils.antsgym.AntsEnv.CHANNEL_AGENT_ANT] == 1)
            for row, col in zip(*locs):
                if row < self.view_radius or col < self.view_radius:
                    continue
                elif row >= (state.shape[2] + self.view_radius) or col >= (state.shape[3] + self.view_radius):
                    continue
                batch_locs.append((row - self.view_radius, col - self.view_radius))
                batch.append(padded_state[0, :, (row - self.view_radius):(row + self.view_radius), (col - self.view_radius):(col + self.view_radius)])
            batch = torch.from_numpy(np.stack(batch)).float().to(self.device)

            out = F.softmax(self.net(batch), dim=1)
            sample = torch.distributions.Categorical(out).sample()
            actions = np.zeros((1, env.game.height, env.game.width))
            for i, (row, col) in enumerate(batch_locs):
                actions[0, row, col] = sample[i]

            state, reward, done, info = env.step(actions)
            turn += 1
            if done:
                break
        return env



# %load_ext autoreload
# %autoreload 2
# import trained_bots.likhit.supervised.policytrain as ptrain
# import torch

# device = torch.device('cuda')

# trainer = ptrain.Trainer(9, device)

# trainer.train('./trained_bots/likhit/data/train/policy/train', 9, 10)