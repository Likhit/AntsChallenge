import ignite
import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

from .utils import ShelveDataset, Trainer
from .. import nets

class PolicyDataset(ShelveDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        winner_inds = [i for i, p in enumerate(item.players) if p == 'xanthis']
        return SimpleNamespace(
            state=torch.from_numpy(item.state[winner_inds]),
            action=torch.from_numpy(item.action[winner_inds]),
        )

class PolicyTrainer(Trainer):
    def train_update_func(self, engine, batch):
        self.optim.zero_grad()
        out = self.net(batch.state)
        loss = self.get_loss(out, batch.action)
        loss.backward()
        self.optim.step()
        return out, batch.action

    def val_update_func(self, engine, batch):
        out = self.net(batch.state)
        return out, batch.action

    def get_shelve_dataset(self, ds_path):
        return PolicyDataset(ds_path)

    def _add_metrics(self):
        super()._add_metrics()
        val_acc = ignite.metrics.Accuracy()
        val_acc.attach(self.evaluator, 'val_acc')

    def _print_val_metrics(self, engine):
        super()._print_val_metrics(engine)
        print(f'Val acc: {engine.state.metrics["val_acc"]}')


class ShallowPolicy1Trainer(PolicyTrainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [8, 8, 8, 3]
        net = nets.nets.ShallowNet1(shape, 3, filters)
        return net, f'ShallowPolicy1_r{self.view_radius}'

    def get_loss(self, output, target):
        return F.cross_entropy(output, target)

    def get_test_action(self, network_output):
        softmax = F.softmax(network_output, dim=1)
        return torch.distributions.Categorical(softmax).sample()


class ShallowPolicy2Trainer(ShallowPolicy1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [8, 8, 8, 8]
        net = nets.nets.ShallowNet2(shape, filters)
        return net, f'ShallowPolicy2_r{self.view_radius}'


class DeepPolicy1Trainer(ShallowPolicy1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [[8, 8, 8, 3], [8, 8, 8, 3]]
        net = nets.nets.DeepNet1(shape, 3, filters)
        return net, f'DeepPolicy1_r{self.view_radius}'


class DeepPolicy2Trainer(ShallowPolicy1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [[8, 8, 8, 8], [8, 8, 8, 8]]
        net = nets.nets.DeepNet2(shape, filters)
        return net, f'DeepPolicy2_r{self.view_radius}'

