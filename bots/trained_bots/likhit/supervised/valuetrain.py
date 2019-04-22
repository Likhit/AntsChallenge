import ignite
import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

from .utils import ShelveDataset, Trainer
from .. import nets

class ValueDataset(ShelveDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return SimpleNamespace(
            state=torch.from_numpy(item.state),
            action=torch.from_numpy(item.action),
            reward=torch.from_numpy(item.reward),
            info=item.info
        )

class ValueTrainer(Trainer):
    def __init__(self, ds_path, save_dir, view_radius, device, reward_scale=1):
        self.reward_scale = reward_scale
        super().__init__(ds_path, save_dir, view_radius, device)

    def train_update_func(self, engine, batch):
        self.optim.zero_grad()
        out = self.net(batch.state)
        out = out[torch.arange(out.shape[0]), batch.action]
        scaled_reward = batch.reward * self.reward_scale
        loss = self.get_loss(out, scaled_reward)
        loss.backward()
        self.optim.step()
        return out, scaled_reward

    def val_update_func(self, engine, batch):
        out = self.net(batch.state)
        scaled_reward = batch.reward * self.reward_scale
        return out[torch.arange(out.shape[0]), batch.action], scaled_reward

    def get_shelve_dataset(self, ds_path):
        return ValueDataset(ds_path)

    def _add_metrics(self):
        super()._add_metrics()
        val_acc = ignite.metrics.MeanAbsoluteError()
        val_acc.attach(self.evaluator, 'val_mae')

    def _print_val_metrics(self, engine):
        super()._print_val_metrics(engine)
        print(f'Val MAE: {engine.state.metrics["val_mae"]}')


class ShallowValue1Trainer(ValueTrainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [8, 8, 8, 3]
        net = nets.nets.ShallowNet1(shape, 3, filters)
        return net, f'ShallowValue1_r{self.view_radius}_s{self.reward_scale}'

    def get_loss(self, output, target):
        return F.smooth_l1_loss(output, target)

    def get_test_action(self, network_output):
        best_action = torch.argmax(network_output, dim=1)
        return best_action


class ShallowValue2Trainer(ShallowValue1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [8, 8, 8, 3]
        net = nets.nets.ShallowNet2(shape, filters)
        return net, f'ShallowValue2_r{self.view_radius}_s{self.reward_scale}'


class DeepValue1Trainer(ShallowValue1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [[8, 8, 8, 3], [8, 8, 8, 3]]
        net = nets.nets.DeepNet1(shape, 3, filters)
        return net, f'DeepValue1_r{self.view_radius}_s{self.reward_scale}'


class DeepValue2Trainer(ShallowValue1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [[8, 8, 8, 8], [8, 8, 8, 8]]
        net = nets.nets.DeepNet2(shape, filters)
        return net, f'DeepValue2_r{self.view_radius}_s{self.reward_scale}'


class MultiValueTrainer(Trainer):
    def __init__(self, ds_path, save_dir, view_radius, device, reward_scale=1):
        self.reward_scale = reward_scale
        super().__init__(ds_path, save_dir, view_radius, device)

    def train_update_func(self, engine, batch):
        self.optim.zero_grad()
        out = self.net(batch.state)
        scaled_reward = (batch.reward * self.reward_scale, batch.info * self.reward_scale)
        loss = self.get_loss((out, batch.action), scaled_reward)
        loss.backward()
        self.optim.step()
        return (out, batch.action), scaled_reward

    def val_update_func(self, engine, batch):
        out = self.net(batch.state)
        scaled_reward = (batch.reward * self.reward_scale, batch.info * self.reward_scale)
        return (out, batch.action), scaled_reward

    def get_shelve_dataset(self, ds_path):
        return ValueDataset(ds_path)

    def _add_metrics(self):
        loss = ignite.metrics.Loss(self.get_loss, batch_size=lambda x: x[0].shape[0])
        train_loss = ignite.metrics.RunningAverage(loss)
        train_loss.attach(self.trainer, 'avg_train_loss')

        val_loss = ignite.metrics.Loss(self.get_loss, batch_size=lambda x: x[0].shape[0])
        val_loss.attach(self.evaluator, 'val_loss')


class ShallowMultiValue1Trainer(MultiValueTrainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [8, 8, 8, 3]
        net = nets.nets.ShallowNet1(shape, 3, filters, 10)
        return net, f'ShallowMultiValue1_r{self.view_radius}_s{self.reward_scale}'

    def get_loss(self, output, target):
        out, act = output
        reward, info = target
        N = out.shape[0]
        score_out = out[:, :5][torch.arange(N), act]
        food_out = out[:, 5:][torch.arange(N), act]
        score_loss = F.smooth_l1_loss(score_out, reward)
        food_loss = F.smooth_l1_loss(food_out, info)
        return score_loss + food_loss

    def get_test_action(self, network_output):
        best_action = torch.argmax(network_output[:, :5], dim=1)
        return best_action


class ShallowMultiValue2Trainer(ShallowMultiValue1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [8, 8, 8, 8]
        net = nets.nets.ShallowNet2(shape, filters, 10)
        return net, f'ShallowMultiValue2_r{self.view_radius}_s{self.reward_scale}'


class DeepMultiValue1Trainer(ShallowMultiValue1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [[8, 8, 8, 3], [8, 8, 8, 3]]
        net = nets.nets.DeepNet1(shape, 3, filters, 10)
        return net, f'DeepMultiValue1_r{self.view_radius}_s{self.reward_scale}'


class DeepMultiValue2Trainer(ShallowMultiValue1Trainer):
    def define_net(self):
        shape = (7, 2 * self.view_radius, 2 * self.view_radius)
        filters = [[8, 8, 8, 8], [8, 8, 8, 8]]
        net = nets.nets.DeepNet2(shape, filters, 10)
        return net, f'DeepMultiValue2_r{self.view_radius}_s{self.reward_scale}'
