import abc
import os
import shelve

import numpy as np
import torch

from types import SimpleNamespace

from ignite.engine.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, RunningAverage
from torch.utils.data import Dataset, DataLoader

from ..nets import netutils
from ... import utils

class ShelveDataset(Dataset):
    """
    Dataset which reads from a shelve where the keys are the index
    of an entry.

    Args:
        shelve_file (str): The file path containing the dataset shelve.
        transform (callable): Transforms to be applied to the data.
    """
    def __init__(self, shelve_file, transform=None):
        self.shelve_file = shelve_file
        self.transform = transform
        self._shelve = shelve.open(self.shelve_file, 'c')

    def __len__(self):
        return len(self._shelve) - 1

    def __getitem__(self, idx):
        item = self._shelve[str(idx)]
        if self.transform:
            item = self.transform(item)
        return item

    def close(self):
        self._shelve.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class DataLoaderTransformer(abc.ABC):
    """
    A dataloader like object that transforms the input dataloader.
    """
    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for batch in self._dataloader:
            for new_batch in self.transform(batch):
                yield new_batch

    @abc.abstractmethod
    def transform(self, batch):
        """
        Implement this function to specify the transform that
        should be performed on the dataloader. The function should
        return an iterator.

        Args:
            batch: A batch from the dataloader to transform.
        """
        pass


class GameSpaceToAntSpaceTransformer(DataLoaderTransformer):
    """
    Transforms the output of a dataloader (which returns batches
    containing the ants game state) into new batches by creating a
    subspace for each of the player's ants in a state. For example,
    if an state contains 32 ants, it is converted into 32
    subspaces for each ant.

    Args:
        sub_space_side (int): The length of the subspace state to
        be created around each ant.
        device (torch.device): The device to move the tensors to.
    """
    def __init__(self, dataloader, sub_space_side, device=None):
        super().__init__(dataloader)
        self.sub_space_side = sub_space_side
        self.device = device or torch.device('cpu')

    def transform(self, batch):
        result = SimpleNamespace(locs=[], state=[], action=[], reward=[], info=[])
        padder = netutils.WrapPadTransform(self.sub_space_side)
        batch_size = len(batch.state)
        for index in range(batch_size):
            state = batch.state[index]
            padded_state = torch.from_numpy(padder(state))
            locs = np.where(state[:, utils.antsgym.AntsEnv.CHANNEL_AGENT_ANT] == 1)
            for i, row, col in zip(*locs):
                result.locs.append((i, row, col))
                view = padded_state[
                    i, :,
                    row:(row + 2 * self.sub_space_side),
                    col:(col + 2 * self.sub_space_side)
                ]
                result.state.append(view)

                if hasattr(batch, 'action'):
                    action = batch.action[index]
                    result.action.append(action[i, row, col])

                if hasattr(batch, 'reward'):
                    num_ants = np.count_nonzero(locs[0] == i)
                    reward = batch.reward[index]
                    result.reward.append(reward[i])

                if hasattr(batch, 'info'):
                    info = batch.info[index]
                    new_info = self.transform_info(info, i, row, col)
                    result.info.append(new_info)

        for i in range(0, len(result.state), batch_size):
            ret = SimpleNamespace()
            ret.locs = result.locs[i:(i + batch_size)]

            ret.state = torch.stack(
                result.state[i:(i + batch_size)]
            ).float().to(self.device)

            if len(result.action) > 0:
                ret.action = torch.stack(
                    result.action[i:(i + batch_size)]
                ).long().to(self.device)

            if len(result.reward) > 0:
                ret.reward = torch.stack(
                    result.reward[i:(i + batch_size)]
                ).float().to(self.device)

            if len(result.info) > 0:
                ret.info = torch.stack(
                    result.info[i:(i + batch_size)]
                ).to(self.device)

            yield ret

    def transform_info(self, info, player_num, row, col):
        new_info = 0
        if hasattr(info, 'food_distr'):
            new_info += info.food_distr[player_num, player_num, row, col]
        if hasattr(info, 'raze_loc') and not any([x is None for x in info.raze_loc]):
            if info.raze_loc[0] - row <= 3 or info.raze_loc[1] - col <= 3:
                new_info += 100
        return torch.tensor(new_info, dtype=torch.float32)


class Trainer(abc.ABC):
    """
    Abstract class that manages the training, and evaluation of a network.
    """
    def __init__(self, ds_path, save_dir, view_radius, device):
        self.ds_path = ds_path
        self.device = device
        self.view_radius = view_radius
        self.net, self.net_name = self.define_net()
        self.net.to(device)
        self.optim = torch.optim.Adam(self.net.parameters())
        self.trainer = Engine(self.train_update_func)
        self.evaluator = Engine(self.val_update_func)
        self.saver = ModelCheckpoint(
            save_dir, self.net_name, save_interval=1,
            n_saved=20, require_empty=False
        )
        self._add_metrics()
        self._add_event_handlers()

    @abc.abstractmethod
    def define_net(self):
        """
        Implement this to return the network that will be used by the trainer.

        Returns:
            Tuple containing the network, and a string name for the network.
        """
        pass

    @abc.abstractmethod
    def get_loss(self, output, target):
        """
        Implement this to return the loss of the network.
        """
        pass

    @abc.abstractmethod
    def train_update_func(self, engine, batch):
        """
        The update step taken for each train iteration.
        """
        pass

    @abc.abstractmethod
    def val_update_func(self, engine, batch):
        """
        The step taken at each validation iteration.
        """
        pass

    @abc.abstractmethod
    def get_shelve_dataset(self, ds_path):
        """
        Return the torch dataset to use.
        """
        pass

    @abc.abstractmethod
    def get_test_action(self, network_output):
        """
        Returns the action to take for the ant given the network output.
        """
        pass

    def train(self, max_epochs, batch_size=32):
        self.net.train()
        ds_path = os.path.join(self.ds_path, 'train')
        with self.get_shelve_dataset(ds_path) as ds:
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=netutils.collate_namespace)
            loader = GameSpaceToAntSpaceTransformer(dl, self.view_radius, self.device)
            self.trainer.run(loader, max_epochs=max_epochs)

    def validate(self):
        self.net.eval()
        ds_path = os.path.join(self.ds_path, 'val')
        with self.get_shelve_dataset(ds_path) as ds, torch.no_grad():
            dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=netutils.collate_namespace)
            loader = GameSpaceToAntSpaceTransformer(dl, self.view_radius, self.device)
            self.evaluator.run(loader, max_epochs=1)

    def test(self, map_file=None, opponent=None, opponent_plays_first=False):
        map_file = map_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../../../ants/maps/example/tutorial_p2_1.map'
            # '../../../../ants/maps/cell_maze/cell_maze_p02_04.map'
        )
        opponent = opponent or utils.enemybots.SampleBots.greedy_bot()
        if opponent_plays_first:
            agent_num, opponent_num = 1, 0
            player_names = [opponent.name, self.net_name]
        else:
            agent_num, opponent_num = 0, 1
            player_names = [self.net_name, opponent.name]

        opts = utils.antsgym.AntsEnvOptions()   \
            .set_map_file(map_file)
        reward_func = utils.reward.ScoreFunc()
        env = utils.antsgym.AntsEnv(opts, [], reward_func, player_names)

        opponent = utils.antsgym.SampleAgent(opponent, env, opponent_num)
        opponent.setup()

        self.net.eval()
        state = env.reset()
        while True:
            transformer = GameSpaceToAntSpaceTransformer(
                [SimpleNamespace(state=[state[[agent_num]]])],
                self.view_radius, self.device
            )
            batch, batch_locs = [], []
            for substate in transformer:
                batch.append(substate.state)
                batch_locs.append(substate.locs[0])
            batch = torch.cat(batch)

            with torch.no_grad():
                out = self.net(batch)
                sample = self.get_test_action(out)
            agent_acts = np.zeros((1, env.game.height, env.game.width))
            for i, (_, row, col) in enumerate(batch_locs):
                agent_acts[0, row, col] = sample[i]

            opponent.update_map(state[opponent_num])
            opponent_acts = opponent.get_moves()

            if agent_num == 0:
                actions = np.concatenate([agent_acts, opponent_acts], axis=0)
            else:
                actions = np.concatenate([opponent_acts, agent_acts], axis=0)
            state, reward, done, info = env.step(actions)
            if done:
                break
        return env

    def benchmark(self, map_files, num_trials):
        opponents = [
            lambda: utils.enemybots.CmdBot.xanthis_bot(),
            lambda: utils.enemybots.SampleBots.random_bot(),
            lambda: utils.enemybots.SampleBots.lefty_bot(),
            lambda: utils.enemybots.SampleBots.hunter_bot(),
            lambda: utils.enemybots.SampleBots.greedy_bot()
        ]
        result = {}
        for map_file in map_files:
            result[map_file] = {}
            for opponent in opponents:
                for trial in range(num_trials):
                    o = opponent()
                    env = self.test(map_file, o, False)
                    game_results = env.get_game_result()
                    if o.name not in result[map_file]:
                        result[map_file][o.name] = {'wins': [0, 0], 'losses': [0, 0], 'turns': [0, 0]}
                    if game_results['score'][0] > game_results['score'][1]:
                        result[map_file][o.name]['wins'][0] += 1
                    elif game_results['score'][0] < game_results['score'][1]:
                        result[map_file][o.name]['losses'][0] += 1
                    result[map_file][o.name]['turns'][0] = len(game_results['replaydata']['scores'][0])
                    env.visualize()
                for trial in range(num_trials):
                    o = opponent()
                    env = self.test(map_file, o, True)
                    game_results = env.get_game_result()
                    if game_results['score'][1] > game_results['score'][0]:
                        result[map_file][o.name]['wins'][1] += 1
                    elif game_results['score'][1] < game_results['score'][0]:
                        result[map_file][o.name]['losses'][1] += 1
                    result[map_file][o.name]['turns'][1] = len(game_results['replaydata']['scores'][0])
                    env.visualize()
        return result

    def restore_net(self, net_path, epoch_num):
        """
        Loads the network parameters from a file.

        Args:
            net_path (str): The path to the pickled network params.
            epoch_num (int): The epoch at which the net was saved (in order to continue from here when saving later).
        """
        self.net.load_state_dict(torch.load(net_path))
        self.saver._iteration = epoch_num
        return self

    def _add_metrics(self):
        train_loss = RunningAverage(Loss(self.get_loss))
        train_loss.attach(self.trainer, 'avg_train_loss')

        val_loss = Loss(self.get_loss)
        val_loss.attach(self.evaluator, 'val_loss')

    def _add_event_handlers(self):
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self._print_train_metrics
        )
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED, lambda e: self.validate()
        )
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.saver,
            {'net': self.net}
        )
        self.evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, self._print_val_metrics
        )

    def _print_train_metrics(self, engine):
        if engine.state.iteration % 100 == 0:
            print(f'Epoch {engine.state.epoch}, iter {engine.state.iteration}. Loss: {engine.state.metrics["avg_train_loss"]}')

    def _print_val_metrics(self, engine):
        print(f'Val loss: {engine.state.metrics["val_loss"]}.')
