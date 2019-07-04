import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import torch.utils.data
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from config import cfg
from model import Net
from utils import get_logger


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg

        self.init_env()
        self.init_device()
        self.init_data()
        self.init_model()
        self.init_optimizer()

    def init_env(self):
        self.exp_dir = Path(
            self.cfg.train_log_root).expanduser().joinpath(self.cfg.exp_id)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir.joinpath(self.cfg.log_subdir)
        self.tb_dir = self.exp_dir.joinpath(self.cfg.tb_subdir)
        self.ckpt_dir = self.exp_dir.joinpath(self.cfg.ckpt_subdir)

        self.logger = get_logger(__name__, self.log_dir)
        self.tb = SummaryWriter(self.tb_dir)
        torch.manual_seed(self.cfg.seed)

        self.epoch = 0
        self.acc = 0.

        self.logger.info('Train log location: {}'.format(self.exp_dir))

    def init_device(self):
        self.use_cuda = not self.cfg.no_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device('cuda')
            self.logger.info('Using gpu')
        else:
            self.device = torch.device('cpu')
            self.logger.info('Using cpu')

    def init_data(self):
        self.logger.info('Initializing data loader...')
        kwargs = {
            'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.cfg.data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=self.cfg.batch_size, shuffle=True, **kwargs)
        self.logger.info('Train loader has been initialized.')
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.cfg.data_root, train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=self.cfg.val_batch_size, shuffle=True, **kwargs)
        self.logger.info('Test loader has been initialized.')

    def init_model(self):
        self.model = Net()
        data, target = next(iter(self.train_loader))
        self.tb.add_graph(self.model, data)
        self.model = self.model.to(self.device)
        self.logger.info('Model has been initialized.')

    def init_optimizer(self):
        cfg_optim = self.cfg.optim
        optim_func = getattr(optim, cfg_optim.type)
        self.optimizer = optim_func(
            self.model.parameters(), **dict(self.cfg.optim.args))
        self.logger.info('Optimizer has been initialized.')

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_loss = loss.item()
            if batch_idx % self.cfg.log_interval == 0:
                self.logger.info(
                    '{:2d}, {}/{} loss: {:.6f}, test acc: {:.2f}%'.format(
                        self.epoch, batch_idx * len(data),
                        len(self.train_loader.dataset), loss.item(), self.acc))
                total_iter = self.epoch * len(self.train_loader) + batch_idx
                self.tb.add_scalar('train/loss', loss.item(), total_iter)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.val_loader.dataset)
        self.acc = 100. * correct / len(self.val_loader.dataset)
        self.logger.info(
            '{:2d}, test loss: {:.4f}, test acc: {}/{} ({:.2f}%)'.format(
                self.epoch, test_loss, correct, len(self.val_loader.dataset),
                self.acc))
        self.tb.add_scalar('test/acc', self.acc, self.epoch)
        self.tb.add_scalar('test/loss', test_loss, self.epoch)

    def load(self, for_resuming_training=True, label='latest'):
        ckpt_path = self.ckpt_dir.joinpath('{}.pt'.format(label))
        if ckpt_path.is_file():
            self.logger.info('Loading model from {}'.format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if for_resuming_training:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.epoch = ckpt['epoch'] + 1
                self.acc = ckpt['acc']
            self.logger.info(
                'Model of epoch {} loaded.'.format(ckpt['epoch']))
        else:
            self.logger.info('No checkpoint found.')

    def save(self, label='latest'):
        self.logger.info('Saving model...')
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_path = self.ckpt_dir.joinpath('{}.pt'.format(label))
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'acc': self.acc
            }, ckpt_path)
        self.logger.info('Model saved to {}.'.format(ckpt_path))

    def start(self):
        self.load(for_resuming_training=True)
        if self.epoch > 0:
            self.logger.info('Training start from epoch {}'.format(self.epoch))
        try:
            for self.epoch in range(self.epoch, self.cfg.epochs):
                self.train()
                self.test()
            self.logger.info('Training is finished.')
        except KeyboardInterrupt:
            self.logger.warning('Keyboard Interrupted.')
        except Exception as e:
            self.logger.exception(repr(e))
        finally:
            if self.epoch > 0:
                self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--exp', default='exps/default.yaml', help='Override parameters.')
    args = parser.parse_args()

    cfg.merge_from_file(args.exp)

    trainer = Trainer(cfg)
    trainer.start()
