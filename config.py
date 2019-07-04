from yacs.config import CfgNode as CN


cfg = CN()

cfg.batch_size = 64
cfg.val_batch_size = 1000
cfg.epochs = 10

cfg.optim = CN()
cfg.optim.type = 'SGD'
cfg.optim.args = CN()
cfg.optim.args.lr = 0.01
cfg.optim.args.momentum = 0.5

cfg.no_cuda = False
cfg.seed = 1
cfg.log_interval = 10
cfg.save_model = False

cfg.data_root = '~/data'
cfg.train_log_root = '~/train_log/pytorch-mnist'
cfg.exp_id = 'base'
cfg.log_subdir = 'log'
cfg.tb_subdir = 'runs'
cfg.ckpt_subdir = 'ckpt'
