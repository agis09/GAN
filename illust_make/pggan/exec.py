from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import math

from config import cfg
from net import DCGAN
import tensorflow as tf

cfg['beta1'] = 0.
cfg['beta2'] = 0.99
cfg['batch_size'] = 16
cfg['save_period'] = 1000
cfg['display_period'] = 100
cfg['n_iters'] = 20000
cfg['n_critic'] = 1
cfg['learning_rate'] = 0.001
cfg['norm_g'] = 'pixel_norm'
cfg['norm_d'] = None
cfg['weight_scale'] = True
cfg['drift_loss'] = True
cfg['loss_mode'] = 'wgan_gp'
cfg['use_tanh'] = True
cfg['save_images'] = True

cfg['resolution'] = 8
cfg['transition'] = False
cfg['load_model'] = None

model = DCGAN(cfg)
model.train()

cfg.resolution = 16
cfg.transition = True
cfg.load_model = None

model = DCGAN(cfg)
model.train()

cfg.resolution = 16
cfg.transition = False
cfg.load_model = '16x16_transition'

model = DCGAN(cfg)
model.train()

cfg.resolution = 32
cfg.transition = True
cfg.load_model = None

model = DCGAN(cfg)
model.train()

cfg.resolution = 32
cfg.transition = False
cfg.load_model = '32x32_transition'

model = DCGAN(cfg)
model.train()

