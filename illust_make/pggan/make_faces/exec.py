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
cfg['display_period'] = 1000
cfg['n_iters'] = 200000
cfg['n_critic'] = 1
cfg['learning_rate'] = 0.001
cfg['norm_g'] = 'pixel_norm'
cfg['norm_d'] = None
cfg['weight_scale'] = True
cfg['drift_loss'] = True
cfg['loss_mode'] = 'wgan_gp'
cfg['use_tanh'] = True
cfg['save_images'] = True

"""
cfg['resolution'] = 4
cfg['transition'] = False
cfg['load_model'] = None

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()


cfg['resolution'] = 8
cfg['transition'] = True
cfg['load_model'] = None

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()


cfg['batch_size'] = 48
cfg['resolution'] = 8
cfg['transition'] = False
cfg['load_model'] = None    # '8x8_transition'

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()


cfg['resolution'] = 16
cfg['transition'] = True
cfg['load_model'] = None

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()

cfg['resolution'] = 16
cfg['transition'] = False
cfg['load_model'] = '16x16_transition'

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()
"""
cfg['batch_size'] = 24
cfg['resolution'] = 32
cfg['transition'] = True
cfg['load_model'] = None

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()

cfg['resolution'] = 32
cfg['transition'] = False
cfg['load_model'] = '32x32_transition'

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()


cfg['batch_size'] = 16
cfg['resolution'] = 64
cfg['transition'] = True
cfg['load_model'] = None

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()

cfg['resolution'] = 64
cfg['transition'] = False
cfg['load_model'] = '64x64_transition'

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()


cfg['resolution'] = 128
cfg['transition'] = True
cfg['load_model'] = None

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()

cfg['resolution'] = 128
cfg['transition'] = False
cfg['load_model'] = '128x128_transition'

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()
"""
cfg['resolution'] = 256
cfg['transition'] = True
cfg['load_model'] = None

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()

cfg['resolution'] = 256
cfg['transition'] = False
cfg['load_model'] = '256x256_transition'

tf.reset_default_graph()
model = DCGAN(cfg)
model.train()
"""
