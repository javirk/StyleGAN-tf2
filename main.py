from utils.model import StyleGAN
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

MAPPING_SIZE = 8
NOISE_SHAPE = [256, 256, 1]
OUT_DIM = 256
INPUT_SHAPE = [256, 256, 3]
ITERATIONS = 1000000
BATCH_SIZE = 1
LATENT_DIM = 512
INPUT_DIR = '../00. Datasets/'
INPUT_DIR = './'
DATASET = 'images.npy'
INPUT_DIM = INPUT_SHAPE[0]

noise_shape_batch = NOISE_SHAPE[:]
noise_shape_batch.insert(0, BATCH_SIZE)

ds = np.load(INPUT_DIR + DATASET).astype('f')
ds = ds / 255.0
ds = ds[0:2,...]

model = StyleGAN(INPUT_DIM, MAPPING_SIZE, NOISE_SHAPE, LATENT_DIM, OUT_DIM, INPUT_SHAPE)

model.train(ds, BATCH_SIZE, ITERATIONS)