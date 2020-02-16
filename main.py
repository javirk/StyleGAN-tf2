from utils.model import StyleGAN
import numpy as np

MAPPING_SIZE = 8
NOISE_SHAPE = [256, 256, 1]
OUT_DIM = 256
INPUT_SHAPE = [256, 256, 3]
ITERATIONS = 1
BATCH_SIZE = 8
INPUT_DIR = '../00. Datasets/'
INPUT_DIM = INPUT_SHAPE[0]

noise_shape_batch = NOISE_SHAPE[:]
noise_shape_batch.insert(0, BATCH_SIZE)

ds = np.load(INPUT_DIR + 'total_three_datasets_sorted_256.npy')

d_loss = []
g_loss = []
gp_loss = []

model = StyleGAN(INPUT_DIM, MAPPING_SIZE, NOISE_SHAPE, 256, OUT_DIM, INPUT_SHAPE)

model.train(ds.astype('f'), 8, 10000)