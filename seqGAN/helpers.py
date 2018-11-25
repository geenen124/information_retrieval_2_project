import torch
from torch.autograd import Variable
from math import ceil
import time
from data_util.batcher import Batcher
from data_util import config


# def batchwise_sample(gen, inputs, targets, num_samples, batch_size):
#     """
#     Sample num_samples samples batch_size samples at a time from gen.
#     Does not require gpu since gen.sample() takes care of that.
#     """

#     samples = []
#     for i in range(int(ceil(num_samples/float(batch_size)))):
#         input_samples, _ = random_from_data(inputs, targets, batch_size)
#         samples.append(gen.sample(input_samples))

#     return torch.cat(samples, 0)[:num_samples]

def random_from_data(n_samples, vocab):
    batcher = Batcher(config.train_data_path, vocab, mode='train',
                               batch_size=n_samples, single_pass=False) # check batch size

    time.sleep(15)

    return batcher.next_batch()