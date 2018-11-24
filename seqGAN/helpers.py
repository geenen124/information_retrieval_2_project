import torch
from torch.autograd import Variable
from math import ceil
from random import shuffle


def batchwise_sample(gen, inputs, targets, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        input_samples, _ = random_from_data(inputs, targets, batch_size)
        samples.append(gen.sample(input_samples))

    return torch.cat(samples, 0)[:num_samples]

def random_from_data(inputs, targets, n_samples):
    indices = [i for i in range(len(inputs))]
    shuffle(indices)
    indices = indices[:n_samples]

    input_samples = torch.zeros(n_samples, inputs.shape[1])
    target_samples = torch.zeros(n_samples, targets.shape[1])
    
    for count, idx in enumerate(indices):
        input_samples[count] = inputs[idx]
        target_samples[count] = targets[idx]
    
    return input_samples, target_samples