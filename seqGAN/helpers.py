import torch
from torch.autograd import Variable
from math import ceil
from random import shuffle

def prepare_generator_batch(inputs, targets, start_letter, pad_id, gpu=False):
    """
    Takes a batch of inputs and targets and returns

    Inputs: inputs, targets, start_letter, cuda
        - inputs: batch_size x seq_len (Tensor with a tokenized article in each row)
        - targets: batch_size x seq_len (Tensor with a tokenized abstract in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as input, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as targets)
    """

    batch_size, i_seq_len = inputs.size()
    _, t_seq_len = targets.size()
    
    assert i_seq_len >= t_seq_len, "Is the abstract longer than the original?"

    inp = torch.zeros(batch_size, i_seq_len)
    inp[:, 0] = start_letter
    inp[:, 1:] = inputs[:, :i_seq_len-1]
    
    target = torch.ones(batch_size, i_seq_len) * pad_id
    target[:, :t_seq_len] = targets[:, :]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """    
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


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
    
#def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
#    s = batchwise_sample(gen, num_samples, batch_size)
#    oracle_nll = 0
#    for i in range(0, num_samples, batch_size):
#        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
#        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
#        oracle_nll += oracle_loss.data.item()
#
#    return oracle_nll/(num_samples/batch_size)

#def batchwise_nll(gen, inputs, targets, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
#    s = batchwise_sample(gen, num_samples, batch_size)
#    nll = 0
#    for i in range(0, num_samples, batch_size):
#        inp, target = prepare_generator_batch(, start_letter, gpu)