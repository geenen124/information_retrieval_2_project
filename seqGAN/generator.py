import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from training_ptr_gen.model import Model
from data_util import config
from helpers import random_from_data
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.seqseq_model = Model(model_file_path=None)

    def init_hidden(self, batch_size=1):
        # h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False) # ToDo: Do we want to initialize with the hidden state from random data?

        if config.use_gpu:
            return h.cuda()
        else:
            return h

    def sample(self, num_samples, vocab):
        """ 
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        batch = random_from_data(num_samples, vocab)

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, config.use_gpu)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, config.use_gpu)

        print(enc_batch)
        print(dec_batch)

        # samples = torch.zeros(num_samples, config.max_enc_steps).type(torch.LongTensor) # ToDo: Verify max seq length

        # h = self.init_hidden(num_samples)
        # inp = torch.LongTensor([start_letter]*num_samples)

        # if config.use_gpu:
        #     samples = samples.cuda()
        #     inp = inp.cuda()

        # for i in range(config.max_enc_steps):
        #     out, h = self.seqseq_model(inp, h)               # out: num_samples x vocab_size
        #     out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
        #     samples[:, i] = out.view(-1).data

        #     inp = out.view(-1)

        # return samples

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

