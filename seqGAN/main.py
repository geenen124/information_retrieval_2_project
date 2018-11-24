from __future__ import print_function
from math import ceil
import sys

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

# from data_util.data import Vocab, example_generator, text_generator, abstract2sents, START_DECODING, STOP_DECODING, PAD_TOKEN
from trainer import TrainSeq2Seq
from data_util import config

#MAX_SEQ_LEN = 20 #TODO: check this
# BATCH_SIZE = 16#32
MLE_TRAIN_EPOCHS = 1#100
ADV_TRAIN_EPOCHS = 50

# DIS_EMBEDDING_DIM = 64
# DIS_HIDDEN_DIM = 64


def train_generator_PG(gen, num_batches, inputs, targets, start_letter, pad_id):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    n_samples = BATCH_SIZE*2    # 64 works best

    for batch in range(num_batches):      
        input_samples, _ = helpers.random_from_data(inputs, targets, n_samples)
        s = gen.sample(input_samples)        
        # inp, target = helpers.prepare_generator_batch(input_samples,
                                                      # s, 
                                                      # start_letter, 
                                                      # pad_id,
                                                      # gpu=config.use_gpu)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

# MAIN
if __name__ == '__main__':
    # Model
    gen = generator.Generator()

    if config.use_gpu:
        gen = gen.cuda()


    seq2seq_checkpoint_file = None

    # Load data
    generator_trainer = TrainSeq2Seq()
    # Prepare for training (e.g. optimizer)
    iter, running_avg_loss = generator_trainer.setup(gen, seq_model_file_path=seq2seq_checkpoint_file)

    # GENERATOR MLE TRAINING - Pretrain
    print('Starting Generator MLE Training...')
    generator_trainer.train_nll(MLE_TRAIN_EPOCHS, iter, running_avg_loss)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, 1, articles, abstracts, start_letter, pad_id)

        # TRAIN DISCRIMINATOR
        # print('\nAdversarial Training Discriminator : ')
        # train_discriminator(dis, dis_optimizer, articles, abstracts, gen, 5, 3)