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
MLE_TRAIN_EPOCHS = 0#100
ADV_TRAIN_EPOCHS = 1#50

# DIS_EMBEDDING_DIM = 64
# DIS_HIDDEN_DIM = 64
    

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
    iter, running_avg_loss = generator_trainer.setup(gen, model_file_path=seq2seq_checkpoint_file)

    # GENERATOR MLE TRAINING - Pretrain
    print('Starting Generator MLE Training...')
    generator_trainer.train_nll(MLE_TRAIN_EPOCHS, iter, running_avg_loss)

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        # train_generator_PG(gen, 1, articles, abstracts, start_letter, pad_id)
        generator_trainer.train_pg(num_batches=1)

        # TRAIN DISCRIMINATOR
        # print('\nAdversarial Training Discriminator : ')
        # train_discriminator(dis, dis_optimizer, articles, abstracts, gen, 5, 3)