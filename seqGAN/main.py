from __future__ import print_function
from math import ceil
import sys

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

from trainer import TrainSeq2Seq

# BATCH_SIZE = 16#32
MLE_TRAIN_EPOCHS = 0#100
ADV_TRAIN_EPOCHS = 1#50


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
        generator_trainer.train_pg(num_batches=1)