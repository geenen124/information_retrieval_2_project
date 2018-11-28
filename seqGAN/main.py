from __future__ import print_function
from math import ceil
import sys

import torch
import torch.optim as optim
import torch.nn as nn

from training_ptr_gen.model import Model
from data_util import config
from trainer import TrainSeq2Seq

MLE_TRAIN_EPOCHS = 0#100
PG_TRAIN_EPOCHS = 2#50


# MAIN
if __name__ == '__main__':
    # Model
    model = Model()

    if config.use_gpu:
        gen = gen.cuda()


    seq2seq_checkpoint_file = None

    # Load data
    trainer = TrainSeq2Seq()
    # Prepare for training (e.g. optimizer)
    iter, running_avg_loss = trainer.setup(model, model_file_path=seq2seq_checkpoint_file)

    # GENERATOR MLE TRAINING - Pretrain
    print('Starting Generator MLE Training...')
    trainer.train_nll(MLE_TRAIN_EPOCHS, iter, running_avg_loss)

    # ADVERSARIAL TRAINING
    print('\nStarting PG Training...')
    trainer.train_pg(PG_TRAIN_EPOCHS)

    # for epoch in range(PG_TRAIN_EPOCHS):
    #     print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    #     # TRAIN GENERATOR
    #     print('\nAdversarial Training Generator : ', end='')
    #     sys.stdout.flush()
    #     trainer.train_pg(num_batches=1)