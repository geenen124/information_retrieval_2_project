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
PG_TRAIN_EPOCHS = 10000


# MAIN
if __name__ == '__main__':

    seq2seq_checkpoint_file = "/home/lgpu0231/information_retrieval_2_project/Seq2Seq_model_50000"

    # Model
    model = Model(seq2seq_checkpoint_file)

    if config.use_gpu:
        model = model.cuda()

    # Load data
    trainer = TrainSeq2Seq()
    # Prepare for training (e.g. optimizer)
    iter, running_avg_loss = trainer.setup(model, model_file_path=seq2seq_checkpoint_file)

    # GENERATOR MLE TRAINING - Pretrain
    print('Starting Generator MLE Training...')
    #trainer.train_nll(MLE_TRAIN_EPOCHS, iter, running_avg_loss)

    # ADVERSARIAL TRAINING
    print('\nStarting PG Training...')
    trainer.train_pg(PG_TRAIN_EPOCHS)
