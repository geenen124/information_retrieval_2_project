from __future__ import print_function
from math import ceil
import sys

import torch
import torch.optim as optim
import torch.nn as nn

from training_ptr_gen.model import Model
from data_util import config
from trainer import TrainSeq2Seq

import pickle
import argparse

MLE_TRAIN_EPOCHS = 0#100
PG_TRAIN_EPOCHS = 10000


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-level", dest="is_word_level", action="store_true")
    parser.add_argument("--combined", dest="is_combined", action="store_true")
    parser.set_defaults(is_word_level=False)
    parser.set_defaults(is_combined=False)

    args = parser.parse_args()

    seq2seq_checkpoint_file = "./Seq2Seq_model_50000"
    pg_losses = []#pickle.load(open("/home/lgpu0231/dumps_model_12_16_11_08/pg_losses_350.p", 'rb'))
    run_avg_losses = []#pickle.load(open("/home/lgpu0231/dumps_model_12_16_11_08/run_avg_losses_350.p", 'rb'))

    # Model
    model = Model(seq2seq_checkpoint_file)
    # model = Model()

    # Load data
    trainer = TrainSeq2Seq(is_word_level=args.is_word_level, is_combined=args.is_combined)
    # Prepare for training (e.g. optimizer)
    iter, running_avg_loss = trainer.setup(model, model_file_path=None)

    # GENERATOR MLE TRAINING - Pretrain
    print('Starting Generator MLE Training...')
    #trainer.train_nll(MLE_TRAIN_EPOCHS, iter, running_avg_loss)

    # ADVERSARIAL TRAINING
    print('\nStarting PG Training...')
    trainer.train_pg(PG_TRAIN_EPOCHS, iter, running_avg_loss, pg_losses, run_avg_losses)
