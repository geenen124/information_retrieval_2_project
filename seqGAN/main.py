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

# CUDA = torch.cuda.is_available()
#MAX_SEQ_LEN = 20 #TODO: check this
# BATCH_SIZE = 16#32
MLE_TRAIN_EPOCHS = 1#100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 100#10000

# GEN_EMBEDDING_DIM = 32
# GEN_HIDDEN_DIM = 32
# DIS_EMBEDDING_DIM = 64
# DIS_HIDDEN_DIM = 64


# VOCAB_SIZE = 50000
# VOCAB_PATH = "cnn-dailymail-master/finished_files/vocab"
# TRAIN_DATA_PATH = "cnn-dailymail-master/finished_files/chunked/train_*"
# TRAIN_DATA_PATH = "cnn-dailymail-master/finished_files/chunked/val_*"


def train_generator_MLE(gen, train_processor, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    
    # Use seq-seq model
    train_processor.train(gen, epochs, model_file_path=None)


def train_generator_PG(gen, gen_opt, dis, num_batches, inputs, targets, start_letter, pad_id):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    n_samples = BATCH_SIZE*2    # 64 works best

    for batch in range(num_batches):      
        input_samples, _ = helpers.random_from_data(inputs, targets, n_samples)
        s = gen.sample(input_samples)        
        inp, target = helpers.prepare_generator_batch(input_samples,
                                                      s, 
                                                      start_letter, 
                                                      pad_id,
                                                      gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
#    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
#                                                   start_letter, gpu=CUDA)
#
#    print(' oracle_sample_NLL = %.4f' % oracle_loss)


# def train_discriminator(discriminator, dis_opt, train_processor, generator, d_steps, epochs):
#     """
#     Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
#     Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
#     """
#     # Grab a random subset of inputs and their corresponding targets
#     n_samples = 100
#     input_samples, target_samples = helpers.random_from_data(inputs, targets, n_samples)

#     # generating a small validation set before training (using real data and generator)
#     pos_val = target_samples
#     neg_val = generator.sample(input_samples).type(torch.FloatTensor)
    
#     val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

#     for d_step in range(d_steps):
#         s = helpers.batchwise_sample(generator, inputs, targets, POS_NEG_SAMPLES, BATCH_SIZE).type(torch.FloatTensor)
#         dis_inp, dis_target = helpers.prepare_discriminator_data(target_samples, s, gpu=CUDA)
        
#         for epoch in range(epochs):
#             print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
#             sys.stdout.flush()
#             total_loss = 0
#             total_acc = 0

#             for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
#                 inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
#                 assert inp.size()[0] > 0, 'huh? ' + i
#                 dis_opt.zero_grad()
#                 out = discriminator.batchClassify(inp)
#                 loss_fn = nn.BCELoss()
#                 loss = loss_fn(out, target)
#                 loss.backward()
#                 dis_opt.step()

#                 total_loss += loss.data.item()
#                 total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

#                 if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
#                         BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
#                     print('.', end='')
#                     sys.stdout.flush()

#             total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
#             total_acc /= float(2 * POS_NEG_SAMPLES)

#             val_pred = discriminator.batchClassify(val_inp)
#             print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
#                 total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

# MAIN
if __name__ == '__main__':
    # Models
    gen = generator.Generator()
    # dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, gpu=CUDA)

    if config.use_gpu:
        gen = gen.cuda()
        # dis = dis.cuda()

    # Set up train, load data
    train_processor = TrainSeq2Seq()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    # gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, train_processor, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    # print('\nStarting Discriminator Training...')
    # dis_optimizer = optim.Adagrad(dis.parameters())
    # train_discriminator(dis, dis_optimizer, train_processor, gen, 50, 3)
    
    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))


    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
#    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
#                                               start_letter, gpu=CUDA)
#    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    # for epoch in range(ADV_TRAIN_EPOCHS):
    #     print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    #     # TRAIN GENERATOR
    #     print('\nAdversarial Training Generator : ', end='')
    #     sys.stdout.flush()
    #     train_generator_PG(gen, gen_optimizer, dis, 1, articles, abstracts, start_letter, pad_id)

    #     # TRAIN DISCRIMINATOR
    #     print('\nAdversarial Training Discriminator : ')
    #     train_discriminator(dis, dis_optimizer, articles, abstracts, gen, 5, 3)