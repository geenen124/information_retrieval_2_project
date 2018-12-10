from __future__ import unicode_literals, print_function, division

import os
import time
import argparse

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad, Adam
from torch.autograd import Variable

import numpy as np

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from data_util import config, data
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch

from rouge import Rouge

rouge = Rouge()


class TrainSeq2Seq(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        #time.sleep(15)

        train_dir = './train_log'
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup(self, seqseq_model, model_file_path):
        self.model = seqseq_model

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        #self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        self.optimizer = Adam(params, lr=initial_lr)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            print("Loading checkpoint .... ")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if config.use_gpu:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch_nll(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, config.use_gpu)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, config.use_gpu)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def train_nll(self, n_iters, iter, running_avg_loss):
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch_nll(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            print("Iteration:", iter, "  loss:", loss, "  Running avg loss:", running_avg_loss)
            iter += 1

            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 1000 == 0:
                self.save_model(running_avg_loss, iter)


    def train_pg(self, n_iters):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """

        pg_batcher = Batcher(config.train_data_path, self.vocab, mode='train',
            batch_size=config.batch_size, single_pass=False)

        time.sleep(15)

        start = time.time()
        running_avg_loss = 0

        for iter in range(n_iters):
            batch = pg_batcher.next_batch()
            loss = self.train_one_batch_pg(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            print("Iteration:", iter, "  PG loss:", loss, "  Running avg loss:", running_avg_loss)
            iter += 1

            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 1000 == 0:
                self.save_model(running_avg_loss, iter)


    def get_rouge_scores(self, ref_sum, pred_sum):
        scores = rouge.get_scores(ref_sum, pred_sum)
        f1_rL = [score['rouge-l']['f'] for score in scores]
        return f1_rL


    def get_rewards(self, orig, pred):
        rewards = []
        for i in range(len(orig)):
            rewards.append([])
            rewards[i] = []
            prev_score = None
            for j in range(len(pred[i])):
                if prev_score is None:
                    prev_score = self.get_rouge_scores(orig[i], pred[i][j])[0]
                    rewards[i].append(0)
                else:
                    score = self.get_rouge_scores(orig[i], pred[i][j])[0]
                    r_gain = (score - prev_score) / score if score > 0 else 0
                    rewards[i].append(r_gain)


        # TODO: Is the rouge gain correct reward? Or should it be 1-rouge)gain?
        return rewards


    def compute_pg_loss(self, orig, pred, sentence_losses):
        rewards = self.get_rewards(orig, pred)

        pg_losses = [[rs * sentence_losses[ri][rsi]  for rsi, rs in enumerate(r)] for ri, r in enumerate(rewards)]
        pg_losses = [sum(pg) for pg in pg_losses]

        return pg_losses



    def compute_batched_loss(self, word_losses, orig, pred):
        orig_sum = []
        new_pred = []
        pred_sum = []
        sentence_losses = []

        # Convert the original sum as one single string per batch
        for i in range(len(orig)):
            orig_sum.append(' '.join(map(str, orig[i])))
            new_pred.append([])
            pred_sum.append([])
            sentence_losses.append([])

        for i in range(len(pred)):
            sentence = []
            sentence = pred[i]
            losses = word_losses[i]
            count = 0
            while len(sentence) > 0:
                try:
                    idx = sentence.index(".")
                except ValueError:
                    idx = len(sentence)

                if count>0:
                    new_pred[i].append(new_pred[i][count-1] + sentence[:idx+1])
                else:
                    new_pred[i].append(sentence[:idx+1])

                sentence_losses[i].append(sum(losses[:idx+1]))

                sentence = sentence[idx+1:]
                losses = losses[idx+1:]
                count += 1

        for i in range(len(pred)):
            for j in range(len(new_pred[i])):
                pred_sum[i].append(' '.join(map(str, new_pred[i][j])))


        pg_losses = self.compute_pg_loss(orig_sum, pred_sum, sentence_losses)

        return torch.Tensor(pg_losses)


    def train_one_batch_pg(self, batch):
        batch_size = batch.batch_size

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, config.use_gpu)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, config.use_gpu)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        output_ids = []
        # Begin with START symbol
        y_t_1 = torch.ones(batch_size, dtype=torch.long) * self.vocab.word2id(data.START_DECODING)
        if config.use_gpu:
            y_t_1 = y_t_1.cuda()

        for _ in range(batch_size):
            output_ids.append([])
            step_losses.append([])

        for di in range(min(max_dec_len, config.max_dec_steps)):
            #y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps) # NLL
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask

            # Move on to next token
            _, idx = torch.max(final_dist, 1)
            idx = idx.reshape(batch_size, -1).squeeze()
            y_t_1 = idx

            for i, pred in enumerate(y_t_1):
                if not pred.item() == data.PAD_TOKEN:
                    output_ids[i].append(pred.item())

            # TODO: Check the correctness of this
            for i, loss in enumerate(step_loss):
                step_losses[i].append(step_loss[i].item())

        # Obtain the original and predicted summaries
        original_abstracts = batch.original_abstracts_sents
        predicted_abstracts = [data.outputids2words(ids, self.vocab, None) for ids in output_ids]

        # Compute the batched loss
        batched_losses = self.compute_batched_loss(step_losses, original_abstracts, predicted_abstracts)
        batched_losses = Variable(batched_losses, requires_grad=True)
        batched_losses = batched_losses / dec_lens_var

        loss = torch.mean(batched_losses)
        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()
