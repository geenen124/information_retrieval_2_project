from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import torch
from torch.utils.data import DataLoader

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.daily_mail_dataset import DailyMailDataset

from data_util.utils import calc_running_avg_loss
from rewards import get_sentence_rewards, get_word_level_rewards
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch, create_batch_collate
from training_ptr_gen.model import Model


import pickle
from rouge import Rouge

rouge = Rouge()

use_cuda = config.use_gpu and torch.cuda.is_available()


class Evaluate_pg(object):
    def __init__(self, model_file_path, is_word_level, is_combined, alpha):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        # self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
        #                        batch_size=config.batch_size, single_pass=True)
        self.dataset = DailyMailDataset("val", self.vocab)
        # time.sleep(15)
        model_name = os.path.basename(model_file_path)

        self.is_word_level = is_word_level
        self.is_combined = is_combined
        self.alpha = alpha

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)

        self.model = Model(model_file_path, is_eval=True)

    def compute_policy_grads_using_rewards(self, sentence_rewards, word_rewards, sentence_losses, word_losses, word_to_sent_ind):
        if self.is_combined:
            pg_losses = [[(self.alpha * word_reward + (1-self.alpha) * sentence_rewards[i][word_to_sent_ind[i][j]])* word_losses[i][j] for j, word_reward in enumerate(abstract_rewards) if j < len(word_to_sent_ind[i])] for i, abstract_rewards in enumerate(word_rewards)]
            pg_losses = [sum(pg) for pg in pg_losses]
        elif self.is_word_level:
            pg_losses = [[word_reward * word_losses[i][j] for j, word_reward in enumerate(abstract_rewards)] for i, abstract_rewards in enumerate(word_rewards)]
            pg_losses = [sum(pg) for pg in pg_losses]
        else:
            pg_losses = [[rs * sentence_losses[ri][rsi]  for rsi, rs in enumerate(r)] for ri, r in enumerate(sentence_rewards)]
            pg_losses = [sum(pg) for pg in pg_losses]
        return pg_losses

    def compute_pg_loss(self, orig, pred, sentence_losses, split_predictions, word_losses, word_to_sent_ind):
        sentence_rewards = None
        word_rewards = None
        # First compute the rewards
        if not self.is_word_level or self.is_combined:
            sentence_rewards = get_sentence_rewards(orig, pred)

        if self.is_word_level or self.is_combined:
            word_rewards = get_word_level_rewards(orig, split_predictions)

        pg_losses = self.compute_policy_grads_using_rewards(
            sentence_rewards=sentence_rewards,
            word_rewards=word_rewards,
            sentence_losses=sentence_losses,
            word_losses=word_losses,
            word_to_sent_ind=word_to_sent_ind
        )

        return pg_losses

    def compute_batched_loss(self, word_losses, orig, pred):
        orig_sum = []
        new_pred = []
        pred_sum = []
        sentence_losses = []

        # Convert the original sum as one single string per article
        for i in range(len(orig)):
            orig_sum.append(' '.join(map(str, orig[i])))
            new_pred.append([])
            pred_sum.append([])
            sentence_losses.append([])

        batch_sent_indices = []
        for i in range(len(pred)):
            sentence = []
            sentence = pred[i]
            losses = word_losses[i]
            sentence_indices = []
            count = 0
            while len(sentence) > 0:
                try:
                    idx = sentence.index(".")
                except ValueError:
                    idx = len(sentence)

                sentence_indices.extend([count for _ in range(idx)])

                if count>0:
                    new_pred[i].append(new_pred[i][count-1] + sentence[:idx+1])
                else:
                    new_pred[i].append(sentence[:idx+1])

                sentence_losses[i].append(sum(losses[:idx+1]))

                sentence = sentence[idx+1:]
                losses = losses[idx+1:]
                count += 1
            batch_sent_indices.append(sentence_indices)

        for i in range(len(pred)):
            for j in range(len(new_pred[i])):
                pred_sum[i].append(' '.join(map(str, new_pred[i][j])))

        pg_losses = self.compute_pg_loss(orig_sum, pred_sum,
                                         sentence_losses,
                                         split_predictions=pred,
                                         word_losses=word_losses,
                                         word_to_sent_ind=batch_sent_indices)

        return pg_losses

    def eval_one_batch(self, batch):
        batch_size = batch.batch_size

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        output_ids = []
        y_t_1 = torch.ones(batch_size, dtype=torch.long) * self.vocab.word2id(data.START_DECODING)

        if config.use_gpu:
            y_t_1 = y_t_1.cuda()

        for _ in range(batch_size):
            output_ids.append([])
            step_losses.append([])

        for di in range(min(max_dec_len, config.max_dec_steps)):
            #y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps) #NLL
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask

            # Move on to the next token
            _, idx = torch.max(final_dist, 1)
            idx = idx.reshape(batch_size, -1).squeeze()
            y_t_1 = idx

            for i, pred in enumerate(y_t_1):
                if not pred.item() == data.PAD_TOKEN:
                    output_ids[i].append(pred.item())

            for i, loss in enumerate(step_loss):
                step_losses[i].append(step_loss[i])

        # Obtain the original and predicted summaries
        original_abstracts = batch.original_abstracts_sents
        predicted_abstracts = [data.outputids2words(ids, self.vocab, None) for ids in output_ids]

        # Compute the batched loss
        batched_losses = self.compute_batched_loss(step_losses, original_abstracts, predicted_abstracts)
        losses = torch.stack(batched_losses)
        losses = losses / dec_lens_var

        loss = torch.mean(losses)

        return loss.item()

    def run_eval(self, model_dir, train_iter_id):
        dataloader = DataLoader(self.dataset, batch_size=config.batch_size,
                                shuffle=False, num_workers=1,
                                collate_fn=create_batch_collate(self.vocab, config.batch_size))
        running_avg_loss, iter = 0, 0
        start = time.time()
        # batch = self.batcher.next_batch()
        pg_losses = []
        run_avg_losses = []
        for batch in dataloader:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            print("Iteration:", iter, "  loss:", loss, "  Running avg loss:", running_avg_loss)
            iter += 1

            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                iter, print_interval, time.time() - start, running_avg_loss))
                start = time.time()

            pg_losses.append(loss)
            run_avg_losses.append(running_avg_loss)

        # Dump val losses
        pickle.dump(pg_losses, open(os.path.join(model_dir, 'val_pg_losses_{}.p'.format(train_iter_id)),'wb'))
        pickle.dump(run_avg_losses, open(os.path.join(model_dir, 'val_run_avg_losses_{}.p'.format(train_iter_id)),'wb'))

        return run_avg_losses


#if __name__ == '__main__':
 #   model_filename = sys.argv[1]
  #  eval_processor = Evaluate(model_filename)
   # eval_processor.run_eval()


