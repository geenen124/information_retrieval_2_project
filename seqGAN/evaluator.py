from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import torch

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch
from training_ptr_gen.model import Model

import pickle
from rouge import Rouge

rouge = Rouge()

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate_pg(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)

        self.model = Model(model_file_path, is_eval=True)

    def get_rouge_scores(self, ref_sum, pred_sum):
        scores = rouge.get_scores(pred_sum, ref_sum)
        f1_rL = [score['rouge-l']['f'] for score in scores]
        return f1_rL

    def get_rewards(self, orig, pred):
        rewards = []
	# We want reward of the whole sentence - reward of the sentence without sentence i

        for i in range(len(orig)):
            # Reward using the whole sentence
            total_score = self.get_rouge_scores(orig[i], ' '.join(pred[i]))[0]

            rewards.append([])
            rewards[i] = []

            for j in range(len(pred[i])):
                # sequence without sentence j
                sub_summary = [sen for idx,sen in enumerate(pred[i]) if idx != j] if len(pred[i]) > 1 else pred[i]

                score = self.get_rouge_scores(orig[i], ' '.join(sub_summary))[0]
                r_weight = ((total_score - score) / total_score) if total_score > 0 else 1

                rewards[i].append(r_weight)

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
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        pg_losses = []
        run_avg_losses = []
        while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            print("Iteration:", iter, "  loss:", loss, "  Running avg loss:", running_avg_loss)
            iter += 1

            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                iter, print_interval, time.time() - start, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()

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


