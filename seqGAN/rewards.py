from rouge import Rouge

rouge = Rouge()


def get_rouge_scores(ref_sum, pred_sum, rouge_type="rouge-l"):
    scores = rouge.get_scores(pred_sum, ref_sum)
    f1_rL = [score[rouge_type]['f'] for score in scores]
    return f1_rL


def get_word_level_rewards(orig, whole_preds):
    rewards = []
    for i in range(len(orig)):
        cached_rouge = {}
        # First get the ROUGE of the whole prediction
        score_whole_pred = get_rouge_scores(orig[i], " ".join(whole_preds[i]),
                                            rouge_type="rouge-1")[0]
        rewards.append([])

        for j in range(len(whole_preds[i])):
            if whole_preds[i][j] in cached_rouge:
                rewards[i].append(cached_rouge[whole_preds[i][j]])
            else:
                # entire prediction without token j
                sub_summary = whole_preds[i][:j] + whole_preds[i][j+1:]
                if len(sub_summary) > 0:
                    score = get_rouge_scores(orig[i], ' '.join(sub_summary))[0]
                else:
                    score = 0
                r_weight = ((score_whole_pred - score) / score_whole_pred) if score_whole_pred > 0 else 1
                cached_rouge[whole_preds[i][j]] = r_weight

                rewards[i].append(r_weight)
    return rewards


def get_sentence_rewards(orig, pred):
    rewards = []
    # We want reward of the whole sentence - reward of the sentence without sentence i
    for i in range(len(orig)):
        # Reward using the whole sentence
        total_score = get_rouge_scores(orig[i], ' '.join(pred[i]))[0]

        rewards.append([])

        for j in range(len(pred[i])):
            # sequence without sentence j
            # sub_summary = [sen for idx,sen in enumerate(pred[i]) if idx != j] if len(pred[i]) > 1 else pred[i]
            sub_summary = pred[i][:j]+pred[i][j+1:]
            if len(sub_summary) > 0:
                score = get_rouge_scores(orig[i], ' '.join(sub_summary))[0]
            else:
                score = 0
            r_weight = ((total_score - score) / total_score) if total_score > 0 else 1

            rewards[i].append(r_weight)

    return rewards
