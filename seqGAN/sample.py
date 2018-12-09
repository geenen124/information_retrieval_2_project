import numpy as np
from rouge import Rouge

rouge = Rouge()

batch_size = 4

orig_sent = [['Greetings .', "I am Rick .", 'I know .', "Yes , it is ."],
            ["Hello , there .", "How are you ?", "I am good .", "Thank you ."],
            ["Hi , there .", "What's up ?", "I am fine .", "Thanks ."],
            ["Hey , there .", "How are you doing ?", "I am doing great .", "Thank you ."]]

pred_sent = [['Greetings', '.', 'I', 'am', 'Rick', '.', 'I', 'Know', '.', 'Yes', ',', 'that', 'it', 'is', '.'],
            ['Hey', '.', 'How', 'are', 'you', '.', 'I', 'am', 'good', '.', 'Thank', 'you', '.'],
            ['Hi', '.', 'Wassup', '.', 'I', 'am', 'fine', '.', 'Thanks', '.'],
            ['Hello', '.', 'How', 'are', 'you', '.', 'I', 'am', 'doing', 'good', '.', 'Thank', 'you', '.']]

     
orig = []
pred = []
for i in range(len(pred_sent)):
    orig.append(' '.join(map(str, orig_sent[i])))
    pred.append(' '.join(map(str, pred_sent[i])))


scores = rouge.get_scores(orig, pred)
scores = [score['rouge-l']['f'] for score in scores]
print(scores)
print("\n\n")

new_pred = [[], [], [], []]
for i in range(len(pred_sent)):
    sents = []
    sent = pred_sent[i]
    count = 0
    while len(sent) > 0:
        try:
            idx = sent.index(".")
        except ValueError:
            idx = len(sent)

        if count > 0:
            new_pred[i].append(new_pred[i][count-1] + sent[:idx+1])
        else:
            new_pred[i].append(sent[:idx+1])
        sent = sent[idx+1:]
        count += 1

Pred = [[], [], [], []]
for i in range(len(pred_sent)):
    for j in range(len(new_pred[i])):
        Pred[i].append(' '.join(map(str, new_pred[i][j])))

out = [p[0] for p in Pred]
scores = rouge.get_scores(orig, out)
scores = [score['rouge-l']['f'] for score in scores]
print(scores)
out = [p[1] for p in Pred]
scores = rouge.get_scores(orig, out)
scores = [score['rouge-l']['f'] for score in scores]
print(scores)
out = [p[2] for p in Pred]
scores = rouge.get_scores(orig, out)
scores = [score['rouge-l']['f'] for score in scores]
print(scores)
out = [p[3] for p in Pred]
scores = rouge.get_scores(orig, out)
scores = [score['rouge-l']['f'] for score in scores]
print(scores)
