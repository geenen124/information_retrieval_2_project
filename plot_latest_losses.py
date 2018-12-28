import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import pickle
import os
import numpy as np

folder = '.'
if len(sys.argv) > 1:
    folder = sys.argv[1]

train_losses_file_prefix = 'train_run_avg_losses_'
val_losses_file_prefix = 'val_run_avg_losses_'
losses_file_suffix = '.p'

# Grab latest versions of the files
all_files = os.listdir(folder)
train_all_files = [f for f in all_files if f.startswith(train_losses_file_prefix) and f.endswith(losses_file_suffix)]
val_all_files = [f for f in all_files if f.startswith(val_losses_file_prefix) and f.endswith(losses_file_suffix)]
train_all_files = sorted(train_all_files)
val_all_files = sorted(val_all_files)

if len(train_all_files) == len(val_all_files):
    train_losses_file = train_all_files[-1]
else: # Grab the latest complete
    assert len(train_all_files) == len(val_all_files) + 1
    train_losses_file = train_all_files[-2]

train_losses_file = '{}/{}'.format(folder, train_losses_file)
train_losses = pickle.load(open(train_losses_file,'rb'))

val_losses = []
for f in val_all_files:
    file = '{}/{}'.format(folder, f)
    per_batch_losses = pickle.load(open(file,'rb'))
    val_losses.append(np.mean(per_batch_losses))

train_iterations = range(len(train_losses))
val_iterations = [10*v for v in range(len(val_losses))]

plt.plot(train_iterations, train_losses, color='blue')
plt.plot(val_iterations, val_losses, color='green')
plt.legend(('Train', 'Validation'))
plt.xlabel('Iteration')
plt.ylabel('Running avg loss')
plt.title('Loss curves')
#plt.show()
file_id = train_losses_file[train_losses_file.rfind('_')+1:-2]
fig_name = '{}/losses_curves_{}.png'.format(folder, file_id)
plt.savefig(fig_name)
print('File {} saved'.format(fig_name))
