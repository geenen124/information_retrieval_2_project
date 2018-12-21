import sys
import pickle
import numpy as np

file_name = sys.argv[1]
x = pickle.load(open(file_name, 'rb'))
print('{} iterations'.format(len(x)))
print(x)

if 'val' in file_name:
    print('Mean average loss: {}'.format(np.mean(x)))
