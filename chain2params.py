'''
chain2params.py
==============================
Extract parameters from trained model on chainer.
'''
from __future__ import print_function
import argparse
import os

try:
    import h5py
except ImportError:
    pass
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('chainer_model', help='Path to the trained model.')
args = parser.parse_args()

print('Loading model: {}'.format(args.chainer_model))
try:
    namedparams = []
    with h5py.File(args.chainer_model, 'r') as f:
        for group, data in f.iteritems():
            for name, param in data.iteritems():
                name = '{}_{}'.format(group, name)
                param = np.asarray(param)
                namedparams.append((name, param))
    print('Serialization type: hdf5')
except:
    namedparams = np.load(args.chainer_model).iteritems()
    print('Serialization type: npz')

save_dir = os.path.join(os.path.dirname(args.chainer_model), 'params')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

for name, param in namedparams:
    filename = os.path.join(save_dir, name.replace('/', '_'))
    np.save(filename, param)
print('Done')