import numpy as np
import os
train_filter_head = np.load(os.path.join('../data/CDR/processed/train_filter_no_split.data.head.npy')).item()
train_filter_dep = np.load(os.path.join('../data/CDR/processed/train_filter_no_split.data.deprel.npy')).item()

dev_filter_head = np.load(os.path.join('../data/CDR/processed/dev_filter_no_split.data.head.npy')).item()
dev_filter_dep = np.load(os.path.join('../data/CDR/processed/dev_filter_no_split.data.deprel.npy')).item()

train_dev_filter_head = {**train_filter_head, **dev_filter_head}
train_dev_filter_dep = {**train_filter_dep, **dev_filter_dep}

for key in dev_filter_head.keys():
    if key in train_filter_head:
        print(key)
        assert (train_filter_head[key] == dev_filter_head[key]).all()

np.save(os.path.join('../data/CDR/processed/train+dev_filter_no_split.data.head.npy'), train_dev_filter_head)
np.save(os.path.join('../data/CDR/processed/train+dev_filter_no_split.data.deprel.npy'), train_dev_filter_dep)

