import torch
import numpy as np
from collections import defaultdict

def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class Partition(object):
    def __init__(self, labels, n_way, n_shot, n_query):
        partition = defaultdict(list)
        cleaned_partition = {}
        for ind, label in enumerate(labels):
            partition[label].append(ind)
        for label in list(partition.keys()):
            if len(partition[label]) >= n_shot + n_query:
                cleaned_partition[label] = np.array(partition[label], dtype=np.int)
        self.partition = cleaned_partition
        self.subset_ids = np.array(list(cleaned_partition.keys()))

    def __getitem__(self, key):
        return self.partition[key]


class TaskLoader(object):
    def __init__(self, data, partitions, n_way, n_shot, n_query, cuda, length):
        self.data = data
        self.partitions = partitions
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.cuda = cuda
        self.length = length
        self.iter = 0

    def reset(self):
        self.iter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.iter == self.length:
            raise StopIteration
        self.iter += 1

        i_partition = torch.randint(low=0, high=len(self.partitions), size=(1,), dtype=torch.int)
        partition = self.partitions[i_partition]
        sampled_subset_ids = np.random.choice(partition.subset_ids, size=self.n_way, replace=False)
        xs, xq = [], []
        for subset_id in sampled_subset_ids:
            indices = np.random.choice(partition[subset_id], self.n_shot + self.n_query, replace=False)
            x = self.data[indices]
            xs.append(x[:self.n_shot])
            xq.append(x[self.n_shot:])
        xs = torch.stack(xs, dim=0)
        xq = torch.stack(xq, dim=0)
        if self.cuda:
            xs = xs.cuda()
            xq = xq.cuda()
        return {'xs': xs, 'xq': xq}
