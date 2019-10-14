# toke a look at and used them implementation of the data structure
# https://github.com/rlcode/per/blob/master/SumTree.py

import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.data = np.zeros(capacity, dtype=object)  # transitions
        self.tree = np.zeros(2 * capacity - 1)  # vector representation of tree
        self.position = 0
        self.capacity = capacity
        self.n_entries = 0

    # total sum of the priorities
    def total(self):
        return self.tree[0]

        # update to the root node

    def _propagate(self, idx, change):
        # recursion to update the sum in the tree till the root
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p  # updates node value
        self._propagate(idx, change)  # propagates recursively to parent nodes

        # store priority and sample

    def add(self, p, data):
        idx = self.position + self.capacity - 1  # get the leaf position

        self.data[self.position] = data
        self.update(idx, p)  # update parent nodes

        self.position += 1
        # reset the writing position to the beginning of the leaves
        if self.position >= self.capacity:
            self.position = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        priority = self.tree[idx]
        data = self.data[data_idx]
        return (idx, priority, data)
