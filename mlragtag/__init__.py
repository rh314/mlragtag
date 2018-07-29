# Author(s): Ruan Havenstein
# License: MIT - see LICENSE file

import random
import numpy as np


class IdxSplitter():
    '''
    This class assists in repeatedly/hierarchically splitting a dataset.
    It splits using indexes instead of working with the entire dataset /
    data object.
    '''

    def __init__(self):
        self.num_rows = 0
        self.idxs = []
        self.batch_loader = None

    @classmethod
    def new_root(self, num_rows, init_idxs=None):
        obj = IdxSplitter()
        if init_idxs is None:
            obj.num_rows = num_rows
            obj.idxs = list(range(num_rows))
        else:
            obj.idxs = init_idxs[:]
            obj.num_rows = len(init_idxs)
        return obj

    def split(self, fraction):
        left = int(fraction * self.num_rows)
        idxs_shuffled = self.idxs[:]
        random.shuffle(idxs_shuffled)
        left_idxs = sorted(idxs_shuffled[:left])
        right_idxs = sorted(idxs_shuffled[left:])
        left_obj = IdxSplitter.new_root(left, left_idxs)
        right_obj = IdxSplitter.new_root(self.num_rows - left, right_idxs)
        return left_obj, right_obj

    def get_batch_idxs(self, batch_num_rows):
        num_rows = self.num_rows
        idxs_pre = [random.randrange(num_rows) for _ in range(batch_num_rows)]
        idxs = [self.idxs[_] for _ in idxs_pre]
        return idxs

    def get_batch(self, batch_num_rows):
        if self.batch_loader is None:
            raise Exception('Need to set a custom batch_loader first')
        idxs = self.get_batch_idxs(batch_num_rows)
        return self.batch_loader(idxs)


def ceil_power2(n):
    return 2 ** int(np.ceil(np.log2(n)))


def random_pad_2n(items, pad_value=None, pad_is_func=False):
    '''
    Randomly intersperses None into the list so that it's length is an integer
    power of 2.
    '''
    old_len = len(items)
    new_len = ceil_power2(old_len)
    mask = [1] * old_len + [0] * (new_len - old_len)
    np.random.shuffle(mask)
    mask_counter = -1
    if not pad_is_func:
        new_items = [pad_value] * new_len
    else:
        new_items = [pad_value() for _ in range(new_len)]
    for k, mask_item in enumerate(mask):
        if mask_item:
            mask_counter += 1
            new_items[k] = items[mask_counter]
    return new_items


def binary_fold(fold_fn, items):
    '''
    Repeatedly combine every two items into one using fold_fn until there is
    only one item left.
    '''
    if len(items) != ceil_power2(len(items)):
        raise Exception('Length of items must be a power of 2')
    items = items[:]
    while len(items) > 1:
        print(items)
        for k in range(len(items) // 2):
            items[k] = fold_fn(items[2 * k], items[2 * k + 1])
        del items[len(items) // 2:]
    print(items)
    return items[0]
