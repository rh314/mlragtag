import random


class IdxSplitter():
    def __init__(self):
        self.num_rows = 0
        self.idxs = []
        self.batch_loader = None

    @classmethod
    def new_root(self, num_rows, init_idxs=None):
        obj = IdxSplitter()
        obj.num_rows = num_rows
        if init_idxs is None:
            obj.idxs = list(range(num_rows))
        else:
            obj.idxs = init_idxs
        return obj

    def split(self, fraction):
        left = int(fraction * self.num_rows)
        idxs_shuffled = self.idxs
        random.shuffle(idxs_shuffled)
        left_idxs = idxs_shuffled[:left]
        right_idxs = idxs_shuffled[left:]
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
