from random import random

class BaseGenerator:

    names = ['John', 'Jane', 'Tom', 'Alice', 'Bob', 'Sonya']

    def __init__(self, sample=1.0):
        self.sample = sample

    def generate_all(self):
        for attr in dir(self):
            if attr.startswith('gen_'):
                gen_fn = self.__getattribute__(attr)
                for p, h, e in gen_fn():
                    if random() < self.sample:
                        yield p, h, e
    
    def __iter__(self):
        for p, h, e in self.generate_all():
            yield (p, h), e

    def batch(self, batch_size=32, drop_last=False, preprocess_x=None):
        cur_batch_x, cur_batch_y = [], []
        for x, y in self:
            cur_batch_x.append(x if preprocess_x is None else preprocess_x(x))
            cur_batch_y.append(y)
            if len(cur_batch_x) == batch_size:
                yield cur_batch_x, cur_batch_y
                cur_batch_x, cur_batch_y = [], []
        if cur_batch_x and not drop_last:
            yield cur_batch_x, cur_batch_y