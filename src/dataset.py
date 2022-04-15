import csv
from random import random

class SpatialNLIDataset:

    def __init__(self, data_filepath, sample=1.0):
        self.tsv_file = open(data_filepath, 'r', encoding='utf8', newline='')
        self.tsv_reader = csv.reader(self.tsv_file, delimiter='\t', lineterminator='\n')
        headers = next(self.tsv_reader)
        self.sample = sample

    def batch(self, batch_size=32, drop_last=False):
        cur_batch_x, cur_batch_y, cur_batch_m = [], [], []
        for p, h, e, r_type, fn_name, id_ in self.tsv_reader:
            if random() < self.sample:
                cur_batch_x.append((p, h))
                cur_batch_y.append(e)
                cur_batch_m.append((r_type, fn_name, id_))
            if len(cur_batch_x) == batch_size:
                yield cur_batch_x, cur_batch_y, cur_batch_m
                cur_batch_x, cur_batch_y, cur_batch_m = [], [], []
        if cur_batch_x and not drop_last:
            yield cur_batch_x, cur_batch_y, cur_batch_m
        
        self.tsv_file.close()