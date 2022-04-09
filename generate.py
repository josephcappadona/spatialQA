from motion import MotionGenerator
from orientation import OrientationGenerator
from distance import DistanceGenerator
from metaphor import MetaphorGenerator
from sklearn.metrics import accuracy_score as accuracy

import csv

def write_to_tsv(tsv, gen):
    
    for (p, h), e, (r_type, fn_name) in gen:
        print(*[p, h, e, r_type, fn_name])
        tsv_writer.writerow([p, h, e, r_type, fn_name])


if __name__ == '__main__':

    tsv_filename = "data.tsv"
    tsv_file = open(tsv_filename, 'w+', encoding='utf8', newline='')
    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    tsv_writer.writerow(['premise', 'hypothesis', 'entailment', 'reasoning_type', 'function_name'])
    
    motion_df = write_to_tsv(tsv_file, MotionGenerator())
    orientation_df = write_to_tsv(tsv_file, OrientationGenerator())
    distance_df = write_to_tsv(tsv_file, DistanceGenerator())
    metaphor_df = write_to_tsv(tsv_file, MetaphorGenerator())

    tsv_file.close()