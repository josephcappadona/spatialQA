from script_utils import configure_path
configure_path(__file__)

from generators.motion import MotionGenerator
from generators.orientation import OrientationGenerator
from generators.distance import DistanceGenerator
from generators.metaphor import MetaphorGenerator
from utils import append_to_tsv
import csv


if __name__ == '__main__':

    tsv_filename = "data.tsv"
    with open(tsv_filename, 'w+', encoding='utf8', newline='') as tsv_file:

        data_headers = ['premise', 'hypothesis', 'entailment', 'reasoning_type', 'function_name']
        
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(data_headers)
        
        motion_df = append_to_tsv(tsv_writer, MotionGenerator())
        orientation_df = append_to_tsv(tsv_writer, OrientationGenerator())
        distance_df = append_to_tsv(tsv_writer, DistanceGenerator())
        metaphor_df = append_to_tsv(tsv_writer, MetaphorGenerator())
        