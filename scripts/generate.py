from script_utils import configure_path
configure_path(__file__)

from generators.containment import ContainmentGenerator
from generators.motion import MotionGenerator
from generators.orientation import OrientationGenerator
from generators.distance import DistanceGenerator
from generators.metaphor import MetaphorGenerator
from generators.containment import ContainmentGenerator
from utils import append_generator_to_tsv, data_headers
import csv


if __name__ == '__main__':

    tsv_filename = "data.tsv"
    with open(tsv_filename, 'w+', encoding='utf8', newline='') as tsv_file:

        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(data_headers)
        
        motion_df = append_generator_to_tsv(tsv_writer, MotionGenerator())
        orientation_df = append_generator_to_tsv(tsv_writer, OrientationGenerator())
        distance_df = append_generator_to_tsv(tsv_writer, DistanceGenerator())
        containment_df = append_generator_to_tsv(tsv_writer, ContainmentGenerator())
        metaphor_df = append_generator_to_tsv(tsv_writer, MetaphorGenerator())