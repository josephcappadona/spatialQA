from script_utils import configure_path
configure_path(__file__)

import os


from visualization import make_figure_analysis, make_figure_summary

if __name__ == '__main__':

    from sys import argv
    df_analysis = argv[1]
    df_summary = argv[2]

    # Output file path
    output_dir = "figures"

    # Check if path have directory, if not create directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print('Creating figures')

    make_figure_analysis(df_analysis, output_dir)
    make_figure_summary(df_summary, output_dir)
    
    print('Figures created')
