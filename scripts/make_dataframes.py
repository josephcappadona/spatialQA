from script_utils import configure_path
configure_path(__file__)

from visualization import get_analysis_df, get_summary_df

if __name__ == '__main__':

    from sys import argv

    analysis_dir = argv[1]
    summary_dir = argv[2]

    print('Combining dataframes...')

    get_analysis_df(analysis_dir)
    get_summary_df(summary_dir)

    print('Dataframes created.')