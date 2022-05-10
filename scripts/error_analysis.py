from sys import argv
import os
import pandas as pd


model_names = {
    'davinci': 'gpt',
    'allenai-unifiedqa-v2-t5-3b-1363200': 'unifiedqav2',
    't5-3b': 't5',
    'microsoft-deberta-v2-xxlarge-mnli': 'deberta',
    'roberta-large-mnli': 'roberta',
    'anirudh21-albert-large-v2-finetuned-mnli': 'albert',
    'textattack-xlnet-base-cased-MNLI': 'xlnet'
}

if __name__ == '__main__':
    
    analysis_dir = argv[1]
    error_analysis_df = pd.DataFrame()

    for model_name in model_names.keys():
        
        analysis_filename = f'analysis-{model_name}.tsv'
        analysis_filepath = os.path.join(analysis_dir, analysis_filename)

        model_analysis_df = pd.read_csv(analysis_filepath, sep='\t', header=0)
        if error_analysis_df.size == 0:
            error_analysis_df['reasoning_type'] = model_analysis_df['reasoning_type']
            error_analysis_df['fn_name'] = model_analysis_df['fn_name']
            error_analysis_df['test_id'] = model_analysis_df['test_id']
            error_analysis_df['num_tests'] = model_analysis_df['num_tests']
            error_analysis_df['avg_test_acc'] = 0
        error_analysis_df[model_names[model_name]] = model_analysis_df['test_acc']
        error_analysis_df['avg_test_acc'] += model_analysis_df['test_acc']
    
    error_analysis_df['avg_test_acc'] /= len(model_names)
    error_analysis_df = error_analysis_df.sort_values('avg_test_acc')
    print(error_analysis_df)
    
    error_analysis_filename = os.path.join(analysis_dir, 'error-analysis.tsv')
    error_analysis_df.to_csv(error_analysis_filename, sep='\t')
    print(f'Error analysis dataframe output to {error_analysis_filename}.')