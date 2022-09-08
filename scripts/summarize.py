from sys import argv
import os
import pandas as pd
from statistics import stdev


model_names = {
    'ada': 'gpt-b',
    'babbage': 'gpt-m',
    'curie': 'gpt-l',
    'davinci': 'gpt-xl',

    'allenai-unifiedqa-v2-t5-small-1363200': 'unifiedqav2-s',
    'allenai-unifiedqa-v2-t5-base-1363200': 'unifiedqav2-b',
    'allenai-unifiedqa-v2-t5-large-1363200': 'unifiedqav2-l',
    'allenai-unifiedqa-v2-t5-3b-1363200': 'unifiedqav2-xl',

    't5-small': 't5-s',
    't5-base': 't5-b',
    't5-large': 't5-l',
    't5-3b': 't5-xl',

    'microsoft-deberta-base-mnli': 'deberta-b',
    'microsoft-deberta-large-mnli': 'deberta-l',
    'microsoft-deberta-xlarge-mnli': 'deberta-xl',
    'microsoft-deberta-v2-xxlarge-mnli': 'deberta-xxl',

    'textattack-roberta-base-MNLI': 'roberta-b',
    'roberta-large-mnli': 'roberta-l',

    'anirudh21-albert-large-v2-finetuned-mnli': 'albert-l',

    'textattack-xlnet-base-cased-MNLI': 'xlnet-b'
}

def df_to_latex(df, ignore_last=0, stdev=False):
    rows = []
    for _, row in df.iterrows():
        vals = [row[0]]
        valid = row[1:-ignore_last]
        zipped = zip(valid[::2], valid[1::2]) if stdev else zip(valid, valid)
        for x, sd in zipped:
            if stdev:
                vals.append(f"{x*100:.1f} ({sd*100:.1f})")
            else:
                vals.append(f"{x*100:.1f}")
        rows.append(' & '.join(vals))
    return '\n'.join(rows)

if __name__ == '__main__':
    
    summary_dir = argv[1]
    wopc_summary_df = pd.DataFrame(columns=['model_name', 'motion', 'orientation', 'distance', 'containment', 'metaphor', 'all'])
    wpc_summary_df = pd.DataFrame(columns=['model_name',
                                            'motion', 'motion_sd',
                                            'orientation', 'orientation_sd',
                                            'distance', 'distance_sd',
                                            'containment', 'containment_sd',
                                            'metaphor', 'metaphor_sd',
                                            'all', 'all_sd'])

    for model_name in model_names.keys():
        
        summary_filename = f'summary-{model_name}.tsv'
        summary_filepath = os.path.join(summary_dir, summary_filename)

        model_summary_df = pd.read_csv(summary_filepath, sep='\t', header=0)

        wopc_model_summary = {'model_name': model_names[model_name]}
        wpc_model_summary = {'model_name': model_names[model_name]}
        wopc_vals = []
        wpc_vals = []
        for i, row in model_summary_df.iterrows():
            wopc_model_summary[row['reasoning_type']] = row['acc_wo_partial_credit']
            wpc_model_summary[row['reasoning_type']] = row['acc_w_partial_credit']
            wpc_model_summary[row['reasoning_type'] + '_sd'] = row['stdev_acc_w_partial_credit']
            wopc_vals.append(row['acc_wo_partial_credit'])
            wpc_vals.append(row['acc_w_partial_credit'])
        wopc_model_summary['all'] = sum(wopc_vals) / len(wopc_vals)
        wpc_model_summary['all'] = sum(wpc_vals) / len(wpc_vals)
        wpc_model_summary['all_sd'] = stdev(wpc_vals)
        
        wopc_summary_df = pd.concat([wopc_summary_df, pd.DataFrame([wopc_model_summary])])
        wpc_summary_df = pd.concat([wpc_summary_df, pd.DataFrame([wpc_model_summary])])

    print(df_to_latex(wopc_summary_df, ignore_last=1))
    print(df_to_latex(wpc_summary_df, ignore_last=2, stdev=False))

    wopc_summary_filename = os.path.join(summary_dir, 'wopc-summary.tsv')
    wopc_summary_df.to_csv(wopc_summary_filename, sep='\t')
    print(f'WOPC summary dataframe output to {wopc_summary_filename}.')

    wpc_summary_filename = os.path.join(summary_dir, 'wpc-summary.tsv')
    wpc_summary_df.to_csv(wpc_summary_filename, sep='\t')
    print(f'WPC summary dataframe output to {wpc_summary_filename}.')