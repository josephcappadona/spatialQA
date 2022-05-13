#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join

# Function that reads in model names and declares model group and model size. 
def model_name_size(model, path):
    temp = pd.read_csv(path + model, delim_whitespace = True, encoding = "ISO-8859-1")
    #print(model)
    if 'allenai-unifiedqa-v2-t5' in model:
        if 'base' in model:
            temp['size'] = 220000000
        elif 'small' in model:
            temp['size'] = 60000000
        elif 'large' in model:
            temp['size'] = 770000000
        elif '3b' in model:
            temp['size'] = 3000000000
        temp['model'] = 'UnifiedQA'
    elif 'anirudh21-albert-large-v2-finetuned-mnli' in model:
        temp['size'] = 17000000
        temp['model'] = 'ALBERT'
    elif 'microsoft-deberta' in model:
        if 'base' in model:
            temp['size'] = 86000000
        elif 'deberta-large' in model:
            temp['size'] = 350000000
        # error with data, temporary delete
        elif 'deberta-xlarge' in model:
            temp['size'] = 700000000
        elif 'deberta-v2-xxlarge' in model:
            temp['size'] = 1320000000
        temp['model'] = 'DeBERTa'
    elif 'cross-encoder-nli-deberta' in model:
#         if 'base' in model:
#             temp['size'] = 86
#         elif 'xsmall' in model:
#             temp['size'] = 22
#         elif 'small' in model:
#             temp['size'] = 44
#         elif 'large' in model:
#             temp['size'] = 350
#         temp['model'] = 'SNLI&MNLI'
        temp['size'] = np.NaN
        temp['model'] = np.NaN
    elif 'cross-encoder-nli-roberta' in model:
        temp['size'] = np.NaN
        temp['model'] = 'cross-encoder-roberta'
    elif 'roberta-large-mnli' in model:
        temp['size'] = 355000000
        temp['model'] = 'RoBERTa'
    elif 'textattack-roberta-base-MNLI' in model:
        temp['size'] = 125000000
        temp['model'] = 'RoBERTa'
    elif 't5' in model:
        if 'base' in model:
            temp['size'] = 220000000
        elif 'small' in model:
            temp['size'] = 60000000
        elif 'large' in model:
            temp['size'] = 770000000
        elif '3b' in model:
            temp['size'] = 3000000000
        temp['model'] = 'T5'
    elif 'textattack-xlnet-base-cased-MNLI' in model:
        temp['size'] = 110000000
        temp['model'] = 'XLNet'
    elif 'ada' in model:
        temp['size'] = 2700000000
        temp['model'] = 'GPT'
    elif 'babbage' in model:
        temp['size'] = 6700000000
        temp['model'] = 'GPT'
    elif 'curie' in model:
        temp['size'] = 13000000000
        temp['model'] = 'GPT'
    elif 'davinci' in model:
        temp['size'] = 175000000000
        temp['model'] = 'GPT'
    else: # ynie models
        temp['size'] = np.NaN
        temp['model'] = np.NaN
    return temp

def get_largest(df):
    df_largest = pd.DataFrame()
    for model in list(df.model.unique()):
        df = df[df['size'].notna()]
        max_size = df[df['model'] == model]['size'].max()
        temp = df[df['size'] == max_size]
        df_largest = pd.concat([df_largest, temp], ignore_index=True)
    return df_largest

def get_analysis_df(analysis_file_path):

    # File path
    analysis_path = analysis_file_path

    # Read all file names 
    analysis_files = [f for f in listdir(analysis_path) if isfile(join(analysis_path, f))]

    # remove existing dataframe if exist
    if 'df_analysis.csv' in analysis_files:
        analysis_files.remove('df_analysis.csv')

    # remove '.DS_Store' file
    if '.DS_Store' in analysis_files:
        analysis_files.remove('.DS_Store')
        
    if 'error-analysis.tsv' in analysis_files:
        analysis_files.remove('error-analysis.tsv')

    # Combine all files to a dataframe
    df_analysis = pd.DataFrame()
    for model in analysis_files:
        temp = model_name_size(model, analysis_path)
        df_analysis = pd.concat([df_analysis, temp], ignore_index=True)
        
    # Output file to folder
    df_analysis.to_csv(analysis_file_path + '/df_analysis.csv', index=False)


def get_summary_df(summary_file_path):

    # File path
    summary_path = summary_file_path

    # Read all file names 
    summary_files = [f for f in listdir(summary_path) if isfile(join(summary_path, f))]

    # remove existing dataframe if exist
    if 'df_summary.csv' in summary_files:
        summary_files.remove('df_summary.csv')

    # remove '.DS_Store' file
    if '.DS_Store' in summary_files:
        summary_files.remove('.DS_Store')
        
    if 'wopc-summary.tsv' in summary_files:
        summary_files.remove('wopc-summary.tsv')

    if 'wpc-summary.tsv' in summary_files:
        summary_files.remove('wpc-summary.tsv')

    # Combine all files to a dataframe
    df_summary = pd.DataFrame()
    for model in summary_files:
        temp = model_name_size(model, summary_path)
        df_summary = pd.concat([df_summary, temp], ignore_index=True)
    
    # Output file to folder
    df_summary.to_csv(summary_file_path + '/df_summary.csv', index=False)


def make_figure_analysis(df_path, output_dir):

    df = pd.read_csv(df_path)
    df = df[df['model'].notna()]

    analysis_mean = df.groupby(['model','reasoning_type']).mean().reset_index()

    # Model vs Mean accuracy by different model.png
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    analysis_mean_plt = sns.catplot(data=analysis_mean, x='model', y='test_acc', hue='reasoning_type', 
                                    alpha=.8, kind='bar', height=6, legend=True, aspect=2);

    analysis_mean_plt.despine(left=True)
    analysis_mean_plt.set_axis_labels("Models", "Mean accuracy")
    analysis_mean_plt.legend.set_title("reasoning_type")
    analysis_mean_plt.set_xticklabels(rotation=10, fontsize=20)
    analysis_mean_plt.set(ylim=(0, 1))
    analysis_mean_plt.savefig('{}/Model vs Mean accuracy by different model.png'.format(output_dir))


    # Box plot of Reasoning type vs Test accuracy
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_1 = sns.boxplot(x='reasoning_type', y='test_acc', data=df, palette='vlag')
    sns.stripplot(x='reasoning_type', y='test_acc', data=df,size=3, color='.3')
    boxplt_1.set_xticklabels(boxplt_1.get_xticklabels(), rotation=20, ha="right", fontsize=15)
    boxplt_1.set(xlabel='Reasoning type', ylabel='Test accuracy')
    boxplt_1.figure.savefig('{}/Box plot of Reasoning type vs test accuracy.png'.format(output_dir), bbox_inches='tight')
    
    # Box plot of Reasoning type vs Test accuracy with the largest size
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_largest_size = sns.boxplot(x='reasoning_type', y='test_acc', data=get_largest(df), palette='vlag')
    sns.stripplot(x='reasoning_type', y='test_acc', data=df,size=3, color='.3')
    boxplt_largest_size.set_xticklabels(boxplt_1.get_xticklabels(), rotation=20, ha="right", fontsize=15)
    boxplt_largest_size.set(xlabel='Reasoning type', ylabel='Test accuracy')
    boxplt_largest_size.figure.savefig('{}/Box plot of Reasoning type vs Test accuracy with the largest size'.format(output_dir), bbox_inches='tight')


    # Box plot of Model vs Test accuracy
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_2 = sns.boxplot(x='model', y='test_acc', data=df,palette='vlag')
    boxplt_2 = sns.stripplot(x='model', y='test_acc', data=df,size=3, color='.3')
    boxplt_2.set_xticklabels(boxplt_2.get_xticklabels(), rotation=20, ha="right", fontsize=15)
    boxplt_2.set(xlabel='Model', ylabel='Test accuracy')
    boxplt_2.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit.png'.format(output_dir), bbox_inches='tight')


    # Box plot of Reasoning type vs Test accuracy with hue = Model
    plt.figure(figsize=(20,3), dpi=120)
    sns.set(font_scale=2)
    boxplt_3 = sns.boxplot(x='model', y='test_acc', data=df,palette='vlag', hue='reasoning_type')
    boxplt_3.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
    boxplt_3.set_xticklabels(boxplt_3.get_xticklabels(), rotation=10, ha="right", fontsize=20)
    boxplt_3.set(xlabel='Model', ylabel='Test accuracy')
    boxplt_3.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit with hue = Model.png'.format(output_dir), bbox_inches='tight')


    # Box plot of Model vs Test accuracy with hue = Reasoning type
    plt.figure(figsize=(20,3), dpi=120)
    sns.set(font_scale=2)
    boxplt_4 = sns.boxplot(x='reasoning_type', y='test_acc', data=df,palette='vlag', hue='model')
    boxplt_4.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
    boxplt_4.set(xlabel='Reasoning type', ylabel='Test accuracy')
    boxplt_4.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit with hue = Reasoning type.png'.format(output_dir), bbox_inches='tight')



def make_figure_summary(df_path, output_dir):

    df = pd.read_csv(df_path)

    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    # Model vs averaged Accuracy without partial credit among all size
    plt_1 = sns.catplot(data=df[df['size'].notna()], kind="bar", 
                    x="model", y="acc_wo_partial_credit", 
                    hue="reasoning_type", ci=None, alpha=.8, 
                    height=6, legend_out=True, aspect = 2
                    )
    plt_1.despine(left=True)
    plt_1.set_axis_labels("Models", "Accuracy w/o partial credit")
    plt_1.set(ylim=(0, 1))
    plt_1.set_xticklabels(rotation=10, fontsize=20)
    plt_1.legend.set_title("Reasoning type")
    plt_1.savefig('{}/Model vs Accuracy without partial credit.png'.format(output_dir))
    
    # Model vs Accuracy without partial credit of the largest model
    plt_2 = sns.catplot(data=get_largest(df), kind="bar", 
                    x="model", y="acc_wo_partial_credit", 
                    hue="reasoning_type", ci=None, alpha=.8, 
                    height=6, legend_out=True, aspect = 2
                    )
    plt_2.despine(left=True)
    plt_2.set_axis_labels("Models", "Accuracy w/o partial credit")
    plt_2.set(ylim=(0, 1))
    plt_2.set_xticklabels(rotation=10, fontsize=20)
    plt_2.legend.set_title("Reasoning type")
    plt_2.savefig('{}/Model vs Accuracy without partial credit of the largest model.png'.format(output_dir))


    # Model vs averaged Accuracy without partial credit among all size
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    plt_3 = sns.catplot(data=df[df['size'].notna()], kind="bar", 
                    x="model", y="acc_w_partial_credit", 
                    hue="reasoning_type", ci=None, alpha=.8, 
                    height=6, legend_out=True, aspect = 2
                    )
    plt_3.despine(left=True)
    plt_3.set_axis_labels("Models", "Accuracy w/ partial credit")
    plt_3.set(ylim=(0, 1))
    plt_3.set_xticklabels(rotation=10, fontsize=20)
    plt_3.legend.set_title("Reasoning type")
    plt_3.savefig('{}/Model vs Accuracy with partial credit.png'.format(output_dir))
    
    
    # Model vs Accuracy with partial credit of the largest model
    plt_4 = sns.catplot(data=get_largest(df), kind="bar", 
                    x="model", y="acc_w_partial_credit", 
                    hue="reasoning_type", ci=None, alpha=.8, 
                    height=6, legend_out=True, aspect = 2
                    )
    plt_4.despine(left=True)
    plt_4.set_axis_labels("Models", "Accuracy w/o partial credit")
    plt_4.set(ylim=(0, 1))
    plt_4.set_xticklabels(rotation=10, fontsize=20)
    plt_4.legend.set_title("Reasoning type")
    plt_4.savefig('{}/Model vs Accuracy with partial credit of the largest model.png'.format(output_dir))



#     # Model vs Accuracy with partial credit with trimmed error bars
#     def barplot(df, x, hue, y , err):
#         plt.figure(figsize=(10,6), dpi=120)
#         u = df[x].unique()
#         x = np.arange(len(u))
#         subx = df[hue].unique()
#         offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
#         width= np.diff(offsets).mean()
#         for i,gr in enumerate(subx):
#             dfg = df[df[hue] == gr]
#             plt.bar(x+offsets[i], dfg[y].values, width=width, 
#                     label="{}".format(gr), yerr=dfg[err].values, alpha=0.7)
#         plt.xlabel('Models')
#         plt.ylabel('Accuracy w/ partial credit')
#         plt.xticks(x, u, fontsize=18)
#         plt.yticks(fontsize=15)
#         plt.ylim([0, 1])
#         plt.legend(title='Reasoning type', title_fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', prop={'size': 10})
#         plt.savefig('{}/Model vs Accuracy with partial credit with trimmed error bars.png'.format(output_dir))


#     x = "model"
#     hue = "reasoning_type"
#     y = "acc_w_partial_credit"
#     err = "stdev_acc_w_partial_credit"
#     barplot(df, x, hue, y, err )


#     # Model vs Accuracy with partial credit with error bars
#     def barplot_trim(df, x, hue, y , err):
#         plt.figure(figsize=(10,6), dpi=120)
#         u = df[x].unique()
#         x = np.arange(len(u))
#         subx = df[hue].unique()
#         offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
#         width= np.diff(offsets).mean()
#         for i,gr in enumerate(subx):
#             dfg = df[df[hue] == gr]
#             plt.bar(x+offsets[i], dfg[y].values, width=width, 
#                     label="{}".format(gr), yerr=dfg[err].values, alpha=0.7)
#         plt.xlabel('Models')
#         plt.ylabel('Accuracy w/ partial credit')
#         plt.xticks(x, u, fontsize=18)
#         plt.yticks(fontsize=15)
#         plt.savefig('{}/Model vs Accuracy with partial credit with error bars.png'.format(output_dir))


#     x = "model"
#     hue = "reasoning_type"
#     y = "acc_w_partial_credit"
#     err = "stdev_acc_w_partial_credit"

#     barplot_trim(df, x, hue, y, err )
    
    # Line plot of Size vs Accuracy w/ partial credit averaged over all catogries.
    plt.figure(figsize = (15,8))
    sns.set(font_scale = 2.5)
    line_plt_1 = sns.lineplot(data=df[df['size'].notna()], x="size", y="acc_w_partial_credit", hue="model", err_style="bars", ci=68, markersize=15, linewidth = 3, style="model", markers=True)
    line_plt_1.set(ylim=(0, 1))
    line_plt_1.set(xscale='log')
    #line_plt.set(xlabel='Size (Number of parameters)', ylabel='Accuracy w/ partial credit')
    line_plt_1.set_xlabel("Size (Number of parameters)",fontsize=30)
    line_plt_1.set_ylabel("Accuracy w/ partial credit",fontsize=25)
    line_plt_1.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize='25').set_title("Models")
    line_plt_1.figure.savefig('{}/Size vs Accuracy with partial credit averaged over all catogries.png'.format(output_dir), bbox_inches='tight')
    
    # Line plot of Size vs Accuracy without partial credit averaged over all catogries
    plt.figure(figsize = (15,8))
    sns.set(font_scale = 2.5)
    line_plt_2 = sns.lineplot(data=df[df['size'].notna()], x="size", y="acc_wo_partial_credit", hue="model", err_style="bars", ci=68, markersize=15, linewidth = 3, style="model", markers=True)
    line_plt_2.set(ylim=(0, 1))
    line_plt_2.set(xscale='log')
    #line_plt.set(xlabel='Size (Number of parameters)', ylabel='Accuracy w/ partial credit')
    line_plt_2.set_xlabel("Size (Number of parameters)",fontsize=30)
    line_plt_2.set_ylabel("Accuracy w/o partial credit",fontsize=25)
    line_plt_2.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize='25').set_title("Models")
    line_plt_2.figure.savefig('{}/Size vs Accuracy without partial credit averaged over all catogries.png'.format(output_dir), bbox_inches='tight')

    
    
    df_with_size = df[df['size'].notna()]
    # Line plots of Size vs Accuracy w/ partial credit averaged by catogries.
    for i in ['motion', 'orientation', 'distance', 'containment', 'metaphor']:
        plt.figure(figsize = (15,8))
        sns.set(font_scale = 2.5)
        
        temp_plt_1 = sns.lineplot(data=df_with_size[df_with_size['reasoning_type'] == i], x="size", y="acc_w_partial_credit", hue="model", estimator = None, markersize=10, linewidth = 3, marker="o")
        temp_plt_1.set(ylim=(0, 1))
        temp_plt_1.set(title=i.capitalize())
        temp_plt_1.set_xlabel("Size (Number of parameters)",fontsize=30)
        temp_plt_1.set_ylabel("Accuracy w/ partial credit",fontsize=25)
        temp_plt_1.set(xscale='log')
        temp_plt_1.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize='25').set_title("Models")
        temp_plt_1.figure.savefig('{}/Size vs Accuracy with partial credit averaged over {}.png'.format(output_dir, i), bbox_inches='tight')
        plt.clf()
        
    for i in ['motion', 'orientation', 'distance', 'containment', 'metaphor']:
#         plt.figure(figsize = (15,8))
#         sns.set(font_scale = 2.5)
        temp_plt_2 = sns.lineplot(data=df_with_size[df_with_size['reasoning_type'] == i], x="size", y="acc_wo_partial_credit", hue="model", estimator = None, markersize=10, linewidth = 3, marker="o")
        temp_plt_2.set(ylim=(0, 1))
        temp_plt_2.set(title=i.capitalize())
        temp_plt_2.set_xlabel("Size (Number of parameters)",fontsize=30)
        temp_plt_2.set_ylabel("Accuracy w/ partial credit",fontsize=25)
        temp_plt_2.set(xscale='log')
        temp_plt_2.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize='25').set_title("Models")
        temp_plt_2.figure.savefig('{}/Size vs Accuracy without partial credit averaged over {}.png'.format(output_dir, i), bbox_inches='tight')
        plt.clf()
    



    # Box plot of Reasoning type vs Accuracy with partial credit.
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_5 = sns.boxplot(x='reasoning_type', y='acc_w_partial_credit', data=df_with_size,palette='vlag')
    sns.stripplot(x='reasoning_type', y='acc_w_partial_credit', data=df,size=3, color='.3')
    boxplt_5.set_xticklabels(boxplt_5.get_xticklabels(), rotation=40, ha="right", fontsize=15)
    boxplt_5.set(xlabel='Reasoning type', ylabel='Accuracy w/ partial credit')
    boxplt_5.figure.savefig('{}/Box plot of Reasoning type vs Accuracy with partial credit.png'.format(output_dir), bbox_inches='tight')


    # Box plot of Model vs Accuracy with partial credit
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_6 = sns.boxplot(x='model', y='acc_w_partial_credit', data=df_with_size,palette='vlag')
    sns.stripplot(x='model', y='acc_w_partial_credit', data=df,size=3, color='.3')
    boxplt_6.set_xticklabels(boxplt_6.get_xticklabels(), rotation=40, ha="right", fontsize=15)
    boxplt_6.set(xlabel='Model', ylabel='Test accuracy')
    boxplt_6.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit.png'.format(output_dir), bbox_inches='tight')
