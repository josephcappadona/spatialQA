#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join



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

    # Combine all files to a dataframe
    df_analysis = pd.DataFrame()
    for model in analysis_files:
        temp = pd.read_csv(analysis_path + model, delim_whitespace=True, encoding = "ISO-8859-1")
        if 'allenai-unifiedqa-v2-t5-large-1363200' in model:
            temp['model'] = 'unifiedQA-t5'
        elif 'anirudh21-albert-large-v2-finetuned-mnli' in model:
            temp['model'] = 'albert'
        elif 'microsoft-deberta-large-mnli' in model:
            temp['model'] = 'deberta'
        elif 'roberta-large-mnli' in model:
            temp['model'] = 'roberta'
        elif 't5-large' in model:
            temp['model'] = 't5'
        elif 'textattack-xlnet-base-cased-MNLI' in model:
            temp['model'] = 'xlnet'
        elif 'ada' in model:
            temp['model'] = 'ada'
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

    # Combine all files to a dataframe
    df_summary = pd.DataFrame()
    for model in summary_files:
        temp = pd.read_csv(summary_path + model, delim_whitespace=True, encoding = "ISO-8859-1")
        if 'allenai-unifiedqa-v2-t5-large-1363200' in model:
            temp['model'] = 'unifiedQA-t5'
        elif 'anirudh21-albert-large-v2-finetuned-mnli' in model:
            temp['model'] = 'albert'
        elif 'microsoft-deberta-large-mnli' in model:
            temp['model'] = 'deberta'
        elif 'roberta-large-mnli' in model:
            temp['model'] = 'roberta'
        elif 't5-large' in model:
            temp['model'] = 't5'
        elif 'textattack-xlnet-base-cased-MNLI' in model:
            temp['model'] = 'xlnet'
        elif 'ada' in model:
            temp['model'] = 'ada'
        df_summary = pd.concat([df_summary, temp], ignore_index=True)
    
    # Output file to folder
    df_summary.to_csv(summary_file_path + '/df_summary.csv', index=False)


def make_figure_analysis(df_path, output_dir):

    df = pd.read_csv(df_path)

    analysis_mean = df.groupby(['model','reasoning_type']).mean().reset_index()

    # Model vs Mean accuracy by different model.png
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    analysis_mean_plt = sns.catplot(data=analysis_mean, x='model', y='test_acc', hue='reasoning_type', 
                                    alpha=.8, kind='bar', height=6, legend=True, aspect=2);

    analysis_mean_plt.despine(left=True)
    analysis_mean_plt.set_axis_labels("Models", "Mean accuracy")
    analysis_mean_plt.legend.set_title("reasoning_type")
    analysis_mean_plt.set(ylim=(0, 1))
    analysis_mean_plt.savefig('{}/Model vs Mean accuracy by different model.png'.format(output_dir))


    # Box plot of Reasoning type vs Test accuracy
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_1 = sns.boxplot(x='reasoning_type', y='test_acc', data=df,palette='vlag')
    sns.stripplot(x='reasoning_type', y='test_acc', data=df,size=3, color='.3')
    boxplt_1.set_xticklabels(boxplt_1.get_xticklabels(), rotation=40, ha="right", fontsize=15)
    boxplt_1.set(xlabel='Reasoning type', ylabel='Test accuracy')
    boxplt_1.figure.savefig('{}/Box plot of Reasoning type vs test accuracy.png'.format(output_dir), bbox_inches='tight')


    # Box plot of Model vs Test accuracy
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_2 = sns.boxplot(x='model', y='test_acc', data=df,palette='vlag')
    boxplt_2 = sns.stripplot(x='model', y='test_acc', data=df,size=3, color='.3')
    boxplt_2.set_xticklabels(boxplt_2.get_xticklabels(), rotation=40, ha="right", fontsize=15)
    boxplt_2.set(xlabel='Model', ylabel='Test accuracy')
    boxplt_2.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit.png'.format(output_dir), bbox_inches='tight')


    # Box plot of Reasoning type vs Test accuracy with hue = Model
    plt.figure(figsize=(20,3), dpi=120)
    sns.set(font_scale=2)
    boxplt_3 = sns.boxplot(x='model', y='test_acc', data=df,palette='vlag', hue='reasoning_type')
    boxplt_3.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
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

    # Model vs Accuracy without partial credit
    plt_1 = sns.catplot(data=df, kind="bar", 
                    x="model", y="acc_wo_partial_credit", 
                    hue="reasoning_type", ci=None, alpha=.8, 
                    height=6, legend_out=True, aspect = 2
                    )
    plt_1.despine(left=True)
    plt_1.set_axis_labels("Models", "Accuracy w/o partial credit")
    plt_1.set(ylim=(0, 1))
    plt_1.legend.set_title("Reasoning type")
    plt_1.savefig('{}/Model vs Accuracy without partial credit.png'.format(output_dir))


    # Model vs Accuracy with partial credit
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    plt_2 = sns.catplot(data=df, kind="bar", 
                    x="model", y="acc_w_partial_credit", 
                    hue="reasoning_type", ci=None, alpha=.8, 
                    height=6, legend_out=True, aspect = 2
                    )
    plt_2.despine(left=True)
    plt_2.set_axis_labels("Models", "Accuracy w/ partial credit")
    plt_2.set(ylim=(0, 1))
    plt_2.legend.set_title("Reasoning type")
    plt_2.savefig('{}/Model vs Accuracy with partial credit.png'.format(output_dir))


    # Model vs Accuracy with partial credit with trimmed error bars
    def barplot(df, x, hue, y , err):
        plt.figure(figsize=(10,6), dpi=120)
        u = df[x].unique()
        x = np.arange(len(u))
        subx = df[hue].unique()
        offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
        width= np.diff(offsets).mean()
        for i,gr in enumerate(subx):
            dfg = df[df[hue] == gr]
            plt.bar(x+offsets[i], dfg[y].values, width=width, 
                    label="{}".format(gr), yerr=dfg[err].values, alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Accuracy w/ partial credit')
        plt.xticks(x, u, fontsize=18)
        plt.yticks(fontsize=15)
        plt.ylim([0, 1])
        plt.legend(title='Reasoning type', title_fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', prop={'size': 10})
        plt.savefig('{}/Model vs Accuracy with partial credit with trimmed error bars.png'.format(output_dir))


    x = "model"
    hue = "reasoning_type"
    y = "acc_w_partial_credit"
    err = "stdev_acc_w_partial_credit"
    barplot(df, x, hue, y, err )


    # Model vs Accuracy with partial credit with error bars
    def barplot_trim(df, x, hue, y , err):
        plt.figure(figsize=(10,6), dpi=120)
        u = df[x].unique()
        x = np.arange(len(u))
        subx = df[hue].unique()
        offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
        width= np.diff(offsets).mean()
        for i,gr in enumerate(subx):
            dfg = df[df[hue] == gr]
            plt.bar(x+offsets[i], dfg[y].values, width=width, 
                    label="{}".format(gr), yerr=dfg[err].values, alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Accuracy w/ partial credit')
        plt.xticks(x, u, fontsize=18)
        plt.yticks(fontsize=15)
        plt.savefig('{}/Model vs Accuracy with partial credit with error bars.png'.format(output_dir))


    x = "model"
    hue = "reasoning_type"
    y = "acc_w_partial_credit"
    err = "stdev_acc_w_partial_credit"

    barplot_trim(df, x, hue, y, err )

    # Box plot of Reasoning type vs Accuracy with partial credit
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_1 = sns.boxplot(x='reasoning_type', y='acc_w_partial_credit', data=df,palette='vlag')
    sns.stripplot(x='reasoning_type', y='acc_w_partial_credit', data=df,size=3, color='.3')
    boxplt_1.set_xticklabels(boxplt_1.get_xticklabels(), rotation=40, ha="right", fontsize=15)
    boxplt_1.set(xlabel='Reasoning type', ylabel='Accuracy w/ partial credit')
    boxplt_1.figure.savefig('{}/Box plot of Reasoning type vs Accuracy with partial credit.png'.format(output_dir), bbox_inches='tight')


    # Box plot of Model vs Accuracy with partial credit
    plt.figure(figsize=(10,6), dpi=120)
    sns.set(font_scale=1.5)
    boxplt_2 = sns.boxplot(x='model', y='acc_w_partial_credit', data=df,palette='vlag')
    sns.stripplot(x='model', y='acc_w_partial_credit', data=df,size=3, color='.3')
    boxplt_2.set_xticklabels(boxplt_2.get_xticklabels(), rotation=40, ha="right", fontsize=15)
    boxplt_2.set(xlabel='Model', ylabel='Test accuracy')
    boxplt_2.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit.png'.format(output_dir), bbox_inches='tight')
