#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import listdir
from os.path import isfile, join
import pandas as pd



# File path
analysis_path = '../analysis'
# Read all file names 
analysis_files = [f for f in listdir(analysis_path) if isfile(join(analysis_path, f))]

# remove existing dataframe if exist
if 'df_analysis.csv' in analysis_files:
    analysis_files.remove('df_analysis.csv')


    
# Combine all files to a dataframe
df_analysis = pd.DataFrame()
# model_lst = ['ada', 'allenai-unifiedqa-v2-t5-large-1363200', 'anirudh21-albert-large-v2-finetuned-mnli', 'microsoft-deberta-large-mnli', 'roberta-large-mnli', 't5-large', 'textattack-xlnet-base-cased-MNLI']
for model in analysis_files:
    temp = pd.read_csv('../analysis/' + model, delim_whitespace=True, encoding = "ISO-8859-1")
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
df_analysis.to_csv('../analysis/df_analysis.csv')

# Output file path
analysis_output_dir = "../analysis/output_fig_analysis"

# Check if path have directory, if not create directory
if not os.path.isdir(analysis_output_dir):
    os.makedirs(analysis_output_dir)

    
analysis_mean = df_analysis.groupby(['model','reasoning_type']).mean().reset_index()

# Model vs Mean accuracy by different model.png
sns.set_theme(style="whitegrid")
sns.set(font_scale=2)

analysis_mean_plt = sns.catplot(data=analysis_mean, x='model', y='test_acc', hue='reasoning_type', 
                                alpha=.8, kind='bar', height=6, legend=True, aspect=2);

analysis_mean_plt.despine(left=True)
analysis_mean_plt.set_axis_labels("Models", "Mean accuracy")
analysis_mean_plt.legend.set_title("reasoning_type")
analysis_mean_plt.set(ylim=(0, 1))
analysis_mean_plt.savefig('{}/Model vs Mean accuracy by different model.png'.format(analysis_output_dir))


# Box plot of Reasoning type vs Test accuracy
plt.figure(figsize=(10,6), dpi=120)
sns.set(font_scale=1.5)
boxplt_1 = sns.boxplot(x='reasoning_type', y='test_acc', data=df_analysis,palette='vlag')
sns.stripplot(x='reasoning_type', y='test_acc', data=df_analysis,size=3, color='.3')
boxplt_1.set_xticklabels(boxplt_1.get_xticklabels(), rotation=40, ha="right", fontsize=15)
boxplt_1.set(xlabel='Reasoning type', ylabel='Test accuracy')
boxplt_1.figure.savefig('{}/Box plot of Reasoning type vs test accuracy.png'.format(analysis_output_dir), bbox_inches='tight')


# Box plot of Model vs Test accuracy
plt.figure(figsize=(10,6), dpi=120)
sns.set(font_scale=1.5)
boxplt_2 = sns.boxplot(x='model', y='test_acc', data=df_analysis,palette='vlag')
boxplt_2 = sns.stripplot(x='model', y='test_acc', data=df_analysis,size=3, color='.3')
boxplt_2.set_xticklabels(boxplt_2.get_xticklabels(), rotation=40, ha="right", fontsize=15)
boxplt_2.set(xlabel='Model', ylabel='Test accuracy')
boxplt_2.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit.png'.format(analysis_output_dir), bbox_inches='tight')


# Box plot of Reasoning type vs Test accuracy with hue = Model
plt.figure(figsize=(20,3), dpi=120)
sns.set(font_scale=2)
boxplt_3 = sns.boxplot(x='model', y='test_acc', data=df_analysis,palette='vlag', hue='reasoning_type')
boxplt_3.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
boxplt_3.set(xlabel='Model', ylabel='Test accuracy')
boxplt_3.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit with hue = Model.png'.format(analysis_output_dir), bbox_inches='tight')


# Box plot of Model vs Test accuracy with hue = Reasoning type
plt.figure(figsize=(20,3), dpi=120)
sns.set(font_scale=2)
boxplt_4 = sns.boxplot(x='reasoning_type', y='test_acc', data=df_analysis,palette='vlag', hue='model')
boxplt_4.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
boxplt_4.set(xlabel='Reasoning type', ylabel='Test accuracy')
boxplt_4.figure.savefig('{}/Box plot of Model vs Accuracy with partial credit with hue = Reasoning type.png'.format(analysis_output_dir), bbox_inches='tight')
