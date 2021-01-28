#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:05:48 2020

@author: nolanlem
"""
import pandas as pd 
import numpy as np 
import csv 
import os 


#%%
# df1.loc[df1['stream'] == 2, ['feat','another_feat']] = 'aaaa'
# print df1
#    stream        feat another_feat
# a       1  some_value   some_value
# b       2        aaaa         aaaa
# c       2        aaaa         aaaa
# d       3  some_value   some_value
 
#%%
### tally participants with nan 
rootdir = '/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/csvs/'
#thecsvfile = '10-19_18-18.csv'
#thecsvfile = '10-29_16-59-w NO beat binning.csv'
thecsvfile = '11-8_15-18-w NO beat binning.csv'
thecsvfile = '11-15_14-9-w NO beat binning.csv'
thecsvfile = '11-17_15-22-w NO beat binning.csv'


#csvfile = os.path.join(rootdir,thecsvfile)
df = pd.read_csv(csv_filename_str)
#################

#idx = df[df.mx == np.nan].index
nan_mx_idx = df[df['mx'].isnull()].index
nan_sx_idx = df[df['sx'].isnull()].index

zero_mx_idx = df[df['mx'] == 0.0].index
zero_sx_idx = df[df['sx'] == 0.0].index

print('zero mx:', zero_mx_idx)
print('zero sx:', zero_sx_idx)
##################
nan_subjs = df.iloc[list(nan_mx_idx.values)]
nan_subjects = []

for subj in nan_subjs.values:
    nan_subjects.append(subj[0])
    
num_nans_per_subject = {i:nan_subjects.count(i) for i in nan_subjects}
print(num_nans_per_subject)
#################
df.insert(df.shape[1], "mx_imputed", 0.0)
df.insert(df.shape[1], "sx_imputed", 0.0)
df.insert(df.shape[1], "good", 1)
##################
idx_nans = []
for particip in nan_subjects: 
    idx_nans = df[df['subject'] == particip].index

for idx in idx_nans:
    df.loc[idx, 'good'] = 0
    # df.loc[idx, 'mx'] = np.nan 
    # df.loc[idx, 'sx'] = np.nan
    # df.loc[idx, 'mx_imputed'] = np.nan 
    # df.loc[idx, 'sx_imputed'] = np.nan
###################
#drop the nan participants 
# for nan_subj in nan_subjects:
#     df = df.drop(df[df['subject'] == nan_subj].index) 

####################3
mx_sum = df.loc[df['mx'] > 0.0, ['mx']].sum(axis=0)
mx_sum = mx_sum/(len(df)-len(zero_mx_idx))

sx_sum = df.loc[df['sx'] != 0.0, ['sx']].sum(axis=0)
sx_sum = sx_sum/(len(df)-len(zero_sx_idx))

print('mx sum:', mx_sum, ' sx sum: ', sx_sum)

    
####################
mxs, sxs = [], []
for row in df.index:
    if df.loc[row, 'mx'] > 0.0:
        #mxs.append(df.loc[row,'mx'])
        df.loc[row, 'mx_imputed'] = df.loc[row, 'mx']
    if df.loc[row, 'mx'] ==  0.0:
        df.loc[row, 'mx_imputed'] = float(mx_sum)
    
    if df.loc[row, 'sx'] > 0.0:
        df.loc[row, 'sx_imputed'] = df.loc[row, 'sx']
    if df.loc[row, 'sx'] == 0.0:
        df.loc[row, 'sx_imputed'] = float(sx_sum)

# create mx[], sx[] to accumulate if mx or sx !== 0 
# if mx or sx !== 0: do  
#   mx,sx_sum.append(num)
#####################
df.to_csv(rootdir + os.path.basename(thecsvfile).split('.')[0] + '-imputed.csv')
#%%
def impute_mx(csv_file):
    df = pd.read_csv(csv_file)
    nan_mx_idx = df[df['mx'].isnull()].index
    nan_sx_idx = df[df['sx'].isnull()].index
    
    zero_mx_idx = df[df['mx'] == 0.0].index
    zero_sx_idx = df[df['sx'] == 0.0].index
    
    print('zero mx:', zero_mx_idx)
    print('zero sx:', zero_sx_idx)
    ##################
    nan_subjs = df.iloc[list(nan_mx_idx.values)]
    nan_subjects = []
    
    for subj in nan_subjs.values:
        nan_subjects.append(subj[0])
        
    num_nans_per_subject = {i:nan_subjects.count(i) for i in nan_subjects}
    print(num_nans_per_subject)
    #################
    df.insert(df.shape[1], "mx_imputed", 0.0)
    df.insert(df.shape[1], "sx_imputed", 0.0)
    df.insert(df.shape[1], "good", 1)
    ##################
    idx_nans = []
    
    for particip in nan_subjects: 
        idx_nans = df[df['subject'] == particip].index
    
    for idx in idx_nans:
        df.loc[idx, 'good'] = 0

    mx_sum = df.loc[df['mx'] > 0.0, ['mx']].sum(axis=0)
    mx_sum = mx_sum/(len(df)-len(zero_mx_idx))
    
    sx_sum = df.loc[df['sx'] != 0.0, ['sx']].sum(axis=0)
    sx_sum = sx_sum/(len(df)-len(zero_sx_idx))

    mxs, sxs = [], []
    for row in df.index:
        if df.loc[row, 'mx'] > 0.0:
            #mxs.append(df.loc[row,'mx'])
            df.loc[row, 'mx_imputed'] = df.loc[row, 'mx']
        if df.loc[row, 'mx'] ==  0.0:
            df.loc[row, 'mx_imputed'] = float(mx_sum)
        
        if df.loc[row, 'sx'] > 0.0:
            df.loc[row, 'sx_imputed'] = df.loc[row, 'sx']
        if df.loc[row, 'sx'] == 0.0:
            df.loc[row, 'sx_imputed'] = float(sx_sum)
    df.to_csv(os.path.basename(thecsvfile).split('.')[0] + '-imputed.csv')
    print('creating %r' %(os.path.basename(thecsvfile).split('.')[0] + '-imputed.csv'))
#%%
impute_mx(csv_filename_str) 