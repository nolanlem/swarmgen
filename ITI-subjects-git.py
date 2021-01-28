#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:38:35 2020
NB: root dir is './psychopy/swarm-tapping-study now (see os.chdir below). Most of the other scripts are 
../../ here '

study
A1 no(1,2)        timbre(1,2)
B1 timbre(1,2)    no(1,2)
A2 no(3,4)        timbre(3,4)
B2 timbre(3,4)    no(3,4)

version: timbre (t), no-timbre (nt)
coupling: none, weak, medium, strong
tempo: 72, 81, 95, 105, 119
@author: nolanlem
"""


import numpy as np 
import pandas as pd
import glob 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker
import sys
import os
from ast import literal_eval
from io import StringIO
import itertools
from scipy.signal import find_peaks
from collections import defaultdict
import librosa
from scipy.stats import sem
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.cm as cm
import datetime 
from collections import defaultdict
import scipy.stats
import csv


import seaborn as sns
sns.set()
os.chdir('/Users/nolanlem/Documents/kura/kura-git/py/psychopy/swarm-tapping-study/')


sr=22050


def removeStrFormatting(str_array):
    for str_arr in str_array:
        str_arr = str_arr[1:-1] # remove "'[" and "]'"
        str_arr = str.split(str_arr, ',') # split strings
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
        #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr


def flatten2DList(thelist):
    flatlist = list(itertools.chain(*thelist))
    return flatlist

def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    for i, tap in enumerate(taps):
        try:
            binnedtaps.append(taps[i][0]) # take first tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    return binnedtaps

def removeStrFormatting(str_arr):
    str_arr = str_arr[1:-1] # remove "'[" and "]'"
    str_arr = str.split(str_arr, ',') # split strings
    try:
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
    except ValueError:
        pass
    #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr

def makeDir(dirname):
    if os.path.exists(dirname) == False:
        print('making directory: ', dirname)
        os.mkdir(dirname)
        
def impute_mx(csv_file):
    rootdir = os.path.dirname(csv_file)
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
    df.to_csv(rootdir + '/' + os.path.basename(csv_file).split('.')[0] + '-imputed.csv')
    print('creating %r' %(rootdir + '/' + os.path.basename(csv_file).split('.')[0] + '-imputed.csv'))



#%%########### get CENTER PERIODS and BEAT BINS from data txt files from generative model ########
##################################################################################################

# dictionaries for center periods, beat bins, etc. from generative model (already generated) 
idealperiods, sndbeatbins, centerbpms, centerperiods = {},{},{},{}

# load beatbins for no-timbre type
datadirs = ['../stim-no-timbre-5', '../stim-timbre-5']
timbre_tags = ['n','t']
stimuli_dirs = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4']
#beatbins_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows')
#centerbpms_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm')


for datadir, ttag in zip(datadirs, timbre_tags):
    for stimuli_dir in stimuli_dirs:
        beatbins_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows')
        for fi in glob.glob(beatbins_dir + '/*.txt'):
            fi_basename = str.split(os.path.basename(fi), '.')[0] # --> weak_79_1
            f = str.split(fi_basename, '_') 
            sync_cond = "_".join([f[0], ttag, f[1], f[2]])
            thebeatbins = np.loadtxt(fi, delimiter='\n')
            sndbeatbins[sync_cond] = thebeatbins
        
        centerbpms_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm')    
        for fi in glob.glob(centerbpms_dir + '/*.txt'):       
            fi_basename = str.split(os.path.basename(fi), '.')[0] # --> weak_79_1
            f = str.split(fi_basename, '_') 
            sync_cond = "_".join([f[0], ttag, f[1], f[2]])    
            thecenterbpm = np.loadtxt(fi)        
            centerbpms[sync_cond] = float(thecenterbpm)
            centerperiods[sync_cond] = 60./float(thecenterbpm)
            

#%% ######### parse subject taps in csv output files and format into dataframes or arrays

# default dictionarya
subject_resps = defaultdict(lambda: defaultdict(list))

ordered_subjects = []

# STRING PROMPTS in HEADER of csv files 
block1taps = 'block1_taps.rt'
block2taps = 'block2_taps.rt'
csv_sndfiles = 'sndfile'
csv_tempo = 'tempo'
csv_coupling_cond = 'cond'
csv_version = 'version'
csv_participant = 'Participant Initials'
csv_type = 'type'

##################################################
######### only take good USABLE csv files #####
#################################################
batch_folder = 'usable-batch-12-7'
#batch_folder = 'usable-stanford-batch'
#batch_folder = 'usable-mturk-batch'
subject = []
csvfiles = []   

### fill up subject with csv basename 
for csv_ in glob.glob('./mturk-csv/' + batch_folder + '/*.csv'):
    #namestripped = os.path.basename(csv_).split('.')[0].split(' ')[0]
    namestripped = os.path.basename(csv_)
    subject.append(namestripped)
    csvfiles.append(csv_)

########## GET ALL STIM NAMES with full path from allstims dir --> allstims list
allstims = []   # allstims is full file path of every stimuli 
for fi in glob.glob('./allstims/*.wav'):
    allstims.append(fi)

#%%########## READ IN THE SUBJECT TAPS #################

for csv_file, person in zip(csvfiles, subject):
    print('SUBJECT: ', person)
    df_block = pd.read_csv(csv_file, keep_default_na=False)
    subject_resps[person] = {}  

    try:

        df_block_1 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block1taps])[4:44]
        df_block_2 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block2taps])[44:-1]
        
        df_block_1_type = df_block_1[csv_type]
        #timbre_type = df_block_1['sndfile'].values
    
        for index, row in df_block_1.iterrows():
            sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = []
        for index, row in df_block_2.iterrows():
            sync_cond_version  = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = []
    
        for index, row in df_block_1.iterrows():
            sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = removeStrFormatting(row[block1taps])
        for index, row in df_block_2.iterrows():
            sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = removeStrFormatting(row[block2taps])
    
    except TypeError:
        print('could not read %r csv file' %(person))
        
       
#####NB: subject_resps are now in this format 
#### subject_resps[person][type(no, timbre)][sync_tempo_version]            

#%% ### reformat trials subjects did not perform with empty list '' -> [] ###
  
subjectplotdir = './analysis-scripts/plots/' + batch_folder + "/subjects/"

# replace all empty trials with [] (tried with np.nan but not good for plotting... )
for person in subject:
    print(person)
    for n, sndfile in enumerate(allstims):
        sync_cond_version = str.split(os.path.basename(sndfile), '.')[0]  
        try:
            if (subject_resps[person][sync_cond_version] == ['']):
                subject_resps[person][sync_cond_version] = []
        except KeyError:
            print('subject %r did not tap to %r' %(person, sndfile))

#%%% ########### UTIL FUNCTIONS FOR BEAT BINNING ########
### this is redefined in case we want to use a different algo for beat binning for ITI
#### but as of now, we are NOT beat binning for ITI analysis 
def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    avg_taps_per_bin = []
    for i, tap in enumerate(taps):
        try:
            num_taps_in_bin = len(taps[i])
            avg_taps_per_bin.append(num_taps_in_bin)
            if num_taps_in_bin > 0:            
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) # take random tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim


#%% ############################################################
##################### ITI of beat sections ########################
##############################################################
## FORM the groupings for analysis (by timbre, by coupling

sync_conds = ['none','weak','medium','strong']   # strings for coupling cond
t_strs = ['t', 'n'] # for delineating timbre ('t') vs. no-timbre ('n') in dictionaries

## separate allstims by timbre ('t','n') and coupling cond ('none', 'weak', 'medium','strong)
t_none = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('n') and os.path.basename(elem).split('_')[1] == 't']
t_weak = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('w') and os.path.basename(elem).split('_')[1] == 't']
t_medium = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('m') and os.path.basename(elem).split('_')[1] == 't']
t_strong = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('s') and os.path.basename(elem).split('_')[1] == 't']

n_none = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('n') and os.path.basename(elem).split('_')[1] == 'n']
n_weak = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('w') and os.path.basename(elem).split('_')[1] == 'n']
n_medium = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('m') and os.path.basename(elem).split('_')[1] == 'n']
n_strong = [os.path.basename(elem).split('.')[0] for elem in allstims if os.path.basename(elem).startswith('s') and os.path.basename(elem).split('_')[1] == 'n']

timbre_conds = [t_none, t_weak, t_medium, t_strong]
notimbre_conds = [n_none, n_weak, n_medium, n_strong]

#############################################
#%% NB: only to split stim by tempo, but can run ITI-subjects-tempo.py instead
# sync_strs = ['none','weak','medium','strong']
# nsnds = [n_none, n_weak, n_medium, n_strong]
# tsnds = [t_none, t_weak, t_medium, t_strong]
# tmps = ['72','81','92','105','119']
# nsync_conds_tmp = {'72':[],"81":[],"92":[], "105":[], "119":[]}
# tsync_conds_tmp = {'72':[],"81":[],"92":[], "105":[], "119":[]}

# for n_, t_, sync_str in zip(nsnds, tsnds, sync_strs):
#     for tmp in tmps:
#         # for no timbre
#         nthetmp_ = [elem for elem in n_ if elem.split('_')[-2] == tmp]
#         nsync_conds_tmp[tmp].append(nthetmp_)
#         # for timbre
#         tthetmp_ = [elem for elem in t_ if elem.split('_')[-2] == tmp]
#         tsync_conds_tmp[tmp].append(tthetmp_)


############ IMPORTANT ###############################################
############## NOW DOING WHOLE BATCH (TIMBRE AND NON TIMBRE) AT ONCE #######
#####################################################################
all_timbre_conds = [timbre_conds, notimbre_conds]

###############################################################

#%%
####################################################
############ ITI ANALYSIS ##############################
########################################################

# make directories for individual ITI subject plots
subject_iti_dir = './analysis-scripts/plots/usable-batch/subject-10-6-plots/'
for person in subject:
    the_subject_iti_dir = subject_iti_dir + person
    if os.path.exists(the_subject_iti_dir) == False:
        print('making dir for ', person, ' in ', the_subject_iti_dir)
        os.mkdir(the_subject_iti_dir)  

# this function plots each subjects individual ITIs per beat and saves into their subject directory  
def plotSubjectITIperTap(means, stds, label_str, person_str):
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    ax.plot(means)
    xrange = np.linspace(0, len(means) -1 , len(means))
    ax.errorbar(xrange, means, yerr=stds, marker='.', label=label_str, capsize=3)
    ax.set_title(' '.join([person_str, sync_str]))
    plt.savefig('./analysis-scripts/plots/usable-batch/subject-10-6-plots/' + person_str + '/' + person_str + ' ' + label_str + '.png', dpi=160)

#%% MUST INTIALIZE DICTIONARIES BEFORE LOOP 
### NB: MUST DO THIS EVERYTIME BEFORE RUNNING ITI ANALYSIS depending on no bb, bb, or outlier algo 
sndfile_strs = 'Timbre and No Timbre Condition' # e.g. 'Timbre Condition', "No Timbre Condition"
binned_subject_taps = {}
subject_sync_cond_taps, subject_sync_cond_itis = {}, {}

for person in subject:
    subject_sync_cond_taps[person], subject_sync_cond_itis[person] = {}, {}

    for sync_str in sync_conds:
        subject_sync_cond_taps[person][sync_str], subject_sync_cond_itis[person][sync_str] = {}, {}
        for version, v in zip(all_timbre_conds, t_strs):
            subject_sync_cond_taps[person][sync_str][v] = []
            subject_sync_cond_itis[person][sync_str][v] = []

#%% !!!!! NB:########### ONE THIS CODE BLOCK OR THE NEXT !!!!!! ##########
####### 1. BEAT BINNING
####### 2. NO BEAT BINNING 
####### 3. OUTLIERS ALGO
#######DONT FORGET
###### HAVE TO INITIALIZE DICTIONARIES IN CODE BLOCK BEFORE!!!!
#################  1. BEAT BINNING:  GET ITIs FROM SUBJECT_RESPS[] WITH BEAT BINNING 
#%%% ########### UTIL FUNCTIONS FOR BEAT BINNING ########
def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    avg_taps_per_bin = []
    for i, tap in enumerate(taps):
        try:
            num_taps_in_bin = len(taps[i])
            avg_taps_per_bin.append(num_taps_in_bin)
            if num_taps_in_bin > 0:            
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) # take random tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim


##################################################################       
######### 2. NB: NO BEAT BINNING.......DEFAULT!!!!!!
beat_binning_flag = 'w NO beat binning'
for version, v in zip(all_timbre_conds, t_strs):
    for sync_cond, sync_str in zip(version, sync_conds):
        #iti_segment_mx[sync_str], iti_segment_sx[sync_str] = {}, {}
       
        for sync_cond_version in sync_cond:
          
            sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
            timbre_version = sync_cond_version.split('_')[1] # get the timbre tag
                        
            binned_subject_taps[sync_cond_version] = []
            
            for person in subject:
                try:
                    tap_resps = subject_resps[person][sync_cond_version] # get subject taps per stim 
                    #binned_taps = binBeats(tap_resps, sndbeatbin)
                    # beat binning for ITI or no? 
                    #binned_taps, _ = binTapsFromBeatWindow(binned_taps)
                    #binned_subject_taps[sync_cond_version].append(binned_taps) # save subject's binned_taps per stim

                    # accumulate subject taps per sync_cond
                    subject_sync_cond_taps[person][sync_str][v].extend(tap_resps)
                    
                    # get normalized ITI vector and add it to the subject array
                    normalized_tap_iti = list(np.diff(tap_resps)/centerperiods[sync_cond_version])
                    # if longer than 19 beats, take last 19 beats 
                    if len(normalized_tap_iti) > 19:
                        normalized_tap_iti = normalized_tap_iti[len(normalized_tap_iti)-19:]
                    subject_sync_cond_itis[person][sync_str][v].append(normalized_tap_iti)
                    #print(subject_sync_cond_itis[person][sync_str][v])

                except KeyError:
                    #print('did not tap to ', sync_cond_version)
                    pass
###############################
###############################################

                    
#%% #%% ############## NEW ITI PLOTS!!!! ##################
######################################################

####### ITI SLICING AND TAKE MEAN/STD

# how many beat sections to analyze?
#beatsegments = [(0,5), (5,10), (10,15), (15,20)] # 4, 5 beat sections
#beatsegments = [(0,4), (4,8), (8,12), (12,16), (16,20)] # 5, 4 beat sections 
beatsegments = [(0,3), (3,6), (6,9), (9,12), (12, 15), (15,18)] # 6,3 beat sections
#beatsegments = [(0,2), (2,4), (4,6), (6,8), (8, 10), (10,12), (12,14), (14,16), (16, 18)] # 9, 2 beat sections


# dictionaries to hold mx, sx, and mx/sx errors 
iti_segment_mx, iti_segment_sx= {},{}
iti_mx, iti_sx = {}, {}
iti_mx_error, iti_sx_error = {}, {}

# beat str array for csv file output 
beat_strs = [str(i) for i in range(len(beatsegments))]

beat_segment_dir = './analysis-scripts/plots/beat-segment-analysis/' + str(len(beatsegments)) + '-beat-segments/' 
itis_dir = beat_segment_dir + '/ITIs/'
pcs_dir = beat_segment_dir + '/PCs/'
csvs_dir = beat_segment_dir + '/csvs/'
subject_itis_dir = itis_dir + 'subject-ITI-plots'

########## make all the directories #################
makeDir(beat_segment_dir)
makeDir(itis_dir) # make ITI subdir
makeDir(pcs_dir) # make PCs subdir 
makeDir(csvs_dir) # make csvs subdir
makeDir(subject_itis_dir) # make subject ITI plots 

# for PC directories, model and subjects subdirs and phases dir 
model_dir = pcs_dir + 'model/'
subject_dir = pcs_dir + 'subject/'
makeDir(model_dir)
makeDir(model_dir + 'phases/')
makeDir(subject_dir)
makeDir(subject_dir + 'phases/')

#####################################
# timestamp for csv file and cross referencing plot with csv 
now = datetime.datetime.now()
timestamp = str(now.month) + '-' + str(now.day) + '_' + str(now.hour) + '-' + str(now.minute)   

makeDir('./analysis-scripts/ITIs/')


#%%#### FILTER OUT SUBJECTS remove subjects with constraint: avg iti across beatsegs on 'weak' < 0.67

badsubjects = []
for person in subject:
    for j, sync_str in enumerate(sync_conds):
        for i, v in enumerate(t_strs): 
            df = pd.DataFrame(subject_sync_cond_itis[person][sync_str][v])
            person_avg_iti = np.nanmean(df.iloc[-18:].mean(axis=0).values)
            person_avg_std = np.nanstd(df.iloc[-18:].mean(axis=0).values)
            
            if sync_str == 'weak':
                if person_avg_iti < 0.67:
                    print(person, sync_str, v, person_avg_iti)
                    badsubjects.append(person)
                if person_avg_iti > 1.33:
                    print(person, sync_str, v, person_avg_iti)
                    badsubjects.append(person)
            
            # if sync_str == 'medium':
            #     if person_avg_std > 0.15:
            #         print('std: ', person, sync_str, person_avg_iti)
            #         badsubjects.append(person)  
# remove duplicates from badsubjects 
badsubjects = list(set(badsubjects))
# remove badsubjects from subject 
usable_subjects = subject.copy() # create copy of subject 
### remove the badsubjects
for subj in subject: 
    if subj in badsubjects:
        usable_subjects.remove(subj)
print('total subjects:', len(subject), '\n', 'usable subjects:', len(usable_subjects), '\n',len(badsubjects), 'subjects were filtered out')
      
#%% Run participant stats with filtered subjects to tally totals per study mturk,stanford 
from utils.parse_subjects import parse_subject_info 

## def parseSubjectInfo(batch_folder, usable_batch, badsubjects)

parse_subject_info('./mturk-csv/' + batch_folder, usable_subjects, badsubjects)  

#%% ############### PLOT WITHIN SUBJECT ITI AVERAGES AGGREGATED ON SAME PLOT WITH FILTERED SUBJECTS
fig_t, ax_t = plt.subplots(nrows=4, ncols =2, figsize=(10,7), sharex=True)
fig_n, ax_n = plt.subplots(nrows=4, ncols =2, figsize=(10,7), sharex=True)


## intialize dictionaries 
mx_all_subject_iti_seg = {}
sx_all_subject_iti_seg = {}

for sync_str in sync_conds:
    mx_all_subject_iti_seg[sync_str], sx_all_subject_iti_seg[sync_str] = {},{}
    for v in t_strs:
        mx_all_subject_iti_seg[sync_str][v], sx_all_subject_iti_seg[sync_str][v] = [],[]
        

for person in usable_subjects:
    sc = 0
    for j, sync_str in enumerate(sync_conds):
        for i, v in enumerate(t_strs):           
            
            person_iti = pd.DataFrame(subject_sync_cond_itis[person][sync_str][v])
            
            mx_subj_iti_seg, sx_subj_iti_seg = [],[]
            for beatseg, beat_str in zip(beatsegments, beat_strs):
                tap_col = person_iti.iloc[:,beatseg[0]:beatseg[1]].values
                mx = np.nanmean(tap_col)
                sx = np.nanstd(tap_col)
                mx_subj_iti_seg.append(mx)
                sx_subj_iti_seg.append(sx)
            
            mx_all_subject_iti_seg[sync_str][v].append(mx_subj_iti_seg)
            sx_all_subject_iti_seg[sync_str][v].append(sx_subj_iti_seg)
            
            ### plot individual subject ITI per beat segment 
        
            if v == 't':
                version_str = 'timbre'
                ax_t[j,0].plot(mx_subj_iti_seg, linewidth=0.5, alpha=0.5, label=sync_str)
                ax_t[j,0].set_title('mean ' + version_str + ' ' + sync_str)
                ax_t[j,1].plot(sx_subj_iti_seg, linewidth=0.5, alpha=0.5, label=sync_str)
                ax_t[j,1].set_title('SD ' + version_str + ' ' + sync_str)
                
            else:
                version_str = 'no-timbre'
                ax_n[j,0].plot(mx_subj_iti_seg, linewidth=0.5, alpha=0.5, label=sync_str)
                ax_n[j,0].set_title('mean ' + version_str + ' ' + sync_str)
                ax_n[j,1].plot(sx_subj_iti_seg, linewidth=0.5, alpha=0.5, label=sync_str)                
                ax_n[j,1].set_title('SD ' + version_str + ' ' + sync_str)


error_range = np.arange(6)
for j, sync_str in enumerate(sync_conds):
    for v in t_strs:
        # mx averages
        mx_all = np.array(mx_all_subject_iti_seg[sync_str][v])
        mx_all_segs = np.nanmean(mx_all, axis=0)
        mx_all_segs_error = np.nanstd(mx_all, axis=0)
        # sx averages
        sx_all = np.array(sx_all_subject_iti_seg[sync_str][v])
        sx_all_segs = np.nanmean(sx_all, axis=0)
        sx_all_segs_error = np.nanstd(sx_all, axis=0)


        if v == 't':
            #ax_t[j,0].plot(mx_all_segs, color='r')
            #ax_t[j,1].plot(sx_all_segs, color='r')
            
            ax_t[j,0].errorbar(error_range, mx_all_segs, yerr=mx_all_segs_error, label=sync_str, marker='.', color='r', capsize=3)
            ax_t[j,1].errorbar(error_range, sx_all_segs, yerr=sx_all_segs_error, label=sync_str, marker='.', color='r', capsize=3)
            
        else:
            #ax_n[j,0].plot(mx_all_segs, color='r')
            #ax_n[j,1].plot(sx_all_segs, color='r')
            ax_n[j,0].errorbar(error_range, mx_all_segs, yerr=mx_all_segs_error, label=sync_str, marker='.', color='r',capsize=3)
            ax_n[j,1].errorbar(error_range, sx_all_segs, yerr=sx_all_segs_error, label=sync_str, marker='.', color='r',capsize=3)
                    
   
fig_t.suptitle('Timbre ITI')
fig_n.suptitle('No-Timbre ITI')        
   
fig_t.tight_layout()
fig_n.tight_layout() 

for ax_1, ax_2 in zip(ax_t[:,0].flat, ax_n[:,0].flat):
    ax_1.set_ylim([0, 2])
    ax_2.set_ylim([0, 2])
for ax_1, ax_2 in zip(ax_t[:,1].flat, ax_n[:,1].flat):
    ax_1.set_ylim([-0.5, 1.5])
    ax_2.set_ylim([-0.5, 1.5])
for ax_1, ax_2 in zip(ax_t[-1].flat, ax_n[-1].flat):
    ax_1.set_xlabel('beat segment')               
    ax_2.set_xlabel('beat segment')
    ax_1.set_xticks(np.arange(0,6))               
    ax_1.set_xticklabels([str(elem) for elem in np.arange(1,7)])               
    ax_2.set_xticks(np.arange(0,6))               
    ax_2.set_xticklabels([str(elem) for elem in np.arange(1,7)])                 

fig_t.savefig(subject_itis_dir + '/t-mx-aggregates.png', dpi=150)                
fig_n.savefig(subject_itis_dir + '/n-mx-aggregates.png', dpi=150)                
        
#%% CREATE CSV of mx, sx for usable subjects
import csv
csv_filename_str = csvs_dir + timestamp + '-' + beat_binning_flag + '.csv'
 
with open(csv_filename_str, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["subject", "study", "version", "condition", "section", "mx", "sx"])
    
    for sync_str in sync_conds:
        iti_segment_mx[sync_str], iti_segment_sx[sync_str] = {}, {}
        iti_mx[sync_str], iti_sx[sync_str] = {}, {}
        iti_mx_error[sync_str], iti_sx_error[sync_str] = {}, {}
        
        for v in t_strs:
            iti_mx[sync_str][v], iti_sx[sync_str][v] = [], []
            iti_mx_error[sync_str][v], iti_sx_error[sync_str][v] = [], []
                        
            for person in usable_subjects: 
                print(person, sync_str, v)
                study_number = person.split('_')[1].split('-')[1]
                print([len(elem) for elem in subject_sync_cond_itis[person][sync_str][v]])
                
                df_taps = pd.DataFrame(subject_sync_cond_itis[person][sync_str][v])
                
                iti_segment_mx[sync_str][v], iti_segment_sx[sync_str][v] = [], []
                
                for beatseg, beat_str in zip(beatsegments, beat_strs):
                    tap_col = df_taps.iloc[:,beatseg[0]:beatseg[1]].values
                    # get mean, std
                    mx = np.nanmean(tap_col)
                    #sx = np.nanmean(np.nanstd(tap_col, axis=1))
                    sx = np.nanstd(tap_col)
                    #sx = np.nanstd(np.nanstd(tap_col, axis=1))
               
                    iti_segment_mx[sync_str][v].append(mx) 
                    iti_segment_sx[sync_str][v].append(sx)
                    
                    writer.writerow([person, study_number, v, sync_str, beat_str, mx, sx])
                
                # accumulate mean iti, and sd iti per person per sync_cond, nb: have to take means later
                iti_mx[sync_str][v].append(iti_segment_mx[sync_str][v])
                iti_sx[sync_str][v].append(iti_segment_sx[sync_str][v])

            subject_iti_means = np.array(iti_mx[sync_str][v])
            subject_iti_stds = np.array(iti_sx[sync_str][v])
                   
            # COMPUTER SUBJECT-TO-SUBJECT ERRORS FOR MX,SX PER SYNC_COND
            iti_mx_error[sync_str][v] = np.nanstd(subject_iti_means, axis=0)
            iti_sx_error[sync_str][v] = np.nanstd(subject_iti_stds, axis=0)
print('created csv file:', csv_filename_str)           
#%%############ PLOT THE SUBJECT MEAN and SD ITIs  #############
################################################################          
sns.set()
sns.set_palette(sns.color_palette("tab10"))

plt.figure()

fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(10,5), sharex=True, sharey='row',gridspec_kw={"height_ratios":[0.02,1,1]})

# labels for beat sections 
xlabels_pos = np.arange(0,len(beatsegments))
beatsectionlabels = [str(elem[0]) + '-' + str(elem[1]) for elem in beatsegments] # form str array for plotting

import matplotlib.ticker as ticker

filemx = open(csvs_dir + timestamp + '-mx-global-table.csv', 'w')
writermx = csv.writer(filemx)
filesx = open(csvs_dir + timestamp + '-sx-global-table.csv', 'w')
writersx = csv.writer(filesx)
file = open(csvs_dir + timestamp + '-global-table.csv', 'w')
writer = csv.writer(file)


writer.writerow(["condition", 'version', "beat section", "mx", "sx"])
header = ['condition', 'version']
[header.append(str(b)) for b in range(len(beatsegments))]
writermx.writerow(header)
writersx.writerow(header)

        

for i, v in enumerate(t_strs):
    for j, sync_str in enumerate(sync_conds[:]):
        iti_mx_arr = np.array(iti_mx[sync_str][v])
        iti_sx_arr = np.array(iti_sx[sync_str][v])
        
        subject_iti_means = np.nanmean(iti_mx[sync_str][v], axis=0)
        subject_iti_stds = np.nanmean(iti_sx[sync_str][v], axis=0)
        
        xrange = np.linspace(0, len(subject_iti_means) - 1, len(subject_iti_means))
        xrange += j/6 # shift over to better see error bars
        # MEANS
        #ax[1,i].plot(subject_iti_means, linewidth=0.8, label=sync_str)
        ax[1,i].errorbar(xrange, subject_iti_means, 
                         yerr=iti_mx_error[sync_str][v], label=sync_str, 
                         capsize=3, linewidth=0.7, 
                         elinewidth=0.5, markeredgewidth=0.5)        
        # STDS
        #ax[2,i].plot(subject_iti_stds, linewidth=0.8, label=sync_str)
        ax[2,i].errorbar(xrange, subject_iti_stds, 
                         yerr=iti_sx_error[sync_str][v], label=sync_str, 
                         capsize=3, linewidth=0.7, 
                         elinewidth=0.5,  markeredgewidth=0.5)        
        
        ax[2,i].set_title('ITI SD')
        ax[1,i].set_title('ITI MEAN')
        

        ax[1,i].set_xticks(xlabels_pos)
        ax[2,i].set_xticklabels(beatsectionlabels, fontsize=6)
        ax[2,i].set_xlabel('beat segment')

        writermx.writerow([sync_str, v, subject_iti_means[0], subject_iti_means[1], 
                           subject_iti_means[2], subject_iti_means[3], 
                           subject_iti_means[4], subject_iti_means[5]])
        writersx.writerow([sync_str, v, subject_iti_stds[0], subject_iti_stds[1], 
                           subject_iti_stds[2], subject_iti_stds[3], 
                           subject_iti_stds[4], subject_iti_stds[5]])

cols = ['{}'.format(col) for col in ['Timbre', 'No Timbre']]
for ax, col in zip(ax[0], cols):
    ax.axis("off")
    ax.set_title(col, fontweight='bold')
    
plt.legend(title='coupling conditions', bbox_to_anchor=(1., 1.05))
fig.tight_layout()
plt.savefig(itis_dir + timestamp + ' timbre-no-timbre-ITI-SD-' + beat_binning_flag + '.png', dpi=160)

file.close()
filemx.close()
filesx.close() 

#%% run impute_mx() function from utils/impute_means.py 
from utils.impute_means import impute_mx
impute_mx(csv_filename_str)
#%% ##### PLOT ALL SUBJECTS TAPS ON CIRCLE MAPS PER STIMULI AND AGGREGATE/POOLED TAPS 

def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins) +1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    avg_taps_per_bin = []
    for i, tap in enumerate(taps):
        try:
            num_taps_in_bin = len(taps[i])
            avg_taps_per_bin.append(num_taps_in_bin)

            if num_taps_in_bin > 1:   
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) # take random tap in window if multiple in one window
            if num_taps_in_bin == 0:
                binnedtaps.append(np.nan)
            if num_taps_in_bin == 1:
                binnedtaps.append(taps[i][0])
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim

maxbeats = 18
maxsubjects = 38
cmapdir = './analysis-scripts/plots/beat-segment-analysis/6-beat-segments/cmaps/'
# sync_cond_version = 'weak_t_72_2'
# sync_cond_version = 'strong_n_81_4'


# fig, ax = plt.subplots(nrows=len(people)+1, ncols=maxbeats, subplot_kw=dict(polar=True), 
#                        gridspec_kw= {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
#                        figsize=(25,20), 
#                        sharex=True)
# for ax_ in ax.flat:
#     ax_.set_thetagrids([])
#     ax_.set_yticklabels([])
#     ax_.set_axisbelow(True)
#     ax_.grid(linewidth=0.1, alpha=1.0)

#person = 'PARTICIPANT_kura-B2_2020-09-25_19h37.04.527.csv'
# sync_cond_version = 'weak_n_81_4'
# sync_cond_version = 'weak_n_105_4'
# sync_cond_version = 'none_n_81_3'
# sync_cond_version = 'weak_n_105_3'
# sync_cond_version = 'strong_n_81_4'

#PARTICIPANT_kura-A1_2020-10-05_09h41.04.130.csv

#sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])

sync_batch = all_timbre_conds[-1][-1][0:2] # nt strong all

for k, sync_cond_version in enumerate(sync_batch):

    people = []
    for person in usable_subjects[:maxsubjects]:
        try:
            subject_resps[person][sync_cond_version]
            people.append(person)
        except:
            pass    
 
    fig, ax = plt.subplots(nrows=len(people)+1, ncols=maxbeats, subplot_kw=dict(polar=True), 
                           gridspec_kw= {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                           figsize=(25,20), 
                           sharex=True)
    for ax_ in ax.flat:
        ax_.set_thetagrids([])
        ax_.set_yticklabels([])
        ax_.set_axisbelow(True)
        ax_.grid(linewidth=0.1, alpha=1.0)
    
    sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
    
    pooltaps = [] # for plotting aggregate taps per beat 
    for n, person in enumerate(people):
        print('working on snd %r/%r for subject %r/%r'%(k, len(sync_batch), n,len(people)))
        taps = subject_resps[person][sync_cond_version]
        
        # plt.figure(figsize=(20,5))
        # plt.vlines(librosa.time_to_samples(sndbeatbin), -0.5, 1, color='red', lw=0.7)
        # plt.vlines(librosa.time_to_samples(taps), -1, 0.5, color='blue', lw=1)
        # y, sr_ = librosa.load('./allstims/' + sync_cond_version + '.wav')
        # plt.plot(y, linewidth=0.5, alpha=0.5, color='grey')
        
        binnedbeats = binBeats(taps, sndbeatbin)
        #bcat = np.concatenate(binnedbeats, axis=0)        
        bb, _ = binTapsFromBeatWindow(binnedbeats)
        bb = [elem if np.isnan(elem) == False else np.nan for elem in bb]
        #plt.vlines(librosa.time_to_samples(bcat),-2,-1,color='green')
        #plt.vlines(librosa.time_to_samples(bb),-3,-2,color='orange')
        
        
        cmaptaps = []
        for i in range(1, len(sndbeatbin)):
            taptobin = bb[i-1]
            binmin = sndbeatbin[i-1]
            binmax = sndbeatbin[i]
            bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
            cmaptaps.append(float(bininterp(taptobin)))
          
        pooltaps.append(cmaptaps)
            
        for b,tap in enumerate(cmaptaps[:maxbeats]):
            ax[n,b].plot(tap, 0.8, 'ro')
            ax[n,b].arrow(0, 0, tap, 0.8, color='r', linewidth=0.5)
            ax[n,b].arrow(0, 0, np.pi, 1, color='black', linewidth=0.5)
        
        ax[n,0].set_title(person)
            

            #print('this person:', person, 'did not tap to', sync_cond_version)
    
        
    df = pd.DataFrame(pooltaps)
    for bseg in range(maxbeats):
        bpool = df.iloc[:,bseg].values
        rnoise = 0.2*np.random.random(size=len(bpool))    
        R = np.abs(np.nanmean(np.exp(1j*bpool)))
        psi = np.angle(np.nanmean(np.exp(1j*bpool)))
                
        ax[len(people), bseg].scatter(bpool,1-rnoise, s=6, alpha=0.3)    # scattered
        ax[len(people), bseg].arrow(0,0,psi,R, color='red',linewidth=0.9) # phase coherence phasor
        ax[len(people), bseg].arrow(0,0,np.pi,1,color='black', linewidth=0.5) # center of beat arrow
        
    #ax_subject[sc + vc,m].scatter(subject_beat_column_pooled_taps, 1-randomnoise_subject, s=12, alpha=0.08, c=subject_version_color, marker='.', edgecolors='none')
        
    
    fig.suptitle('circle map taps per beat ' + sync_cond_version)
    fig.savefig(cmapdir + sync_cond_version + '.png', dpi=150)
    plt.close(fig)
#%%
fig, ax = plt.subplots(2,2)
ax[0].plot(np.random.random(size=10))
ax[1].plot(np.random.random(size=10))
plt.close(fig)
#%%
#########################################################################
################### PHASE COHERENCE ANALYSIS ###########################################
###############################################################################




def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins) +1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    avg_taps_per_bin = []
    for i, tap in enumerate(taps):
        try:
            num_taps_in_bin = len(taps[i])
            avg_taps_per_bin.append(num_taps_in_bin)

            if num_taps_in_bin > 1:   
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) # take random tap in window if multiple in one window
            if num_taps_in_bin == 0:
                binnedtaps.append(np.nan)
            if num_taps_in_bin == 1:
                binnedtaps.append(taps[i][0])
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim


#%%
         

sns.set()
sns.set_palette(sns.color_palette("Paired"))

#pc_beat_windows = [(0,4),(4,8),(8,12),(12,16)] # beat windows to form beat columns 


binned_taps_per_cond = {}
subject_binned_taps_per_cond = {}
subject_binned_taps_per_stim = {}

all_osc_binned_taps_per_stim = {}

all_subject_binned_taps_per_stim = {}
all_subject_binned_taps_per_cond = {}

all_subject_taps_per_cond = {}

plt.figure()

fig_subject, ax_subject = plt.subplots(nrows=8, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)
plt.figure()
fig_model, ax_model = plt.subplots(nrows=8, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)

plt.figure()
fig_combined, ax_combined = plt.subplots(nrows=8, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)


for ax_s, ax_m, ax_c in zip(ax_subject.flat, ax_model.flat, ax_combined.flat):
    ax_s.set_thetagrids([])
    ax_s.set_yticklabels([])
    ax_s.set_axisbelow(True)
    ax_s.grid(linewidth=0.1, alpha=1.0)

    ax_m.set_thetagrids([])
    ax_m.set_yticklabels([])
    ax_m.set_axisbelow(True)
    ax_m.grid(linewidth=0.1, alpha=1.0)

    ax_c.set_thetagrids([])
    ax_c.set_yticklabels([])
    ax_c.set_axisbelow(True)
    ax_c.grid(linewidth=0.1, alpha=1.0)



sns.set(style='darkgrid')

the_osc_phases = {}
osc_phases_cond = {}

random_color = np.random.random(4000)

import csv

R_csv = open('./analysis-scripts/plots/beat-segment-analysis/' + str(len(beatsegments)) + '-beat-segments/PCs/R_csv.csv', 'w')
R_writer = csv.writer(R_csv)
R_writer.writerow(['coupling condition', 'beat section', 'timbre version', 'R model', 'R subject', 'psi model', 'psi subject'])

vc = 0
for version,v in zip(all_timbre_conds, t_strs):
    the_osc_phases[v] = {}
    osc_phases_cond[v] = {}
    
    all_subject_binned_taps_per_stim[v], all_subject_binned_taps_per_cond[v] = {}, {}
    all_subject_taps_per_cond[v] = {}
        
    sc = 0
    for sync_cond, sync_str in zip(version, sync_conds):
        osc_phases_cond[v][sync_str] = []
        all_subject_binned_taps_per_cond[v][sync_str] = []
        
        for n, sync_cond_version in enumerate(sync_cond):
            #print('working on %r %r/%r'%(sync_cond_version, n, len(sync_cond)))
            
            osc_phases = {}
            stim_phases_sec = {}
            
            sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
            y, _ = librosa.load('./allstims/' + sync_cond_version + '.wav')
            phases = np.load('./all-stim-all-zcs/' + sync_cond_version + '.npy', allow_pickle=True)

            the_osc_phases[v][sync_cond_version] = []

            ################## GENERATIVE MODEL ##################################
            for p, osc in enumerate(phases):
                binned_zcs = binBeats(osc, sndbeatbin)
                binned_zcs, _ = binTapsFromBeatWindow(binned_zcs)
                osc_phases[str(p)] = []
                
                for i in range(1, len(sndbeatbin)):
                    zctobin = binned_zcs[i-1]
                    binmin = sndbeatbin[i-1]
                    binmax = sndbeatbin[i]
                    bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                    osc_phases[str(p)].append(float(bininterp(zctobin)))
                
                the_osc_phases[v][sync_cond_version].append(osc_phases[str(p)])
            
            osc_phases_cond[v][sync_str].extend(the_osc_phases[v][sync_cond_version])
            
            ################# SUBJECTS TAPS ###################################
            all_subject_binned_taps_per_stim[v][sync_cond_version] = []  
            
            for person in usable_subjects:
                try:
                    taps = subject_resps[person][sync_cond_version]
                    
                    tap_iti = np.diff(taps)
                    tap_mean = np.nanmean(tap_iti)
                    tap_std = np.nanstd(tap_iti)
                    
                    #tap_resps_algo = [tap for tap in tap_iti if (tap < tap_mean + 2*tap_std) and (tap >= tap_mean - 2*tap_std)]
                    tap_resps_secs = [taps[t] for t, tap in enumerate(tap_iti) if (tap < tap_mean + 2*tap_std) and (tap >= tap_mean - 2*tap_std)]
                    #binned_taps = tap_resps_secs                   
 
                    
                    #binned_taps = binBeats(tap_resps_secs, sndbeatbin)
                    binned_taps = binBeats(taps, sndbeatbin)

                    binned_taps, avg_taps_per_bin = binTapsFromBeatWindow(binned_taps) 
                     
                    subject_binned_taps_per_stim[person] = []
                    
                    for i in range(1, len(sndbeatbin)):
                        taptobin = binned_taps[i-1]
                        binmin = sndbeatbin[i-1]
                        binmax = sndbeatbin[i]
                        bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                        subject_binned_taps_per_stim[person].append(float(bininterp(taptobin)))
                        
                    all_subject_binned_taps_per_stim[v][sync_cond_version].append(subject_binned_taps_per_stim[person])
      
                except:
                    pass
                                
            all_subject_binned_taps_per_cond[v][sync_str].extend(all_subject_binned_taps_per_stim[v][sync_cond_version])        
                
        ############### take dataframes from subject, model per coupling cond ###################################
        df_model  = pd.DataFrame(osc_phases_cond[v][sync_str]) 
        df_subject = pd.DataFrame(all_subject_binned_taps_per_cond[v][sync_str])     # (470,19) = (5*tempo*2versions * 47 subjects) for coupling cond and timbre version                        
                     
        
        #### COLORS FOR SUBJECT AND MODEL ###########                       
        if v == "t":
            model_version_color = 'steelblue'
            subject_version_color = 'steelblue'            
        else:
            model_version_color = 'firebrick'
            subject_version_color = 'firebrick' 
                        
        for m, beatwindow in enumerate(beatsegments):
            # SUBJECT
            subject_beat_column = df_subject.iloc[:, beatwindow[0]:beatwindow[1]].values
            subject_beat_column_pooled_taps = subject_beat_column.flatten()
                        
            # MODEL
            model_beat_column = df_model.iloc[:, beatwindow[0]:beatwindow[1]].values
            model_beat_column_pooled_taps = model_beat_column.flatten()            
            
            # calculate SUBJECT phase coherence
            R_subject = np.abs(np.nanmean(np.exp(1j*subject_beat_column_pooled_taps)))
            psi_subject = np.angle(np.nanmean(np.exp(1j*subject_beat_column_pooled_taps)))
            
            # calculate MODEL phase coherence
            R_model = np.abs(np.nanmean(np.exp(1j*model_beat_column_pooled_taps)))
            psi_model = np.angle(np.nanmean(np.exp(1j*model_beat_column_pooled_taps)))
            #print(v, sync_str, R_model)
            
            randomnoise_subject = np.random.random(len(subject_beat_column_pooled_taps))*0.3
            randomnoise_model = np.random.random(len(model_beat_column_pooled_taps))*0.3
            
            ##### CENTER THE SUBJECT TAPS SO THAT THEY ARE NORMALIZED TO THE MODEL TAPS PSI @ 0 deg? 
            # psi_model_centered = psi_model - np.pi + (np.pi - psi_model)
            # psi_subject_centered = psi_subject - np.pi + (np.pi - psi_model)
            # if psi_subject_centered > np.pi: 
            #     psi_subject_centered = 2*np.pi - psi_subject_centered
            # if psi_subject_centered < -np.pi:
            #     psi_subject_centered = 2*np.pi + psi_subject_centered
                      
            # subject_beat_column_pooled_taps -= psi_model
            # model_beat_column_pooled_taps = model_beat_column_pooled_taps - np.pi + (np.pi - psi_model)
            
            ## or make center beat on right hemisphere of circle
            #psi_model_centered = psi_model - np.pi 
            psi_subject_centered = psi_subject - psi_model 
            subject_beat_column_pooled_taps -= np.pi 
            model_beat_column_pooled_taps -= psi_model
    
            # PLOT AX OF SUBJECT
            ax_subject[sc + vc,m].plot(np.arange(2), np.arange(2), alpha=0, color='white') # this is a phantom line, for some reason there's a bug with the arrow in this polar plot so just make it transparant and length 1                      
            ax_subject[sc + vc,m].scatter(subject_beat_column_pooled_taps, 1-randomnoise_subject, s=12, alpha=0.08, c=subject_version_color, marker='.', edgecolors='none')
            ax_subject[sc + vc,m].arrow(0, 0.0, psi_subject_centered, R_subject, color='firebrick', linewidth=1)
    
    
            # PLOT AX OF MODEL 
            ax_model[sc + vc, m].plot(np.arange(2), np.arange(2), alpha=0, color='white') # this is a phantom line, for some reason there's a bug with the arrow in this polar plot so just make it transparant and length 1                      
            ax_model[sc + vc,m].scatter(model_beat_column_pooled_taps, 1-randomnoise_model, s=12, alpha=0.05, c=model_version_color, marker='.', edgecolors='none')
            ax_model[sc + vc,m].arrow(0, 0.0, 0, R_model, color='black', linewidth=1)
    
    
            # COMBINED SUBJECT + MODEL 
            ax_combined[sc + vc, m].plot(np.arange(2), np.arange(2), alpha=0, color='white') # this is a phantom line, for some reason there's a bug with the arrow in this polar plot so just make it transparant and length 1            
            ax_combined[sc + vc,m].scatter(subject_beat_column_pooled_taps, 0.7-randomnoise_subject, s=12, alpha=0.05, c='blueviolet', marker='.', edgecolors='none', zorder=0)
            ax_combined[sc + vc,m].arrow(0, 0.0, psi_subject_centered, R_subject, color='red', linewidth=0.7, zorder=2)            
            ax_combined[sc + vc,m].scatter(model_beat_column_pooled_taps, 1-randomnoise_model, s=12, alpha=0.05, c='steelblue', marker='.', edgecolors='none', zorder=0)
            ax_combined[sc + vc,m].arrow(0, 0.0, 0, R_model, color='blue', linewidth=0.7, zorder=1)
    
            ##### WRITE POOLED PHASES PER BEAT SEG TO TXT FILE TO RUN STATS IN R script 
            model_phases_txtfile = model_dir + "/phases/" + v + '-' + sync_str + "-" + str(m) + ".txt"
            np.savetxt(model_phases_txtfile, model_beat_column_pooled_taps, delimiter=',')            
            subject_phases_txtfile = subject_dir + "/phases/" + v + '-' + sync_str + "-" + str(m) + ".txt" 
            np.savetxt(subject_phases_txtfile, subject_beat_column_pooled_taps, delimiter=',')
    
            ### write R and ang per beat to csv
            if psi_model < 0:
                psi_model += 2*np.pi
            if psi_subject < 0: 
                psi_subject += 2*np.pi
            #R_writer.writerow([sync_str, str(m), str(v), R_model, R_subject, str(np.degrees(psi_model_centered)), str(np.degrees(psi_subject_centered))])            
            R_writer.writerow([sync_str, str(m), str(v), R_model, R_subject, str(np.degrees(psi_model)), str(np.degrees(psi_subject))])            
                
        sc += 1
    vc += 4

R_csv.close()

fig_combined.suptitle('Timbre and No Timbre Phase Coherence Per Beat Segment')
fig_subject.suptitle('Subject Phase Coherence Per Beat Segment')
fig_model.suptitle('Generative Model Phase Coherence Per Beat Segment')



colabels = [str(beatsegment) for beatsegment in beatsegments]

for ax, col in zip(ax_subject[0], colabels):
    ax.set_title(col, fontsize=10)
for ax, col in zip(ax_model[0], colabels):
    ax.set_title(col, fontsize=10)
for ax, col in zip(ax_combined[0], colabels):
    ax.set_title(col, fontsize=10)



rowlabels = ['none', 'weak', 'medium', 'strong', 'none', 'weak', 'medium', 'strong']
cnt = 0
for ax, row in zip(ax_subject[:,0], rowlabels):
    ax.set_ylabel(row, rotation=90, size='large', fontsize=8)
    
for ax, row in zip(ax_model[:,0], rowlabels):
    ax.set_ylabel(row, rotation=90, size='large', fontsize=8)
for ax, row in zip(ax_combined[:,0], rowlabels):
    ax.set_ylabel(row, rotation=90, size='large', fontsize=8)
   
fig_model.text(0.5, 0.04, 'beat segment', ha='center', va='center')
fig_subject.text(0.5, 0.04, 'beat segment', ha='center', va='center')
fig_combined.text(0.5, 0.04, 'beat segment', ha='center', va='center')
     
#plt.savefig(model_dir + 'gen model distributions.png', dpi=160)

fig_subject.savefig(subject_dir + 'subject  distributions.png', dpi=200)

fig_model.savefig(model_dir + 'model distributions.png', dpi=200)

fig_combined.savefig(model_dir + '../subject and model distributions.png', dpi=200)


#%%

import itertools
from astropy.stats import rayleightest, kuiper 

def rnd(num):
    return np.round(num,2)

def wrap(ph):
    phases = (ph + np.pi) % (2 * np.pi) - np.pi
    return phases

f = open(model_dir + './R_psi.csv', 'w')
writer = csv.writer(f)
writer.writerow(['coupling', 'timbre', 'R_m', 'R_s', 'psi_m', 'psi_s', 'rayleigh model', 'rayleigh subject'])

# fig, ax = plt.subplots(nrows=3, ncols=1, 
#                        gridspec_kw={'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
#                             figsize=(10,10), 
#                             sharex=True)

fig, ax = plt.subplots(nrows=2, ncols=4, subplot_kw=dict(polar=True), 
                       gridspec_kw={'wspace':0.2,'hspace':0.01,'top':0.9, 
                                    'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(15,7), 
                            sharex=True)

fig_h, ax_h = plt.subplots(nrows=2, ncols=4, 
                           sharey=True, sharex=True, figsize=(15,5))

# for 
for ax_, sync_str in zip(ax[0].flat, sync_conds):
    ax_.set_thetagrids([])
    ax_.set_yticklabels([])
    ax_.grid(linewidth=0.1, alpha=1.0)
    ax_.set_title('timbre ' + sync_str)
for ax_, sync_str in zip(ax[1].flat, sync_conds):
    ax_.set_thetagrids([])
    ax_.set_yticklabels([])
    ax_.grid(linewidth=0.1, alpha=1.0)
    ax_.set_title('no timbre ' + sync_str)

# for histograms 
for ax_, sync_str in zip(ax_h[0].flat, sync_conds):
    ax_.set_yticklabels([])
    # ax__.set_axisbelow(True)
    ax_.set_title('timbre ' + sync_str)    
    ax_.set_xticks(np.linspace(-np.pi,np.pi,7))
    xlabels = [r'$\pi$', r'-2$\pi /3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'2$\pi/3$', r'$\pi$']
    ax_.set_xticklabels(xlabels, fontsize=8)
for ax_, sync_str in zip(ax_h[1].flat, sync_conds):
    ax_.set_yticklabels([])
    # ax__.set_axisbelow(True)
    ax_.set_title('no timbre ' + sync_str)    
    ax_.set_xticks(np.linspace(-np.pi,np.pi,7))
    xlabels = [r'$\pi$', r'-2$\pi /3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'2$\pi/3$', r'$\pi$']
    ax_.set_xticklabels(xlabels, fontsize=8)
    

from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde, norm
import matplotlib.mlab as mlab
import scipy.stats as stats


for i, v in enumerate(t_strs):    
    for j, sync_str in enumerate(sync_conds[:]):
        ######## MODEL ##########
        mtaps = np.array(list(itertools.chain(*osc_phases_cond[v][sync_str])))
        mtaps = mtaps[~np.isnan(mtaps)] # remove nans
        noise = np.random.random(len(mtaps))*0.3
        
        R_m = np.abs(np.nanmean(np.exp(1j*mtaps)))
        psi_m = np.angle(np.nanmean(np.exp(1j*mtaps)))
    
        mtaps = mtaps - psi_m 
    
        
        ax[i,j].scatter(mtaps, 1-noise, s=20, alpha=0.05, 
                      c='blue', marker='.', edgecolors='none', 
                      zorder=0)
        ax[i,j].arrow(0, 0.0, 0, R_m, color='black', linewidth=1, zorder=2)            
    
        phases = wrap(mtaps) # wrap around -pi to pi
        print(v, sync_str, len(phases), i, j)
        nbins = 60 # num bins for histo
        n, bins, p = ax_h[i,j].hist(phases, bins=nbins, 
                                    density=True, 
                                    alpha=0.8, lw=0)

        # get best fit line 
        (mu, sigma) = norm.fit(phases)
        y = stats.norm.pdf(bins, mu, sigma)
        ax_h[i,j].plot(bins, y, 'r--', linewidth=1 )
    
    
        ######### SUBJECTS #################
        staps = np.array(list(itertools.chain(*all_subject_binned_taps_per_cond[v][sync_str])))
        staps = staps[~np.isnan(staps)] # remove nans
    
        snoise = np.random.random(len(staps))*0.3
    
        staps = staps - psi_m    
    
        R_s = np.abs(np.nanmean(np.exp(1j*staps)))
        psi_s = np.angle(np.nanmean(np.exp(1j*staps)))
        
    
        ax[i,j].scatter(staps, 0.7-snoise, s=20, alpha=0.05, 
                      c='purple', marker='.', edgecolors='none', 
                      zorder=0)
        ax[i,j].arrow(0, 0.0, psi_s, R_s, color='red', linewidth=1, zorder=2)            
        
        #write to csv
       
        raymod = rayleightest(mtaps)
        submod = rayleightest(staps)
        writer.writerow([sync_str, v, rnd(R_m), rnd(R_s), rnd(psi_m), rnd(psi_s), raymod, submod])
    

fig.tight_layout()    
fig.savefig(model_dir + './model all beat sections collapsed.png', dpi=200)
fig_h.savefig(model_dir + './model histogram densities.png', dpi=160)
f.close()
#%%





