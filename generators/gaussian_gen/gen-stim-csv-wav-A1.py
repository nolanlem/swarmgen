#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:21:26 2020

@author: nolanlem
"""

from random import shuffle
import glob 
import csv 
import os 
import string
import numpy as np
import librosa
os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/psychopy-A1') 


#%%#%%# NB: need to run on python 3 spyder environment e.g. conda activate py3


notimbre_basedir = 'stim-no-timbre-all-20/' # where to read audio stims for notimbre
timbre_basedir = 'stim-timbre-all-20/'      # '' timbre

block_1 = 'no'
block_2 = 'timbre'

numbers_1 = [1,2] # block 1
numbers_2 = [3,4] # block 2
#psychopydir = './psychopy/'

stims_1 = [] # snd path strings for block 1
stims_2 = [] # snd path strings for block 2

#NB: have to change glob wc *.wav to *.mp3 depending on mp3 or wav
wavdir = '/wavs/' 
mp3dir = '/mp3s/'

#%%

# for A1 (no timbre 1,2       timbre 1,2)
# for num in numbers_1: 
#     stimdir = notimbre_basedir + 'stimuli_' + str(num)  
# #    for stim in glob.glob(stimdir + "*.wav"):
# #        stims_1.append(stim)
#     for stim in glob.glob(stimdir + "/*.wav"):
#         stims_1.append(stim)

# for num in numbers_1: 
#     stimdir = timbre_basedir + 'stimuli_' + str(num) 
# #    for stim in glob.glob(stimdir + "*.wav"):
# #        stims_2.append(stim)
#     for stim in glob.glob(stimdir + "/*.wav"):
#         stims_2.append(stim)

# # randomize all the stims in block 1,2
# shuffle(stims_1)
# shuffle(stims_2)

# stims_practice = stims_1 + stims_2 
# shuffle(stims_practice)
# stims_practice = stims_practice[:4]
# stim_practice_types = [str.split(str.split(os.path.split(snd)[0], '/')[0], '-')[1] for snd in stims_practice]
# stim_1_types = [str.split(str.split(os.path.split(snd)[0], '/')[0], '-')[1] for snd in stims_1]
# stim_2_types = [str.split(str.split(os.path.split(snd)[0], '/')[0], '-')[1] for snd in stims_2]


stims_1 = []
stims_2 = []

for stim in glob.glob('block_1/*.wav'):
    stims_1.append(stim)
for stim in glob.glob('block_2/*.wav'):
    stims_2.append(stim)
    
stims_practice = stims_1 + stims_2 
shuffle(stims_practice)
stims_practice = stims_practice[:4]


#%%
conditions_practice = 'conditions_practice_wav.csv'
conditions_test_1 = 'conditions_test_1_wav.csv'
conditions_test_2 = 'conditions_test_2_wav.csv'

# write csv conditions practice file
with open(conditions_practice, 'w') as file:
    writer = csv.writer(file) 
    writer.writerow(["sndfile", "type", "tempo", "cond", "version", "delay", "dur"])
    for i, stim in enumerate(stims_practice):
        
        y, sr = librosa.load(stim)
        duration = len(y)/float(sr)
        
        tempo = str.split(os.path.basename(stim), "_")[1]
        cond = str.split(os.path.basename(stim), "_")[0]
        version = str.split(str.split(os.path.basename(stim), "_")[2], ".")[0]          
        delay = np.random.uniform(low=1.0, high=2.5) 
        total_dur = duration + delay
        
        if os.path.split(stim)[0] == 'block_1':
            timbretype = block_1
        if os.path.split(stim)[0] == 'block_2':
            timbretype = block_2 
        writer.writerow([stim, timbretype , tempo, cond, version, delay, total_dur])

# write csv conditions file for block 1
with open(conditions_test_1, 'w') as file:
    writer = csv.writer(file) 
    writer.writerow(["sndfile", "type", "tempo", "cond", "version", "delay", "dur"])
    for i, stim in enumerate(stims_1):
        y, sr = librosa.load(stim)
        duration = len(y)/float(sr)        
        
        tempo = str.split(os.path.basename(stim), "_")[1]
        cond = str.split(os.path.basename(stim), "_")[0]
        version = str.split(str.split(os.path.basename(stim), "_")[2], ".")[0]          
        delay = np.random.uniform(low=1.0, high=2.5)  
        total_dur = duration + delay        
        
        if os.path.split(stim)[0] == 'block_1':
            timbretype = block_1
        if os.path.split(stim)[0] == 'block_2':
            timbretype = block_2         
        writer.writerow([stim, timbretype, tempo, cond, version, delay, total_dur])

# write csv conds file for block 2
with open(conditions_test_2, 'w') as file:
    writer = csv.writer(file) 
    writer.writerow(["sndfile", "type", "tempo", "cond", "version", "delay", "dur"])
    for i, stim in enumerate(stims_2):
        y, sr = librosa.load(stim)
        duration = len(y)/float(sr)          
        
        tempo = str.split(os.path.basename(stim), "_")[1]
        cond = str.split(os.path.basename(stim), "_")[0]
        version = str.split(str.split(os.path.basename(stim), "_")[2], ".")[0]          
        delay = np.random.uniform(low=1.0, high=2.5)        
        total_dur = duration + delay                
        
        if os.path.split(stim)[0] == 'block_1':
            timbretype = block_1
        if os.path.split(stim)[0] == 'block_2':
            timbretype = block_2         
        writer.writerow([stim, timbretype, tempo, cond, version, delay, total_dur])
#%%

import librosa
all_stim_durs = []
thedir = './psychopy/'

for num in numbers_1: 
    stimdir = thedir + 'stimuli_' + str(num) + '/wavs/'
    for stim in glob.glob(stimdir + "*.mp3"):
        y, sr = librosa.load(stim)
        length_sec = len(y)/float(sr)
        all_stim_durs.append(length_sec)

for num in numbers_2: 
    stimdir = thedir + 'stimuli_' + str(num) + '/wavs/'
    for stim in glob.glob(stimdir + "*.mp3"):
        y, sr = librosa.load(stim)
        length_sec = len(y)/float(sr)
        all_stim_durs.append(length_sec)
#%%
total_time = np.sum(all_stim_durs)
total_time = total_time + np.sum(np.random.uniform(low=1.0, high=2.5, size=(len(stims_1)+len(stims_2))))
print 'total time of stims %r minutes' %(total_time/60.)