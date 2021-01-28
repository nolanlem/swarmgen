#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Thu Dec 10 15:56:32 2020

@author: nolanlem
"""

import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import soundfile as sf
import os
import sys
import librosa
import glob
from scipy.interpolate import interp1d
os.chdir('/Users/nolanlem/Documents/kura/kura-git/swarm-tapping-study/generators')
import seaborn as sns 
from scipy.ndimage import gaussian_filter1d
if '/Users/nolanlem/Documents/kura/kura-git/swarm-tapping-study/' not in sys.path:
    sys.path.insert(0,'/Users/nolanlem/Documents/kura/kura-git/swarm-tapping-study/' )
from utils.utils import makeDir, round2dec
import config 
import random 

#%%
# have to run this codeblock everytime you change config file 
%reload_ext autoreload
%autoreload 2

#%%

sns.set()
sns.set_palette('tab10')

sr_audio = config.sr_audio 
# load mono metronome click, non-spatialized 
thesample = './sampleaudio/woodblock_lower.wav'
y, _ = librosa.load(thesample, sr=int(sr_audio))
y = y*0.5 # reduce amp audiofile 

###### load binaural metronome clicks
samples = []
binaural_metros_dir = './sampleaudio/binaural-metros/'
#binaural_metros_dir = '/Users/nolanlem/Documents/TEACHING/jupyter-nbs/conv-outputs-finland/'
ir_flag = config.flags['ir']
#############
spatial_metros_dir = '/Users/nolanlem/Documents/TEACHING/jupyter-nbs/conv-outputs-finland/'
###########

# load audio (spatial or mono) into samples list 
for fi in glob.glob(binaural_metros_dir + '*.wav'):
    y_, _ = librosa.load(fi, mono=False)
    samples.append(y_)

# get largest sample in samples array 
tmpmax = samples[0][0].shape[0]
for deg in range(len(samples)):
    maxnum = samples[deg][0].shape[0]
    if maxnum > tmpmax:
        tmpmax = maxnum
largestsampnum = tmpmax
 

# PER STIMULUS TIME HISTORGRAM 
def PSTH(x, N, len_y):
    # x = spikes/taps matrix 
    # N = 0.1 seconds = 0.1*22050 = 2205 samples
    spike_blocks = np.linspace(0, x.shape[1], int(len_y/N))
    spike_blocks_int = [int(elem) for elem in spike_blocks]
    mx = []
    for i in range(1, len(spike_blocks_int)):
        tapblock_mx = np.nanmean(spikes[:, spike_blocks_int[i-1]:spike_blocks_int[i]])
        block_mx = tapblock_mx*np.ones(spike_blocks_int[i] - spike_blocks_int[i-1])
        mx.extend(block_mx)
    return mx 

def calculateCOP(window, period_samps, dist_type = 'uniform'):
    if dist_type == 'uniform':
        window = [elem for elem in window if elem <= period_samps]
    if dist_type == 'gaussian':
        window = np.array(window) + int(period_samps/2)
        window = [elem for elem in window if elem <= period_samps and elem >= 0]
        
    bininterp = interp1d([0, period_samps], [0, 2*np.pi])
    phases = bininterp(window)   
    R = np.nansum(np.exp(phases*1j))/N
    R_mag = np.abs(R)
    R_ang = np.angle(R)
    R_mag_traj.append(R_mag)
    R_ang_traj.append(R_ang)
    return R_mag, R_ang
        

def makeAudio(events, iteration, stimdir, spatial_flag=False):   
    eventsinsamples = librosa.time_to_samples(events,sr=int(sr_audio))
    # audiobufffers for spatial and mono audio     
    audiobuffer_L = np.zeros(max(eventsinsamples) + largestsampnum)
    audiobuffer_R = np.zeros(max(eventsinsamples) + largestsampnum)  
    y_mono = y
    
    for startpos in eventsinsamples:
        random_deg = np.random.randint(N)
        y_l = samples[random_deg][0]
        y_r = samples[random_deg][1]

        if spatial_flag == True:
            audiobuffer_L[startpos:(startpos + len(y_l))] = audiobuffer_L[startpos:(startpos + len(y_l))] + y_l
            audiobuffer_R[startpos:(startpos + len(y_r))] = audiobuffer_R[startpos:(startpos + len(y_r))] + y_r
        
        if spatial_flag == False:
            audiobuffer_L[startpos:(startpos + len(y_mono))] = audiobuffer_L[startpos:(startpos + len(y_mono))] + y_mono
            audiobuffer_R[startpos:(startpos + len(y_mono))] = audiobuffer_R[startpos:(startpos + len(y_mono))] + y_mono

    audio_l = 0.8*audiobuffer_L/max(audiobuffer_L)
    audio_r = 0.8*audiobuffer_R/max(audiobuffer_R)
    
    audio = np.array([audio_l, audio_r])
#    audiofi = os.path.join(stimdir, iteration + '.wav') # put in tempo folder 
    audiofi = os.path.join(stimdirname, iteration + '.wav') # put in tempo folder 
    
    sf.write(audiofi, audio.T, samplerate=int(sr_audio))
    print('creating', audiofi)
    return audio

def makePulses(events, iteration, stimdir, spatial_flag=False):
    eventsinsamples = librosa.time_to_samples(events,sr=int(sr_audio))
    # audiobufffers for spatial and mono audio     
    audiobuffer_L = np.zeros(max(eventsinsamples) + largestsampnum)
    audiobuffer_R = np.zeros(max(eventsinsamples) + largestsampnum)    
    y_mono = np.array([1])
    
    for startpos in eventsinsamples:
        random_deg = np.random.randint(100)
        y_l = np.array([1])
        y_r = np.array([1])

        if spatial_flag == True:
            random_chan = np.random.randint(2)
            if random_chan == 1:
                audiobuffer_L[startpos:(startpos + len(y_l))] = audiobuffer_L[startpos:(startpos + len(y_l))] + y_l
            else:
                audiobuffer_R[startpos:(startpos + len(y_r))] = audiobuffer_R[startpos:(startpos + len(y_r))] + y_r
        
        if spatial_flag == False:
            audiobuffer_L[startpos:(startpos + len(y_mono))] = audiobuffer_L[startpos:(startpos + len(y_mono))] + y_mono
            audiobuffer_R[startpos:(startpos + len(y_mono))] = audiobuffer_R[startpos:(startpos + len(y_mono))] + y_mono

    audio_l = 0.8*audiobuffer_L/max(audiobuffer_L)
    audio_r = 0.8*audiobuffer_R/max(audiobuffer_R)
    
    audio = np.array([audio_l, audio_r])
    audiofi = os.path.join(stimdir, config.dist_type + '_' + config.strs['quantize'] + '-' + str(config.qsteps) + '_' + config.strs['binaural'][0] + '_' + str(config.N) + '_' + iteration + '.wav')
    sf.write(audiofi, audio.T, samplerate=int(sr_audio))
    print('creating', audiofi)
    return audio   

def getKDE(events):
    eventsinsamples = librosa.time_to_samples(events)
    taps = np.zeros(max(eventsinsamples)+1)
    
    for spike in eventsinsamples:
        np.put(taps, spike, 1)
    blocksize = int(sr_audio/10)
    blocks = np.arange(0, len(taps), blocksize)
    mx = []
    
    for j in range(1, len(blocks)):
        tapblock_mx = np.mean(taps[blocks[j-1]:blocks[j]])
        block_mx = tapblock_mx*np.ones(blocksize)
        mx.extend(block_mx)
        
    gmx = gaussian_filter1d(mx, 1000)
    gmx = gmx/max(gmx)
    gmx -= 2 # move it down below wf amplitude space   
    return gmx

def removeAudioStims(thestimdir):
    for folder in thestimdir:
        if os.path.exists(folder):
            for fi in glob.glob(folder + "/*.wav"):
                os.remove(fi)
            for png in glob.glob(folder + '/*.png'):
                os.remove(png)

def quantizeEvents(window, nsteps = 8):
    #window = np.random.normal(0, 0.37, size=40)
    #plt.plot(window)
    qfunct = np.linspace(-1, 1, nsteps)
    qidx = np.digitize(window, qfunct)
    qwindow = [qfunct[idx-1] for idx in qidx]
    #plt.plot(qwindow, color='orange')
    return qwindow


#%%
#region 
#endregion


#%%##################################################
########## INITIALIZE GLOBAL PARAMS ################
################################################

N = config.N
sr_model = config.sr_model

####### set the binaural flag and DISTRIBUTION TYPE (uniform, gaussian) #######
binaural_flag = config.flags['binaural'] # binaural audio or no? 
binaural_flag = True
ir_flag = config.flags['ir'] # use concert hall ir convolution as well? 
pulse_flag = config.flags['pulse']
quantize_flag = config.flags['quantize']
qsteps = config.qsteps
#### SET THE SCALING FLAG #####
scaling_flag = True 
if scaling_flag == True:
    scaling_str = 'sc'
else:
    scaling_str = 'ns'

##### input the VERSION #########
version = 2
#####################
dist_type = config.dist_type # which probability density function (uniform, or gaussian)
stimdirname = str(version) + '-' + scaling_str + '-' + 'stim-step' + '-' + config.strs['quantize'] + '-' + config.strs['pulse'] + '-' + config.strs['binaural']
trigsdir = stimdirname + '/triggers/'
beatcntdir = stimdirname + '/beat-centers/'
Rmagdir = stimdirname + '/Rmag/'
Rangdir = stimdirname + "/Rang/"
mp3sdir = stimdirname + '/mp3s/'

print('stirdirname:', stimdirname)

for dir_ in [trigsdir, beatcntdir, Rmagdir, Rangdir]:
    try:
        if os.path.exists(stimdirname + dir_) == False:
            print('making', stimdirname + dir_, ' dir')
            os.mkdir(dir_)
    except:
        pass


print('------ intialization -------')
print('distribution type', dist_type)
print("binaural:", config.strs['binaural'])
print('sample audio:', config.strs['pulse'])
print('quantization:', config.strs['quantize'], 'num steps:', config.qsteps)
print('audio stimuli dir:', stimdirname)
print('trigs npy dir:', trigsdir)
print('R mag npy dir:', Rmagdir)
print('R ang npy dir:', Rangdir)
print('beat centers npy dir:', beatcntdir)



#### LOOP PARAMS
target_tempos = np.geomspace(start=60, stop=100, num=5)
freq_conds = target_tempos/60. # tempo to distribute events around
period_conds = 1/freq_conds 
num_beats = 15  # number of beats
seconds = (1./freq_conds)*num_beats   # length of audio to generate 
beg_delay = 0.5 # time (secs) to insert in beginning of stim audio
end_delay = 0.5 # time to insert at end fo audio
period_samps = np.array(sr_audio*1./freq_conds, dtype=np.int) # num of samples for 1 Hz isochronous beat where events are distributed aroudn

totalsamps = beg_delay*sr_audio + seconds*sr_audio + end_delay*sr_audio
totalsamps = totalsamps.astype(int)
totalsecs = totalsamps/sr_audio

## make the directories to hold the audio stim and plots
if os.path.exists(trigsdir) == False:
    os.makedirs(trigsdir)
    print('making ', trigsdir)

stimdirs = []
for tmp in target_tempos:
    rootdir = os.path.join(stimdirname + '/' + str(config.N) + '_' + config.strs['binaural'] + '_' + config.strs['pulse'],  str(int(tmp))) # hold R, or ramps?  
    stimdirs.append(rootdir)
    if os.path.exists(rootdir) == False:
        print('making dirs for:', rootdir)
        os.makedirs(rootdir)
    else:
        print(rootdir, 'already exists')

########## RANGE of UNIFORM low and high (l,r) range for uniform distribution
num_grads = 8   # number of SD gradations for each tempo cond

l = np.linspace(-0.1, -1, num_grads)
r = np.linspace(0.1, 1, num_grads)
#brange = np.linspace(1, num_beats, num_beats)
#brange_secs = np.linspace(0.1, 0.1 + num_beats*1./period_conds, num_beats)

####### RANGE of Stand Dev for NORMAL dist. increments upon each iteration scaled for each tempo condition  

##########

sd_start = 0.4                                  # starting SD, wide enough for no synchrony 
sd_step_targets = np.linspace(0.1, 0.3, num_grads) # step targets to change SD to after starting_beats 

# replace first step at beginning to create perfectly isochronous tempo change 
# replace last step as 'no step change' as last step  
sd_step_targets_arr = []
for tmp in freq_conds:
    ratiotmp = tmp/freq_conds[0]
    sc_sd_step_targets = sd_step_targets/ratiotmp
    sc_sd_step_targets = list(sc_sd_step_targets[1:-1])
    sc_sd_step_targets.insert(0, 0.00)
    sc_sd_step_targets.insert(len(sc_sd_step_targets), sd_start)
    sd_step_targets_arr.append(sc_sd_step_targets)
    
sd_step_targets_arr = np.array(sd_step_targets_arr)

if scaling_flag == False:
    sd_step_targets = np.linspace(0.1, 0.3, num_grads) # step targets to change SD to after starting_beats     
    for i,tmp in enumerate(freq_conds):
        steps = list(sd_step_targets[1:-1])
        steps.insert(0, 0.00)
        steps.insert(len(steps), sd_start)
        sd_step_targets_arr[i] = steps
        

#removeAudioStims(stimdirs)

#region plot simple gaussian generator and function
#%% plot simple gaussian generator and function 

mu = 1
variance = 0.4
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, N)
#plt.plot(x, stats.norm.pdf(x, mu, sigma))
norm_pdf = stats.norm.pdf(x, mu, sigma)
window = np.random.normal(1, 0.37, size=N)
plt.hist(window, bins=20, density=True)
plt.plot(x,norm_pdf)
plt.show()

#%%
fig, ax = plt.subplots(nrows=len(freq_conds), ncols=1, sharex=True, sharey=True,
                       figsize=(5,10))
mu = 0
for n, starr in enumerate(sd_step_targets_arr):
    i = 0
    for sd, b in zip(starr, np.arange(1, num_beats)):
        x = np.linspace(mu - 3*sd, mu + 3*sd, N)        
        norm_pdf = stats.norm.pdf(x, mu, sd)
        ax[n].plot(norm_pdf)
        

# another way to do it 
# sigma = 0.4
# mu = 0
# window = np.random.normal(mu, sigma, size=N)
# count, bins, ignored = plt.hist(window, bins=30, density=True)
# pdfcont = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(bins-mu)**2 / (2*sigma**2))
# plt.plot(bins, pdfcont, linewidth=2, color='r')

#endregion

#%%
#%%

#region PLOT PDF OF GAUSSIAN DISTRIBUTION AT STEP CHANGE
#%% NB: don't have to run .......
# ################################################################################
################PLOT PDF of GAUSSIAN DISTRIBUTION AT STEP CHANGE 
##########################################################################
import logging 
log = logging.getLogger() 
console = logging.StreamHandler()
log.addHandler(console)

freq = 1
brange_secs = np.linspace(beg_delay, num_beats*1./freq, num_beats)    
#sd = np.linspace(0.1,0.35,num_grads)    # range of SDs to increment through to make stims
#sd_start = 0.4                               # starting SD, wide enough for no synchrony 
#sd_step_targets = np.linspace(0.1, 0.3, num_grads) # step targets to change SD to after starting_beats 

mu = 0
fig, ax = plt.subplots(nrows=(num_grads), ncols=1, figsize=(10,13), sharex=True, sharey=True)

#len(sd_target)

for n, sd_target in enumerate(sd_step_targets):
    #log.warning(str(n) + ', sigma target' + ' ' + str(sd_target))
    
    buffers = np.zeros((num_beats, int(brange_secs[-1]*N+N)))
    trigs = np.zeros((num_beats,int(brange_secs[-1]*N+N)))
    events, R_traj = [], []
    
    
    #startbeats = np.random.randint(low=3, high=6)   # number of beats to wait until to apply SD step change
    startbeats = random.choice([3,6,9])
    print(startbeats)
    sd_traj = [sd_start for elem in range(startbeats)] # fill up SD trajectory (per beat) with starting SDZ
    sd_traj.extend([sd_target for elem in range(num_beats - startbeats)]) # fill up with SD target 

    i = 0
    for b, sd_ in zip(brange_secs, sd_traj):
        x = np.linspace(mu - 3*sd_, mu + 3*sd_, N)
        norm_pdf = stats.norm.pdf(x, mu, sd_)
        loc = int(b*N)
        buffers[i, loc:(loc+N)] = norm_pdf
        
        window = np.random.normal(0, sd_, size=N)
        norm_vals = stats.norm.rvs(loc=mu, scale=sd_, size=N)      
        #trigs[i, loc:(loc+N)] = N*(norm_vals + b)
        trigsecs = norm_vals + b
        events.extend(trigsecs)
        i+=1
    
    events = np.array(events)
    events_N = np.array(N*events, dtype=np.int)
    minevents = int(np.abs(min(events_N[events_N<0])))
    # shift forward half of a beat so that events line up on beat 1 not beat 0
    events_N = events_N + int(N/2) 
    #events_N = events_N[:int(brange_secs[-1]*N+N)]

    for buffer, tr in zip(buffers, trigs):
        ax[n].plot(buffer)
    ax[n].vlines(events_N, -2, -1, linewidth=0.5, alpha=0.6)
    ax[n].set_title('SD ' + str(round2dec(sd_traj[0])) + '->' + str(round2dec(sd_traj[-1])))

    ymin, ymax = ax[n].get_ylim()
    ax[n].vlines(int(brange_secs[1]*N + startbeats*N), 0, ymax, color='green', linestyle='--')

plt.tight_layout()
fig.savefig(os.path.join(stimdirname, config.dist_type + ' trajectories' + '.png'))
print('saving file ', os.path.join(stimdirname, config.dist_type + ' trajectories' + '.png'))
fig.subplots_adjust(top=0.925)
fig.suptitle('Audio Onset Events PDFs with step changes')    
        
#endregion       

#%% ######################################################################
# ############# GENERATE AUDIO STIMULI SD STEPS AT RANDOM STEP ###########
##########################################################################
removeAudioStims(stimdirs)
R_mag_traj, R_ang_traj, width_traj = [], [], []
events, trigs = [], [] 


plt.figure()
for tmp, stimdir, totalsec, period_samp, sig_step_targets in zip(freq_conds, stimdirs, totalsecs, np.array(period_samps, dtype=np.float), sd_step_targets_arr):
    
    fig, ax = plt.subplots(nrows=len(sd_step_targets), ncols=2, figsize=(10, 10), sharex='col', sharey='col')
    fig_r, ax_r = plt.subplots(1,1)
        
    beatlocations = np.linspace(beg_delay, num_beats*1./tmp, num_beats)
    
    i = 0
    for n, sd_target in enumerate(sig_step_targets):
        events, R_mag_traj, R_ang_traj = [], [], []  
        #iteration_str = str(round2dec(sd_traj[0])) + '->' + str(round2dec(sd_traj[-1])) + '_' + str(startbeats) # this names the audio 
        #startbeats = np.random.randint(low=3, high=6)   # number of beats to wait until to apply SD step change
        startbeats = random.choice([3,6,9])
        sd_traj = [sd_start for elem in range(startbeats)] # fill up SD trajectory (per beat) with starting SDZ
        sd_traj.extend([sd_target for elem in range(num_beats - startbeats)]) # fill up with SD target 

        # form audio name str (version_tempo_gradation_stepbeat)
        iteration_str = str(version) + "_" + str(round(60./(1./tmp))) + '_' + str(startbeats) + '_' + str(n)
        
        # distribute gaussian window over beats 
        for b, sd_ in zip(beatlocations, sd_traj):
            if dist_type == 'gaussian':
                window = np.random.normal(0, sd_, size=N)
            if quantize_flag == True:
                window = quantizeEvents(window, nsteps=qsteps)
            # if step is into isochrony (sigma = 0), then only fill up window with single onset event at center of beat 
            if sd_ == 0.0:
                window = np.array([0.0]) 
            
            b_window = np.array(window)+ b # distribute prob events window around beats depending on tempo (in sec)            
            events.extend(b_window)
            
            R_m, R_ang = calculateCOP(window, period_samp/sr_audio, dist_type=dist_type)
            # if sd_ == 0.0:
            #     R_m = R_m*N
                
            R_mag_traj.append(R_m)
            R_ang_traj.append(R_ang)
                        
        #print('R_traj:', R_traj)  
        events = np.array(events) # -> nparray
        events = events[events < totalsec] # remove events > max sec
        events = events[events > 0.0]  # remove events < 0 sec
            
        # center beat locations
        #beatlocations = np.linspace(beg_delay, (1./tmp)*num_beats, num_beats)

        ax[i,0].hist(events, linewidth=0.3, bins=100) ## bins in a way mean what how rhythmic acuity is per second (we can cohere 30 events within a second)
        ax[i,0].vlines(beatlocations, 0, 18, color='red', linewidth=0.5, alpha=0.5)
        ax[i,0].vlines(startbeats, 0, 18, color='green', linewidth=1.5, linestyle='--') # beat where step happens
        ax[i,0].set_title('SD=' + iteration_str, fontsize=10)
        
        print('making', config.strs['pulse'], 'audio')
        if pulse_flag == False:
            wf = makeAudio(events, iteration_str, stimdir, spatial_flag=config.flags['binaural'])
        else:
            wf = makePulses(events, iteration_str, stimdir, spatial_flag=config.flags['binaural'])
        
        # write all trigger event onsets, beat centers, R mag and R ang trajectories into npy files at stimdirname root 
        np.save(trigsdir + iteration_str + '.npy', events)        
        np.save(beatcntdir + iteration_str + '.npy', beatlocations) 
        np.save(Rmagdir + iteration_str + '.npy', R_mag_traj)
        np.save(Rangdir + iteration_str + '.npy', R_ang_traj)
        
        print('get KDE')
        mx = getKDE(events)
        print('plotting KDE..')
        ax[i,1].plot(mx, linewidth=0.7, color='orange')
        
        wf_mono = wf[0] + wf[1]
        ax[i,1].plot(wf_mono, linewidth=0.5)
        beatlocs_samps = librosa.time_to_samples(beatlocations)
        ax[i,1].vlines(beatlocs_samps, -2, 1, color='red', linewidth=0.5, alpha=0.5, zorder=1)          
        ax[i,1].plot(beatlocs_samps, R_mag_traj[:len(beatlocs_samps)], linewidth=1, color='red')
        ax[i,1].vlines(int((startbeats-0.5)*sr_audio), -2, 1, color='green', linewidth=1.5, linestyle='--') # beat where step happens
        
        #ax[i,1].annotate(str(round2dec(R_traj[-1])), xy=(beatlocs_samps[-1], R_traj[-1]), color='red', fontsize=8)

        # R comparisons figure
        ax_r.plot(beatlocations, R_mag_traj[:len(beatlocs_samps)], linewidth=1)
       
        i+=1
    ax_r.set_ylim([0,1])
    ax_r.set_title(str(round2dec(60/(1/tmp))) + ' phase coherence magnitudes |R| Comparison')
    fig_r.savefig(os.path.join(stimdir, 'R' + '_' + config.dist_type + '_' + config.strs['binaural'][0] + '-' + str(config.N) + '-' + iteration_str + '.png'))
    # remove x-ticks on right col
    for ax_ in ax[:,1].flat:
        currxticks = ax_.get_xticks()
        ax_.set_xticklabels(np.round(currxticks/sr_audio,1))
        ax_.set_yticklabels([])
    
    fig_r.suptitle(str(round2dec(60/(1/tmp))) + " Hz", fontsize=12)
    fig.suptitle(str(round2dec(60/(1/tmp))) + " Hz", fontsize=12)
    fig_r.tight_layout()
    fig.tight_layout()
    #plt.savefig('./plots/inc-uniform-window.png', dpi=150)
    #plt.suptitle(dist_type + ' distribution: sd increments += ' + str(np.mean(np.diff(sd))))
    fig.savefig(os.path.join(stimdir, config.dist_type + '_' + config.strs['binaural'][0] + '-' + str(config.N) + '-' + iteration_str + '.png'), dpi=150)

# make mp3s
#%%
# mp3dir = stimdirname + '/mp3s' 
# if os.path.exists(mp3dir) == False:
#     os.mkdir(mp3dir)

# import subprocess 
# subprocess.call(['./' + stimdirname + '/wav2mp3.sh'])


# %%

# %%

#%%##################################################################
##### get Kernel Density Estimate based off of triggers and PLOT
###################################################################

plt.figure()
fig, ax = plt.subplots(2,2)
for ax_ in ax[:,1].flat:
    



    
#%%

n = 0
for beat, width, width_n in zip(beats_1hz, g_width, uniform_width):
    print('iter:', n, 'R:', R_m, 'w:', w_init)
    # uncomment if you wanna use a gaussian window 
    # gaussian_window = np.random.normal(0, int(width), N)
    # gaussian_window[gaussian_window < -sr_audio] = -sr_audio
    # gaussian_window = [int(elem) for elem in gaussian_window]
   
    # uniform distribution that decreases width
    # uniform_window = np.random.uniform(low=0, high=width_n, size=N)
    # uniform_window = [int(elem) for elem in uniform_window]
    
    # ADAPTIVE GAUSSIAN WINDOW 
    #R_m, R_a = calculateCOP(gaussian_window, period_samps, dist_type=dist_type)

    # ADAPTIVE uniform distribution that tries to keep R at R_target
    R_m, R_a = calculateCOP(uniform_window, period_samps)
   
    if R_m < R_target:
        width_init -= width_inc
        uniform_window = np.random.uniform(low=0, high=width_init, size=N)
        
        w_init -= w_inc
        gaussian_window = np.random.normal(0, w_init, size=N)
    if R_m >= R_target:
        width_init += width_inc 
        uniform_window = np.random.uniform(low=0, high=width_init, size=N)
        
        w_init += w_inc
        gaussian_window = np.random.normal(0, w_init, size=N)

    width_traj.append(width_init)
    
    # reinit distribution windows
    gaussian_window = [int(elem*sr_audio) for elem in gaussian_window]
    uniform_window = [int(elem) for elem in uniform_window]
         
    trigs.extend(uniform_window + beat)
    
    distributeSamples(binaural_flag, dist_type)

    n+=1
    
    
audio_l = np.sum(audiobuffer_L, axis=0)
audio_r = np.sum(audiobuffer_R, axis=0)

audio_l = 0.8*audio_l/max(audio_l)
audio_r = 0.8*audio_r/max(audio_r)

audio = np.array([audio_l, audio_r])
audiofi = rootdir + dist_type + '_' + str(N) + '_' + binaural_str + '-' + str(R_target) + '-' + str(width_inc) + '.wav'
sf.write(audiofi, audio.T, samplerate=sr_audio)
print('creating', audiofi)
plt.plot(R_mag_traj)
plt.suptitle('Phase Coherence |R| Trajectory')
plt.savefig(rootdir + os.path.basename(audiofi)[0] + '.png', dpi=150)

#%%
##################################################################
##### get Kernel Density Estimate based off of triggers and PLOT
###################################################################
from scipy.ndimage import gaussian_filter1d

def getKDE(events):
    eventsinsamples = librosa.time_to_samples(events)
    taps = np.zeros(max(eventsinsamples)+1)
    
    for i, spike in enumerate(eventsinsamples):
        np.put(taps, spike, 1)
    blocksize = int(sr_audio/10)
    blocks = np.arange(0, len(taps), blocksize)
    mx = []
    
    for i in range(1, len(blocks)):
        tapblock_mx = np.mean(taps[blocks[i-1]:blocks[i]])
        block_mx = tapblock_mx*np.ones(blocksize)
        mx.extend(block_mx)
     
    gmx = gaussian_filter1d(mx, 1000)
    gmx = gmx/max(gmx)
    gmx -= 2 # move it down below wf amplitude space 
    
    return gmx
    
    #y, sr_ = librosa.load(audiofi) # plot waveform 
    #plt.plot(y)
    #plt.plot(gmx, linewidth=0.5)
    #plt.savefig(rootdir + os.path.splitext(os.path.basename(audiofi))[0] + '.png', dpi=150)


#%%%%

taps = np.zeros(totalsamps + 4*sr_audio)

#spikes = np.zeros((tap_mat.shape[0], 1 + int(np.nanmax(tap_mat))))
# for j, taps in enumerate(tap_mat):
#     #non_nan_taps = taps[np.logical_not(np.isnan(taps))]
#     #non_nan_taps = [int(tap) for tap in non_nan_taps]
#     np.put(spikes[j], non_nan_taps, 1.0)
    
for i, spike in enumerate(trigs):
    np.put(taps, trigs[i], 1)

blocksize = int(sr_audio/10)
blocks = np.arange(0, len(taps), blocksize)
mx = []

for i in range(1, len(blocks)):
    tapblock_mx = np.mean(taps[blocks[i-1]:blocks[i]])
    block_mx = tapblock_mx*np.ones(blocksize)
    mx.extend(block_mx)
 
gmx = gaussian_filter1d(mx, 1000)
gmx = gmx/max(gmx)
gmx -= 2 # move it down below wf amplitude space 

y, sr_ = librosa.load(audiofi) # plot waveform 
plt.plot(y)
plt.plot(gmx, linewidth=0.5)
plt.savefig(rootdir + os.path.splitext(os.path.basename(audiofi))[0] + '.png', dpi=150)


#%%   
plt.figure()
for i, b in enumerate(beats_1hz):
    for pt in uwindow[i]:
        plt.vlines(pt,-1,1)
#%%
mx = PSTH(np.array(spikes).T, 1102, len(y_[0])) # per stimulus time histogram, 2205 = 0.1 seconds 
plt.plot(mx)
#%%


    
    
    