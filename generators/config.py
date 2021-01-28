#!/usr/bin/env python
N  = 40
sr_audio = 22050.
sr_model = 20
qsteps = 48
dist_type = 'gaussian'
sr_model = 20
###### set FLAGS HERE ########
flags = dict(
    binaural = False,
    ir = False,
    pulse = False,
    quantize = False
)
####### form strings for file saving / dirs
strs = {}
if flags['binaural'] == True:
    strs['binaural'] = 'binaural'
if flags['binaural'] == False and flags['ir'] == True:
    strs['binaural'] = "binaural-concert"
else:
    strs['binaural'] = 'mono'
if flags['quantize'] == True:
    strs['quantize'] = 'q'
else:
    strs['quantize'] = 'nq'
if flags['pulse'] == True:
    strs['pulse'] = 'pulse'
else:
    strs['pulse'] = 'sample'


#%%
