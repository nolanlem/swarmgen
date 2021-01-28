#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:13:03 2019
config file for different setups
@author: nolanlem
"""

# for kura-rhythmic-synth.py
vweak = 0.175 
weak = 0.2 
medium = 0.22 
strong = 0.3 
# don't worry about kn_end for kura-rhythmic-synth.py
config = dict(
   sr = 20,
   numoscs = 40,
   mean = 0.1,
   std = 0.1,
   kn_start = weak,
   kn_end = 0.3,
   seconds = 30
)

#config = dict(
#    sr = 8000,
#    numoscs = 100,
#    mean = 0.1,
#    std = 0.1,
#    kn_start = 0.05,
#    kn_end = 0.2,
#    seconds = 20
#)

#config = dict(
#    sr = 8000,
#    numoscs = 10,
#    mean = 0.1,
#    std = 0.01,
#    kn_start = 0.05,
#    kn_end = 0.1,
#    seconds = 6
#)

#config = dict(
#    sr = 8000,
#    numoscs = 20,
#    mean = 0.3,
#    std = 0.1,
#    kn_start = 0.05,
#    kn_end = 0.5,
#    seconds = 5
#)

# for use with input audio
# config = dict(
#     sr = 8000,
#     numoscs = 129,
#     mean = 0.3,
#     std = 0.1,
#     kn_start = 0.00,
#     kn_end = 0.5,
#     seconds = 8
# )
