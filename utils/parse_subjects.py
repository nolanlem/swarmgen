#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:12:40 2020

@author: nolanlem
"""
import os 
import numpy as np
import pandas as pd
import glob

def parse_subject_info(batch_folder, usable_batch, filtered_subjects):
    a1,a2,b1,b2 = [],[],[],[]
    for csv in usable_batch:
        csv_ = os.path.basename(csv).split('.')[0] + ".csv"
        study = csv_.split('.')[0].split('_')[1].split('-')[1]
        if study == 'A1':
            a1.append(csv_)
        if study == 'A2':
            a2.append(csv_)
        if study == 'B1':
            b1.append(csv_)
        if study == 'B2':
            b2.append(csv_)
            
    print('A1:', len(a1), 'A2:', len(a2), 'B1:', len(b1), 'B2:', len(b2), 'total:', len(a1) + len(a2) + len(b1) + len(b2))
    
    mturk_prompts = ['If you are a MTurk worker, what is your MTurk worker ID?', "If you are a MTurk worker, what is your MTurk worker ID?'"]
    email_prompt = "If you are not an MTurk worker, please provide your email address" 
     
    
    stanford_subjects, mturk_subjects = [], []
    
    for csv_file in glob.glob(batch_folder + '/*.csv'):
        if os.path.basename(csv_file) not in filtered_subjects:
            csvbasename = os.path.basename(csv_file)        
            csv_data = pd.read_csv(csv_file, keep_default_na=False)
                    
            mturk_id = 'none'
            email = 'none'
            
            try:
                mturk_id = csv_data[mturk_prompts[0]][0]
            except:
                pass    
            try:
                mturk_id = csv_data[mturk_prompts[1]][0]
            except:
                pass
            try:
                email = csv_data[email_prompt][0]
            except:
                pass
      
            if (mturk_id != 'none') and (mturk_id != '') and (mturk_id != 'Na'):
                mturk_subjects.append(csvbasename)
                subjectplotdir = './analysis-scripts/plots/' + batch_folder + '/subjects/' + mturk_id        
            else:
    
                if (email != 'none') and (email != ''):
                    #print('email given', email)
                    stanford_subjects.append(csvbasename)
                    subjectplotdir = './analysis-scripts/plots/' + batch_folder + '/subjects/' + email         
        
                else:
                    inits = csv_data['Participant Initials'][0] 
                    stanford_subjects.append(csvbasename)
                                     
    print('stanford total:', len(stanford_subjects), 'mturk total:', len(mturk_subjects))
    
    a1,a2,b1,b2 = [],[],[],[]
    for csv in stanford_subjects:
        csv_ = os.path.basename(csv).split('.')[0] + ".csv"
        study = csv_.split('.')[0].split('_')[1].split('-')[1]
        if study == 'A1':
            a1.append(csv_)
        if study == 'A2':
            a2.append(csv_)
        if study == 'B1':
            b1.append(csv_)
        if study == 'B2':
            b2.append(csv_)
    print('STANFORD TOTALS PER STUDY')
    print('A1:', len(a1), 'A2:', len(a2), 'B1:', len(b1), 'B2:', len(b2), 'total:', len(a1) + len(a2) + len(b1) + len(b2))
    
    a1,a2,b1,b2 = [],[],[],[]
    for csv in mturk_subjects:
        csv_ = os.path.basename(csv).split('.')[0] + ".csv"
        study = csv_.split('.')[0].split('_')[1].split('-')[1]
        if study == 'A1':
            a1.append(csv_)
        if study == 'A2':
            a2.append(csv_)
        if study == 'B1':
            b1.append(csv_)
        if study == 'B2':
            b2.append(csv_)
    print('MTURK TOTALS PER STUDY')
    print('A1:', len(a1), 'A2:', len(a2), 'B1:', len(b1), 'B2:', len(b2), 'total:', len(a1) + len(a2) + len(b1) + len(b2))
