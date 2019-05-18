#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:16:30 2019

@author: Max Shen
"""

import numpy as np
import os
import csv

def ReadFile():
    oldfolder = os.getcwd()
    
    os.chdir("train") #Go to train data folder
   
    firstfolder = os.getcwd()
    num_file = 12500
    rawx = []
    rawy = np.hstack((np.ones(num_file),np.zeros(num_file))) #Create y label, 0=neg 1=pos
    
    os.chdir("pos")
    for filename in os.listdir(os.getcwd()): #Read the comments from pos
        obj = open(filename, 'r')
        rawx.append(obj.read())
    os.chdir(firstfolder)
    
    os.chdir("neg")
    for filename in os.listdir(os.getcwd()): #Read the comments from neg
        obj = open(filename, 'r')
        rawx.append(obj.read())
    os.chdir(oldfolder)
    
    return rawx,rawy

rawx,rawy = ReadFile()
with open('training.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    writer.writerows(zip(rawx, rawy))

def ReadTest():
    oldfolder = os.getcwd()
    
    os.chdir("test") #Go to test data folder
    
    rawx,name = [],[]
    
    for filename in os.listdir(os.getcwd()): #Read the comments from pos
        obj = open(filename, 'r')
        rawx.append(obj.read())
        namy = os.path.splitext(filename)[0]
        name.append(namy)
    os.chdir(oldfolder)
    
    return rawx, name

testx,name = ReadTest()

sort_index = np.argsort(np.float_(name))
sorted_testx = []
for i in range(len(testx)):
    sorted_testx.append(testx[sort_index[i]])

with open('test.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    writer.writerows(zip(sorted_testx, np.sort(np.float_(name))))
