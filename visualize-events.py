#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:55:23 2021

@author: poppy
"""
import numpy as np
import cv2
import glob

#function to load in the txt file line by line
def iter_loadtxt(filename, delimiter=' ', skiprows=1, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

#load txt file
data = iter_loadtxt('perlin_out.txt')
print(data.shape)
maxElement = np.amax(data)
print('Video width : ', maxElement + 1)

#loop through and find the max number of accumulated events in a frame (for normalization)
a = np.zeros((346, 260), dtype="uint8")
dt = 0.03
t = 0
event_num = []
for x, d in enumerate(data):
    if (d[0] > t + dt):
        event_num.append(a.max())
        a = np.zeros((346, 260), dtype="uint8")
        t = d[0]
    y = int(d[2])
    x = int(d[1])
    a[y][x] = a[y][x] + 1
max_event_num = max(event_num)
print(max_event_num)

#convert timestamped events to numpy array and display as frames
a = np.zeros((346, 260), dtype="uint8")
dt = 0.0001 #change this number to allow more events to accumulate before frame is displayed
t = 0
for x, d in enumerate(data):
    if (d[0] > t + dt):
        np.save("./frames/{:020}.npy".format(x), a)
        cv2.imshow("frame", a)
        cv2.waitKey(50) #change this number for different frame speed
        a = np.zeros((346, 260), dtype="uint8")
        t = d[0]
    y = int(d[2])
    x = int(d[1])
    a[y][x] = a[y][x] + 10
    a[y][x]= a[y][x]/(max_event_num/127) #normalize
cv2.destroyAllWindows()

#convert saved numpy arrays to a video
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, (260,346))
files = sorted(glob.glob('./frames/*.npy'))
for f in files:
    image = np.load(f)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out.write(image)
out.release()
