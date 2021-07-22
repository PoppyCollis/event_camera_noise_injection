from datetime import datetime
from multiprocessing import Process
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import tables
import stl
import cv2
import dv

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

def get_noise_events(f_name='./data/noise_event.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f_x = f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
    f_y = f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))

    #data = iter_loadtxt('perlin_out_1.txt')
    #print(len(data))
    
    with open('perlin_out_4.txt', 'r') as infile:
        for _ in range(1):
            next(infile)
        

        while True:
            try:
                data = next(infile)
                iter = 0
                for line in infile:
                    line = line.rstrip().split(" ")
                    timestamp = line[0]
                    #timestamp = timestamp[2:]
                    timestamp = timestamp.replace('.','0')
                    timestamp = timestamp[:-5]

                    #timestamp = str(int(timestamp) / 1000)
                    #print(timestamp)
                    #x = line[1]
                    x = line[2]
                    #y = line[2]    
                    y = line[1]
                    # ^ x and y are swapped around here because noise file was the right dimension but portrait - should have been landscape...
                    polarity = line[3]

                    sample_rate = 3

                    if iter % sample_rate == 0:
                        #print('append'* 10)
                    
                        f_timestamp.append([timestamp])
                        f_polarity.append([polarity])
                        f_x.append([x])
                        f_y.append([y])
                        
                    iter = iter + 1
                    
                data = next(infile)

            except StopIteration:

                break

            
    f.close()
    return

get_noise_events()

    
