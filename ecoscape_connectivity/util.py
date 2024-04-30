import numpy as np
import csv
import os

# We collect here a few generally useful functions.

def dict_translate(np_arr, my_dict, default_val=0):
    """Translates the terrain type according to a dictionary mapping
    terrain type to values.
    """
    u,inv = np.unique(np_arr,return_inverse = True)
    return np.array([(my_dict.get(x, default_val)) for x in u])[inv].reshape(np_arr.shape)

def read_resistance_csv(fn):
    """Reads a dictionary of terrain to resistance in csv, producing a dictionary."""
    d = {}
    with open(fn) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d[int(row['map_code'])] = float(row['resistance'])
    return d

def read_transmission_csv(fn):
    """Reads a dictionary of terrain resistance or transmission in csv, producing a dictionary."""
    d = {}
    with open(fn) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'resistance' in row:
                d[int(row['map_code'])] = 1. - float(row['resistance'])
            else:
                d[int(row['map_code'])] = float(row['transmission'])
    return d

def rescale_resistance(d, resolution_m, hop_length):
    """Resistance dictionaries are based on decay over 100m.
    This function rescales the resistance to the value of the
    actual hop length."""
    alpha = resolution_m * hop_length / 100.
    return {k: v ** alpha for k, v in d.items()}

def createdir_for_file(fn):
    """Ensures that the path to a file exists."""
    dirs, ffn = os.path.split(fn)
    # print("Creating", dirs)
    os.makedirs(dirs, exist_ok=True)

class SingleIterator(object):
    """Given an iterator, this class builds an iterator that returns 
    pairs of the form (None, i), where i is given by the iterator."""
    
    def __init__(self, it):
        self.it = iter(it)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        x = next(self.it)
        return (None, x)
