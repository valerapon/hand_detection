import subprocess
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

names = []
for root, dirs, files in os.walk("./training"):
    for filename in files:
        names.append(filename)

for filename in sys.argv:
    if filename[-3:] != 'tif':
        continue
    out = str(subprocess.check_output(['python', 'get_features.py', filename, 'True'])).split('@')[1]
    out = [float(i) for i in out.strip().split()]

    X = np.array(pd.read_excel('data.xlsx'))[:, 1:]
    indeces = ((X - out) ** 2).sum(axis=1).argsort()[-3:][::-1]

    print(names[indeces[0]], names[indeces[1]], names[indeces[2]])
    f = open('output/test_ans' + filename[-7:-4] +'.txt', 'w')
    f.write(names[indeces[0]] + ' ' + names[indeces[1]] + ' ' + names[indeces[2]])
    f.close()