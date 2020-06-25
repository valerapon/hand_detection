import subprocess
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

status = 'False'
if len(sys.argv) >= 2 and sys.argv[1] == 'True':
        status = 'True'

X = np.zeros((1, 11), dtype=float)
names = []
for root, dirs, files in os.walk("./training"):
    for filename in files:
        names.append(filename)
        out = str(subprocess.check_output(['python', 'get_features.py', './training/' + filename, status])).split('@')[1]
        out = [float(i) for i in out.strip().split()]
        X = np.concatenate([X, [out]])

X = X[1:]

pd.DataFrame(X).to_excel('data.xlsx')

inertia = [1000000]
n_clusters = 0
for k in range(5, 30):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))
    if inertia[-2] - inertia[-1] < 2.0:
            n_clusters = k
            break

model = KMeans(n_clusters=n_clusters)
model.fit(X)

ans = [0] * X.shape[0]

for i in range(X.shape[0]):
        ans[i] = np.array([names[i], model.labels_[i]])
ans = pd.DataFrame(np.array(ans), columns=['img', 'class'])
ans.to_excel('output/ans.xlsx')

open('output/n_clusters.txt', 'w').write(str(n_clusters)).close()
print(n_clusters)