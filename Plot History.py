# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import matplotlib.pyplot as plt
import pandas as pd

log_csvs = sorted(os.listdir('../Thingy52/combined/logs'))
print(log_csvs)

labels = ['Conv 1D', 'Conv 2D', 'LSTM']
colors = ['r', 'g', 'b']

# +
fig = plt.figure(figsize=(15, 10))
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Accuracy',fontsize=18)
plt.axis([0, 20, 0.60, 1.00])


for (fn, label, c) in zip(log_csvs, labels, colors):
    csv_path = os.path.join('../Thingy52/combined/logs', fn)
    df = pd.read_csv(csv_path)
    plt.plot(df.accuracy, color=c, label=label+' train')
    plt.plot(df.val_accuracy, ls='--', color=c, label=label+' test')
    
plt.legend(loc='lower right')
    
plt.show()
# -




