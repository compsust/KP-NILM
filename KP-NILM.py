#!/usr/bin/env python3
#

"""
    Supervised KP-NILM
    Copyright (C) 2019 Alejandro Rodriguez and Stephen Makonin
"""

import numpy as np
import pandas as pd
import time
import appliance as app
from scipy.signal import medfilt
from statistics import median
from prettytable import PrettyTable

start_time = time.time()

debug = True
step = 60
fs = 1.
start_sample = 0
stop_sample = 5000#86400
num_samples = stop_sample - start_sample

min_conv = 60
labels = ['FRE', 'FGE', 'HPE',  'TVE', 'CWE', 'DWE', 'CDE', 'DWE']#, 'HPE', 'TVE', 'CWE', 'DWE', 'CDE', 'DWE']#'HPE', 'TVE', 'CWE, 'DWE', 'CDE', 'DWE']

num_loads = len(labels)

appl = [] # list containing the samples per appliance
loads = [] # list containing Load object
#opt_sol = [] # list containing the optimal KP solution per sample
app_values = [] # multidimensional list that weill contain the list of values that each appliance has
#measure_acc = np.zeros((num_loads, 4)) # matrix containing: precision, recall, and F1score

for label in labels:
    appl.append(app.load_data(label, start_sample, stop_sample))

zs_agg = np.sum(appl, axis=0)

"""
    Check where does the appliances
"""
# zs_agg = medfilt(zs_agg, 99) # apply median to agg. signal

# for i in range(len(appl)):
#     appl[i] = medfilt(appl[i], 99) # apply median to indiv. appliances

"""
    Calculate mean and variance for the individual appliances
"""
for i in range(len(appl)):
    load = app.Load()
    loads.append(load)
    for t in range(len(appl[i])):
        loads[i].check_on(appl[i][t]) # check if the i-th appliance is ON
        if appl[i][t] > 0:
            loads[i].update(appl[i][t], i, t)
    table = PrettyTable([labels[i], 'Mu', 'Variance', 'Std'])
    table.add_row(['-', loads[i].mu, loads[i].sigma_sq, loads[i].sigma])
    print(table)
    app_values.append(loads[i].values)

dataset_1 = pd.DataFrame({'app':labels, 'values':app_values})
dataset_1.to_csv('values.csv')
print('\n\n\tData values ready, check your folder for values.csv\n\n')
dataset_1 = None

"""
    Apply NILM via KP
"""
# table = PrettyTable(['Iteration', 'Load 0', 'Load 1', 'Agg', 'KP solution', 'Load 0 ON/OFF', 'Load 1 ON/OFF'])
T = len(zs_agg)
tenth = T // 100
dcount = 0

print('\n\nStaring KP-NILM disaggregation on', T, 'time samples...')
print('\tProgress updated every %1 (or', tenth, 'timesteps).')

for t in range(T):
    o_p, m, opt_w = app.find_load_knapsack(zs_agg[t], loads, t)

    gt = []#[True if appl[i][t] > 0 else False for i in range(num_loads)]

    infer = [False] * num_loads
    for mm in m:
        infer[mm] = True

    #print(m, gt, infer)

    for i in range(num_loads):
        gt = True if appl[i][t] > 0 else False
        if infer[i] and gt:
            loads[i].count_tp += 1
        elif not infer[i] and not gt:
            loads[i].count_tn += 1
        elif infer[i] and not gt:
            loads[i].count_fp += 1
        elif not infer[i] and gt:
            loads[i].count_fn += 1

    if t > 0 and not t % tenth:
        dcount += 1
        print('\t', dcount,'% done of', T, 'samples...')


f = open('accuracy.csv', 'w')
f.write(','.join(['Label','F1-Score','Precision', 'Recall', 'Accuracy','TP', 'TN', 'FP', 'FN']))
f.write('\n')
for i in range(num_loads):
    p = loads[i].precision()
    r = loads[i].recall()
    acc = loads[i].accuracy()
    f_s = loads[i].f1_score(p, r)
    f.write('%s,%f,%f,%f,%f,%d,%d,%d,%d\n' % (labels[i], f_s, p, r, acc, loads[i].count_tp, loads[i].count_tn, loads[i].count_fp, loads[i].count_fn))
f.close()


print('\n\n\tAccuracy measures ready, check your folder for accuracy.csv\n\n')

end_time = time.time()
print('Elapsed time:', end_time - start_time)
