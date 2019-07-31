"""
    Supervised KP-NILM
    Copyright (C) 2019 Alejandro Rodriguez and Stephen Makonin
"""

import numpy as np
import pandas as pd
from math import sqrt, pi, exp
from copy import deepcopy

debug = True
"""
    Computes the recursive mean
"""
# with this computation, we ignore the value of t = 0
def mean_r(x, t, mu):
    if t==1:
        return x # ADDED THIS LINE IN MARCH/12/2019
    return (t-1)/t*mu+1/t*x
"""
    Computes the recursive variance
"""
def variance_r(x, t, mu, sigma_sq):
    if t==1:
        return 0
    return (t-1)*sigma_sq/t+((x-mu)**2)/(t-1)
"""
    Computes the recursive covariance
    matrix for two random variables
"""
def covariance_r(x, t, mu, cov_r):
    if t == 1:
        return np.zeros((2,2))
    else:
        cov_r[0,0] = (t-1)*cov_r[0,0]/t + (x[0] - mu[0])**2/(t-1)
        cov_r[0,1] = (t-1)*cov_r[0,1]/t + ((x[0] - mu[0])*(x[1] - mu[1]))/(t-1)
        cov_r[1,0] = cov_r[0,1]
        cov_r[1,1] = (t-1)*cov_r[1,1]/t + (x[1] - mu[1])**2/(t-1)
    return cov_r
"""
    Class that implements Gaussian PDFs
    and it's properties
"""
class Load():
    def __init__(self):
        self.t = 0
        self.mu = 0
        self.sigma_sq = 0
        self.sigma = np.sqrt(self.sigma_sq)
        self.values = []
        self.is_on = []
        self.count_tp = 0
        self.count_tn = 0
        self.count_fp = 0
        self.count_fn = 0

    def update(self, x, appliance, iter):
        """
            Receives the derivative at time t and computes
            the recursive mean and variance.
        """
        self.t += 1
        self.mu = mean_r(x, self.t, self.mu)
        self.sigma_sq = variance_r(x, self.t, self.mu, self.sigma_sq)
        self.sigma = sqrt(self.sigma_sq)

        if int(x) not in self.values:
            self.values.append(int(x))
            self.values.sort()
        # if iter%500 == 0:
        #     print('Appliance', appliance, 'values:', self.values, 'iteration:', iter)
        #     print('---------------------------------------------')

    def prob(self, x):
        if self.sigma == 0 or self.sigma_sq == 0: # to avoid divisions between zero
            self.sigma = 1
            self.sigma_sq = 1
        f = lambda x, mu, sigma: 1/(sqrt(2*pi)*sigma)*exp(-0.5*(1/sigma*(x-mu))**2)
        if isinstance(x, (list,tuple, range)):
            return [f(_x, self.mu, self.sigma) for _x in x]
        return f(x, self.mu, self.sigma)

    def possible_values(self):
        return self.values

    def check_on(self, x):
        if x > 0:
            self.is_on.append(1)
        else:
            self.is_on.append(0)
        return self.is_on

    def count_predictions(self, appliance, t, aggr_solution):
        current_state = self.is_on[t]

        # true positives
        if appliance in aggr_solution and current_state == 1:
            self.count_tp += 1
        # true negatives
        elif appliance not in aggr_solution and current_state == 0:
            self.count_tn += 1
        # false positive
        elif appliance in aggr_solution and current_state == 0:
            self.count_fp += 1
        # false negatives
        else:
            self.count_fn += 1

    def precision(self):
        if self.count_tp + self.count_fp == 0:
            p = 0
        else:
            p = self.count_tp/(self.count_tp+self.count_fp)
        return p

    def recall(self):
        if self.count_tp + self.count_fp == 0:
            r = 0
        else:
            r = self.count_tp/(self.count_tp+self.count_fn)
        return r
    def accuracy(self):
        if (self.count_tp + self.count_tn + self.count_fp + self.count_fn) == 0:
            a = 0
        else:
            a = (self.count_tp + self.count_tn) / (self.count_tp + self.count_tn + self.count_fp + self.count_fn)
        return a

    def f1_score(self, precision, recall):
        if precision+recall == 0:
            f_s = 0
        else:
            f_s = 2*(precision*recall)/(precision+recall)
        return f_s

def find_load_knapsack(x, loads, iter):
    range_ = 1 #available stds for each appliance
    M = len(loads)
    w = []
    w_all = []
    w_id = []
    w_id_all = []
    m = []
    opt_w = []

    for i in range(M):
        w = loads[i].possible_values()
        w_all = w_all + w
        w_id = [i] * len(w)
        w_id_all = w_id_all + w_id

    w_all = np.array(w_all)
    w_id_all = np.array(w_id_all)
    p = (w_all*100//abs(x))
    # if iter%500 == 0:
    #     print('----------------------------')
    #     print('Iteration:', iter)
    #     print('\tWeights: ', w_all)
    #     print('\tID weights: ', w_id_all)
    #     print('\tProfits: ', p)
    #     print('----------------------------')
    opt_set, opt_p = knapsack_dynamic(int(abs(x)), w_all, p, w_id_all)
    if not opt_set:
        # print('\tKP: No solution found, empty set.')
        return (0, [], [])
    else:
        # print('\tKP solution: The optimal set is:')
        for i in range(len(opt_set)):
            # print('\t', w_id_all[opt_set[i]-1])
            m.append(w_id_all[opt_set[i]-1])
            opt_w.append(w_all[opt_set[i]-1])

    ########!!!!!!
    # print('Optimal profit', opt_p)
    # print('Appliance(s)', m)
    # print('Optimal weights', opt_w)

    # if opt_p < 90:
    #     print('\tWARNING!\n\tThe item(s) in the KP but the solution is not the best')
    return (opt_p, m, opt_w)

def knapsack_dynamic(capacity, weight, profit, weight_id):
    c = capacity
    w = weight
    p = profit
    w_a = weight_id
    n = len(p)
    X = []
    Z = np.zeros((c+1, n+1)) # matrix containing the profit solutions (remember this technique solves for all capacities)
    A = np.copy(Z)

    for d in range(0, c+1):
        for j in range(0, n+1):
            if d == 0 or j == 0:
                continue
            # elif w_a[j] == w_a[j-1]:
            #     continue
            elif d >= w[j-1]:
                same = Z[d, j-1]
                new = Z[d - w[j-1], j-1] + p[j-1]
                Z[d, j] = max(same, new)
                if same > new:
                    A[d, j] = 0
                else:
                    A[d,j] = 1
            else:
                Z[d, j] = Z[d, j-1]
                A[d, j] = 0
    d = c

    for n_item in range(n, -1, -1):
        # check if the max profit belongs to the last item
        if A[d,n_item] == 1:
            dup_item = False
            for x in X:
                if w_a[x-1] == w_a[n_item-1]:
                    dup_item = True

            if not dup_item: #len(X) == 0 or (len(X) > 0 and w_a[n_item] != w_a[n_item-1]): #!= w_a[X[-1]]):
                X.append(n_item)
                #print('Item', n_item, 'was added to the bag. The weight of the item is', w[n_item-1])
                d = d - w[n_item-1]
                #print('Now we have to check the item', n_item-1, 'with available weight ', d)
        #     else:
        #         #print('Item', n_item, 'was not added to the bag. It is actually the same appliance!')
        # else:
        #         print('Item', n_item, 'was not added to the bag. The weight of the item is', w[n_item-1])

    # print('\n\tThe individual profits are: ', p)
    # print('\tThe individual weights are: ', w)
    # print('\tThe solution matrix is\n', Z)
    # print('\n\tThe maximum profit for the knapsack with a capacity of', c, 'is:', Z[c, n],'\n')
    # print('\tThe optimal solution set is:', X)
    # for i in range(len(X)):
    #     print('\tThe appliance that is ON/OFF is:', w_a[X[i]-1])
    return X, Z[c,n]

###
#   Load data
###

df = None

def load_data(appliance, start, stop):
    ###
    # This function returns the specified appliance with
    # 'num_samples' number of samples
    ###

    global df

    if df is None:
        print('Loading data... \n' )
        datafile = './house1_power_blk1.csv'

        df = pd.read_csv(datafile)
        df.replace("", np.nan, inplace=True)#df = df.fillna(0)
        df = df.fillna(0)
        df = df.set_index('unix_ts')
        print('Data loaded! \n')
        print('You chose ' + datafile)

        sub = 'sub'
        total_appliances = 24   # we know this by looking at the (house2_labels.txt)
        for i in range(1, total_appliances+1):
            if i == 1:
                df['WHE'] = df[sub + str(i)].values
            else:
                df['WHE'] = df['WHE'] + df[sub + str(i)].values

        df['FRE'] = df['sub11']
        df['FGE'] = df['sub8']
        df['HPE'] = df['sub13'] + df['sub14']
        df['TVE'] = df['sub19'] # (TV/Amp/DVD/PVR)
        df['CWE'] = df['sub9']
        df['CDE'] = df['sub5'] + df['sub6']
        df['DWE'] = df['sub10']
        df['WOE'] = df['sub1'] + df['sub2']
        #df['COOL'] = df['sub8'] + df['sub20']

    return df[appliance].values[start:stop]

def load_data_2018(appliance, start, stop):

    print('Loading data... \n' )
    datafile = './2018_breaker_1hz.csv'

    df = pd.read_csv(datafile)
    df.replace("", np.nan, inplace=True)#df = df.fillna(0)
    df = df.fillna(0)
    df = df.set_index('unix_ts')
    print('Data loaded! \n')
    print('You chose ' + datafile)
    sub = 'sub'
    total_appliances = 21 # we know this because of 2018_breaker_1hz
    for i in range(1, total_appliances+1):
        if i == 1:
            df['WHE'] = df[sub + str(i)].values[start:stop]
        else:
            df['WHE'] = df['WHE'] + df[sub + str(i)].values[start:stop]

        # df['FRE'] = df['sub11']$
        # df['FGE'] = df['sub11']
        # df['HPE'] = df['sub13'] + df['sub14']
        # df['TVE'] = df['sub19'] # (TV/Amp/DVD/PVR)
        # df['CWE'] = df['sub10']
        # df['CDE'] = df['sub4'] + df['sub5']
        # df['DWE'] = df['sub19']
        # df['WOE'] = df['sub1'] + df['sub2']
    return df[appliance].values[start:stop]
###
# Function that calculates the mean and std of
# the specified appliance when its ON
# inputs: zs(measurement), th_up(upper threshold), th_down(lower threshold)
# outputs: mean of appliance when ON, std when appliance ON
###
def on_stats(zs, th_up, th_down):
    # take the derivative to check where the appliance turns ON and OFF
    zs_diff = np.diff(zs)
    # look for the peak position by thresholding the relevant events
    zs_on_pos = np.where(zs_diff>=th_up) # for fridge
    zs_on_pos = np.array(zs_on_pos)
    zs_off_pos = np.where(zs_diff<=-th_down) # for fridge
    zs_off_pos = np.array(zs_off_pos)

    # add one because we took the derivartive
    zs_on_pos = zs_on_pos+1
    zs_off_pos = zs_off_pos+1
    # check first position of ON/OFF to see if the appliance turn ON or off first
    if(zs_on_pos[0,0] < zs_off_pos[0,0]):
        print('The appliance was ON')
        zs_on = zs[zs_on_pos[0,0]:zs_off_pos[0,0]]
        # calculate the statistics for ON state
        zs_on_mean = np.mean(zs_on)
        zs_on_std = np.std(zs_on)
    elif(zs_on_pos[0,0] > zs_off_pos[0,0]):
        print('The appliance was OFF')
        zs_off = zs[zs_off_pos[0,0]:zs_on_pos[0,0]]
        zs_on = zs[zs_on_pos[0,0]:zs_off_pos[0,1]]
        zs_on_mean = np.mean(zs_on)
        zs_on_std = np.std(zs_on)
    else:
        print('Could not identify the ON/OFF state, try with different parameters')

    print()
    print('\t Mean of the appliance when ON: ', zs_on_mean)
    print('\t Std of the appliance when ON: ', zs_on_std)

    return zs_on_mean, zs_on_std
