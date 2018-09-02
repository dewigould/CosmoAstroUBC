#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:49:44 2018

@author: DewiGould
@location: Hennings, UBC

Script to perform parameter analysis on MCMC chain obtained from 'FullTask.py' script.
"""
import cProfile, pstats, StringIO

TIMING_TESTING = False
if TIMING_TESTING:
    pr = cProfile.Profile()
    pr.enable()

import time as time
start = time.time()

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

read_in = True
data_file = "MCMC_Chain.txt"
if read_in:
    with open (data_file, 'rb') as fp:
        chain = pickle.load(fp)
        NLL_vals = chain[-2]
        list_vals_NLL_results = chain[-1]
    print "read in"

burn_in = 0


def statistics_hist(data_list,conf,what):
    """
    Function to return median and confidence interval of whatever parameter chain.
    Finds confidence interval numerically by slowing moving from top of histogram down (in a LEVEL fashion)

    Confidence Level Calculation is RIGHT - as in, it does what I think it should be doing
    It takes a horizontal line, and scans down - calculating the area between the histogram and the line until area = 68%,
    then takes the end points of the line as the confidence intervals.
    The main subtelty is not visible if you plot a bar chart - the two intervals DO CORRESPOND TO THE SAME HEIGHT IN THE HISOGRAM,
    (look at scatter) - the issue is just the non-uniformity means that some bins lying under the scanning line are (intentionally) avoided.

    I'M HAPPY THAT THIS IS WORKING.
    """
    counts, bin_edges = np.histogram(data_list,"auto")
    W = sum(counts)
    counts = [i/float(W) for i in counts]
    W = sum(counts)
    bins = [(bin_edges[i]+bin_edges[i+1])/2.0 for i in range(len(bin_edges)-1)]
    #rv = st.rv_discrete(values=(bins,counts))
    def data_function(x):
        bin_edge = max([i for i in bin_edges if x >= i])
        return counts[np.where(bin_edges == bin_edge)[0][0]]

    median = bins[counts.index(max(counts))] #value with highest count.

    x_range = np.linspace(min(bins),max(bins),1000)
    y = max(counts)
    area = 0.0
    dummy_data = [data_function(i) for i in x_range]
    C_I = conf*sum(dummy_data)


    ind_max = dummy_data.index(max(dummy_data))
    lower_interval, upper_interval = ind_max, ind_max
    bin_width = (max(bin_edges) - min(bin_edges))/float(len(bins))

    while area < C_I and y >0.0:
        area = 0.0
        y_vals = [y for i in x_range]
        data_vals = [data_function(i) for i in x_range]
        result = [data_vals[i] - y_vals[i] for i in range(1000)]
        zeros = []
        for j in range(len(result)):
            if result[j]<0.0:
                zeros.append(j)
                result[j] = 0.0
        if len([i for i in zeros if i < ind_max]) != 0.0:
            lower_interval = max([i for i in zeros if i < ind_max])
        if len([i for i in zeros if i > ind_max]) != 0.0:
            upper_interval = min([i for i in zeros if i > ind_max])

        result = result[lower_interval:upper_interval]
        area = float(sum(result))
        y -= max(counts)*0.001

    lower_interval = x_range[lower_interval]
    upper_interval = x_range[upper_interval]

    """
    Keep this plotting - useful to convince myself this is working.
    """
    #plt.scatter(bins,counts)
    #plt.plot([lower_interval]*4,np.linspace(0,max(counts),4),color="black")
    #plt.plot([upper_interval]*4,np.linspace(0,max(counts),4),color="black")
    #plt.plot([median]*4,np.linspace(0,max(counts),4),color="black")
    #plt.plot(np.linspace(min(bins),max(bins),100),[y + (max(counts)*0.001) for i in range(100)],color="black")
    #plt.xlabel("Parameter")
    #plt.show()
    if what == "all":
        return median, (lower_interval,upper_interval),y+(max(counts)*0.001),counts,bins
    if what == "stats":
        return median, (lower_interval,upper_interval),y+(max(counts)*0.001)
    if what == "plot":
        return counts,bins

#************************************************************************
"""
Basic parameter plots over time - i.e. how the parameter is modified by algorithm.
"""

parameter_time_plots = True

if parameter_time_plots:


    plt.figure()
    plt.title("Temperature Parameters")
    for i in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
        plt.plot(range(len(chain[i][int(burn_in*len(chain[i])):])),chain[i][int(burn_in*len(chain[i])):],label=i)
    plt.legend()
    plt.savefig('temps.png')
    plt.show()
    print "done"

    plt.figure()
    plt.title("Normalisation Parameters")
    for i in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]:
        plt.plot(range(len(chain[i][int(burn_in*len(chain[i])):])),chain[i][int(burn_in*len(chain[i])):],label=i)
    plt.legend()
    plt.savefig('constants.png')
    plt.show()
    print "done"


    plt.figure()
    plt.title("Calibration Parameters")
    for i in np.arange(32,40):
        plt.plot(range(len(chain[i][int(burn_in*len(chain[i])):])),chain[i][int(burn_in*len(chain[i])):],label=i)
    plt.legend()
    plt.savefig('calibs.png')
    plt.show()
    print "done"

    plt.figure()
    plt.title("Background Parameters")
    for i in np.arange(40,47):
        plt.plot(range(len(chain[i][int(burn_in*len(chain[i])):])),chain[i][int(burn_in*len(chain[i])):],label=i)
    plt.legend()
    plt.savefig('background.png')
    plt.show()
    print "done"

    plt.figure()
    plt.title("New parameters")
    for i in np.arange(47,52):
        plt.plot(range(len(chain[i][int(burn_in*len(chain[i])):])),chain[i][int(burn_in*len(chain[i])):],label=i)
    plt.legend()
    plt.savefig('new.png')
    plt.show()

    plt.figure()
    plt.title("Beta parameters")
    for i in np.arange(52,68):
        plt.plot(range(len(chain[i][int(burn_in*len(chain[i])):])),chain[i][int(burn_in*len(chain[i])):],label=i)
    plt.legend()
    plt.savefig('new.png')
    plt.show()


    plt.figure()
    plt.title("NLL Values")
    plt.plot(range(len(NLL_vals)),NLL_vals)
    plt.yscale("log")
    plt.show()
    print "done"


#************************************************************************
"""
Plot individual NLL variation over time.
"""
ind_NLL = False
if ind_NLL:
    plt.figure()
    for i in range(len(list_vals_NLL_results[0])):
        y = [j[i] for j in list_vals_NLL_results]
        plt.plot(range(len(y)),y,label=i)
    plt.yscale("log")
    plt.legend()
    plt.show()
#************************************************************************
"""
Plot parameter development for singled out parameter.
"""

plot_single = False
param = 0
if plot_single:

    for i in range(52):

        #plt.figure()
        #plt.hist(chain[param][int(burn_in*len(chain[param])):],np.linspace(min(chain[param][int(burn_in*len(chain[param])):]),max(chain[param][int(burn_in*len(chain[param])):]),25))
        #plt.show()

        plt.figure()
        plt.title("Parameter Values")
        param = i
        plt.plot(range(len(chain[param][int(burn_in*len(chain[param])):])),chain[param][int(burn_in*len(chain[param])):])
        plt.show()

#************************************************************************
"""
Individual Parameter histograms, Median and CI calculation.
"""

import scipy.stats as st

plot_histos = False
med_and_confs = False
parameter_index = 0
if plot_histos:

    for i in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
        parameter_index = i

        st1,st2,st3,data , bins = statistics_hist(chain[parameter_index][int(burn_in*len(chain[parameter_index])):],0.68,"all")
        t = (st1,st2,st3)
        max_ = max(data)
        plt.bar(bins,data)
        plt.plot([st2[0]]*4,np.linspace(0,max_,4),color="black")
        plt.plot([st2[1]]*4,np.linspace(0,max_,4),color="black")
        plt.plot([st1]*4,np.linspace(0,max_,4),color="black")
        plt.plot(np.linspace(min(bins),max(bins),100),[st3 for i in range(100)],color="black")
        plt.xlabel("Parameter")
        plt.show()
        print "Maximum Likelihood Value and 68% Confidence Interval: ", st1, st2

    for i in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
        parameter_index = i
        plt.figure(i)
        plt.hist(chain[parameter_index][int(burn_in*len(chain[parameter_index])):],np.linspace(min(chain[parameter_index][int(burn_in*len(chain[parameter_index])):]),max(chain[parameter_index][int(burn_in*len(chain[parameter_index])):]),25))
        plt.show()

if med_and_confs:
    for i in chain:
        st = statistics_hist(i[int(burn_in*len(i)):],0.68,"stats")
        print "Maximum Likelihood Value and 68% Confidence Interval: ", st[0],st[1]
#************************************************************************
"""
Parameter density estimation and pair correlations.
"""

from plot2Ddist import plot2Ddist, plot2DdistsPairs

plot_conts = False
if plot_conts:
    for i in [12,13,14,15]:
        for j in [12,13,14,15]:
            if i !=j:
                plot2Ddist([chain[i][int(burn_in*len(chain[i])):],chain[j][int(burn_in*len(chain[j])):]],plotcontours=True,plothists=True,contourFractions = [0.6827,0.9545,0.9973],labelcontours=False)

    plt.show()

#************************************************************************
"""
Pair correlation between specific pair
"""

one_pair = False
param_1 = 0
param_2 = 12
if one_pair:
    plot2Ddist([chain[param_1][int(burn_in*len(chain[param_1])):],chain[param_2][int(burn_in*len(chain[param_2])):]],plotcontours=True,plothists=True,contourFractions=[0.6827,0.9545,0.9973],labelcontours=False)
    plt.show()

#************************************************************************
"""
Get flux density at any wavelength for all 16 galaxies, and associated distribution.
"""
#Code tested with chain of length 1500 metres on ALMA data - to try and get same flux vals at 1.3mm
#correct orders of magnitude (obviously chain should be run longer to get close to right numbers).

from SED import SED,nu_cut
from astropy import constants as const

flux_densities = False
redshifts = [3.00,2.794,2.541,2.43,1.759,1.411,2.59,1.552,0.667,2.086,1.996,5.000,2.497,0.769,1.721,1.314]
c = const.c.value
wavelength = 1.3e-3
freq = c/wavelength
if flux_densities:
    N = len(chain[0][int(burn_in*len(chain[0])):])

    Temps, C_vals = [],[]
    for i in range(16):
        Temps.append(chain[2*i][int(burn_in*len(chain[2*i])):])
        C_vals.append(chain[(2*i)+1][int(burn_in*(len(chain[(2*i)+1]))):])
    SED_chain = [[] for i in range(16)]
    for i in range(int(N)):
        for j in range(16):
            SED_chain[j].append(SED(freq,Temps[j][i],1.5,redshifts[j],C_vals[j][i],2.0, nu_cut(Temps[j][i],1.5,redshifts[j],C_vals[j][i],2.0)))
    for i in SED_chain:
        print "Maximum Likelihood Value and 68% Confidence Interval: ", statistics_hist(i,0.68,"stats")

    #plt.figure()
    #plt.hist(SED_chain[0],np.linspace(min(SED_chain[0]),max(SED_chain[0]),50))
    #plt.xlabel("Flux Density (Jy)")
    #plt.show()

#************************************************************************
"""
Star Formation Rate
- testing not adequate at the moment: wait for longer chain.
"""

from scipy.integrate import simps
L_solar = 3.828e26 #Watts
M_solar = 1.988e30 #kg
factor = 1.08e-10 #for L_FIR to SFR
H_0 = 67.8*1e3 #Hubble Parameter (m/s/Mpc)
omega_m = 0.308
omega_lambda_ = 0.6911
omega_r = 1. - 0.308 - 0.6911 #Flat universe

SFR = False
aiming_for = [326,247,195,94,102,87,56,149,23,45,162,37,66,44,38,40]


def integrand(x):
    """
    integrand for luminosity distance calculation.
    """
    return 1.0/np.sqrt((omega_r*((1.+x)**4))+(omega_m*((1.+x)**3))+omega_lambda_)

def d_l(z):
    """
    Evaluate luminosity distance at specific redshift in Flat Universe with Plank Cosmological Parameters.
    """
    z_range = np.linspace(0,z,10000)
    integrand_range = [integrand(i) for i in z_range]
    return simps(integrand_range,z_range)*((c*(1.+z))/(H_0))*3.086e22 #from Mpc to metres

if SFR:
    d_lums = [d_l(i) for i in redshifts]
    freqs = np.linspace(c/1000e-6,c/8e-6,1000) #integral range by definition (8um to 1000um) converted to
    burn_index = int(burn_in*len(chain[0]))
    Temps,C_vals,beta_vals = [],[],[]
    for i in range(16):
        Temps.append(chain[2*i][burn_index:])
        C_vals.append(chain[(2*i)+1][burn_index:])
        beta_vals.append(chain[52+i])
    SFR_chain = [[] for i in range(16)] #solar masses per year.
    for i in range(int(len(chain[0])-burn_index)):
        for j in range(16):
            nu_cutoff = nu_cut(Temps[j][i],beta_vals[j][i],redshifts[j],C_vals[j][i],2.0)
            integrand_SED = [SED(k/(1.+redshifts[j]),Temps[j][i],beta_vals[j][i],redshifts[j],C_vals[j][i],2.0,nu_cutoff)*1e-26 for k in freqs] #conversion from Jy to W/m^2/Hz
            I_one = simps(integrand_SED,freqs)
            I_one /= (1.+redshifts[j]) #change of variables from integral in emitted frequency
            d_lum = d_lums[j] # luminosity distance
            I_one *= 4.*np.pi*(d_lum**2.0) #convert to a luminosity
            SFR_chain[j].append((factor*I_one)/L_solar) #solar masses per year
        A = int(len(chain[0])-burn_index)
        q = np.arange(1,100)
        if i in [int((A*y)/100) for y in q]:
            print (i*100)/A, "%"
    for k in range(16):
        print "Maximum Likelihood Value and 68% Confidence Interval for Galaxy SFR: ", statistics_hist(SFR_chain[k],0.68,"stats"), aiming_for[k]
        data,bins =  statistics_hist(SFR_chain[k],0.68,"plot")
        #plt.figure()
        #plt.bar(bins,data)
        #plt.show()


#************************************************************************
#Testing process with ~final parameters.
if TIMING_TESTING:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats()
    print s.getvalue()
