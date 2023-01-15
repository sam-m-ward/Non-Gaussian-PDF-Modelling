#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import copy, os, re, pickle
from glob import glob
import yaml
import sys
sys.path.append('model_files/')
from NONGAUSS import *
from sigmaRelPosteriors import *

#Import analysis choices
with open(f'choices.yaml') as f:
	choices = yaml.load(f, Loader=yaml.FullLoader)
globals().update(choices)

#Load up MCMC samples and collect data points that will form histogram which we wish to approximate
files = glob(choices['DATAPATH']+'*.npy') ; hist_samples = []
for f in files:
	samples        = np.load(f,allow_pickle=True).item()
	distance_error = (np.std(samples['mu'])**2 - np.std(samples['delM'])**2)**0.5
	hist_samples.append(distance_error)


#Create nonGaussian Class
pdf = nonGauss(hist_samples)

#Get the CDF coordinates that will be inverted to create ICDF
pdf.get_ICDF_coords()

#Get the Monotonic cubic spline interpolation
pdf.get_ICDF_parametric_model()

'''
#Plot up the distribution we will approximate
pdf.plot_samples(show=False)
#Show the CDF quantiles
pdf.plot_CDF(create_new_plot=False,show=True,rescale=True)

#Plot ICDF
pdf.plot_ICDF()
'''
sgr = sigmaRelPosteriors(choices)
#sgr.create_posterior_files(pdf)
sgr.plot_intervals()









#sigmaFit_dist = Mega_dict['fitting_unc']['std']
#sigmaFit_dist.sort()
