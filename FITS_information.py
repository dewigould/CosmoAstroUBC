#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 09:03:19 2018

@author: DewiGould
"""
#************************************************************************
#Gather list of errors at each pixel location

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np

def get_instrumental_error(err_filename,x,y,conv_factor):

    with fits.open(err_filename) as hdul:
        data = hdul[0].data
        if type(data) != np.ndarray:
            data = data.astype(float)
        if err_filename == "s_mips_1_s1_v0.30_wht.fits":
            return conv_factor/(np.sqrt(data[x,y]))

        if err_filename == "less_laboca_ecdfs_rms_v1.0.fits":
            flux_data = get_flux_density("less_laboca_ecdfs_flux_v1.0.fits",x,y,0,1.0)
            return np.sqrt((data[x,y]**2)-(np.mean(flux_data)**2))*conv_factor
        else:
            if conv_factor != 1.0:
                data = np.multiply(data,conv_factor)
            return data[x,y]

def get_flux_density(survey_filename,x,y,index,conv_factor):

    with fits.open(survey_filename) as hdul:
        data = hdul[index].data
        if type(data) != np.ndarray:
            data = data.astype(float)
        if conv_factor != 1.0:
            data = np.multiply(data,conv_factor)
    return data[x,y]

def get_image_dimension(filestring,index):

    filename =  get_pkg_data_filename(filestring)
    image_data = fits.open(filename)[index].data

    return image_data.shape
