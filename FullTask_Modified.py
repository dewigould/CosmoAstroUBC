#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:29:37 2018

@author: DewiGould
@location: Hennings, UBC

Main script is split into many subsections denoted with stars.
These subsections also have their own individual scripts in the Working_Codes.py folder,
but it made computational sense to put all functions in the same script to avoid costly importing during MCMC phase (especially when run on remote machine).
"""
#************************************************************************
"""
Log Likelihood Function.
"""

import numpy as np
import itertools as itertools

def T_prior(T):
    """
    Implement Bayesian prior on T_d; dust temperature parameter.
    Hard Prior between 10-100K implemented.
    Input: List of temperatures in K.
    Output: 1 or 0. (0 if any temperatures lie outside range).
    """
    for t in T:
        if t>100 or t<10: #Top Hat Function between 10 and 100K
            return 0.0
    return 1.0

def B_prior(list_c):
    """
    Implement Bayesian prior on background constants.
    Hard Prior >0 implemented.
    Input: list of background parameters.
    Output: 1 or 0. (0 if any background parameters outside range).
    """
    for c in list_c:
        if c<0:
            return 0.0
    return 1.0

def C_prior(C_list):
    """
    Implement Bayesian prior on background constants.
    Hard Prior >0 implemented.
    Input: list of background parameters.
    Output: 1 or 0. (0 if any background parameters outside range).
    """
    for c in C_list:
        if c<0.0:
            return 0.0
    return 1.0

def amp_prior(parameter_list):
    """
    Prior on subtraction amplitudes
    """
    for c in parameter_list:
        if c<0.0 or c>100.0:
            return 0.0
    return 1.

def beta_prior(beta_list):
    """
    Prior on beta values, to keep them above 0
    """
    for c in beta_list:
        if c<0.0:
            return 0.0
    return 1.

def gauss_prior(x,err):
    """
    Implement Bayesian prior on Calibration Constants.
    Gaussian prior used with mean 1.0 and given error.
    Input: value of parameter (x),
           standard deviation of Gaussian (err).
    Output: Normalised Gaussian value of prior.
    """
    return (1.0/(err*np.sqrt(2*np.pi)))*np.exp(-1*0.5*(((x-1.0)/err)**2))

def prior(params,calib_params,gaussian_errors,background_params,new_params):
    """
    Gather values from all priors in log(sum).
    Input: params - MCMC chain parameters,
           calib_params - calibration Parameters,
           gaussian_errors - standard deviations of all gaussian priors,
           background_params - background parameter values.
    Output: Prior Value if all parameters lie within hard-set ranges.
            'OutsideRange' if not - this is picked up by a later function to ensure MCMC understands this "infinity"
    """
    T_result = T_prior([i[0] for i in params])
    c_B_result = 0
    for i in range(len(calib_params)):
        g = gauss_prior(calib_params[i],gaussian_errors[i])
        if g <= 0.0: #this shouldn't happen, but check for negative gaussian prior value by Python quirk.
            return "OutsideRange"
        else:
            c_B_result += np.log(g)
    B_result = B_prior(background_params)
    C_result = C_prior([i[2] for i in params])
    D_result = amp_prior(new_params)
    E_result = beta_prior([i[1] for i in params])

    if T_result == 0.0 or B_result == 0.0 or C_result == 0.0 or D_result == 0.0 or E_result == 0.0: #if parameteres outside hard range.
        return "OutsideRange"
    else:
        return -1*c_B_result #negative log likelihood requires negative log (prior).

def NLL_MAIN(X_pix,Y_pix,reconstructed_data, calib_factor,survey_filename,err_filename,conv_factor,units,D,sigma):
    """
    Function to evaluate Negative Log Likelihood (NLL) for PACS, LABOCA and MIPS data.
    Input: X_pix - np.meshgrid values of x-axis pixels,
           Y_pix - np.meshgrid values of y-axis pixels,
           reconstructed_data - matrix of pixel values reconstructed using model SEDs and obtained PSFs,
           calib_factor - Calibation Factor (nuisance parameter fit by MCMC),
           survey_filename - path location to .fits file containing flux density data,
           err_filename - path location to .fits file containing pixel error data,
           conv_factor - for conversion of data from Jy/Beam, Jy/pixel, Jy/sr etc,
           units - n/a,
           D - matrix of flux density data,
           sigma - matrix of pixel error data.
    Output: Negative Log Likelihood Value.
    """
    result = 0.0
    pixels_X, pixels_Y = X_pix, Y_pix
    indices_x, indices_y = np.meshgrid(range(len(pixels_X[0])),range(len(pixels_X)))
    M = reconstructed_data[indices_y,indices_x]
    result_list = [((D[j][i]-(M[j][i]/calib_factor))**2)/(2*(sigma[j][i]**2)) for i,j in list(itertools.product(range(len(X_pix[0])),range(len(X_pix)))) if sigma[j][i] != 0.0 and sigma[j][i] >1e-6 and sigma[j][i] < 1e4]
    result += np.sum(result_list)
    return result

def NLL_LABOCA(X_pix, Y_pix,reconstructed_data,calib_factor,survey_filename,err_filename,conv_factor,D,sigma):
    """
    Evaluate NLL for LABOCA.
    Simply calls NLL_MAIN with specific parameters required for LABOCA (most - if not all - of these options are now redundant anyway)
    """
    result = NLL_MAIN(X_pix, Y_pix,reconstructed_data,calib_factor,survey_filename,err_filename,conv_factor,False,D,sigma)
    return result

def NLL_ALMA(reconstructed_data,flux, calib_factor, errors):
    """
    Evaluate NLL for ALMA.
    Input: reconstructed_data - synthetic data matrix obtained using model SEDs only,
           flux - list of ALMA fluxes (obtained from ALMA paper),
           calib_factor - calibration factor,
           errors - list of errors for each galaxy flux, also obtained from ALMA paper.
    Output: NLL value.
    """
    result = 0.0
    for i in range(16):
        result += (1/(2*(errors[i]**2)))*((reconstructed_data[i]-(flux[i]/calib_factor))**2)
    return result

def NLL_MIPS(X_pix, Y_pix,reconstructed_data,calib_factor,survey_filename,err_filename,conv_factor,D,sigma):
    """
    Evaluate NLL for MIPS data - basically just implements NLL_MAIN with specific switches required for MIPS data typeself.
    """
    result = NLL_MAIN(X_pix, Y_pix,reconstructed_data,calib_factor,survey_filename,err_filename,conv_factor,False,D,sigma)
    return result

def NLL_SPIRE(reconstructed_data, pix_SPIRE, calib_factor_SPIRE, survey_filename,conv_factor,D_list):
    """
    Evaluate total SPIRE NLL over the three wavelengths.
    Input: reconstructed_data - synthetic data produced from SEDs and PSFs,
           pix_SPIRE - pixel windows used to sample over,
           calib_factor_SPIRE - list of calibration factors for the three wavelengths,
           survey_filename - filepath for .fits file of flux densities,
           conv_factor - conversion factor to move from Jy/Beam, Jy/Pixel etc,
           D_list - list of flux density matrices.
    Output: NLL value for SPIRE.
    NOTE: C_inv is inverse correlation matrix obtained from 'convariance_matrix.py'
    """
    result = 0.0
    R = []
    for i in range(3):
        D = D_list[i]
        R_i = [((D[y][x] - (reconstructed_data[i][y][x]/calib_factor_SPIRE[i])))*0.5 for x,y in list(itertools.product(range(len(reconstructed_data[0])),range(len(reconstructed_data))))]
        R.append(np.sum(R_i))
    R_tran = np.transpose(R)
    C_inv = [[44420.11627607, -50283.26944907, 11073.72085387],
             [-50283.26944907, 111440.67372466, -54375.35682845],
             [11073.72085387, -54375.35682845, 51771.68347397]]
    result += np.dot(R_tran,(np.dot(C_inv,R)))
    return result

#************************************************************************
"""
Model SEDs.
Two separate models included here: one with BB and PL modes joined at manually fixed nu_cutoff,
                                   a second one where PL amplitude and nu_cutoff are obtained numerically by matching function and first derivative.
"""

from astropy import constants as const
from scipy.optimize import root
c = const.c.value
h = const.h.value
k_B = const.k_B.value
nu_zero = c/(250e-6)

def bb_spectra(nu,T_d,beta,z,C):
    """
    Returns Planck BB spectrum at a specific frequency for the given parameters.
    (only relevant to first method).
    """
    nu_zero = c/(250e-6)
    term_1 = (((nu*(1.+z))/(nu_zero))**beta)
    term_2 = ((nu*(1+z))**3)
    term_3 = (1/((np.exp((h*nu*(1+z))/(k_B*T_d)))-1))
    return C*term_1*term_2*term_3

def power_law(nu,T_d,beta,z,alpha):
    """
    Returns power law side of radiation spectrum at specific frequency for the given parameters.
    (only relevant to first method).
    """
    return nu**(-alpha)

def P_eval(T_d,beta,z,C,alpha,nu_cutoff):
    """
    Fix power law amplitude by matching BB and PL functions at manually given cut-off wavelength.
    (only relevant to first method).
    """
    return bb_spectra(nu_cutoff,T_d,beta,z,C)/power_law(nu_cutoff,T_d,beta,z,alpha)

def SED_old(nu,T_d,beta,z,C,alpha,P,nu_cutoff):
    """
    Function to match SEDs and return value for given wavelength (first method).
    P - power law amplitude.
    """
    F = 1e-39 #fiddle parameter to fix values C values to be sensible order of magnitude (based on ALMA data)
    if nu >= nu_cutoff:
        return nu**(-alpha)*P*F
    else:
        nu_zero = c/(250e-6)
        term_1 = (((nu*(1.+z))/(nu_zero))**beta)
        term_2 = ((nu*(1+z))**3)
        term_3 = (1/((np.exp((h*nu*(1+z))/(k_B*T_d)))-1))
        return C*term_1*term_2*term_3*F

def B(beta,z,C):
    """
    Evaluation of a bunch of terms that comes up multiple times in below calculations - easier to write separate function.
    """
    return (C*((1.0+z)**(3.0+beta)))/(nu_zero**(beta))

def D(beta,z,C,alpha):
    """
    Evaluation of a bunch of terms that comes up multiple times in below calculations - easier to write separate function.
    """
    return 3.0 + beta + alpha

def A(T_d,z):
    """
    Evaluation of a bunch of terms that comes up multiple times in below calculations - easier to write separate function.
    """
    return (h*(1.0+z))/(k_B*T_d)

def to_min(x,T_d,beta,z,C,alpha):
    """
    Function to find the root of  - root finding algorithm to obtain nu_cutoff for second SED matching technique.
    Input: x is a frequency value,
           All other parameters as usual.
    Output: Value of function - trying to find 'x' value that returns function value of 0.0 (i.e. the root).
    """
    D_val = D(beta,z,C,alpha)
    A_val = A(T_d,z)
    return ((A_val*x)- D_val)*np.exp(A_val*x) + D_val


def nu_cut(T_d,beta,z,C,alpha):
    """
    Find cut-off frequency at which SED routine swtiches from BB to powerlaw.
    The parameter in the root finding function '5e12' is the start value for the root finder - sometimes worth fiddling with if algorithm gets stuck.
    """
    return root(to_min,5e12,args=(T_d,beta,z,C,alpha)).x[0]

def SED(nu,T_d,beta,z,C,alpha,nu_cutoff):
    """
    Spectral Energy Distribution (SED), using second technique with matched first derivative.
    """

    if nu_cutoff <= 0.0:
        nu_cutoff = c/100e-6
        P = P_eval(T_d,beta,z,C,alpha,nu_cutoff)
        return SED_old(nu,T_d,beta,z,C,alpha,P,nu_cutoff)

    nu_zero = c/(250e-6)
    F = 1e-39 #fiddle parameter to fix values C values to be sensible order of magnitude (based on ALMA data)
    P = B(beta,z,C) * (nu_cutoff**(3.+beta+alpha)) * (1.0/((np.exp(A(T_d,z)*nu_cutoff))-1.0))
    if nu >= nu_cutoff:
        return nu**(-alpha)*P*F
    else:
        term_1 = (((nu*(1.+z))/(nu_zero))**beta)
        term_2 = ((nu*(1.+z))**3.)
        term_3 = (1./((np.exp((h*nu*(1.+z))/(k_B*T_d)))-1.))
        return C*term_1*term_2*term_3*F

#************************************************************************
"""
FITS information - functions written to scrape pixel data from .fits maps. (flux densities and errors).
"""

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

def get_instrumental_error(err_filename,x,y,conv_factor):
    """
    Return value of instrumental error at pixel coordiantes (x,y) from .fits file at err_filename extension.
    conv_factor - conversion between Jy/beam, Jy/pixel etc.
    """
    with fits.open(err_filename) as hdul:
        data = hdul[0].data
        if type(data) != np.ndarray:
            data = data.astype(float)
        if conv_factor != 1.0:
            data = np.multiply(data,conv_factor)
    return data[x,y]

def get_flux_density(survey_filename,x,y,index,conv_factor):
    """
    Return value of flux density at pixel coordiantes (x,y) from .fits file at survey_filename extension.
    Index to find the correct data within the .fits file.
    conv_factor - conversion between Jy/beam, Jy/pixel etc.
    """
    with fits.open(survey_filename) as hdul:
        data = hdul[index].data
        if type(data) != np.ndarray:
            data = data.astype(float)
        if conv_factor != 1.0:
            data = np.multiply(data,conv_factor)
    return data[x,y]

def get_image_dimension(filestring,index):
    """
    Return pixel dimension of .fits file image, with data at specific .fits files index 'index'.
    """
    filename =  get_pkg_data_filename(filestring)
    image_data = fits.open(filename)[index].data
    return image_data.shape

#************************************************************************
"""
Full-Width-Half-Maximum calculation
"""

from astropy.wcs import WCS
from astropy import units as u

def FWHM_pix(FWHM,survey_filename,survey):
    """
    Convert a FWHM in arcseconds to pixel coordinates.
    Input: FWHM - FWHM in arcseconds,
           survey_filename - .fits file from which the pixel coordinates transformations can be obtained,
           survey - string ("PACS","LABOCA",...) representing map.
    Output: Return FWHM in pixels.
    """
    filename =  get_pkg_data_filename(survey_filename)
    hdu = fits.open(filename)[0]
    w = WCS(hdu.header)
    if survey == "PACS":
        return (FWHM/((w.wcs.cdelt[1]*u.degree).to(u.arcsecond))).value
    else:
        matrix = w.wcs.cd
        matrix_new = []
        for i in matrix:
            for j in i:
                if j < 0:
                    matrix_new.append(-1*j)
                else:
                    matrix_new.append(j)
        dist = (np.sqrt(matrix_new[0]*matrix_new[3] - matrix_new[1]*matrix_new[2])*u.degree).to(u.arcsecond)
        return (FWHM/dist).value

#************************************************************************
"""
Image Plane Reconstruction
"""

from astropy import constants as const
redshifts = [3.00,2.794,2.541,2.43,1.759,1.411,2.59,1.552,0.667,2.086,1.996,5.000,2.497,0.769,1.721,1.314]
c = const.c.value

def twoD_Gauss(x,y,mu,FWHM):
    """
    PSF with given mean and FWHM.
    Input: x,y - pixel values at which to evaluate PSF,
           mu - mean of Gaussian,
           FWHM - full-width-half-maximum of PSF.
    Output: Not-normalised Gaussian value (not normalised by design).
    """
    return (np.exp(-1.0*((((x-mu[0])**2.0)+((y-mu[1])**2.0))/(2.0*((np.sqrt((FWHM**2.0)/(8.0*np.log(2))))**2.0)))))

def image_plane(x,y,B_b,sources,source_locations,parameters,FWHM,filename, channel):
    """
    Reconstruct image plane using PSF convolution.
    Input: x,y - pixel values at which to evaluate image_plane value,
           B_b - background parameter (fit by MCMC)
           sources - list of frequency averaged flux densities (i.e. passed throguh telescope bandpass filter)
           source_locations - list of pixel values of galaxies (to be input as mean of PSFs)
           parameters - from MCMC,
           FWHM - FWHM,
           filename - filepath,
           channel - wavelength of specific survey.
    Output: Perform convolution with PSF to get reconstructed data.
    """
    result = B_b
    for i in range(len(sources)):
        result += sources[i] * twoD_Gauss(x,y,[source_locations[i][0],source_locations[i][1]],FWHM)
    return result

def get_reconstr_img(survey,survey_filename,filename_filter,channel,FWHM,source_locations_pix,params,background,wavelengths,T_frac,X,Y,nuisance):
    """
    Get full reconstructed image through convolving model SED data, passed through telescope filter, with PSFs for each survey.
    Input: survey - string ("PACS","LABOCA",...) representing map,
           survey_filename - .fits file path extension to flux data,
           filename_filter - filepath to .txt file containing telescope bandpass filter information,
           channel - wavelength of specific survey,
           FWHM - FWHM of PSF,
           source_locations_pix - pixel locations of 16 galaxies in specific survey,
           params - parameters from MCMC,
           background - background parameters (MCMC),
           wavelengths - wavelengths from telescope bandpass filter file,
           T_frac - transmission fraction from telescope bandpass filter file,
           X,Y - np.meshgrid coordinates over which to reconstruct model image,
           nuisance - nuisance parameters/ things to pass into function to allow faster MCMC evaluation (just follow examples where I've used this function elsewhere).
    Output: reconstructed image (for ALMA there is no convolution or bandpass filter - just model SED values).
    """
    sources = []
    if survey == "ALMA":
        recon_image = [SED(c/channel,params[j][0],params[j][1],redshifts[j],params[j][2],params[j][3],nu_cut(params[j][0],params[j][1],redshifts[j],params[j][2],params[j][3])) for j in range(len(params))]
        return recon_image
    else:
        sources = [pass_through(params[j][0],params[j][1],redshifts[j],params[j][2],params[j][3],filename_filter,channel,survey,wavelengths,T_frac,nuisance) for j in range(len(params))]
        return image_plane(X,Y,background,sources,source_locations_pix,params,FWHM,filename_filter,channel)

#************************************************************************
"""
Obtain location of 16 galaxies in pixel coordinates of specific .fits file - transformed from RA and DEC (positions obtained from ALMA paper)
"""

ra = [53.18348,53.18137,53.16062,53.17090,53.15398,53.14347,53.18051,53.16559,
      53.18092,53.16981,53.16695,53.17203,53.14622,53.17067,53.14897,53.17655] * u.degree
dec = [-27.77667,-27.77757,-27.77627,-27.77544,-27.79087,-27.78327,-27.77970, -27.76990,
       -27.77624,-27.79697,-27.79884,-27.79517,-27.77994,-27.78204,-27.78194,-27.78550] * u.degree

def get_source_locations_pix(survey_filename):
    """
    Transform RA and DEC coordinate of galaxies into pixel coordinates.
    Input: survey_filename - filepath to .fits file with required pixel coordinate system
    Output: list of pixel coordinates.
    """
    filename =  get_pkg_data_filename(survey_filename)
    hdu = fits.open(filename)[0]
    w = WCS(hdu.header)
    transformed = w.wcs_world2pix(ra*u.degree,dec*u.degree,0)
    source_locations_pix = [(transformed[0][i],transformed[1][i]) for i in range(len(ra))]
    return source_locations_pix

def get_location(ra,dec,survey_filename):
    """
    Transform RA and DEC coordinates of interest into pixel coordinates.
    Input: ra,dec and survey filepath
    Output: pixel coordinates
    """
    filename = get_pkg_data_filename(survey_filename)
    hdu = fits.open(filename)[0]
    w = WCS(hdu.header)
    transformed = w.wcs_world2pix(ra*u.degree,dec*u.degree,0)
    return int(transformed[0]),int(transformed[1])
#************************************************************************
"""
Telescope Bandpass Filter - scrape data from text files and 'pass' model SED through filter to obtain frequency averaged values.
"""

from scipy.integrate import simps
c = const.c.value

def get_bandpass(filename, survey,channel):
    """
    Get transmission filter information for each telescope and corresponding wavebands.
    Input: filename - file path to .txt file (usually obtained from online archives),
           survey - string to allow function to follow *slightly different* methods for each survey (different units, data representations etc.),
           channel - wavelength of survey.
    Ouput: list of wavelengths and list of transmission coefficients
    """
    if survey == "PACS":
        r1,r2 = [] ,[]
        with open(filename) as f:
            for line in f:
                lambda_i, T_i = line.split()
                r1.append(float(lambda_i)*1e-10) #convert wavelength to metres
                r2.append(float(T_i))
            return r1,r2
    if survey == "LABOCA":
        r1,r2 = [], []
        with open(filename) as f:
            count = 0
            for line in f:
                count +=1
                if count>11:
                    freq_i, T_i = line.split()
                    r1.append(c/(float(freq_i)*1e9))
                    r2.append(float(T_i))
            return np.array(r1)[::-1].tolist(),np.array(r2)[::-1].tolist() #reverse ordering.
    if survey == "IRAC":
        r1,r2 = [],[]
        with open(filename) as f:
            count = 0
            for line in f:
                count +=1
                if count>3:
                    lambda_i, T_i = line.split()
                    r1.append(c/(1e9*float(lambda_i)))
                    r2.append(float(T_i))
            return r1,r2
    if survey == "SPIRE":
        r1,r2, = [],[]
        with open(filename) as f:
            for line in f:
                lambda_i, T_i = line.split()
                r1.append(float(lambda_i)*1e-10) #convert wavelength to metres
                r2.append(float(T_i))
            return r1,r2
    if survey == "MIPS":
        r1, r2 = [],[]
        with open(filename) as f:
            count = 0
            for line in f:
                count +=1
                if count>6:
                    a = line.split()
                    a = [float(i.strip().strip("'")) for i in a]
                    r1.append(a[0]*1e-6)
                    r2.append(a[1])
        return r1,r2

def pass_through(T_d,beta,z,C,alpha,filename,channel,survey,wavelengths,T_frac,nuisance):
    """
    'Pass' model SED data 'Through' telescope bandpass filter.
    Input: parameters - from MCMC,
           filename - .fits file path,
           channel - wavelength of survey,
           wavelengths - wavelengths in telescope bandpass filter .txt file,
           T_frac - transmission filter fractions in telescope bandass filter .txt file,
           nuisance - random parameteres calculated outside MCMC loop to save time.
    Output - bandpass-averaged flux density (calculated using integral in SEDeblend paper).
    """
    nus,factors,y_two = nuisance
    nu_cut_val = nu_cut(T_d,beta,z,C,alpha)
    sed = np.vectorize(SED)
    y_one = sed(nus,T_d,beta,z,C,alpha,nu_cut_val)
    y_one = np.multiply(y_one,T_frac)
    y_one = np.multiply(y_one,factors) #change of variables for integral to wavelength units
    I_one = simps(y_one, wavelengths)
    I_two = simps(y_two, wavelengths)
    S_bar =  I_one/I_two
    return S_bar

#************************************************************************
"""
NLL and MCMC Scripts below - using functions introduced above.
"""

import cProfile, pstats, StringIO

TIMING_TESTING = False
if TIMING_TESTING:
    pr = cProfile.Profile()
    pr.enable()
#************************************************************************
"""
Data Storage - filenames, unit conversions, ALMA data, preliminary calculations that can be done outside MCMC loop.
"""

from astropy import units as u
import matplotlib.pyplot as plt
import time as time
from astropy import constants as const
c = const.c.value

#************************************************************************
"""
ALMA paper data (positions and redshifts)
"""
ra = [53.18348,53.18137,53.16062,53.17090,53.15398,53.14347,53.18051,53.16559,
      53.18092,53.16981,53.16695,53.17203,53.14622,53.17067,53.14897,53.17655] * u.degree
dec = [-27.77667,-27.77757,-27.77627,-27.77544,-27.79087,-27.78327,-27.77970, -27.76990,
       -27.77624,-27.79697,-27.79884,-27.79517,-27.77994,-27.78204,-27.78194,-27.78550] * u.degree
redshifts = [3.00,2.794,2.541,2.43,1.759,1.411,2.59,1.552,0.667,2.086,1.996,5.000,2.497,0.769,1.721,1.314]

#************************************************************************
"""
Telescope Bandpass Filter .txt files and PSF FWHMs.
"""
#PACS link:
#http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Herschel/Pacs.red&&mode=browse&gname=Herschel&gname2=Pacs
"PACS: use all three of these"
PACS_70_extension = "Herschel_Pacs.70.dat.txt"
PACS_100_extension = "Herschel_Pacs.100um.dat.txt"
PACS_160_extension = "Herschel_Pacs.160um.dat.txt"
PACS_100_FWHM = 9.0 * u.arcsec
PACS_160_FWHM = 13.0 * u.arcsec

"LABOCA"
#LABOCA - 870um
#http://www.apex-telescope.org/bolometer/laboca/technical/
LABOCA_870_extension = "laboca_passband_normalized.txt"
#https://watermark.silverchair.com/mnras0411-0505.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAbkwggG1BgkqhkiG9w0BBwagggGmMIIBogIBADCCAZsGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMD5zJn-gDp06bCw9DAgEQgIIBbPgTfQTUCIkfYG0L3UJ040QV33xTqyK_qkLOQ-I6L9QVVh0S9shvfMIMBoaWnfe1kPRRkGT8a9IrzNmVewPzIHkxvR74MWMo4ayY86eY_8mabtobtsKKYPBfq4o-fl8cEOBduGRKz4AFwrACzvlztrY1NUeKNzY-OEW-cFPQd9ZYRRkLOnClgMjEz4WqzSaFC6KkGdmAokYfeFIW7t1a8_kdbiIQ2d6kEgwglLOqn-8E7mni91Ff1_67X5BHDtbUL0uZwWJ1Nkj_noCH3CWER8qOb6JtVpEexIdzR6kyqIIcHTvisMxSMO3bVt_yGHxOo7FISmM_D0olA6ESJbalPBlvfpr7bR1bLV4D_6l84WbTE0HbEd_Q2HWYxLdh8aGt7D1MY5NggcoODo5yWXyTlpVPTo55hMrEcFNRc1bP5BFcp9O7ovpYHJ9qs56iNBbk3uZWXXeqWbrYtCEsubg1e4fSgqJfxjgeYiXqGn0
LABOCA_870_FWHM = 19 * u.arcsec

"SPIRE"
#HERSCHEL SPIRE:
#https://academic.oup.com/mnras/article/433/4/3062/1749440
SPIRE_250_extension = "Herschel_SPIRE_250.PSW.dat.txt"
SPIRE_350_extension = "Herschel_SPIRE_350.PMW.dat.txt"
SPIRE_500_extension = "Herschel_SPIRE_500.PLW.dat.txt"
SPIRE_250_FWHM = 18.1 * u.arcsec
SPIRE_350_FWHM = 24.9 * u.arcsec
SPIRE_500_FWHM = 36.2 * u.arcsec

"MIPS"
#Spitzer MIPS
#http://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/MIPSfiltsumm.txt
MIPS_24_extension = "MIPSfiltsumm24_70_160.txt"
#http://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/50/
MIPS_24_FWHM = 5.9 * u.arcsec

#************************************************************************
"""
Instrumental Error .fits files
"""
#Jy/pixel
PACS_100_err_filename = "gh_goodss_dr1_100_err.fits"
PACS_160_err_filename = "gh_goodss_dr1_160_err.fits"

#Jy/beam
LABOCA_err_filename = "less_laboca_ecdfs_rms_v1.0.fits"

#MJy/sr
MIPS_24_err_filename = "s_mips_1_s1_v0.30_wht.fits"

#micro Jansky.
ALMA_err = [76,87,84,46,49,49,48,46,39,46,46,40,45,44,46,44]
ALMA_err = [float(i)*1e-6 for i in ALMA_err]

#************************************************************************
"""
FWHM to Beam Area (for unit conversions of maps)
"""
def FWHM_to_beam(fwhm):
    """
    Convert FWHM in arcsec^2 to beam area in arcsec^2
    """
    beam_fwhm = fwhm
    fwhm_in_sigma = 1. /(8*np.log(2))**0.5
    beam_sigma = beam_fwhm*fwhm_in_sigma
    return 2 * np.pi *beam_sigma**2

#************************************************************************
"""
Store of .fits files of flux density.
All units converted to Jy/Beam.
"""
#Jy/pixel
PACS_100_flux_filename = "gh_goodss_dr1_100_sci.fits"
PACS_160_flux_filename = "gh_goodss_dr1_160_sci.fits"
beam_area_100 = FWHM_to_beam(PACS_100_FWHM)
beam_area_160 = FWHM_to_beam(PACS_160_FWHM)
pixel_area_100 = (3.33333333333333E-4*u.degree)**2
pixel_area_100 = pixel_area_100.to(u.arcsec**2)
pixel_area_160 = (6.66666666666666E-4*u.degree)**2
pixel_area_160 = pixel_area_160.to(u.arcsec**2)
no_beams_100 = (pixel_area_100/beam_area_100).value
no_beams_160 = (pixel_area_160/beam_area_160).value
conv_fact_100 = 1.0/no_beams_100
conv_fact_160 = 1.0/no_beams_160

# Jy/beam - leaving in Jansky/Beam
LABOCA_flux_filename = "less_laboca_ecdfs_flux_v1.0.fits"

# Jy/beam
SPIRE_250_flux_filename = "NE-CDFS-SWIRE_nested-image_SMAP250_DR2.fits"
SPIRE_350_flux_filename = "NE-CDFS-SWIRE_nested-image_SMAP350_DR2.fits"
SPIRE_500_flux_filename = "NE-CDFS-SWIRE_nested-image_SMAP500_DR2.fits"

#Reprojected .fits files.
SPIRE_250_flux_filename = "SPIRE_250_reprojected.fits"
SPIRE_350_flux_filename = "SPIRE_350_reprojected.fits"
SPIRE_500_flux_filename = "SPIRE_500_reprojected.fits"

#Site for improved image.
#http://irsa.ipac.caltech.edu/cgi-bin/Atlas/nph-atlas?mission=SGOODS&hdr_location=%2Fwork%2FTMP_TNj1gR_26532%2FAtlas%2F03h_32m_44.04s_-27d_46m_36.0s_Equ_J2000_28658.v0001&collection_desc=The+Great+Observatories+Origins+Deep+Survey+%28GOODS%29&locstr=03h+32m+44.04s+-27d+46m+36.0s+Equ+J2000&regSize=0.1&covers=on&radius=0.3&radunits=deg&searchregion=on
MIPS_24_flux_filename = "SPITZER_M1_19926528_0043_0016_5_ebcd.fits" #old crappy image
MIPS_24_flux_filename = "s_mips_1_s1_v0.30_sci.fits" #new image
beam_area_24 = FWHM_to_beam(MIPS_24_FWHM)
Jy_per_DNsec = 6.691e-6
pixelarea_24 = (0.000333333333333**2)*(u.degree**2) #Area gathered from new, good image.
pixelarea_24 = pixelarea_24.to(u.arcsec**2)
num_pix_in_beam = beam_area_24/pixelarea_24
conv_factor_MIPS_24 = num_pix_in_beam*Jy_per_DNsec
conv_factor_MIPS_24 = conv_factor_MIPS_24.value

#micro Jy - conversion to Jy in list.
flux_vals_ALMA = [924,996,863,303,311,239,231,208,198,184,186,154,174,160,166,155]
flux_vals_ALMA = [float(i)*1e-6 for i in flux_vals_ALMA]

#************************************************************************
"""
Use ALMA data to obtain sensible starting guesses for constants C
"""
initial_C_guesses = []
for i in range(16):
    cut_off_ALMA = nu_cut(30,1.5,redshifts[i],10,2.0)
    C_i = flux_vals_ALMA[i]/(SED(c/1.3e-3,30,1.5,redshifts[i],10.0,2.0,cut_off_ALMA))
    initial_C_guesses.append(C_i)

#************************************************************************
"""
Information collection - things dobale outside MCMC loop for time saving.
"""
#Wavelengths and T_frac to avoid repeatedly calling "get_bandpass"
wavelengths_PACS_100, T_frac_PACS_100 = get_bandpass(PACS_100_extension,"PACS",100e-6)
wavelengths_PACS_160, T_frac_PACS_160 = get_bandpass(PACS_160_extension,"PACS",160e-6)
wavelengths_LABOCA, T_frac_LABOCA = get_bandpass(LABOCA_870_extension,"LABOCA",870e-6)
wavelengths_SPIRE_250, T_frac_SPIRE_250 = get_bandpass(SPIRE_250_extension,"SPIRE",250e-6)
wavelengths_SPIRE_350, T_frac_SPIRE_350 = get_bandpass(SPIRE_350_extension,"SPIRE",350e-6)
wavelengths_SPIRE_500, T_frac_SPIRE_500 = get_bandpass(SPIRE_500_extension,"SPIRE",500e-6)
wavelengths_MIPS, T_frac_MIPS = get_bandpass(MIPS_24_extension,"MIPS",24e-6)

#Source Locations pix to avoid repeatedly calling "get_source_location_pix"
sl_PACS_100 = get_source_locations_pix(PACS_100_flux_filename)
sl_PACS_160 = get_source_locations_pix(PACS_160_flux_filename)
sl_LABOCA = get_source_locations_pix(LABOCA_flux_filename)
sl_SPIRE_250 = get_source_locations_pix(SPIRE_250_flux_filename)
sl_SPIRE_350 = get_source_locations_pix(SPIRE_350_flux_filename)
sl_SPIRE_500 = get_source_locations_pix(SPIRE_500_flux_filename)
sl_MIPS = get_source_locations_pix(MIPS_24_flux_filename)

#FWHM conversions to avoid repeatedly calling FWHM_to_pix
fw_PACS_100 = FWHM_pix(PACS_100_FWHM,PACS_100_flux_filename,"PACS")
fw_PACS_160 = FWHM_pix(PACS_160_FWHM,PACS_160_flux_filename,"PACS")
fw_LABOCA = FWHM_pix(LABOCA_870_FWHM,LABOCA_flux_filename,"LABOCA")
fw_SPIRE_250 = FWHM_pix(SPIRE_250_FWHM,SPIRE_250_flux_filename,"SPIRE")
fw_SPIRE_350 = FWHM_pix(SPIRE_350_FWHM,SPIRE_350_flux_filename,"SPIRE")
fw_SPIRE_500 = FWHM_pix(SPIRE_500_FWHM,SPIRE_500_flux_filename,"SPIRE")
fw_MIPS = FWHM_pix(MIPS_24_FWHM,MIPS_24_flux_filename,"MIPS")

#Get pixel windows that box galaxies to avoid repeatedly doing the calculation within the Image Reconstruction Phase.
def pixel_window(sl,filename):
    """
    Generate grid that encompasses all 16 galaxies with a little wiggle room on either side (+/- N pixels)
    Input: sl - source locations
    """
    N = 10
    if int(min([i[0] for i in sl])) < N:
        min_x = 0
    else:
        min_x = int(min([i[0] for i in sl]))-N
    if int(min([i[1] for i in sl])) < N:
        min_y = 0
    else:
        min_y = int(min([i[1] for i in sl]))-N
    image_dim = get_image_dimension(filename,0)
    if int( max([i[0] for i in sl])) > image_dim[0]-N:
        max_x = image_dim[0]-1
    else:
        max_x =int( max([i[0] for i in sl]))+N
    if int(max([i[1] for i in sl])) > image_dim[1] -N:
        max_y = image_dim[1]-1
    else:
        max_y = int(max([i[1] for i in sl]))+N
    x = np.arange(min_x,max_x)
    y = np.arange(min_y,max_y)
    X,Y = np.meshgrid(x,y)
    return X,Y


def fixed_pixel_window(survey_filename):

    """
    Covering a fixed area of the sky in each case (making sure this area encompasses all ALMA sources).
    """
    far_left = max(ra)
    far_right = min(ra)
    top = min(dec)
    bottom = max(dec)
    #expand window
    far_left += 0.5*u.arcminute
    far_right -= 0.5*u.arcminute
    top -= 0.5*u.arcminute
    bottom += 0.5*u.arcminute
    far_left,bottom = get_location(far_left,bottom,survey_filename)
    far_right,top = get_location(far_right,top,survey_filename)
    x = np.arange(far_left, far_right)
    y = np.arange(top,bottom)
    X,Y = np.meshgrid(x,y)
    return X,Y

"""
Getting pixel windows the old way
"""

X_pix_PACS_100, Y_pix_PACS_100 = pixel_window(sl_PACS_100,PACS_100_flux_filename)
X_pix_PACS_160, Y_pix_PACS_160 = pixel_window(sl_PACS_160,PACS_160_flux_filename)
X_pix_LABOCA,Y_pix_LABOCA = pixel_window(sl_LABOCA,LABOCA_flux_filename)
X_pix_SPIRE_250, Y_pix_SPIRE_250 = pixel_window(sl_SPIRE_250,SPIRE_250_flux_filename)
X_pix_SPIRE_350, Y_pix_SPIRE_350 = pixel_window(sl_SPIRE_350,SPIRE_350_flux_filename)
X_pix_SPIRE_500, Y_pix_SPIRE_500 = pixel_window(sl_SPIRE_500,SPIRE_500_flux_filename)
X_pix_MIPS_24, Y_pix_MIPS_24 = pixel_window(sl_MIPS,MIPS_24_flux_filename)

"""
Getting pixel windows the new way - covering same sky area in each case.
"""
X_pix_PACS_100, Y_pix_PACS_100 = fixed_pixel_window(PACS_100_flux_filename)
X_pix_PACS_160, Y_pix_PACS_160 = fixed_pixel_window(PACS_160_flux_filename)
X_pix_LABOCA,Y_pix_LABOCA = fixed_pixel_window(LABOCA_flux_filename)
X_pix_SPIRE_250, Y_pix_SPIRE_250 = fixed_pixel_window(SPIRE_250_flux_filename)
X_pix_SPIRE_350, Y_pix_SPIRE_350 = fixed_pixel_window(SPIRE_350_flux_filename)
X_pix_SPIRE_500, Y_pix_SPIRE_500 = fixed_pixel_window(SPIRE_500_flux_filename)
X_pix_MIPS_24, Y_pix_MIPS_24 = fixed_pixel_window(MIPS_24_flux_filename)

#Get flux density and error matrices here to avoid repeatedly obtaining them inside NLL functions
import FITS_information as f

D_PACS_100, sigma_PACS_100 = np.abs(f.get_flux_density(PACS_100_flux_filename,X_pix_PACS_100,Y_pix_PACS_100,0,conv_fact_100)), np.abs(f.get_instrumental_error(PACS_100_err_filename,X_pix_PACS_100,Y_pix_PACS_100,conv_fact_100))
D_PACS_160, sigma_PACS_160 = np.abs(f.get_flux_density(PACS_160_flux_filename,X_pix_PACS_160,Y_pix_PACS_160,0,conv_fact_160)), np.abs(f.get_instrumental_error(PACS_160_err_filename,X_pix_PACS_160,Y_pix_PACS_160,conv_fact_160))
D_LABOCA, sigma_LABOCA =  np.abs(f.get_flux_density(LABOCA_flux_filename,X_pix_LABOCA,Y_pix_LABOCA,0,1.0)), np.abs(f.get_instrumental_error(LABOCA_err_filename,X_pix_LABOCA,Y_pix_LABOCA,1.0))
D_SPIRE_250 = np.abs(f.get_flux_density(SPIRE_250_flux_filename,X_pix_SPIRE_250,Y_pix_SPIRE_250,0,1.0))
D_SPIRE_350 = np.abs(f.get_flux_density(SPIRE_350_flux_filename,X_pix_SPIRE_350,Y_pix_SPIRE_350,0,1.0))
D_SPIRE_500 = np.abs(f.get_flux_density(SPIRE_500_flux_filename,X_pix_SPIRE_500,Y_pix_SPIRE_500,0,1.0))
D_MIPS, sigma_MIPS = np.abs(f.get_flux_density(MIPS_24_flux_filename,X_pix_MIPS_24,Y_pix_MIPS_24,0,conv_factor_MIPS_24)), np.abs(f.get_instrumental_error(MIPS_24_err_filename,X_pix_MIPS_24,Y_pix_MIPS_24,conv_factor_MIPS_24))

nuisance_PACS_100 = [[c/i for i in wavelengths_PACS_100],[c/(i**2) for i in wavelengths_PACS_100],[(wavelengths_PACS_100[i]/(100e-6))*T_frac_PACS_100[i]*c*(1/wavelengths_PACS_100[i]**2) for i in range(len(wavelengths_PACS_100))]]
nuisance_PACS_160 = [[c/i for i in wavelengths_PACS_160],[c/(i**2) for i in wavelengths_PACS_160],[(wavelengths_PACS_160[i]/(160e-6))*T_frac_PACS_160[i]*c*(1/wavelengths_PACS_160[i]**2) for i in range(len(wavelengths_PACS_160))]]
nuisance_LABOCA = [[c/i for i in wavelengths_LABOCA],[c/(i**2) for i in wavelengths_LABOCA],[(wavelengths_LABOCA[i]/(870e-6))*T_frac_LABOCA[i]*c*(1/wavelengths_LABOCA[i]**2) for i in range(len(wavelengths_LABOCA))]]
nuisance_SPIRE_250 = [[c/i for i in wavelengths_SPIRE_250],[c/(i**2) for i in wavelengths_SPIRE_250],[(wavelengths_SPIRE_250[i]/(250e-6))*T_frac_SPIRE_250[i]*c*(1/wavelengths_SPIRE_250[i]**2) for i in range(len(wavelengths_SPIRE_250))]]
nuisance_SPIRE_350 = [[c/i for i in wavelengths_SPIRE_350],[c/(i**2) for i in wavelengths_SPIRE_350],[(wavelengths_SPIRE_350[i]/(350e-6))*T_frac_SPIRE_350[i]*c*(1/wavelengths_SPIRE_350[i]**2) for i in range(len(wavelengths_SPIRE_350))]]
nuisance_SPIRE_500 = [[c/i for i in wavelengths_SPIRE_500],[c/(i**2) for i in wavelengths_SPIRE_500],[(wavelengths_SPIRE_500[i]/(500e-6))*T_frac_SPIRE_500[i]*c*(1/wavelengths_SPIRE_500[i]**2) for i in range(len(wavelengths_SPIRE_500))]]
nuisance_MIPS = [[c/i for i in wavelengths_MIPS],[c/(i**2) for i in wavelengths_MIPS],[(wavelengths_MIPS[i]/(24e-6))*T_frac_MIPS[i]*c*(1/wavelengths_MIPS[i]**2) for i in range(len(wavelengths_MIPS))]]


#************************************************************************
#Try and remove sources from MIPS 24um map.

"""
Sources removed from MIPS map by creating dummy map with sources at locations,
then convolving sources with MIPS beam to get spreads. then subtracting this result from MIPS map.

"""

#project MIPS files.
mips_pacs_flux_filename = "mips_into_pacs.fits"
mips_lab_flux_filename = "mips_into_laboca.fits"
mips_sp250_flux_filename = "mips_into_sp250.fits"
mips_sp350_flux_filename = "mips_into_sp350.fits"
mips_sp500_flux_filename = "mips_into_sp500.fits"


"""
Do some data extraction here to avoid doing it inside loops
"""
X_pix_mips_pacs,Y_pix_mips_pacs = fixed_pixel_window(mips_pacs_flux_filename)
D_mips_pacs = np.abs(f.get_flux_density(mips_pacs_flux_filename,X_pix_mips_pacs,Y_pix_mips_pacs,0,conv_fact_100))
sl_mips_pacs = get_source_locations_pix(mips_pacs_flux_filename)

X_pix_mips_lab,Y_pix_mips_lab = fixed_pixel_window(mips_lab_flux_filename)
D_mips_lab = np.abs(f.get_flux_density(mips_lab_flux_filename,X_pix_mips_lab,Y_pix_mips_lab,0,1.0))
sl_mips_lab = get_source_locations_pix(mips_lab_flux_filename)

X_pix_mips_sp25,Y_pix_mips_sp25 = fixed_pixel_window(mips_sp250_flux_filename)
D_mips_sp25 = np.abs(f.get_flux_density(mips_sp250_flux_filename,X_pix_mips_sp25,Y_pix_mips_sp25,0,1.0))
sl_mips_sp25 = get_source_locations_pix(mips_sp250_flux_filename)

X_pix_mips_sp35,Y_pix_mips_sp35 = fixed_pixel_window(mips_sp350_flux_filename)
D_mips_sp35 = np.abs(f.get_flux_density(mips_sp350_flux_filename,X_pix_mips_sp35,Y_pix_mips_sp35,0,1.0))
sl_mips_sp35 = get_source_locations_pix(mips_sp350_flux_filename)

X_pix_mips_sp5,Y_pix_mips_sp5 = fixed_pixel_window(mips_sp500_flux_filename)
D_mips_sp5 = np.abs(f.get_flux_density(mips_sp500_flux_filename,X_pix_mips_sp5,Y_pix_mips_sp5,0,1.0))
sl_mips_sp5 = get_source_locations_pix(mips_sp500_flux_filename)


def subtracted_data(flux_filename,conv_factor,D_to_be_subtracted,A,map):
    """
    Function takes reprojected MIPS map and subtracts the non-sources in this map from the data in D_to_be_subtracted
    which is either PACS, SPIRE, or LABOCA.

    """
    if map == "PACS":
        X_pix,Y_pix = deepcopy(X_pix_mips_pacs), deepcopy(Y_pix_mips_pacs)
        D_matrix = deepcopy(D_mips_pacs)
        sl = deepcopy(sl_mips_pacs)
    elif map == "LABOCA":
        X_pix,Y_pix = deepcopy(X_pix_mips_lab), deepcopy(Y_pix_mips_lab)
        D_matrix = deepcopy(D_mips_lab)
        sl = deepcopy(sl_mips_lab)
    elif map == "250":
        X_pix,Y_pix = deepcopy(X_pix_mips_sp25), deepcopy(Y_pix_mips_sp25)
        D_matrix = deepcopy(D_mips_sp25)
        sl = deepcopy(sl_mips_sp25)
    elif map == "350":
        X_pix,Y_pix = deepcopy(X_pix_mips_sp35), deepcopy(Y_pix_mips_sp35)
        D_matrix = deepcopy(D_mips_sp35)
        sl = deepcopy(sl_mips_sp35)
    elif map == "500":
        X_pix,Y_pix = deepcopy(X_pix_mips_sp5), deepcopy(Y_pix_mips_sp5)
        D_matrix = deepcopy(D_mips_sp5)
        sl = deepcopy(sl_mips_sp5)

    recreated_sources = np.zeros((len(D_matrix),len(D_matrix[0])))
    subtracted_data = np.zeros((len(D_MIPS),len(D_MIPS[0])))
    #Convolve each MIPS source with MIPS beam, then subtract this fake map off MIPS map original.
    x_min = X_pix_MIPS_24[0][0]
    y_min = Y_pix_MIPS_24[0][0]

    for i in range(len(recreated_sources)):
        for j in range(len(recreated_sources[i])):
            for k in range(16):
                sl = int(sl_MIPS[k][0]), int(sl_MIPS[k][1])
                data = D_MIPS[sl[0]-x_min][sl[1]-y_min]
                recreated_sources[i][j] += data*twoD_Gauss(X_pix_MIPS_24[0][j],Y_pix_MIPS_24[i][0],sl,fw_MIPS)
            r = D_MIPS[i][j] - recreated_sources[i][j]
            if r <= 0.0:
                r = 0.0
            subtracted_data[i][j] += r
            if (A*subtracted_data[i][j]) <= D_to_be_subtracted[i][j]:
                D_to_be_subtracted[i][j] -= (A*subtracted_data[i][j])
            else:
                val = np.abs(min([min(a) for a in D_to_be_subtracted if min(a) != 0.0]))
                D_to_be_subtracted[i][j] = np.random.normal(val,val*0.05)
    return 0.0

from copy import deepcopy
#Keep originals
D_PACS_160_original = deepcopy(D_PACS_160)
D_LABOCA_original = deepcopy(D_LABOCA)
D_SPIRE_250_original = deepcopy(D_SPIRE_250)
D_SPIRE_350_original = deepcopy(D_SPIRE_350)
D_SPIRE_500_original = deepcopy(D_SPIRE_500)

def do_subtraction(amplitudes,p):
    """
    Create subtracted fits maps for each map with updated parameters at each step of MCMC chain.
    """
    global D_PACS_160
    global D_LABOCA
    global D_SPIRE_250
    global D_SPIRE_350
    global D_SPIRE_500
    if p == "all":
        #Restore data to original values before subtracting with new parameters.
        D_PACS_160 = deepcopy(D_PACS_160_original)
        D_LABOCA = deepcopy(D_LABOCA_original)
        D_SPIRE_250 = deepcopy(D_SPIRE_250_original)
        D_SPIRE_350 = deepcopy(D_SPIRE_350_original)
        D_SPIRE_500 = deepcopy(D_SPIRE_500_original)

        return subtracted_data(mips_pacs_flux_filename,conv_fact_160,D_PACS_160,amplitudes[0],"PACS"), subtracted_data(mips_lab_flux_filename,1.0,D_LABOCA,amplitudes[1],"LABOCA"), subtracted_data(mips_sp250_flux_filename,1.0,D_SPIRE_250,amplitudes[2],"250"), subtracted_data(mips_sp350_flux_filename,1.0,D_SPIRE_350,amplitudes[3],"350"), subtracted_data(mips_sp500_flux_filename,1.0,D_SPIRE_500,amplitudes[4],"500")

    if p == 0:
        D_PACS_160 = deepcopy(D_PACS_160_original)
        subtracted_data(mips_pacs_flux_filename,conv_fact_160,D_PACS_160,amplitudes[0],"PACS")
    elif p == 1:
        D_LABOCA = deepcopy(D_LABOCA_original)
        subtracted_data(mips_lab_flux_filename,1.0,D_LABOCA,amplitudes[1],"LABOCA")
    elif p == 2:
        D_SPIRE_250 = deepcopy(D_SPIRE_250_original)
        subtracted_data(mips_sp250_flux_filename,1.0,D_SPIRE_250,amplitudes[2],"250")
    elif p == 3:
        D_SPIRE_350 = deepcopy(D_SPIRE_350_original)
        subtracted_data(mips_sp350_flux_filename,1.0,D_SPIRE_350,amplitudes[3],"350")
    elif p ==4:
        D_SPIRE_500 = deepcopy(D_SPIRE_500_original)
        subtracted_data(mips_sp500_flux_filename,1.0,D_SPIRE_500,amplitudes[4],"500")


    #during all other steps nothing is changed, and previous subtraction is maintained.
#************************************************************************
"""
Wrapping everything up into one NLL value.
"""
def NLL_total(params, calib_params,background_params,amplitudes,gaussian_params,test,amp_param):
    """
    Return single combined NLL value for all surveys for a given set of parameters as given by step of MCMC algorithm.
    Input: params - SED parameters,
           calib_params - calibration parameters,
           background_params - background parameters,
           gaussian_params - gaussian parameters,
           test - Boolean (redundant).
    Output - NLL value.
    """

    do_subtraction(amplitudes,amp_param)
    prior_value = prior(params,calib_params,gaussian_params,background_params,amplitudes)
    if prior_value == "OutsideRange":
        return "OutsideRange", 0.0
    #recon_data_PACS_100 = get_reconstr_img("PACS",PACS_100_flux_filename,PACS_100_extension,100e-6,fw_PACS_100,sl_PACS_100,params,background_params[0],wavelengths_PACS_100,T_frac_PACS_100,X_pix_PACS_100,Y_pix_PACS_100,nuisance_PACS_100)
    recon_data_PACS_160 = get_reconstr_img("PACS",PACS_160_flux_filename,PACS_160_extension,160e-6,fw_PACS_160,sl_PACS_160,params,background_params[1],wavelengths_PACS_160,T_frac_PACS_160,X_pix_PACS_160,Y_pix_PACS_160,nuisance_PACS_160)
    calib_factor_PACS_100, calib_factor_PACS_160 = calib_params[0], calib_params[1]
    #nll_pacs_100 = NLL_MAIN(X_pix_PACS_100,Y_pix_PACS_100,recon_data_PACS_100,calib_factor_PACS_100,PACS_100_flux_filename,PACS_100_err_filename,conv_fact_100,False,D_PACS_100,sigma_PACS_100)
    nll_pacs_160 = NLL_MAIN(X_pix_PACS_160,Y_pix_PACS_160,recon_data_PACS_160,calib_factor_PACS_160,PACS_160_flux_filename,PACS_160_err_filename,conv_fact_160,False,D_PACS_160,sigma_PACS_160)
    recon_data_laboca = get_reconstr_img("LABOCA",LABOCA_flux_filename,LABOCA_870_extension,870e-6,fw_LABOCA,sl_LABOCA,params,background_params[2],wavelengths_LABOCA,T_frac_LABOCA,X_pix_LABOCA,Y_pix_LABOCA,nuisance_LABOCA)
    calib_factor_laboca = calib_params[2] #dummy value.
    nll_laboca = NLL_MAIN(X_pix_LABOCA,Y_pix_LABOCA,recon_data_laboca, calib_factor_laboca,LABOCA_flux_filename,LABOCA_err_filename,1.0,False,D_LABOCA,sigma_LABOCA)
    recon_data_SPIRE_250 = get_reconstr_img("SPIRE",SPIRE_250_flux_filename,SPIRE_250_extension,250e-6,fw_SPIRE_250,sl_SPIRE_250,params,background_params[3],wavelengths_SPIRE_250,T_frac_SPIRE_250,X_pix_SPIRE_250,Y_pix_SPIRE_250,nuisance_SPIRE_250)
    recon_data_SPIRE_350 = get_reconstr_img("SPIRE",SPIRE_350_flux_filename,SPIRE_350_extension,350e-6,fw_SPIRE_350,sl_SPIRE_350,params,background_params[4],wavelengths_SPIRE_350,T_frac_SPIRE_350,X_pix_SPIRE_350, Y_pix_SPIRE_350,nuisance_SPIRE_350)
    recon_data_SPIRE_500 = get_reconstr_img("SPIRE",SPIRE_500_flux_filename,SPIRE_500_extension,500e-6,fw_SPIRE_500,sl_SPIRE_500,params,background_params[5],wavelengths_SPIRE_500,T_frac_SPIRE_500,X_pix_SPIRE_500,Y_pix_SPIRE_500,nuisance_SPIRE_500)
    calib_factor_SPIRE= calib_params[3],calib_params[4], calib_params[5] #dummy values.
    survey_filenames = [SPIRE_250_flux_filename,SPIRE_350_flux_filename,SPIRE_500_flux_filename]
    recon_data_SPIRE = [recon_data_SPIRE_250, recon_data_SPIRE_350, recon_data_SPIRE_500]
    pix_SPIRE = [(X_pix_SPIRE_250,Y_pix_SPIRE_250),(X_pix_SPIRE_350,Y_pix_SPIRE_350),(X_pix_SPIRE_500,Y_pix_SPIRE_500)]
    nll_SPIRE = NLL_SPIRE(recon_data_SPIRE, pix_SPIRE, calib_factor_SPIRE, survey_filenames,[1.0,1.0,1.0],[D_SPIRE_250,D_SPIRE_350,D_SPIRE_500])
    recon_data_alma =  get_reconstr_img("ALMA",0.0,0.0,1.3e-3,0,0,params,0.0,0,0,0,0,0) #most parameters irrelevant for ALMA
    calib_factor_alma = calib_params[6] #dummy values
    nll_alma = NLL_ALMA(recon_data_alma,flux_vals_ALMA,calib_factor_alma,ALMA_err)
    #recon_data_MIPS_24 = get_reconstr_img("MIPS",MIPS_24_flux_filename,MIPS_24_extension,24e-6,fw_MIPS,sl_MIPS,params,background_params[6],wavelengths_MIPS,T_frac_MIPS,X_pix_MIPS_24,Y_pix_MIPS_24,nuisance_MIPS)
    #calib_factor_MIPS_24 = calib_params[7] #dummy values
    #nll_mips_24 = NLL_MAIN(X_pix_MIPS_24,Y_pix_MIPS_24,recon_data_MIPS_24,calib_factor_MIPS_24,MIPS_24_flux_filename,MIPS_24_err_filename,conv_factor_MIPS_24,False,D_MIPS,sigma_MIPS)
    results =  [nll_pacs_160, nll_laboca, nll_SPIRE, nll_alma]
    return sum(results)+prior_value, results

#************************************************************************
"""
Dummy starting values for parameters.
"""
params = [[30,1.5,1.0,2.0]]*16 #assume all parameters are same for all sources
calib_params = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] # calibration parameters for all.

#Using average data from fits files to estimate starting point for background parameters.
background_params = [np.mean(D_PACS_100),np.mean(D_PACS_160),np.mean(D_LABOCA),np.mean(D_SPIRE_250),np.mean(D_SPIRE_350),np.mean(D_SPIRE_500),np.mean(D_MIPS)] #2 PACS, LABOCA, 3 SPIRE,  MIPS. (not alma)
"""
HERSCHEL PACS 100: 0.05 (SED paper)
PACS 160: 0.05 (from SED paper)
LABOCA: 0.05 (SED paper)
SPIRE: all 0.04 (taken from diagonal entries in SED paper).
ALMA: 0.05 https://arxiv.org/pdf/1503.07647.pdf
MIPS 24: 0.04 http://iopscience.iop.org/1538-4357/700/2/L73/suppdata/apjl313633t2_mrt.txt
"""
gaussian_params = [0.05,0.05,0.05,0.04,0.04,0.04,0.05,0.04] #7 values for sigma of gaussian priors.

#************************************************************************
"""
Markov Chain Monte Carlo w/ Gibbs Sampling.

params is a list of all starting values of parameters (in order presribed in README file).
The different lists below are just a bunch of different informed starting values.
"""
params = []
for i in range(16):
    params.append([30,initial_C_guesses[i]]) #fitting temperature and constants

params = []
params = [15,25,15,30,28,10,30,3,15,8,15,15,20,1.0,45,2.5,40,4,15,7,18,10,30,0.3,20,2.5,15,6,15,5,20,5]
params = [12,120,12,120,12,100,12,40,13,45,12,45,12.5,25,40,3,25,15,13,25,13,20,15,10,15,20,15,30,12,25,14,30]
params = [40,120,40,120,40,100,40,40,40,45,40,45,40,25,40,3,40,15,40,25,40,20,40,10,40,20,40,30,40,25,40,30]
amplitudes = [1.0,1.0,1.0,1.0,1.0]
betas = [1.0 for i in range(16)]
#params_0 = [item for sublist in params for item in sublist] + calib_params + background_params #32 SED + 8 calib + 7 background = 47 parameters
params_0 = params + calib_params + background_params + amplitudes + betas
def pack_up(params):
    """
    Transforms strung out list of parameters into bundled up format as taken by NLL function
    """
    SED_params = params[:32]
    calib_params = params[32:40]
    background_params = params[40:47]
    amplitude_params = params[47:52]
    betas = params[52:68]
    #manual_input = [[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.5,2.0],[1.5,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0],[1.0,2.0]]
    #manual_input = [[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[1.5,2.0],[1.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0],[0.5,2.0]]
    manual_input = []
    for i in betas:
        manual_input.append([i,2.0])
    #manual_input = [[1.5,2.0] for i in range(16)] #inputting parameters that I will have found and fixed (alpha and beta)
    temps = []
    consts = []
    for i in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
        temps.append(SED_params[i])
        consts.append(SED_params[i+1])
    SED_values = []
    for i in range(16):
        SED_values.append([temps[i],manual_input[i][0],consts[i],manual_input[i][1]])
    return SED_values, calib_params, background_params, amplitude_params

import pickle
import copy

def MCMC(p_0, N,acc_params):
    """
    Performs N iterations of MCMC algorithm from starting values p_0
    """
    step_size_0,N_attempts,target_accep = acc_params
    step_size = step_size_0[:]
    step_size_list = [[] for i in range(N+1)]
    step_size_list[0] = step_size_0[:]
    count_steps = dict([(i,[0,0]) for i in range(len(step_size))]) #count Nrejections and Ntrials for each parameter.
    count=0
    p_list = p_0[:]
    accep_changes = []
    param_list = []
    for i in p_0:
        param_list.append([i])
        accep_changes.append([i])
    p1,p2,p3,p4 = pack_up(p_list)
    prev_NLL,list_values_NLL = NLL_total(p1,p2,p3,p4,gaussian_params,False,"all")
    list_values_NLL = [list_values_NLL]
    num_evals = 0
    NLL_results = []
    prev_step_size= step_size_0[:]
    prev_p_list = p_list[:]
    step_size_stop = step_size_0[:]

    while count < N:
        for j in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,41,42,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]:
        #for j in range(len(p_list)):
            if count == 1999:
                step_size_stop[j] = copy.copy(step_size[j])

            if count < 2000:
                if count_steps[j][1] >= N_attempts:
                    accep_ratio = float(count_steps[j][1]-count_steps[j][0])/float(count_steps[j][1])
                    #update step size to improve/ reduce acceptance rate.
                    if accep_ratio > target_accep:
                        step_size[j] *= 1.1
                    elif accep_ratio < target_accep:
                        step_size[j] *= 0.9
                    #reset counters.
                    count_steps[j][1] = 0
                    count_steps[j][0] = 0

            #elif count >= 2000 and count <=8000:
            #        step_size[j] = step_size_stop[j] + (count-2000)*((0.1*step_size_stop[j])-step_size_stop[j])*(1.0/6000.)
            #don't change the step size after that

            step_size_list[count+1].append(step_size[:][j])
            count_steps[j][1] += 1 #update number of trials in counter.
            NLL_results.append(prev_NLL)
            new_param = np.random.normal(p_list[j],np.abs(step_size[j]*p_0[j]))
            p_list_possible = p_list[:]
            p_list_possible[j] = new_param
            p1,p2,p3,p4 = pack_up(p_list_possible)
            NLL_now,results = NLL_total(p1,p2,p3,p4,gaussian_params,False,j-47)
            list_values_NLL.append(results)

            if NLL_now != "OutsideRange":
                #Do Gauss and probability stuff together to avoid overflows.
                P_new_old = np.exp(prev_NLL - NLL_now)
                #A_old_new = (np.abs(step_size[j]*p_0[j])/np.abs(prev_step_size[j]*p_0[j]))*np.exp((-1*0.5*(((prev_p_list[j]-new_param)/np.abs(step_size[j]*p_0[j]))**2))-(-1*0.5*(((new_param-prev_p_list[j])/np.abs(prev_step_size[j]*p_0[j]))**2)))
                #a = A_old_new*P_new_old
                a = P_new_old

            if NLL_now == "OutsideRange": #reject
                param_list[j].append(p_list[j])
                count_steps[j][0] +=1 #count the rejection

            elif a >=1.0: #accept always
                p_list[j] = new_param
                prev_NLL = NLL_now
                param_list[j].append(new_param)
                accep_changes[j].append(new_param)
            elif np.random.random() < a: #accept with probability
                p_list[j] = new_param
                prev_NLL = NLL_now
                param_list[j].append(new_param)
                accep_changes[j].append(new_param)
            else: #reject
                param_list[j].append(p_list[j])
                count_steps[j][0] +=1 #count the rejection
            #fill in the blanks
            for k in range(len(p_list)):
                if k !=j:
                    param_list[k].append(p_list[k])
            num_evals +=1


        count +=1
        print "Chain Link: ", count
        if count == 100:
            with open('MCMC_chain_intermediate_100_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 200:
            with open('MCMC_chain_intermediate_200_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 300:
            with open('MCMC_chain_intermediate_300_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 400:
            with open('MCMC_chain_intermediate_400_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 500:
            with open('MCMC_chain_intermediate_500_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 1000:
            with open('MCMC_chain_intermediate_1000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 1500:
            with open('MCMC_chain_intermediate_1500_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 2000:
            with open('MCMC_chain_intermediate_2000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 3000:
            with open('MCMC_chain_intermediate_3000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 4000:
            with open('MCMC_chain_intermediate_4000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 5000:
            with open('MCMC_chain_intermediate_5000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 10000:
            with open('MCMC_chain_intermediate_10000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 15000:
            with open('MCMC_chain_intermediate_15000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 20000:
            with open('MCMC_chain_intermediate_20000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 25000:
            with open('MCMC_chain_intermediate_25000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
        if count == 30000:
            with open('MCMC_chain_intermediate_30000_sdf.txt', 'wb') as fp:
                result = param_list[:]
                result.append(NLL_results)
                result.append(list_values_NLL)
                pickle.dump(result, fp)
                print "Written File!"
    return param_list,accep_changes,NLL_results,list_values_NLL

#************************************************************************
"""
Testing & Running Code.
Set boolean run_MCMC to True to use MCMC algorithm.
'steps' is a list of all the fractional step sizes that are input at the start of the MCMC.
N_chain is the number of chains.
N_trials - number of chains performed before testing acceptance rate and updating step-size adaptively.
target_accep_rate - target acceptance rate.

This will output a text file 'MCMC_Chain.txt' that can be interogated using 'Output.py' script.

"""
from matplotlib.colors import LogNorm


run_MCMC = True
N_chain = 10000
N_trials = 25
target_accep_rate = 0.234 #0.234 in Gelman paper (https://projecteuclid.org/DPubS?verb=Display&version=1.0&service=UI&handle=euclid.aoap/1034625254&page=record)

if run_MCMC:
    start = time.time()
    steps = []
    for i in range(32):
        steps.append(0.01)
        steps.append(0.1) #make constant parameters step sizes a little larger.
    for i in range(8):
        steps.append(0.01)
    for i in range(7):
        steps.append(0.1)
    for i in range(5):
        steps.append(0.1)
    for i in range(16):
        steps.append(0.1)

    result,acc,acc_NLL_vals,list_vals_NLL_results = MCMC(params_0,N_chain,[steps,N_trials,target_accep_rate])
    result.append(acc_NLL_vals) #save NLL values throughout chain.
    result.append(list_vals_NLL_results)
    with open('MCMC_chain.txt', 'wb') as fp:
        pickle.dump(result, fp)
    print "MCMC Details"
    print "Number of chain links: ", N_chain
    print "Finished in time: ", time.time() - start

#************************************************************************
#Everything below here is just random little things to plot/test stuff during development phase.
test_NLL = False
if test_NLL:
    print "Completed Process for: "
    s = time.time()
    params,calib_params,background_params = pack_up(params_0)
    #final_vals = [51.03951204076105, 1.295997164487083, 11.852168374788642, 53.470610173508575, 32.52281567594406, 6.902375004607385, 46.920444343970416, 1.0897995715330298, 10.000000071051172, 49.87175667274804, 10.000000092453663, 51.621386339478434, 58.246409732340915, 0.20799356944515948, 48.84963852657615, 1.5079820724241013, 65.12781883251586, 0.18136769158942204, 10.910570961177891, 21.499858065212237, 13.16594095976401, 15.358157307883426, 20.83229566758569, 0.9297342738442739, #10.00000002046981, 23.49313462122445, 46.017425707239184, 0.1978457796402932, 10.000000140063122, 26.513123904018762, 26.485403211794214, 3.6984557114927177, 0.9154863434095917, 1.096810430817227, 1.0314154466282202, 0.9942236252176718, 0.9120555315138649, 1.0043626200994442, 1.1812829483229816, 1.0000023262552444, 0.0008089152457051789, 0.002029575464594463, 0.0012234744336325266, 0.006476958126902229, 0.006231492917405989, 0.006536797738130663, 7.5355221745007e-06]
    #params,calib_params,background_params = pack_up(final_vals)
    #final_vals_2000 = [11.014829173698518, 79.29460421904142, 10.099155581821256, 122.79916278820852, 31.16045496124679, 8.058360907457981, 43.6954262026664, 1.3885750854646253, 10.641982465226196, 41.573858887991385, 10.47286351076415, 45.563613393400196, 52.95571963846283, 0.32096730197681417, 54.434020654611146, 0.7910489030193013, 43.767135501420356, 1.3447102092407117, 10.633701126090147, 24.441634063084948, 12.110569752391996, 17.572713447828583, 17.6907230680834, 1.8136599177271646, #13.452413388966013, 8.481629844975691, 64.67875710587683, 0.05295512609936473, 12.344695845053081, 16.497038430168544, 25.68491060209542, 4.1150355890919075, 0.9575100011321509, 1.0310537378219418, 1.001540283896861, 0.9345668899859912, 0.8854382993906474, 1.024759441820604, 1.0793012820198398, 0.9919490874040097, 0.0008182064309263971, 0.0024556848846767616, 0.0011235404380064372, 0.006908797291627328, 0.006812250278399308, 0.007682146113775505, 8.417651735702076e-06]
    #params,calib_params,background_params = pack_up(final_vals_2000)
    value = NLL_total(params, calib_params,background_params,gaussian_params,False)
    print "Total Time Taken: ", time.time()-s
    print "...Finished"
    print "HERSCHEL PACS, 100um = %.2E" % value[1][0]
    print "HERSCHEL PACS, 160um = %.2E" % value[1][1]
    print "LABOCA = %.2E" % value[1][2]
    print "SPIRE = %.2E" % value[1][3]
    print "ALMA = %.2E" % value[1][4]
    print "MIPS, 24um = %.2E" % value[1][5]
    print "Total NLL = %.2E" % value[0]

final_vals_2000 = [11.014829173698518, 79.29460421904142, 10.099155581821256, 122.79916278820852, 31.16045496124679, 8.058360907457981, 43.6954262026664, 1.3885750854646253, 10.641982465226196, 41.573858887991385, 10.47286351076415, 45.563613393400196, 52.95571963846283, 0.32096730197681417, 54.434020654611146, 0.7910489030193013, 43.767135501420356, 1.3447102092407117, 10.633701126090147, 24.441634063084948, 12.110569752391996, 17.572713447828583, 17.6907230680834, 1.8136599177271646, 13.452413388966013, 8.481629844975691, 64.67875710587683, 0.05295512609936473, 12.344695845053081, 16.497038430168544, 25.68491060209542, 4.1150355890919075, 0.9575100011321509, 1.0310537378219418, 1.001540283896861, 0.9345668899859912, 0.8854382993906474, 1.024759441820604, 1.0793012820198398, 0.9919490874040097, 0.0008182064309263971, 0.0024556848846767616, 0.0011235404380064372, 0.006908797291627328, 0.006812250278399308, 0.007682146113775505, 8.417651735702076e-06]
final_vals_2000 = [31.49,1.75,10.27,80.07,15.9,29.0,10.01,28.91,10.01,33.99,10.1,34.21,22.26,1.48,10.1,4.23,23.04,
0.55,10.01,17.31,10.05,22.08,10.42,14.34,10.02,14.68,10.03,3.53e-11,10.07,4.21,10.00,0.003,0.91,0.94,1.022,0.92,
0.98,1.10,1.19,0.91,0.00079,0.00176,0.0012,0.0058,0.0063,0.0067,1.323e-5]

test_shape = False
if test_shape:
    """
    Consider that just taking a flux value at a point doesn't really represnt the "blobs" NECESSARILY.
    Should look more carefully at this if I come back to it later.....
    """
    for i in range(16):
        plt.figure()
        UDF_2 = (ra[i],dec[i])
        X,Y = get_location(UDF_2[0],UDF_2[1],PACS_100_flux_filename)
        plt.scatter(c/100e-6, np.abs(f.get_flux_density(PACS_100_flux_filename,Y,X,0,conv_fact_100))-final_vals_2000[40])
        X,Y = get_location(UDF_2[0],UDF_2[1],PACS_160_flux_filename)
        plt.scatter(c/160e-6,np.abs(f.get_flux_density(PACS_160_flux_filename,Y,X,0,conv_fact_160))-final_vals_2000[41])
        X,Y = get_location(UDF_2[0],UDF_2[1],LABOCA_flux_filename)
        plt.scatter(c/870e-6,np.abs(f.get_flux_density(LABOCA_flux_filename,Y,X,0,1.0))-final_vals_2000[42])
        X,Y = get_location(UDF_2[0],UDF_2[1],SPIRE_250_flux_filename)
        plt.scatter(c/250e-6,np.abs(f.get_flux_density(SPIRE_250_flux_filename,Y,X,0,1.0))-final_vals_2000[43])
        X,Y = get_location(UDF_2[0],UDF_2[1],SPIRE_350_flux_filename)
        plt.scatter(c/350e-6,np.abs(f.get_flux_density(SPIRE_350_flux_filename,Y,X,0,1.0))-final_vals_2000[44])
        X,Y = get_location(UDF_2[0],UDF_2[1],SPIRE_500_flux_filename)
        plt.scatter(c/500e-6,np.abs(f.get_flux_density(SPIRE_500_flux_filename,Y,X,0,1.0))-final_vals_2000[45])
        plt.scatter(c/1.3e-3,flux_vals_ALMA[i]-final_vals_2000[46])
        wavelengths = np.logspace(-5,-1,10000) #return evenly spaced numbers in logspace
        frequencies = [c/w for w in wavelengths]
        T_d,beta,z,C,alpha = final_vals_2000[int(2*i)],1.5,redshifts[i],final_vals_2000[int(2*i)+1],2.0
        nu_cut_val = nu_cut(T_d,beta,z,C,alpha)
        plt.plot(frequencies,[SED(n,T_d,beta,z,C,alpha,nu_cut_val) for n in frequencies])
        plt.xscale("log")
        plt.yscale("log")
    plt.show()

from matplotlib.colors import LogNorm

galaxy_overlay = False
if galaxy_overlay:
    filestring = MIPS_24_flux_filename
    FWHM_input = MIPS_24_FWHM
    X_vals,Y_vals = X_pix_MIPS_24,Y_pix_MIPS_24
    name = "MIPS"
    filename =  get_pkg_data_filename(filestring)
    hdu = fits.open(filename)[0]
    image_data = fits.open(filename)[0].data
    w = WCS(hdu.header)
    transformed = w.wcs_world2pix(ra*u.degree,dec*u.degree,0)
    x_vals = np.linspace(min(X_vals[0]),max(X_vals[0]),100)
    y_vals = np.linspace(Y_vals[0][0],Y_vals[-1][0],100)

    results = []
    for i in range(len(ra)):
        """
        x_list = np.linspace(min(ra)-3*u.arcmin,max(ra)+3*u.arcmin,100)
        y_list = np.linspace(min(dec)-3*u.arcmin,max(dec)+3*u.arcmin,100)
        X,Y = np.meshgrid(x_list,y_list)
        z = twoD_Gauss(X,Y,[ra[i],dec[i]],FWHM_input)
        X,Y = w.wcs_world2pix(X,Y,0) #transform to pixel
        results.append((X,Y,z))
        """
        X,Y = np.meshgrid(x_vals,y_vals)
        z = twoD_Gauss(X,Y,[transformed[0][i],transformed[1][i]],FWHM_pix(FWHM_input,filestring,name))
        results.append((X,Y,z))


    Z = np.zeros((100,100))
    for j in range(100):
        for k in range(100):
            for l in range(len(results)):
                Z[j][k] += results[l][2][j][k]

    plt.figure()
    plt.subplot(projection=w)
    plt.imshow(image_data,cmap='gray',norm=LogNorm())
    #plt.imshow(image_data,norm=LogNorm())
    plt.colorbar()
    for j in range(16):
        plt.annotate(j+1,(transformed[0][j],transformed[1][j]),color='black')
        plt.scatter(transformed[0][j],transformed[1][j],color='red')
    plt.contour(X,Y,Z,levels=[10**i for i in np.linspace(-20,8,28)],norm=LogNorm())
    plt.colorbar()
    plt.plot([min(X_vals[0])]*100,y_vals,color="black")
    plt.plot([max(X_vals[0])]*100,y_vals,color="black")
    plt.plot(x_vals,[Y_vals[0][0]]*100,color="black")
    plt.plot(x_vals,[Y_vals[-1][0]]*100,color="black")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.show()


#************************************************************************
if TIMING_TESTING:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats()
    print s.getvalue()
