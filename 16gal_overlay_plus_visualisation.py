#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:08:12 2018

@author: DewiGould
Created: Hennings, UBC
"""

"""
Code to overlay galaxy positions on different maps to qualitatively look at what's there.
There is also some garbage code at the bottom which overlays some convolved Gaussians etc.. (not really useful anymore).
"""

from astropy.io import fits
from astropy.wcs import WCS, utils
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
import numpy as np
from matplotlib.colors import LogNorm

ra = [53.18348,53.18137,53.16062,53.17090,53.15398,53.14347,53.18051,53.16559,
      53.18092,53.16981,53.16695,53.17203,53.14622,53.17067,53.14897,53.17655]*u.degree
dec = [-27.77667,-27.77757,-27.77627,-27.77544,-27.79087,-27.78327,-27.77970, -27.76990,
       -27.77624,-27.79697,-27.79884,-27.79517,-27.77994,-27.78204,-27.78194,-27.78550]*u.degree

def get_image_dimension(filestring,index):
    """
    Return pixel dimension of .fits file image, with data at specific .fits files index 'index'.
    """
    filename =  get_pkg_data_filename(filestring)
    image_data = fits.open(filename)[index].data
    return image_data.shape

def twoD_Gauss(x,y,mu,FWHM):
    """
    PSF with given mean and FWHM.
    Input: x,y - pixel values at which to evaluate PSF,
           mu - mean of Gaussian,
           FWHM - full-width-half-maximum of PSF.
    Output: Not-normalised Gaussian value (not normalised by design).
    """
    return (np.exp(-1.0*((((x-mu[0])**2.0)+((y-mu[1])**2.0))/(2.0*((np.sqrt((FWHM**2.0)/(8.0*np.log(2))))**2.0)))))

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

PACS_100_flux_filename = "gh_goodss_dr1_100_sci.fits"
PACS_160_flux_filename = "gh_goodss_dr1_160_sci.fits"
LABOCA_flux_filename = "less_laboca_ecdfs_flux_v1.0.fits"
SPIRE_250_flux_filename = "SPIRE_250_reprojected.fits"
SPIRE_350_flux_filename = "SPIRE_350_reprojected.fits"
SPIRE_500_flux_filename = "SPIRE_500_reprojected.fits"
MIPS_24_flux_filename = "s_mips_1_s1_v0.30_sci.fits"

PACS_100_FWHM = 9.0 * u.arcsec
PACS_160_FWHM = 13.0 * u.arcsec
LABOCA_870_FWHM = 19 * u.arcsec
SPIRE_250_FWHM = 18.1 * u.arcsec
SPIRE_350_FWHM = 24.9 * u.arcsec
SPIRE_500_FWHM = 36.2 * u.arcsec
MIPS_24_FWHM = 5.9 * u.arcsec

sl_PACS_100 = get_source_locations_pix(PACS_100_flux_filename)
sl_PACS_160 = get_source_locations_pix(PACS_160_flux_filename)
sl_LABOCA = get_source_locations_pix(LABOCA_flux_filename)
sl_SPIRE_250 = get_source_locations_pix(SPIRE_250_flux_filename)
sl_SPIRE_350 = get_source_locations_pix(SPIRE_350_flux_filename)
sl_SPIRE_500 = get_source_locations_pix(SPIRE_500_flux_filename)
sl_MIPS = get_source_locations_pix(MIPS_24_flux_filename)

X_pix_PACS_100, Y_pix_PACS_100 = pixel_window(sl_PACS_100,PACS_100_flux_filename)
X_pix_PACS_160, Y_pix_PACS_160 = pixel_window(sl_PACS_160,PACS_160_flux_filename)
X_pix_LABOCA,Y_pix_LABOCA = pixel_window(sl_LABOCA,LABOCA_flux_filename)
X_pix_SPIRE_250, Y_pix_SPIRE_250 = pixel_window(sl_SPIRE_250,SPIRE_250_flux_filename)
X_pix_SPIRE_350, Y_pix_SPIRE_350 = pixel_window(sl_SPIRE_350,SPIRE_350_flux_filename)
X_pix_SPIRE_500, Y_pix_SPIRE_500 = pixel_window(sl_SPIRE_500,SPIRE_500_flux_filename)
X_pix_MIPS_24, Y_pix_MIPS_24 = pixel_window(sl_MIPS,MIPS_24_flux_filename)



filestring = MIPS_24_flux_filename
FWHM_input = MIPS_24_FWHM
X_vals,Y_vals = X_pix_MIPS_24,Y_pix_MIPS_24


filename =  get_pkg_data_filename(filestring)
hdu = fits.open(filename)[0]
image_data = fits.open(filename)[0].data
w = WCS(hdu.header)

#convert from eq to pixel coordinates
#1 required for FITS origin being upper left
transformed = w.wcs_world2pix(ra*u.degree,dec*u.degree,0)
x_vals = np.linspace(min(X_vals[0]),max(X_vals[0]),100)
y_vals = np.linspace(Y_vals[0][0],Y_vals[-1][0],100)

results = []
for i in range(len(ra)):
    x_list = np.linspace(min(ra)-3*u.arcmin,max(ra)+3*u.arcmin,200)
    y_list = np.linspace(min(dec)-3*u.arcmin,max(dec)+3*u.arcmin,200)
    X,Y = np.meshgrid(x_list,y_list)
    z = twoD_Gauss(X,Y,[ra[i],dec[i]],FWHM_input)
    X,Y = w.wcs_world2pix(X,Y,0) #transform to pixel
    results.append((X,Y,z))

Z = np.zeros((200,200))
for j in range(200):
    for k in range(200):
        for l in range(len(results)):
            Z[j][k] += results[l][2][j][k]

image_data = np.abs(image_data)
plt.figure()
plt.subplot(projection=w)
plt.imshow(image_data,norm=LogNorm())
plt.colorbar()

#plt.contourf(X,Y,Z,20)

for j in range(16):
    plt.annotate(j+1,(transformed[0][j],transformed[1][j]),color='black')
    plt.scatter(transformed[0][j],transformed[1][j],color='red')


#plt.colorbar()
plt.plot([min(X_vals[0])]*100,y_vals,color="black")
plt.plot([max(X_vals[0])]*100,y_vals,color="black")
plt.plot(x_vals,[Y_vals[0][0]]*100,color="black")
plt.plot(x_vals,[Y_vals[-1][0]]*100,color="black")
plt.xlabel("RA")
plt.ylabel("Dec")
plt.title("24um")
plt.show()

#Doing some silly shit here - overlaying Gaussian's centred at each galaxy with mean
#obtained from paper (something randomish) - and then looking at total distribution
#kind of cool!
other=False
if other:
    def twoD_Gauss(x,y,mu,sigma):
        """
        mu and sigma are 2d arrays which take error in x and y separately
        I only included this because I'm not sure whether the Gaussian is in pixels or World coordinates
        """
        mu_x, mu_y = mu
        sigma_x, sigma_y = sigma

        A = 1.
        pre_factor = A
        exp = np.exp(-1*((((x-mu_x)**2)/(2*(sigma_x**2)))+(((y-mu_y)**2)/(2*(sigma_y**2)))))
        return pre_factor*exp


    #Manually inputting sigma from paper of different surveys etc....
    PACS_160_sigma = 11.6 * u.arcsec

    #Error is really small - so Gaussian circles are confined quite tightly (decaying away to zero very quickly)
    #Defined WINDOW as +/- 3arcmins beyond min and max range of observed galaxies
    results = []
    for i in range(len(ra)):
        x_list = np.linspace(min(ra)-3*u.arcmin,max(ra)+3*u.arcmin,100)
        y_list = np.linspace(min(dec)-3*u.arcmin,max(dec)+3*u.arcmin,100)
        X,Y = np.meshgrid(x_list,y_list)
        z = twoD_Gauss(X,Y,[ra[i],dec[i]],[PACS_160_sigma,PACS_160_sigma])
        X,Y = w.wcs_world2pix(X,Y,0) #transform to pixel
        results.append((X,Y,z))

    Z = np.zeros((100,100))
    for j in range(100):
        for k in range(100):
            for l in range(len(results)):
                Z[j][k] += results[l][2][j][k]

    plt.figure()
    plt.subplot(projection=w)
    plt.imshow(image_data,cmap='prism')
    for j in range(16):
        plt.scatter(transformed[0][j],transformed[1][j],color='red')
        plt.contour(results[j][0],results[j][1],results[j][2])
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.show()

    plt.figure()
    plt.subplot(projection=w)
    plt.imshow(image_data,cmap='prism')
    for j in range(16):
        plt.scatter(transformed[0][j],transformed[1][j],color='red')
    plt.contour(X,Y,Z,10)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.show()

    plt.figure()
    plt.subplot(projection=w)
    plt.contourf(X,Y,Z,25)
    for j in range(16):
        plt.scatter(transformed[0][j],transformed[1][j],color='red')
    plt.show()
