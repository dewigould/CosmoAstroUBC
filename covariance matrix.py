#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:56:18 2018

@author: DewiGould
"""


"""
Probably using the wrong survey as my background
Also: check logic. I am building "background subtracted" maps for the three wavelengths, then finding covariance between
each pair and building 3x3 matrix. I *think* this should be fine...?

CAREFUL: if units ever change, or I modify what my background is - I am doing unit conversion to Jy/Beam here - so this needs 
to be udpated for whatever is put in.
"""
from astropy import units as u

SPIRE_250_flux_filename = "SPIRE_250_reprojected.fits"
SPIRE_350_flux_filename = "SPIRE_350_reprojected.fits"
SPIRE_500_flux_filename = "SPIRE_500_reprojected.fits"


#Need dummy parameters just to set it off - the covariance matrix DOESN'T DEPEND on these.
#generating dummy parameters are passing each source through filter.
#T_d, beta, z, C, alpha for each source.
params = [[100,1.5,2,1e-40,2.8]]*16 #assume all parameters are same for all sources


#Getting the pixels of the window in SPIRE data under consideration
from ImagePlaneReconstruction import get_reconstr_img
from FWHM_to_pix import FWHM_pix
from SourceLocations import get_source_locations_pix
SPIRE_250_extension = "/Users/DewiGould/Desktop/iUROP_UBC_ResearchFolder/LogLikelihood Codes/TelescopeBandPassFilters_txt_files/Herschel_SPIRE_250.PSW.dat.txt"
SPIRE_250_FWHM = 18.1 * u.arcsec
X_pix_SPIRE_250, Y_pix_SPIRE_250, recon_data_SPIRE_250 = get_reconstr_img("SPIRE",SPIRE_250_flux_filename,SPIRE_250_extension,250e-6,FWHM_pix(SPIRE_250_FWHM,SPIRE_250_flux_filename,1.0),get_source_locations_pix(SPIRE_250_flux_filename),params)
  


from numpy import meshgrid


pixel_window_size = (109,111)
x_pixels = range(pixel_window_size[0])
y_pixels = range(pixel_window_size[1])
index_grid_X,index_grid_Y = meshgrid(x_pixels, y_pixels)

#For the moment: just use this file as the "general background thing over which to take cuts"
PACS_100_flux_filename = "gh_goodss_dr1_100_sci.fits"

#Unit conversion to Jy/Beam for PACS file.
PACS_100_FWHM = 9.5 * u.arcsec
from FullTask import FWHM_to_beam
beam_area_100 = FWHM_to_beam(PACS_100_FWHM)
pixel_area_100 = (3.33333333333333E-4*u.degree)**2
pixel_area_100 = pixel_area_100.to(u.arcsec**2)

no_beams_100 = (pixel_area_100/beam_area_100).value
conv_fact_100 = no_beams_100

#background image dimensions
from FITS_information import get_image_dimension
pixels = get_image_dimension(PACS_100_flux_filename,0)

from numpy import random, arange, zeros
X = []
Y = []

#Number of times we are cutting-out sections are running test.
N_cutouts = 1000

#Pick random start pixel (bottom left corner of cut out)
for i in range(N_cutouts):
    X.append(random.randint(0,pixels[0]-pixel_window_size[0]))
    Y.append(random.randint(0,pixels[1]-pixel_window_size[1]))
    
#build grid of background image pixels forming different cut outs
pixelgrids = []
for i in range(N_cutouts):
    horiz = arange(X[i],X[i]+pixel_window_size[0])
    vert = arange(Y[i],Y[i]+pixel_window_size[1])
    pixelgrids.append(meshgrid(horiz,vert))

#evaluate flux density at each point on background using these generated grids
from FITS_information import get_flux_density
background = []
for grids in pixelgrids:
    background.append(get_flux_density(PACS_100_flux_filename,grids[0],grids[1],0,conv_fact_100))

#Evaluate the flux density at all designated pixels within the SPIRE images for comparison with the dummy windows above
flux_data = [get_flux_density(SPIRE_250_flux_filename,X_pix_SPIRE_250,Y_pix_SPIRE_250,0,1.0),get_flux_density(SPIRE_350_flux_filename,X_pix_SPIRE_250,Y_pix_SPIRE_250,0,1.0),get_flux_density(SPIRE_500_flux_filename,X_pix_SPIRE_250,Y_pix_SPIRE_250,0,1.0)]
#Form the residuals
from numpy import stack,cov
subtracted_data = [zeros(pixel_window_size),zeros(pixel_window_size),zeros(pixel_window_size)]
subtracted_data =[subtracted_data]*N_cutouts
for j in range(N_cutouts):
    for x in range(pixel_window_size[0]):
        for y in range(pixel_window_size[1]):
            subtracted_data[j][0][x][y] = flux_data[0][x][y] - background[j][y][x]
            subtracted_data[j][1][x][y] = flux_data[1][x][y] - background[j][y][x]
            subtracted_data[j][2][x][y] = flux_data[2][x][y] - background[j][y][x]
C = []
for j in range(N_cutouts):
    data_250 = subtracted_data[j][0].flatten().tolist()
    data_350 = subtracted_data[j][1].flatten().tolist()
    data_500 = subtracted_data[j][2].flatten().tolist()
    stk_250_350 = stack((data_250,data_350),axis=0)
    stk_250_500 = stack((data_250,data_500),axis=0)
    stk_350_500 = stack((data_350,data_500),axis=0)
    C_j = zeros((3,3))
    cov_2_1 = cov(stk_250_350)
    cov_2_2 = cov(stk_250_500)
    cov_2_3 = cov(stk_350_500)
    C_j[0][0] = cov_2_1[0][0]
    C_j[1][1] = cov_2_1[1][1]
    C_j[2][2] = cov_2_2[1][1]
    C_j[0][1] = cov_2_1[0][1]
    C_j[1][0] = C_j[0][1]
    C_j[0][2] = cov_2_2[0][1]
    C_j[2][0] = C_j[0][2]
    C_j[1][2] = cov_2_3[0][1]
    C_j[2][1] = C_j[1][2]
    C.append(C_j)

cov_matrix = zeros((3,3))
for j in range(3):
    for k in range(3):  
        for p in range(N_cutouts):
            cov_matrix[j][k] += C[p][j][k]
        cov_matrix[j][k] /= float(N_cutouts)
        
print "Covariance Matrix: ", cov_matrix
    
#In updated units of (Jy/beam)(Jy/beam).

import numpy as np
C_matrix = [[6.87155240e-05, 4.88864679e-05, 3.66471105e-05],
            [4.88864679e-05, 5.31852295e-05, 4.54034054e-05],
            [3.66471105e-05, 4.54034054e-05, 5.91637415e-05]]
C_inv = np.linalg.inv(C_matrix)
print C_inv  
