#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:46:30 2018

@author: DewiGould
"""


"""
ONLY needs to be run once (I've already done it) - generates new .fits files with reprojected headers.

"""

import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

file_250 = "NE-CDFS-SWIRE_nested-image_SMAP250_DR2.fits"
file_350 = "NE-CDFS-SWIRE_nested-image_SMAP350_DR2.fits"
file_500 = "NE-CDFS-SWIRE_nested-image_SMAP500_DR2.fits"

#For SPIRE reprojection
#hdu1 = fits.open(get_pkg_data_filename(file_250))[1]
#hdu2 = fits.open(get_pkg_data_filename(file_350))[1]
#hdu3 = fits.open(get_pkg_data_filename(file_500))[1]



from reproject import reproject_interp

#For SPIRE reprojection
#array_1, footprint_1 = reproject_interp(hdu1, hdu3.header)
#array_2, footprint_2 = reproject_interp(hdu2, hdu3.header)
#array_3 ,footprint_3 = reproject_interp(hdu3,hdu3.header)

#For SPIRE reprojection
#fits.writeto("SPIRE_250_reprojected.fits",array_1,hdu3.header, clobber = True)
#fits.writeto("SPIRE_350_reprojected.fits",array_2,hdu3.header, clobber = True)
#fits.writeto("SPIRE_500_reprojected.fits",array_3,hdu3.header,clobber=True)


PACS_160_flux_filename = "gh_goodss_dr1_160_sci.fits"
LABOCA_flux_filename = "less_laboca_ecdfs_flux_v1.0.fits"
SPIRE_250_flux_filename = "SPIRE_250_reprojected.fits"
SPIRE_350_flux_filename = "SPIRE_350_reprojected.fits"
SPIRE_500_flux_filename = "SPIRE_500_reprojected.fits"
MIPS_24_flux_filename = "s_mips_1_s1_v0.30_sci.fits"


def do_thing(name1,name2,FILENAME):
    hdu_file = fits.open(get_pkg_data_filename(name1))[0]
    hdu_file2 = fits.open(get_pkg_data_filename(name2))[0]
    array_file,footprint_file = reproject_interp(hdu_file,hdu_file2.header)
    fits.writeto(FILENAME,array_file,hdu_file2.header,clobber=True)


do_thing(MIPS_24_flux_filename,PACS_160_flux_filename,"mips_into_pacs.fits")
do_thing(MIPS_24_flux_filename,LABOCA_flux_filename,"mips_into_laboca.fits")
do_thing(MIPS_24_flux_filename,SPIRE_250_flux_filename,"mips_into_sp250.fits")
do_thing(MIPS_24_flux_filename,SPIRE_350_flux_filename,"mips_into_sp350.fits")
do_thing(MIPS_24_flux_filename,SPIRE_500_flux_filename,"mips_into_sp500.fits")



#etc.

"""
Consider also including PACS 160, if the improvement is substantial??
ALSO: what am I doing about ALMA - no subtraction because just comparing flux values at specific points.

Haven't run any of this yet....try tomorrow
NOT sure if this is doing the correct thing, or if its just changing pixel SIZE.. but we'll see...
(by printing out sizes of data arrays etc..)
"""
