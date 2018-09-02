# IROPUBC

# FullTask_Modified.py

This is the main script that performs virtually all data pre-processing, log-likelihood calculations and MCMC algorithm. This includes but is not limited to:

⁃	Modified SED functions
⁃	Telescope Bandpass Filter integrating functions
⁃	Image Reconstruction (convolution) functions
⁃	Functions to interrogate .fits files containing flux density and instrumental errors.
⁃	Calculate Log-Likelihoods for ALMA, Spitzer MIPS, Herschel SPIRE (3 wavebands), Herschel PACS, LABOCA maps.
⁃	Calculate priors on parameters as required.
⁃	Perform MCMC using ^log-likelihoods to find best-fit parameters.

The top half of the script is all functions that are used throughout the project later on.
The middle section is a bunch of data reading in and processing (i.e. reading .fits files, changing units, calculating quantities outside of MCMC loops to save computing time).
Final section is where user can manipulate MCMC parameters.

# Output.py

This is the script was used to look at the output of the ^script. There are a bunch of different plotting functions and functions to calculate Far-IR luminosty/ SFR rates. These are also all explained individually in the script.
The user just needs to change the variable ‘data_file’ to the name of the text file spat out by ‘FullTask_Modified.py’, and everything else should be fine.

 
# Other Stuff.

There are loads of other individual scripts. But in virtually all cases these can be ignored (they have all be put into the FullTask_Modified.py script to avoid loads of importing functions when running on remote machine).
There are a few that might be useful to know about:

⁃	FITS_information.py - script to get flux densities, errors etc. from fits files (imported by FullTask_Modified.py)
⁃	covariance_matrix.py - script to calculate the covariance matrix across the three SPIRE wavebands
⁃	16gal_overlay_plus_visualisation.py - this doesn’t really do anything required for the method, but produces some pictures of galaxy overlays and allows visualisation of fits files and smeared Gaussians etc.
⁃	SPIRE_pixel_resizing.py - script to match pixel sizes in the three SPIRE wavebands for correlation matrix procedure and log-likelihood comparison.
