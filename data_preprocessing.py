"""
Steps involved for preprocessing the Sentinel-1, ERS and Envisat scenes:
1. Set a naming convention: [SAT]_[YYYYMMDD]_[POLARISATION]_[EXTRA]
2. Resize all images and masks to the same size. Downscale Envisat and ERS to 40 m pixels (currently 30 m)
3. Images are greyscale, masks are RBG - convert masks to greyscale as only 2 classes
4. Patch the images and masks

"""

# import libraries
import os




# set the path to the data
"""Structure of data:

benchmark_data_CB
-- Sentinel-1
------ masks
------ scenes
-- ERS
------ masks
------ scenes
------ vectors
-- Envisat
------masks
------ scenes
------ vectors

"""

parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB"
S1_dir = os.path.join(parent_dir, "Sentinel-1")
ERS_dir = os.path.join(parent_dir, "ERS")
Envisat_dir = os.path.join(parent_dir, "Envisat")

