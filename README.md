# Implied Heat Transport data processing code
This repository stores the code used in Pearce and Bodas-Salcedo, submitted to JClim.

There are two scripts in the src directory that have to be run in the following order:

   * iht_paper_results.py: calculates the energy flux potentials and meridional heat transports from the input CERES data.
   * iht_paper_figures.py: produces the figures.

The input CERES data can be obtained from the the NASA Langley Research Center CERES ordering tool at https://ceres.larc.nasa.gov/data/.
For convenience, here we provide the NetCDF header of the file used in this study.
