import os
import logging
import iris
import numpy as np
import cubeUtils as cu
from pathlib import Path
from poisson_solver import spherical_poisson

##### Modify to match your directory structure and location of CERES monthly data
dir_transports = "/work_dir/transports"
ceres_fluxes = "/ceres_data_dir/CERES_EBAF_Ed4.1_Subset_200003-202002.nc"
##### End modifications

# Mapping between CERES variable names and long names
var_mapping = {"toa_tot_net": {"ceres_varname":"toa_net_all_mon", "long_name":"toa_net_downward_radiative_flux",
                               "subtract":None, "change_sign":False},
             "toa_sw_net": {"ceres_varname":"solar_mon", "long_name":"toa_net_shortwave_radiative_flux",
                            "subtract": "toa_sw_all_mon", "change_sign":False},
             "toa_lw_all": {"ceres_varname":"toa_lw_all_mon", "long_name":"toa_outgoing_longwave_flux",
                            "subtract":None, "change_sign":False},
             "toa_lw_net": {"ceres_varname":"toa_lw_all_mon", "long_name":"toa_net_longwave_radiative_flux",
                            "subtract":None, "change_sign":True},
             "toa_tot_cre": {"ceres_varname":"toa_cre_net_mon", "long_name":"toa_cloud_radiative_effect",
                             "subtract":None, "change_sign":False},
             "toa_sw_cre": {"ceres_varname":"toa_cre_sw_mon", "long_name":"toa_shortwave_cloud_radiative_effect",
                            "subtract":None, "change_sign":False},
             "toa_lw_cre": {"ceres_varname":"toa_cre_lw_mon", "long_name":"toa_longwave_cloud_radiative_effect",
                            "subtract":None, "change_sign":False},
             "toa_swup_clear_minus": {"ceres_varname":"toa_sw_clr_t_mon", "long_name":"minus_toa_outgoing_shortwave_flux_assuming_clear_sky",
                                      "subtract":None, "change_sign":True},
             "toa_swup_minus": {"ceres_varname":"toa_sw_all_mon", "long_name":"minus_toa_outgoing_shortwave_flux",
                                "subtract":None, "change_sign":True},
             "toa_tot_net_clear": {"ceres_varname":"toa_net_clr_t_mon", "long_name":"toa_net_downward_radiative_flux_assuming_clear_sky",
                                   "subtract":None, "change_sign":None},
             "toa_sw_net_clear": {"ceres_varname": "solar_mon", "long_name": "toa_net_shortwave_radiative_flux_assuming_clear_sky",
                              "subtract": "toa_sw_clr_t_mon", "change_sign": False},
             "toa_lw_net_clear": {"ceres_varname": "toa_lw_clr_t_mon", "long_name": "toa_net_longwave_radiative_flux_assuming_clear_sky",
                              "subtract": None, "change_sign": True}}


# Initialise logger
logger = logging.getLogger(Path(__file__).stem)


def call_poisson(flux_cube, latitude='latitude', longitude='longitude'):

    if flux_cube.coord(latitude).bounds is None: flux_cube.coord(latitude).guess_bounds()
    if flux_cube.coord(longitude).bounds is None: flux_cube.coord(longitude).guess_bounds()

    # Remove average of flux field to account for storage term
    data = flux_cube.data.copy()
    grid_areas = iris.analysis.cartography.area_weights(flux_cube)
    data_mean = flux_cube.collapsed(["longitude", "latitude"], iris.analysis.MEAN, weights=grid_areas).data
    data -= data_mean

    logger.info("Calling spherical_poisson")
    poisson, mht = spherical_poisson(logger,
                                     forcing=data * (6371e3**2.0),
                                     tolerance=2.0e-4)
    logger.info("Ending spherical_poisson")

    # Energy flux potential (P)
    p_cube = flux_cube.copy()
    p_cube.var_name = "{}_efp".format(flux_cube.var_name)
    p_cube.long_name = "energy_flux_potential_of_{}".format(flux_cube.long_name)
    p_cube.units = 'J s-1'
    p_cube.data = poisson[1:-1, 1:-1]

    # MHT data cube
    mht_cube = flux_cube.copy()
    mht_cube = mht_cube.collapsed('longitude', iris.analysis.MEAN)
    mht_cube.var_name = "{}_mht".format(flux_cube.var_name)
    mht_cube.long_name = "meridional_heat_transport_of_{}".format(flux_cube.long_name)
    mht_cube.units = 'W'
    mht_cube.data = mht

    return p_cube, mht_cube


def ceres_flux_climatology(var_name, new_name, long_name, subtract=None):
    vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == var_name)
    flux_monthly = iris.load_cube(ceres_fluxes, vname_constraint)
    flux_cube = cu.annual_climatology(flux_monthly)
    if subtract is not None:
        vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == subtract)
        flux_monthly = iris.load_cube(ceres_fluxes, vname_constraint)
        flux_cube_subtract = cu.annual_climatology(flux_monthly)
        flux_cube.data -= flux_cube_subtract.data
    flux_cube.var_name = new_name
    flux_cube.long_name = long_name
    flux_cube.standard_name= None
    return flux_cube


def produce_efp_and_mht_climatology(var_name, new_name, long_name, subtract=None, change_sign=False):
    flux_cube = ceres_flux_climatology(var_name, new_name, long_name, subtract=subtract)
    if change_sign: flux_cube.data = -flux_cube.data
    p_cube, mht_cube = call_poisson(flux_cube)
    cubes = [flux_cube, p_cube, mht_cube]
    iris.save(cubes, os.path.join(dir_transports, "{}.nc".format(new_name)))


def produce_efp_and_mht_time_series(var_name, new_name, long_name, subtract=None, change_sign=False):
    # Read monthly data and calculate rolling annual time series
    vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == var_name)
    flux_monthly = iris.load_cube(ceres_fluxes, vname_constraint)
    flux_annual = flux_monthly.rolling_window('time', iris.analysis.MEAN, 12)
    if subtract is not None:
        vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == subtract)
        flux_monthly = iris.load_cube(ceres_fluxes, vname_constraint)
        flux_annual_subtract = flux_monthly.rolling_window('time', iris.analysis.MEAN, 12)
        flux_annual.data -= flux_annual_subtract.data
    flux_annual.var_name = new_name
    flux_annual.long_name = long_name
    flux_annual.standard_name= None
    if change_sign: flux_annual.data = -flux_annual.data
    # Iterate over time dimension and calculate EFP
    p_annual = iris.cube.CubeList()
    mht_annual = iris.cube.CubeList()
    for flux_cube in flux_annual.slices_over('time'):
        p_cube, mht_cube = call_poisson(flux_cube)
        p_annual.append(p_cube)
        mht_annual.append(mht_cube)
    #Merge cubes and save to file
    mht_annual = mht_annual.merge_cube()
    p_annual = p_annual.merge_cube()
    cubes = [flux_annual, p_annual, mht_annual]
    iris.save(cubes, os.path.join(dir_transports, "{}_annual_time_series.nc".format(new_name)))


def produce_climatologies(variables=["toa_tot_net","toa_sw_net","toa_lw_net",
                                     "toa_tot_cre","toa_sw_cre","toa_lw_cre",
                                     "toa_swup_clear_minus","toa_swup_minus","toa_tot_net_clear",
                                     "toa_sw_net_clear","toa_lw_net_clear"]):
    for v in variables:
        produce_efp_and_mht_climatology(var_mapping[v]["ceres_varname"], v, var_mapping[v]["long_name"],
                                        subtract=var_mapping[v]["subtract"],
                                        change_sign=var_mapping[v]["change_sign"])


def produce_annual_time_series(variables=["toa_tot_net", "toa_tot_net_clear", "toa_sw_net", "toa_sw_net_clear",
                                          "toa_lw_net", "toa_lw_net_clear"]):
    for v in variables:
        produce_efp_and_mht_time_series(var_mapping[v]["ceres_varname"], v, var_mapping[v]["long_name"],
                                        subtract=var_mapping[v]["subtract"],
                                        change_sign=var_mapping[v]["change_sign"])


def main():
    produce_climatologies()
    produce_annual_time_series()


if __name__ == "__main__":
    main()