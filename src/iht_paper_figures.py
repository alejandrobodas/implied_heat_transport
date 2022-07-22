import numpy as np
import iris
import os
import matplotlib.pyplot as plt
import iris.coord_categorisation
from scipy.optimize import curve_fit
import cartopy.crs as ccrs
from matplotlib import gridspec
from matplotlib import rcParams

##### Modify to match your directory structure
dir_transports = "/work_dir/transports"
dir_plots = "/work_dir/plots"
##### End modifications

rcParams.update(
    {'font.size': 14, 'text.latex.preamble': [r"\usepackage{amsmath}"], 'xtick.major.pad': 10, 'ytick.major.pad': 10,
     'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.minor.size': 5, 'ytick.minor.size': 5, 'axes.linewidth': 2,
     'lines.markersize': 8, 'lines.linewidth': 2})


class implied_heat_transport:
    def __init__(self):
        self.var_name_clim = {'toa_net_downward_radiative_flux': 'toa_tot_net',
                         'toa_net_shortwave_radiative_flux': 'toa_sw_net',
                         'toa_net_longwave_radiative_flux': 'toa_lw_net',
                         'toa_cloud_radiative_effect': 'toa_tot_cre',
                         'toa_shortwave_cloud_radiative_effect': 'toa_sw_cre',
                         'toa_longwave_cloud_radiative_effect': 'toa_lw_cre',
                         'minus_toa_outgoing_shortwave_flux_assuming_clear_sky': 'toa_swup_clear_minus',
                         'minus_toa_outgoing_shortwave_flux': 'toa_swup_minus',
                         'toa_net_downward_radiative_flux_assuming_clear_sky': 'toa_tot_net_clear',
                         'toa_net_shortwave_radiative_flux_assuming_clear_sky': 'toa_sw_net_clear',
                         'toa_net_longwave_radiative_flux_assuming_clear_sky': 'toa_lw_net_clear'}
        self.var_name_year = {'toa_net_downward_radiative_flux': 'toa_tot_net',
                              'toa_net_downward_radiative_flux_assuming_clear_sky': 'toa_tot_net_clear',
                              'toa_net_shortwave_radiative_flux': 'toa_sw_net',
                              'toa_net_shortwave_radiative_flux_assuming_clear_sky': 'toa_sw_net_clear',
                              'toa_net_longwave_radiative_flux': 'toa_lw_net',
                              'toa_net_longwave_radiative_flux_assuming_clear_sky': 'toa_lw_net_clear'}

        # Load data
        self.load_ceres_data()

        # Grid
        v = "toa_tot_net"
        fname = os.path.join(dir_transports, v+".nc")
        vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == v)
        flux = iris.load_cube(fname, vname_constraint)
        self.add_bounds(flux)
        self.grid = iris.analysis.cartography.area_weights(flux, normalize=True)
        self.lat = flux.coord('latitude').points
        self.lon = flux.coord('longitude').points
        del flux
        return

    def add_bounds(self, cube):
        # Add bounds to lat and long coords
        lat_bounds = np.asarray([np.linspace(-90, 89, 180), np.linspace(-89, 90, 180)]).T
        lon_bounds = np.asarray([np.linspace(0, 359, 360), np.linspace(1, 360, 360)]).T

        cube.coord('latitude').bounds = lat_bounds
        cube.coord('longitude').bounds = lon_bounds
        return


    def load_ceres_data(self):
        # Climatologies
        self.clim_mht = {}
        self.clim_fld = {}
        self.flux_clim = {}
        for v in self.var_name_clim.values():
            fname = os.path.join(dir_transports, "{}.nc".format(v))
            #MHT
            vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == "{}_mht".format(v))
            self.clim_mht[v] = iris.load_cube(fname, vname_constraint).data
            #EFP
            vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == "{}_efp".format(v))
            self.clim_fld[v] = iris.load_cube(fname, vname_constraint).data
            #Flux
            vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == v)
            self.flux_clim[v] = iris.load_cube(fname, vname_constraint).data

        # Annual time series
        self.year_mht = {}
        for v in self.var_name_year.values():
            fname = os.path.join(dir_transports, "{}_annual_time_series.nc".format(v))
            #MHT
            vname_constraint = iris.Constraint(cube_func=lambda c: c.var_name == "{}_mht".format(v))
            self.year_mht[v] = iris.load_cube(fname, vname_constraint).data
        return

    ### Figures 2, 4, 5 ###
    def quiver_subplot(self, p_field, var_name, nsf, wmin, wmax, nwlevs, wlevstep, plt_name=None):
        x, y = np.meshgrid(self.lon, self.lat)

        vmin = 0
        vmax = 0

        for i, p in enumerate(p_field):
            tmp = np.average(p) # Arbitrary choice of origin

            fig = plt.figure()
            plt.axes(projection=ccrs.PlateCarree())
            plt.contourf(self.lon, self.lat, p - tmp, levels=10,
                         transform=ccrs.PlateCarree(central_longitude=0))
            plt.gca().coastlines()
            cbar = plt.colorbar()
            cmin, cmax = cbar.mappable.get_clim()
            vmin = np.min([vmin, cmin])
            vmax = np.max([vmax, cmax])
            plt.close()

            fig = plt.figure()
            plt.axes(projection=ccrs.PlateCarree())
            plt.contourf(self.lon, self.lat, nsf[i], levels=10,
                         transform=ccrs.PlateCarree(central_longitude=0))
            plt.gca().coastlines()
            cbar = plt.colorbar()
            plt.close()

        levels1 = np.linspace(vmin / 1e15, vmax / 1e15, 11)
        levels2 = np.linspace(wmin, wmax, nwlevs)

        len_p = len(p_field)
        if len_p == 3:
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(22, 2)
            gs.update(wspace=0.25, hspace=1.5)
        elif len_p == 2:
            fig = plt.figure(figsize=(10, 6.5))
            gs = gridspec.GridSpec(15, 2)
            gs.update(wspace=0.25, hspace=1.5)

        label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

        for i, p in enumerate(p_field):
            tmp = np.average(p)
            v, u = np.gradient(p, 1e14, 1e14)
            u = u[1:-1, 1:-1]
            v = v[1:-1, 1:-1]

            ax1 = plt.subplot(gs[i * 7:(i * 7) + 7, 0], projection=ccrs.PlateCarree())
            cb1 = ax1.contourf(self.lon, self.lat, (p - tmp) / 1e15, levels=levels1,
                               transform=ccrs.PlateCarree(central_longitude=0))
            plt.gca().coastlines()
            xq = x[10::20, 10::20]
            yq = y[10::20, 10::20]
            uq = u[10::20, 10::20]
            vq = v[10::20, 10::20]
            if i == 0:
                Q = ax1.quiver(xq, yq, uq, vq, pivot='mid', color='w', width=0.005)
                Q._init()
            else:
                ax1.quiver(xq, yq, uq, vq, pivot='mid', scale=Q.scale, color='w')
            ax1.set_xticks(np.arange(-180, 190, 60))
            ax1.set_xticklabels(['180', '120W', '60W', '0', '60E', '120E', '180'])
            ax1.set_yticks(np.arange(-90, 100, 30))
            ax1.set_yticklabels(['90S', '60S', '30S', 'Eq', '30N', '60N', '90N'])
            ax1.annotate(label[i], xy=(0.05, 1.05), xycoords=ax1.get_xaxis_transform(), color='k')
            del tmp, u, v, uq, vq

            tmp = np.average(nsf[i])
            ax2 = plt.subplot(gs[i * 7:(i * 7) + 7, 1], projection=ccrs.PlateCarree())
            cb2 = ax2.contourf(self.lon, self.lat, nsf[i] - tmp, levels=levels2,
                               transform=ccrs.PlateCarree(central_longitude=0),
                               cmap='RdBu_r')
            plt.gca().coastlines()
            ax2.set_xticks(np.arange(-180, 190, 60))
            ax2.set_xticklabels(['180', '120W', '60W', '0', '60E', '120E', '180'])
            ax2.set_yticks(np.arange(-90, 100, 30))
            ax2.set_yticklabels(['90S', '60S', '30S', 'Eq', '30N', '60N', '90N'])
            ax2.annotate(label[i + len_p], xy=(0.05, 1.05), xycoords=ax2.get_xaxis_transform(), color='k')
            del tmp

        ax1 = plt.subplot(gs[-1, 0])
        plt.colorbar(cb1, cax=ax1, orientation='horizontal', label='Energy flux potential (PJ s$^{-1}$)')

        ax2 = plt.subplot(gs[-1, 1])
        plt.colorbar(cb2, cax=ax2, orientation='horizontal', label=r'Flux (Wm$^{-2}$)',ticks=levels2[1::wlevstep])

        if len_p == 3:
            plt.subplots_adjust(left=0.1, right=0.94, top=1.0, bottom=0.11)
        elif len_p == 2:
            plt.subplots_adjust(left=0.11, right=0.9, top=1.0, bottom=0.13)

        if plt_name is not None:
            '{0}_p_field_flux_comparison.png'.format(plt_name)
            plt.savefig(os.path.join(dir_plots,plt_name))
            plt.close()
        else:
            plt.show()
        return

    ### Figure 1 ###
    def mht_plot(self, mht, var_name):
        fig = plt.figure()
        for i, x in enumerate(mht):
            plt.plot(self.lat, mht[i] / 1e15, label=var_name[i])
        plt.hlines(0, -90, 90, color='k', linestyles=':')
        plt.vlines(0, -10, 10, color='k', linestyles=':')
        plt.xlim(-90, 90)
        plt.ylim(-10, 10)
        plt.xticks(np.arange(-90, 120, 30))
        plt.xlabel('Latitude')
        plt.ylabel('MHT (PW)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dir_plots, 'toa_all_mht_comparison.png'))
        plt.close()
        return

    ### Figure 3 ###
    def cre_mht_plot(self, cre, cre_name, toa, toa_name):
        fig = plt.figure(figsize=(11, 5))
        ax1 = plt.subplot(121)
        for i, x in enumerate(cre):
            ax1.plot(self.lat, cre[i] / 1e15, label=cre_name[i])
        ax1.axhline(0, color='k', ls=':')
        ax1.axvline(0, color='k', ls=':')
        ax1.set_xlim(-90, 90)
        ax1.set_xticks(np.arange(-90, 120, 30))
        ax1.set_xlabel('Latitude')
        ax1.set_ylim(-1.3, 1.3)
        ax1.set_ylabel('MHT (PW)')
        plt.legend()

        ax2 = plt.subplot(122)
        col = ['C3', 'C7']
        for i, x in enumerate(toa):
            ax2.plot(self.lat, toa[i] / 1e15, label=toa_name[i], color=col[i])
        ax2.axhline(0, color='k', ls=':')
        ax2.axvline(0, color='k', ls=':')
        ax2.set_xlim(-90, 90)
        ax2.set_xticks(np.arange(-90, 120, 30))
        ax2.set_xlabel('Latitude')
        # ax2.set_ylim(-6.3,6.3)
        ax2.set_ylim(-1.4, 1.4)
        ax2.set_ylabel('MHT (PW)')
        # plt.legend(loc='upper left')
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(dir_plots, 'osr_cre_mht_comparison.png'))
        plt.close()
        return

    def y_intercept(self, data, lat):
        # Calculate cross equatorial heat-transport
        # Assumes 1x1 degree grid
        # It could be done by taking difference between hemispheric means,
        # but this approach can be easily generalised to other grids.
        data = data[89:91]
        lat = lat[89:91]

        def func(x, a, b):
            return a + b * x

        popt, pciv = curve_fit(func, lat, data)
        yint = popt[0]
        return yint

    def hemispheric_symmetry(self, data):
        # Calculates hemispheric symmetry value
        # S = 0 is perfectly symmetrical

        nh = data[90:]
        sh = data[:90]
        sh = -1 * sh[::-1]
        grid = np.sum(self.grid, axis=1)[90:]

        diff = np.abs((nh - sh) * grid)
        hem = np.sum(diff)
        trop = np.sum(diff[:30])
        mid = np.sum(diff[30:60])
        high = np.sum(diff[60:])
        return hem, trop, mid, high

    ### Figure 6 ###
    def ceht_time_series(self):
        lat = np.arange(-0.5 * np.pi + 0.5 * np.pi / 180., 0.5 * np.pi, np.pi / 180.)

        fig = plt.figure(figsize=(12, 9))
        xvar = np.arange(0, 229, 1)
        xticks = np.arange(10, 229, 36)
        xlabels = ['2001','2004','2007','2010','2013','2016','2019']

        var_list = [['toa_tot_net', 'toa_tot_net_clear'], ['toa_sw_net', 'toa_sw_net_clear'],
                    ['toa_lw_net', 'toa_lw_net_clear']]
        var_name = [['NET (all)', 'NET (clear)'], ['SW (all)', 'SW (clear)'], ['LW (all)', 'LW (clear)']]
        for j, x in enumerate(var_list):
            yint = np.zeros([2, 229])
            yint_clim = np.zeros([2])
            ax = plt.subplot(2, 2, j + 1)
            col = ['C0', 'C1']
            for k, y in enumerate(x):
                data = self.year_mht[y]
                clim = self.clim_mht[y] / 1e15  #
                yint_clim[k] = self.y_intercept(clim, lat)
                for i, z in enumerate(data):
                    tmp = z / 1e15
                    yint[k, i] = self.y_intercept(tmp, lat)
                del data, clim
                ax.plot(xvar, yint[k], lw=4, linestyle='-', label=var_name[j][k])
                ax.axhline(yint_clim[k], 0, 229, linestyle='--' if k == 0 else ':', color='k', zorder=10)

            ax.set_xlim(0, 229)
            ax.set_xticks(ticks=xticks)
            ax.set_xticklabels(labels=xlabels)
            ymin, ymax = ax.get_ylim()
            ax.fill_between(np.arange(118, 131, 1), ymin, ymax, color='k', alpha=0.25)
            ax.fill_between(np.arange(178, 191, 1), ymin, ymax, color='k', alpha=0.25)
            ax.set_ylim(ymin, ymax)
            if j > 1:
                ax.set_xlabel('Year')
            if j in [0, 2]:
                ax.set_ylabel('Cross-equatorial transport (PW)')
            if j != 2:
                plt.legend(loc=6)
            else:
                plt.legend(loc=2)
            print(np.std(yint, axis=1))
            del yint, yint_clim

        plt.tight_layout()
        plt.savefig(os.path.join(dir_plots,'cross_equatorial_heat_transport.png'))
        plt.close()
        return

    ### Figure 7 ###
    def symmetry_time_series(self, var=None):
        # Produces extra plots to give table values
        color = ['C0', 'C1']
        var_list = [['toa_tot_net', 'toa_tot_net_clear']]
        xvar = np.arange(0, 229, 1)
        xticks = np.arange(10, 229, 36)
        xlabels = ['2001','2004','2007','2010','2013','2016','2019']

        for j, x in enumerate(var_list):
            trop = np.zeros([2, 229])
            mid = np.zeros([2, 229])
            high = np.zeros([2, 229])
            hem = np.zeros([2, 229])
            sym = np.zeros([2, 4])

            for k, y in enumerate(x):
                data = self.year_mht[y]
                clim = self.clim_mht[y] / 1e15
                sym[k] = self.hemispheric_symmetry(clim)
                for i, z in enumerate(data):
                    tmp = z / 1e15
                    hem[k, i], trop[k, i], mid[k, i], high[k, i] = self.hemispheric_symmetry(tmp)
                    del tmp
                del data

            sym_var = [hem, trop, mid, high]
            print(np.shape(np.asarray(sym_var)))
            print(np.mean(np.asarray(sym_var), axis=2))
            sym_std = np.std(np.asarray(sym_var), axis=2)

            label = ['Hemispheric symmetry', 'Eq-30 symmetry', '30-60 symmetry', '60-90 symmetry']
            col = ['C0', 'C1']
            fig = plt.figure(figsize=(12, 9))
            for h in range(0, 4):
                ax = plt.subplot(2, 2, h + 1)
                ax.plot(xvar, sym_var[h][0], lw=4, linestyle='-', label=x[0])
                ax.plot(xvar, sym_var[h][1], lw=4, linestyle='-', label=x[1])
                for l in range(-1, 1):
                    ax.axhline(sym[l, h], 0, 230, color='k', linestyle='--' if l == 0 else ':', zorder=10)
                    ax.annotate(r'$\sigma$: {0}'.format(np.round(sym_std[h][l], 3)),
                                (0.05, 0.4 - (0.1 * l)) if h < 3 else (0.05, 0.8 - (0.1 * l)), xycoords='axes fraction',
                                color=col[l])
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                ax.set_xlim(0, 229)
                ax.set_xticks(ticks=xticks)
                ax.set_xticklabels(labels=xlabels)
                if h in [0, 2]:
                    ax.set_ylabel(r'$\Sigma$(NH-SH)')
                if h in [2, 3]:
                    ax.set_xlabel('Year')
                ax.set_title(label[h])
                if h == 0:
                    plt.legend(loc=5)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_plots,'{0}_yearly_hemispheric_symmetry.png'.format(x[0])))
            plt.close()
        return


    """ Climatology plots: MHT/maps """
    def climatology_plots(self):

        """ Map plots """
        # Figure 2
        self.quiver_subplot([self.clim_fld['toa_tot_net'], self.clim_fld['toa_sw_net'], self.clim_fld['toa_lw_net']],
                            var_name=['(a) Net', '(b) SW', '(c) LW'],
                            nsf=[self.flux_clim['toa_tot_net'], self.flux_clim['toa_sw_net'],
                                 self.flux_clim['toa_lw_net']], wmin=-180, wmax=180, nwlevs=19, wlevstep=4,
                            plt_name='toa_all')  #
        # Figure 4
        self.quiver_subplot([self.clim_fld['toa_tot_cre'], self.clim_fld['toa_sw_cre'], self.clim_fld['toa_lw_cre']],
                            var_name=['(a) Net CRE', '(b) SW CRE', '(c) LW CRE'],
                            nsf=[self.flux_clim['toa_tot_cre'], self.flux_clim['toa_sw_cre'],
                            self.flux_clim['toa_lw_cre']], wmin=-60, wmax=60, nwlevs=13, wlevstep=2,
                            plt_name='toa_cre')
        # Figure 6
        self.quiver_subplot([self.clim_fld['toa_swup_clear_minus'], self.clim_fld['toa_swup_minus']],
                            var_name=['(a) OSR (clear)', '(b) OSR (all)'],
                            nsf=[self.flux_clim['toa_swup_clear_minus'], self.flux_clim['toa_swup_minus']],
                            wmin = -100, wmax = 100, nwlevs = 21, wlevstep = 3,
                            plt_name='toa_osr')

        return


    def mht_plots(self):
        # Figure 1
        self.mht_plot([self.clim_mht['toa_tot_net'], self.clim_mht['toa_sw_net'], self.clim_mht['toa_lw_net']],
                      var_name = ['Net', 'SW', 'LW'])
        # Figure 3
        self.cre_mht_plot([self.clim_mht['toa_tot_cre'], self.clim_mht['toa_sw_cre'], self.clim_mht['toa_lw_cre']],
                          ['Net CRE', 'SW CRE','LW CRE'],
                          [self.clim_mht['toa_swup_minus'], self.clim_mht['toa_swup_clear_minus']],
                          ['-1 x OSR (all-sky)','-1 x OSR (clear-sky)'])
        return


def main():
    iht = implied_heat_transport()
    iht.climatology_plots()
    iht.mht_plots()
    iht.ceht_time_series()
    iht.symmetry_time_series()


if __name__ == "__main__":
    main()