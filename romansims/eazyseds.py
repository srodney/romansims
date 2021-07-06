from astropy.io import fits
from astropy.table import Table, Column
from astropy import units
from astropy import constants


import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

import sncosmo
from scipy import interpolate

__EAZY_FLAMBDA_UNIT__ = units.erg / units.second / (units.cm*units.cm) / units.Angstrom
__EAZY_WAVE_UNIT__ = units.Angstrom
#__FNU_UNIT__ = units.erg / units.second / (units.cm*units.cm) / units.Hz

__EAZYPY_DATADIR__ =  os.path.abspath('../data/eazypy/')
__EAZY_TEMPLATE_FILENAME__ =  os.path.join(
__EAZYPY_DATADIR__, 'eazy_13_spectral_templates.dat')
__NEAZYTEMPLATES__ = 13

__FILTER_DATADIR__ = os.path.abspath('../data/roman_filters')


def register_roman_filters(filterdir=__FILTER_DATADIR__):
    """Read in the Roman filter transmission curves and register them into
    sncosmo.
    """
    # read in the bandpasses and add them to the sncosmo registry
    filterfilelist = glob(os.path.join(filterdir,"*.dat"))
    for filterfile in filterfilelist:
        bandname = os.path.splitext(os.path.basename(filterfile))[0]
        filterdata = np.loadtxt(filterfile)
        wave = filterdata[:,0]
        trans = filterdata[:,1]
        try :
            band = sncosmo.get_bandpass(bandname)
        except:
            sncosmo.registry.register(sncosmo.Bandpass(wave,trans,name=bandname))
    return

class EazyData(object):
    """ EAZY data from gabe brammer """
    # TODO : this should probably just inherit from a fits BinTableHDU class
    def __init__(self, fitsfilename):
        hdulist = fits.open(fitsfilename)
        self.namelist = [hdu.name for hdu in hdulist]
        for name in self.namelist:
            self.__dict__[name] = hdulist[name].data
        hdulist.close()

class EazySpecSim(Table):
    """ a class for running G.Brammer's EAZY spec simulator code to recreate
    simulated SEDs from a set of 'specbasis' coefficients.
    """
    def __init__(self, eazytemplatefilename=__EAZY_TEMPLATE_FILENAME__,
                 verbose=False, **kwargs):
        """Read in the galaxy SED templates (basis functions for the
        eazypy SED fitting / simulation) and store as the 'eazytemplatedata'
        property.

        We read in an astropy Table object with N rows and M+1 columns, where
        N is the number of wavelength steps and M is the
        number of templates (we expect 13).
        The first column is the  wavelength array, common to all templates.

        We translate the Nx(M+1) Table data into a np structured array,
        then reshape as a (M+1)xN numpy ndarray, with the first row giving
        the wavelength array and each subsequent row giving a single
        template flux array.
        See the function simulate_eazy_sed_from_coeffs() to construct
        a simulated galaxy SED with a linear combination from this matrix.
        """
        eazytemplates = Table.read(eazytemplatefilename,
                                   format='ascii.commented_header',
                                   header_start=-1, data_start=0, **kwargs)
        tempdata = eazytemplates.as_array()
        self.eazytemplatedata = tempdata.view(np.float64).reshape(
            tempdata.shape + (-1,)).T
        if verbose:
            print("Loaded Eazypy template SEDs from {0}".format(
                eazytemplatefilename))
        return

    def simulatesed(self, eazycoeffs, z=0,
                    wavemin=2000, wavemax=25000,
                    savetofile='', **outfile_kwargs):
        """
        Generate a simulated SED from a given set of input eazy-py coefficients
        and a specified redshift (defaults to rest-frame with z=0).

        NB: Requires the eazy-py package to apply the IGM absorption!
        (https://github.com/gbrammer/eazy-py)

        Optional Args:
        wavemin: [Angstroms] limit the output to wavelengths >wavemin.
                  set to 0 for no limit.
        wavemin: [Angstroms] limit the output to wavelengths <wavemax
                  set to 0 for no limit.
        savetofile: filename for saving the output spectrum as a two-column
            ascii data file

        Returns
        -------
            obswave : observed-frame wavelength, Angstroms or  nm
            obsflux : flux density of best-fit template, erg/s/cm2/A or AB mag
        """
        # the input data units are Angstroms for wavelength
        # and cgs for flux: erg/cm2/s/Ang
        obswave = self.eazytemplatedata[0] * (1 + z)
        obsfluxmatrix = self.eazytemplatedata[1:]
        sedsimflux = np.dot(eazycoeffs, obsfluxmatrix)
        fnu_factor = 10 ** (-0.4 * (25 + 48.6))
        flam_spec = 1. / (1 + z) ** 2
        obsflux = sedsimflux * fnu_factor * flam_spec

        try:
            import eazy.igm
            igmz = eazy.igm.Inoue14().full_IGM(z, obswave)
            obsflux *= igmz
        except:
            pass

        if wavemin or wavemax:
            ilimit = np.where((obswave > wavemin) & (obswave < wavemax))[0]
            obswave = obswave[ilimit]
            obsflux = obsflux[ilimit]

        if savetofile:
            out_table = Table()
            outcol1 = Column(data=obswave, name='wave')
            outcol2 = Column(data=obsflux, name='flux')
            out_table.add_columns([outcol1, outcol2])
            out_table.write(savetofile, **outfile_kwargs)

        return SimulatedSED(obswave, obsflux,
                            waveunit=__EAZY_WAVE_UNIT__,
                            fluxunit=__EAZY_FLAMBDA_UNIT__)



class SimulatedSED(object):
    def __init__(self, wave, flux,
                 waveunit=__EAZY_WAVE_UNIT__,
                 fluxunit=__EAZY_FLAMBDA_UNIT__):
        self.wave = wave
        self.flux = flux
        self.waveunit = waveunit
        self.fluxunit = fluxunit
        return

    def plot_sed(self, plotwaveunit=__EAZY_WAVE_UNIT__,
                 showmags = True, showbands = True,
                 *args, **kwargs):
        """ Plot the simulated SED
        TODO : adapt to different flux density or AB mag units
        (need a conversion function to get from fnu to flambda to AB mag)
        """
        fig = plt.gcf()
        ax1 = fig.gca()
        wave = self.wave * self.waveunit
        if plotwaveunit != self.waveunit:
            wave = wave.to(plotwaveunit)
        flux = self.flux * self.fluxunit

        ax1.plot(wave, flux, *args, **kwargs)

        bandnamelist = ['r062', 'z087', 'y106', 'j129', 'h158', 'w146', 'f184']
        colorlist = ['m', 'b', 'g', 'darkorange', 'r', '0.8', '0.3' ]
        if showmags:
            magdict = self.integrate_bandpasses()
            for bandname,color in zip(bandnamelist, colorlist):
                w, f = magdict[bandname]
                wave = (w * self.waveunit).to(plotwaveunit)
                fluxdensity = (f * self.fluxunit)
                ax1.plot(wave, fluxdensity,
                         marker='d', mfc=color, mec='k', ms=8, ls=' ')
        if showbands:
            ax2 = ax1.twinx()
            band = sncosmo.get_bandpass(bandname)
            bandwave_sedunits = (band.wave * units.Angstrom).to(self.waveunit)
            ax2.plot(bandwave_sedunits, band.trans, color=color, ls='-', marker=' ')
            ax2.set_ylim(0,2.4)
            ax2.yaxis.set_ticks([])

        ax1.set_xlabel(f'observed wavelength [{plotwaveunit}]')
        ax1.set_ylabel(f'flux density [{self.fluxunit}]')
        plt.tight_layout()
        return


    def integrate_bandpasses(self):
        """Integrate the SED across the given bandpass.
        Returns a dictionary of tuples, with each entry keyed by the bandpass
        name, and giving:
           (effective wavelength, flux density)
        in the units defined by this SimulatedSED object's waveunit and
        fluxunit attributes.
        """
        register_roman_filters()
        sedinterp = interpolate.interp1d(self.wave, self.flux,
                                         bounds_error=False,
                                         fill_value='extrapolate')
        bandnamelist = ['r062', 'z087', 'y106', 'j129', 'h158', 'w146', 'f184']
        returndict = {}
        for bandname in bandnamelist:
            band = sncosmo.get_bandpass(bandname)  # sncosmo works in Angstroms
            bandwave_sedunits = (band.wave * units.Angstrom).to(self.waveunit).value
            flux_through_bandpass = band.trans * sedinterp(bandwave_sedunits)
            # TODO : this is not robust against user changes to flux units.
            # variables carrying units here are appended with _u
            bandwave_u = band.wave_eff * units.Angstrom
            flambda_u = (np.sum(flux_through_bandpass) / np.sum(band.trans) )* self.fluxunit
            fnu =  (bandwave_u**2) * flambda_u / constants.c
            mAB_u = (-2.5*np.log10(fnu/units.Jansky) + 8.90) * units.mag
            returndict[bandname] = [bandwave_u, mAB_u]
        return returndict



