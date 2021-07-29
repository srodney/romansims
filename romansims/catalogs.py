import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as scinterp
from datetime import datetime

from astropy.io import fits
from astropy.table import Table, Column, Row, MaskedColumn
from astropy import table
from astropy import units as u

import seaborn as sns

import eazyseds

__VERBOSE__ = True
__3DHST_DATADIR__ =  os.path.abspath('../data/3DHST/')
__3DHST_MASTERCAT__ =  '3dhst.v4.1.5.master.fits'
__3DHST_PHOTCAT__ =  '3dhst_master.phot.v4.1.cat.fits'
__3DHST_GALFIT_ROOTNAME__ = '_3dhst.v4.1_f160w.galfit'
__3DHST_MASSCORRECTIONS__ = 'whitaker2014_table5_mass_corrections.txt'
__EAZYPY_DATADIR__ =  os.path.abspath('../data/eazypy')

_LOGSSFR_MIN=-14
_LOGSSFR_MAX=-2
_LOGMASS_MIN=0.01
_LOGMASS_MAX=15
_LOGSFR_MIN = -6
_LOGSFR_MAX = 4


class GalaxyCatalog(object):
    """ A generic catalog of galaxies, constructed from any one of the
    many CANDELS/CLASH catalogs.

    Necessary properties:
    ra, dec, field, catalogfile, idx_catalog

    Optional:
    z, zerr, photometry, mass, luminosity, restframe_photometry

    """
    def find_nearest_galaxy(self, location, tolerance_arcsec=0):
        """ Find the galaxy in this catalog that is closest to the given
        position.  Returns a Table object with a single row, giving all
        available information for the galaxy with the smallest angular
        separation from the specified location.

        :param location: a SkyCoord coordinate location to search near
        :param tolerance_arcsec: return None if there are no matches within
           this radius, given in arcseconds
        :return: Table object with one row.
        """
        idx_match, angsep_match, distance_skymatch = \
            location.match_to_catalog_sky(self.locations)
        if isinstance(idx_match, np.ndarray):
            idx_match = int(idx_match)

        if ((tolerance_arcsec > 0) and
                (angsep_match.to(u.arcsec) > tolerance_arcsec * u.arcsec)):
            return None

        match_row = self.catalog_data[idx_match]
        output_table = Table(rows=match_row)
        return(output_table)



class Catalog3DHST(GalaxyCatalog):
    """ Galaxy photometry, redshifts and derived parameters
    from 3DHST for all CANDELS fields.

    User can initialize this with a fully-formed catalog by providing
    'load_simulation_catalog'.   When the user provides this input filename,
    we assume it is a completely ready catalog: photcat parameters and eazypy
    coefficients appended, masses corrected, subset selected, etc.

    When this is not provided, the 'raw' 3DHST catalog is loaded from the user-
    specified 'mastercatfile'.   The subsequent adjustments can then be
    executed by calling the methods one by one, or all together by using
    the method prepare_sn_simulation_catalog().
    """

    def __init__(self, load_simulation_catalog=None,
                 datadir = __3DHST_DATADIR__,
                 mastercatfile = __3DHST_MASTERCAT__,
                 verbose=__VERBOSE__):
        self.verbose = verbose
        self.EazySpecSimulator = None
        self.simulation_catalog = None
        self.ids_adjusted = False
        self.mass_corrected = False
        self.subset_selected = False
        self.eazy_coefficients_appended = False
        self.photcat_params_appended = False
        self.galfit_params_appended = False

        if load_simulation_catalog is not None:
            # when the user provides an input filename, we assume it is
            # a completely ready catalog: masses corrected, subset selected,
            # and eazypy coefficients appended.  TODO: should check!
            self.mastercat = Table.read(load_simulation_catalog)
            self.simulation_catalog = self.mastercat
            self.mass_corrected = True
            self.subset_selected = True
            self.eazy_coefficients_appended = True
            self.photcat_params_appended = False
            return

        self.mastercat = Table.read(os.path.join(datadir, mastercatfile))

        # adjust the mastercat 'id' columns to dtype='int'
        idcol = Column(name='id', data=self.mastercat['phot_id'], dtype=int)
        self.mastercat.remove_columns(['phot_id'])
        self.mastercat.add_column(idcol, index=0)


    def prepare_simulation_catalog(self):
        """ This is the primary function for creating a simulation catalog
        (or HOSTLIB).  It runs through all the steps, resulting in the
        production of a 'simulation_catalog' attribute as an astropy Table
        object.  To write it out to a fits file or a SNANA HOSTLIB, use the
        modules `write_simulation_catalog_as_fits()' or
        'write_simulation_catalog_as_hostlib()'

        WARNING: this code is not optimized or parallelized!
        The step for creating photometric
        data in Roman bandpasses is particularly slow (the module for that is
        'append_eazy_magnitudes()').   It can take >30 minutes on my mac pro
         (3.5GHz 6-core Intel Xeon).

        """

        # Append columns next: have to do these before adjusting the galaxy IDs
        self.append_galfit_params()

        # Select the 'clean' subset of galaxies first (reduce computation time)
        self.select_clean_galaxies()

        self.append_photcat_params()
        self.append_eazy_coefficients()

        # Add a column with unique galaxy IDs
        self.append_unique_sim_ids()

        # SLOW: Append the Roman magnitudes derived from EAZY SED sims
        self.append_eazy_magnitudes()

        self.apply_mass_corrections()
        self.trim_simulation_catalog()
        return

    @property
    def has_unique_sim_ids(self):
        return 'id_sim' in self.mastercat.colnames

    def append_unique_sim_ids(self):
        """Make unique entries  in a new 'id_sim' column, by adding the
        field number x10^5 to each entry
        """
        if self.has_unique_sim_ids:
            print("IDs are already adjusted.")
            return
        idsimcol = Column(name='id_sim',
                          data=self.mastercat['id'] + \
                               (int(1e5) * self.mastercat['ifield'])
                          )
        self.mastercat.add_column(idsimcol)
        return


    def get_rownumber_from_id_sim(self, id_sim):
        """ Returns the row number for the given galid.
        NOTE: requires that the galids have been adjusted, so they are unique.
        """
        if not self.has_unique_sim_ids:
            print("You must adjust the galaxy ids before fetching row numbers")
            print("Run append_unique_sim_ids()")
            return
        irow = np.where(self.mastercat['id_sim']==id_sim)[0][0]
        return(irow)



    def select_clean_galaxies(self, verbose=__VERBOSE__):
        """Select the subset of galaxies that have measurements for
        redshift, mass, star formation rate and no flags of concern
        """
        if not self.galfit_params_appended:
            print("Missing required galfit params. Run append_galfit_params() then try again.")
            return
        if self.subset_selected:
            print("Subset selections already applied. Doing nothing.")
            return
        if verbose:
            print("Applying subset selection...")

        igotz = self.mastercat['z_best']>0
        igotmass = self.mastercat['lmass']>0
        igotsfr = self.mastercat['sfr']>-90
        ilowsfr = self.mastercat['sfr']<10000
        igotgalfit = self.mastercat['n0_sersic']>0
        igood = igotz & igotmass & igotsfr & ilowsfr & igotgalfit
        self.mastercat = self.mastercat[igood]

        if verbose:
            print(f"Selected {len(self.mastercat)} clean galaxies.")
        self.subset_selected = True
        return

    def apply_mass_corrections(self,datadir=__3DHST_DATADIR__,
                               masscorrectionfile=__3DHST_MASSCORRECTIONS__,
                               interpolate=False, verbose=True):
        """ Apply the emission-line corrections to the estimated
        stellar masses, from Whitaker et al. 2014,
        Appendix A, Table 5, Figure 14.

        Options:
        -------
        interpolate: if True, use 2D interpolation over M,z space (slow).
           If False, use the mean value in each M,z bin for all galaxies in
           that bin (fast).
        """
        if self.mass_corrected:
            print("Mass corrections already applied. Doing nothing.")
            return
        if not self.subset_selected:
            print("Subset of 'clean' galaxies not yet selected. Doing that now...")
            self.select_clean_galaxies()

        if verbose:
            print(f"Applying mass corrections to {len(self.mastercat):d} "
                  "galaxies in the mastercat...")

        # read in the Table5 data
        masscorrtab = Table.read(os.path.join(datadir, masscorrectionfile),
                                 format="ascii.commented_header",
                                 header_start=-1, data_start=0)

        if interpolate:
            # create a 2D interpolation function in (M,z) space
            M = np.mean( np.array([masscorrtab['logMmin'], masscorrtab['logMmax']]), axis=0)
            z = np.array([0.75,1.25, 1.75, 2.25])
            deltaM = np.array([masscorrtab['z<1'], masscorrtab['1<z<1.5'],
                               masscorrtab['1.5<z<2'], masscorrtab['2<z']])
            masscorrectionfunc = interpolate.interp2d(M, z, deltaM)
        else:
            # make a function that looks up the mass correction value from the
            # table for an array of given M, z values
            Mbinmax = np.append(masscorrtab['logMmax'][:-1], [99])
            def masscorrectionfunc(Mgal, zgal):
                iMbin = np.argmax(Mgal < Mbinmax)
                if zgal<1:
                    deltam = masscorrtab['z<1'][iMbin]
                elif zgal<1.5:
                    deltam = masscorrtab['1<z<1.5'][iMbin]
                elif zgal<2:
                    deltam = masscorrtab['1.5<z<2'][iMbin]
                else:
                    deltam = masscorrtab['2<z'][iMbin]
                return(deltam)

        # apply the mass corrections directly to the self.mastercat table
        zbest = self.mastercat['z_best']
        Morig = self.mastercat['lmass']
        deltam = np.array([masscorrectionfunc(Morig[i], zbest[i])
                           for i in range(len(Morig))])
        lmass_corr_col = Column(data=Morig - deltam,
                                name='lmass_corrected')
        self.mastercat.add_column(lmass_corr_col)
        self.deltam = deltam
        self.mass_corrected = True
        if verbose:
            print(f"Corrected masses for all {len(self.mastercat)} galaxies.")
        return

    def append_photcat_params(self,
                              datadir = __3DHST_DATADIR__,
                              photcatfile = __3DHST_PHOTCAT__,
                              verbose=__VERBOSE__):
        """Read in the photcatfile and extract columns that are not
        present in the master catalog:  kron_radius, a_image, b_image.
        TODO:  Speed it up
        """
        if self.has_unique_sim_ids:
            print("You can not append photcat parameters after the IDs \n"
                  " have been adjusted. ")
            return
        if self.photcat_params_appended:
            print("Photcat parameters already appended. Doing nothing.")
            return
        if verbose:
            print(f"Appending photcat parameters (a,b,kron_radius)...")

        # Load the 3DHST photcat
        photcatfilename = os.path.join(datadir, photcatfile)
        photcat = Table.read(photcatfilename)

        # adjust the photcat 'id' and 'field' columns to match the mastercat
        idcol = Column(name='id', data=photcat['id'], dtype=int)
        photcat.remove_columns(['id'])
        photcat.add_column(idcol, index=0)
        for i in range(len(photcat)):
            photcat['field'][i] = str.lower(photcat['field'][i]).replace('-','')

        # Extract just the columns of interest from the photcat
        photcatsubset = photcat[['field','id','a_image','b_image','kron_radius']]

        # Join the two tables
        self.mastercat = table.join(self.mastercat, photcatsubset,
                                    keys=['field', 'id'])


        # Add a new column that approximates the FWHM of each galaxy, using
        # the kron_radius * semiminor axis.
        fwhmvals = 2 * (self.mastercat['a_image'] + self.mastercat['b_image'])
        # ALTERNATE FWHM APPROXIMATION :
        # fwhmvals = self.mastercat['kron_radius'] *  self.mastercat['b_image']
        fwhmcol = Column(name='fwhm', data=fwhmvals)
        self.mastercat.add_column(fwhmcol)

        self.photcat_params_appended = True
        if verbose:
            print(f"photcat parameters appended for each galaxy.")
            print(f"length of mastercat = {len(self.mastercat):d}")
        return

    def append_galfit_params(self,
                             datadir = __3DHST_DATADIR__,
                             galfitfile_rootname = __3DHST_GALFIT_ROOTNAME__,
                             verbose=__VERBOSE__):
        """Read in the galfit info and append columns that provide useful
        shape info:  re, n, q, and pa

        Default uses the 3DHST galfit catalogs produced by Arjen van der Wel:
         https://www2.mpia-hd.mpg.de/homes/vdwel/3dhstcandels.html

        Parameters:
        select_clean :: bool
           If True, remove any galaxy from the master catalog that does not
           have a galfit result (flag==3 and therefore missing n)
           TODO: allow for higher levels of cleaning, like remove flag==2?
        """
        if self.has_unique_sim_ids:
            print("You can not append galfit parameters after the IDs \n"
                  " have been adjusted. ")
            return
        if self.galfit_params_appended:
            print("Galfit parameters already appended. Doing nothing.")
            return
        if verbose:
            print(f"Appending galfit parameters (re,n,q,pa)...")

        # Load the 3DHST galfit catalogs, field by field
        galfitcat = None
        for field in ['goodss', 'goodsn', 'uds', 'aegis', 'cosmos']:
            galfitfilename = os.path.join(
                datadir, f'{field}' + galfitfile_rootname)
            fieldcat = Table.read(galfitfilename,
                                   format='ascii.commented_header')
            fieldcat.rename_column('NUMBER', 'id')
            fieldcat.rename_column('n', 'n0_sersic')
            fieldcat.rename_column('q', 'q_sersic')
            fieldcat.rename_column('re', 're_sersic')
            fieldcat.rename_column('pa', 'pa_sersic')
            fieldnamecol = Column(data=[field for i in range(len(fieldcat))],
                                  name='field')
            fieldcat.add_column(fieldnamecol)

            # Extract just the columns of interest from the galfitcat
            fieldsubset = fieldcat[['field','id','re_sersic','n0_sersic',
                                    'q_sersic','pa_sersic']]
            if galfitcat is None:
                galfitcat = fieldsubset
            else:
                galfitcat = table.vstack([galfitcat, fieldsubset])

        # Join the composite galfit cat to the mastercat
        self.mastercat = table.join(self.mastercat, galfitcat,
                                    keys=['field', 'id'])

        self.galfit_params_appended = True
        if verbose:
            print(f"galfit parameters appended for each galaxy.")
            print(f"length of mastercat = {len(self.mastercat):d}")
        return


    def append_eazy_coefficients(self, eazypydatadir=__EAZYPY_DATADIR__,
                                 verbose=__VERBOSE__):
        """Read in all the 3DHST EAZY specbasis coefficient data
         and append as new columns on the mastercat.  Cut out any galaxies
         that have zeroes for all coefficients.
        """
        # TODO : make a real check here
        if self.eazy_coefficients_appended:
            print("EAZY coefficients already appended. Doing nothing.")
            return
        if verbose:
            print("Appending EAZY specbasis coefficients...")

        specbasistablelist = []
        for fld, field in zip(
                ['gs','gn','cos','egs','uds'],
                ['goodss', 'goodsn', 'cosmos', 'aegis', 'uds']):

            field_for_filename = field.lower().replace('-', '')
            coeffdatfilename = os.path.join(
                eazypydatadir,
                f"{field_for_filename}_3dhst.v4.1.eazypy.data.fits")
            assert(os.path.exists(coeffdatfilename))
            hdu = fits.open(coeffdatfilename)
            coefftable = Table(hdu['COEFFS'].data)
            for icol in range(len(coefftable.columns)):
                coefftable.rename_column(f'col{icol}',f'COEFF_SPECBASIS{icol:02d}')
            idcol = Column(name='id', data=hdu['ID'].data)
            field = [field] * len(idcol)
            fld = [fld] * len(idcol)
            fieldcol = Column(name='field', data=field)
            fldcol = Column(name='fld', data=fld)

            coefftable.add_columns([fieldcol, fldcol, idcol], indexes=[0, 0, 0])
            specbasistablelist.append(coefftable)

        # Stack all tables together as a single composite catalog
        specbasiscat = table.vstack(specbasistablelist)

        # remove any that have only zeros for coefficients
        ivalid = np.ones(len(specbasiscat), dtype=bool)
        for irow in range(len(ivalid)):
            coeffsum = np.sum([specbasiscat[irow][i] for i in range(-13,0)])
            if coeffsum==0:
                ivalid[irow] = False
        specbasiscat = specbasiscat[ivalid]

        # Join the two tables
        ngalorig = len(self.mastercat)
        self.mastercat = table.join(self.mastercat, specbasiscat,
                                    keys=['field', 'id'])

        self.eazy_coefficients_appended = True
        if verbose:
            NEAZYPYCOEFFS = len([col for col in specbasiscat.columns
                                 if col.startswith('COEFF')])
            print(f"{NEAZYPYCOEFFS:d} EAZY specbasis coefficients "
                  f"appended for each galaxy.\n Mastercat reduced to "
                  f"{len(self.mastercat)} galaxies (was {ngalorig})")
        return


    def make_eazy_specsimulator(self):
        """Load the 13 EAZYpy templates, for use in simulating SEDs """
        self.EazySpecSimulator = eazyseds.EazySpecSim()
        return

    def fetch_eazy_coefficients(self, irow):
        """Returns an np.array with the 13 EAZYpy coefficients for the galaxy
        in the master catalog on the given row.
        """
        coeffcolnames = [f"COEFF_SPECBASIS{i:02d}"
                         for i in range(eazyseds.__NEAZYTEMPLATES__)]
        coeffs = np.array([self.mastercat[irow][colname]
                           for colname in coeffcolnames])
        z = self.mastercat['z'][irow]
        return coeffs, z

    def make_eazy_simulated_sed(self, irow,  returnfluxunit='AB',
                                returnwaveunit='nm'):
        """Generate a simulated SED (as an eazyseds.SimulatedSED object)
        from the set of 13 EAZYpy coefficients in the spec sim input catalog
        entry for the galaxy on the specified row.

        Parameters
        ----------
        irow: [int] row number in the mastercat for the galaxy of interest
        """
        coeffs, z = self.fetch_eazy_coefficients(irow)
        if self.EazySpecSimulator is None:
            self.make_eazy_specsimulator()
        simsed = self.EazySpecSimulator.simulatesed(
            coeffs, z=z,
            returnfluxunit=returnfluxunit, returnwaveunit=returnwaveunit)
        return simsed

    def append_eazy_magnitudes(self, rowlist=None):
        """ Determine AB mags in Roman bandpasses for every galaxy in the
        catalog, using the simulated SED from the EAZYpy fit.  Each simulated
        SED provides 7 magnitudes in the R,Z,Y,J,H,W,F bandpasses.  These
        are appended onto the master catalog as new columns, keyed by the
        bandpass names.

        Arguments:
        ----------
        rowlist : [list of ints]  row numbers to populate with magnitudes.
            If `None` then all rows are populated.
        """
        if rowlist is None:
            rowlist = range(len(self.mastercat))

        if 'h158' not in self.mastercat.colnames:
            bandnames = ['r062', 'z087', 'y106', 'j129',
                         'h158', 'w146', 'f184']
            for band in bandnames:
                bandcol = Column(name=band,
                                 data= -99 * np.ones(len(self.mastercat)),
                                 dtype=float)
                self.mastercat.add_column(bandcol)

        for irow in rowlist:
            simsed = self.make_eazy_simulated_sed(irow)
            magdict = simsed.integrate_bandpasses()
            for band, val in magdict.items():
                self.mastercat[band][irow] = val[1].value # magnitude; val[0] is wave
        return

    def trim_simulation_catalog(self):
        """Select just the columns of interest for input to Akari pixel-level
        simulations and/or SNANA catalog sims.
        """
        assert(self.subset_selected)
        assert(self.mass_corrected)
        assert(self.eazy_coefficients_appended)
        assert(self.has_unique_sim_ids)

        # extract the columns
        idsim = Column(name='id_sim', data=self.mastercat['id_sim'], dtype=int)
        z = Column(name='z', data=self.mastercat['z_best'], dtype=float)
        ra = Column(name='ra', data=self.mastercat['ra'], dtype=float)
        dec = Column(name='dec', data=self.mastercat['dec'], dtype=float)
        id3dhst = Column(name='id_3dhst', data=self.mastercat['id'], dtype=int)
        ifield = Column(name='ifield', data=self.mastercat['ifield'], dtype=int)
        field = Column(name='field', data=self.mastercat['field'], dtype=str)
        logsfr = Column(name='logsfr', data=np.log10(self.mastercat['sfr']), dtype=float)
        logmass = Column(name='logmass', data=self.mastercat['lmass_corrected'], dtype=float)
        logssfr = Column(name='logssfr', data=logsfr.data-logmass.data, dtype=float)
        av = Column(name='Av', data=self.mastercat['Av'], dtype=float)
        a = Column(name='a', data=self.mastercat['a_image'], dtype=float)
        b = Column(name='b', data=self.mastercat['b_image'], dtype=float)
        kronrad = Column(name='kron_radius', data=self.mastercat['kron_radius'], dtype=float)
        fwhm = Column(name='fwhm', data=self.mastercat['fwhm'], dtype=float)
        n0 = Column(name='n0_sersic', data=self.mastercat['n0_sersic'], dtype=float)
        re = Column(name='re_sersic', data=self.mastercat['re_sersic'], dtype=float)
        q = Column(name='q_sersic', data=self.mastercat['q_sersic'], dtype=float)
        pa = Column(name='pa_sersic', data=self.mastercat['pa_sersic'], dtype=float)

        # Make the Table
        columnlist = [idsim, z, ra, dec, id3dhst, ifield, field,
                      logsfr, logmass, logssfr, av, a, b, kronrad, fwhm,
                      n0, re, q, pa
                      ]

        for band, snanabandname in zip(
                ['r062', 'z087', 'y106', 'j129','h158', 'w146', 'f184'],
                ['R_obs', 'Z_obs', 'Y_obs', 'J_obs','H_obs', 'W_obs', 'F_obs']):
            newbandcol = Column(data=self.mastercat[band], name=snanabandname)
            columnlist.append(newbandcol)

        for icoeff in range(13):
            coeffcolname = f'COEFF_SPECBASIS{icoeff:02d}'
            columnlist.append(self.mastercat[coeffcolname])

        self.simulation_catalog = Table(data=columnlist)
        return

    def write_simulation_catalog_as_fits(self, fitstable_filename, **kwargs):
        """ Write out the simulation input catalog as a .fits table
        """
        #TODO : check for existence. Require clobbering. Add verbosity

        # Write out a FITS table
        self.simulation_catalog.write(fitstable_filename, format='fits', **kwargs)
        return

    def write_simulation_catalog_as_hostlib(self, hostlib_filename, **kwargs):
        """Write out the simulation input catalog as a SNANA HOSTLIB file,
        including a sSNR-based WGTMAP.
        """
        hostlib = SNANAHostLib(self.simulation_catalog)
        hostlib.write_hostlib(hostlib_filename, **kwargs)
        return



otherplots="""
ax2.hexbin(mastercat['z'], mastercat['logsfr'], extent=[0,6,-4,4], cmap='inferno', bins='log')
ax2.set_title('log(SFR) vs z')


ax3.hexbin(mastercat['logmass'], mastercat['logsfr'], extent=[4,12,-4,4], cmap='inferno', bins='log')
ax3.set_xlim(4, 12)
ax3.set_ylim(-4, 4)
ax3.set_xlabel('log(M) : stellar mass')
ax3.set_ylabel('log(SFR) : star formation rate')
ax3.set_title('log(SFR) vs log(M)')


plt.tight_layout()
"""



class SNANAHostLib():
    """Class for reading/writing a SNANA HOSTLIB file.
    The file may contain a weight map, and must contain host galaxy data,
    following the standard SNANA HOSTLIB format.
    """
    def __init__(self, inputdata=None):
        """inputdata can be a string, giving the filename of a fits data
        table or SNANA hostlib text file.  Or it can be an astropy Table
        object with the galaxy data needed to create a HOSTLIB."""
        if type(inputdata) is str:
            filename = inputdata
            if filename.endswith('.fits'):
                self.__init_from_fitstable__(filename)
            else:
                self.__init_from_hostlib__(filename)
        elif inputdata is not None:
            self.galdatatable = inputdata
        return

    def __init_from_fitstable__(self, filename):
        """Read in a fits table with a host galaxy catalog"""
        # TODO allow for a MEF with wgtmap and galdatatable in extensions
        self.galdatatable = table.Table.read(filename)
        return

    def __init_from_hostlib__(self, filename):
        """Read in a SNANA HOSTLIB file"""
        # find the 'VARNAMES' line, use it to define the start of the hostlib
        # section (as opposed to the wgtmap section)
        nwgtmapstart = -1
        ngaldataheader = -1
        ngaldatastart = -1
        ncommentlines = 0
        iline = 0

        # TODO : need to debug the handling of comment lines so the
        #  count for starting sections matches with astropy.table
        with open(filename, 'r') as read_obj:
            for line in read_obj:
                if len(line.strip().lstrip('#'))==0:
                    ncommentlines += 1
                    #continue
                if line.strip().startswith('NVAR_WGTMAP:'):
                    wgtmaphdrline = line.split()
                    varnames_wgtmap = wgtmaphdrline[3:]
                if line.strip().startswith('WGT:') and nwgtmapstart<0:
                    nwgtmapstart = iline - ncommentlines
                if line.strip().startswith('GAL:') and ngaldatastart<0:
                    ngaldatastart = iline - ncommentlines
                if line.strip().startswith('VARNAMES:'):
                    ngaldataheader = iline - ncommentlines
                iline += 1
        if ngaldataheader < 0:
            raise RuntimeError(r"{filename} is not an SNANA HOSTLIB file")

        if nwgtmapstart >= 0:
            self.wgtmaptable = table.Table.read(
                filename, format='ascii.basic',
                names=['label']+varnames_wgtmap+['wgt','snmagshift'],
                data_start=nwgtmapstart,#-1,
                data_end=ngaldataheader,#-2,
                comment='#'
                )
        else:
            self.wgtmaptable = None

        galdatatable = table.Table.read(
            filename, format='ascii.basic',
            header_start= ngaldataheader, # max(0,ngaldataheader-1),
            data_start=ngaldatastart, # max(ngaldataheader+1,ngaldatastart-1),
            comment='#'
            )
        galdatatable.remove_columns(['VARNAMES:'])
        self.galdatatable = galdatatable
        return

    def mk_wgtmap_block(self, snr_model='AH18PW',
                        logssfr_stepsize=0.5,
                        logmass_stepsize=0.5,
                        massstep_threshold=10, massstep_value=0):
        """
        Construct the HOSTLIB weight map: a block of text, with
        each line including N-2 observable host galaxy parameters, and the
        last two giving WGT and SNMAGSHIFT.

        WGT is the weight (relative probability of hosting a SN Ia)
        assigned for the preceding set of galaxy parameters.
        SNMAGSHIFT is the magnitude shift applied toa SN with the matching
        set of host galaxy parameters.

        SNANA will do interpolation between the host galaxy parameter
        values given in the WGTMAP to assign a WGT and SNMAGSHIFT to each
        simulated SN host galaxy in the HOSTLIB section, which follows
        below the WGTMAP.

        snr_model='AH18S' : the smooth logarithmic sSFR model (Andersen & Hjorth 2018)
                 ='AH18PW' : the piecewise sSFR model (Andersen & Hjorth 2018)
        """

        wgtmap_str = "\n\nNVAR_WGTMAP: 2 VARNAMES_WGTMAP: logssfr logmass WGT SNMAGSHIFT\n\n"
        logssfr_gridpoints = np.arange(_LOGSSFR_MIN,
                                       _LOGSSFR_MAX + logssfr_stepsize,
                                       logssfr_stepsize)
        logmass_gridpoints = np.arange(_LOGMASS_MIN,
                                       _LOGMASS_MAX + logmass_stepsize,
                                       logmass_stepsize)
        massstep_values = np.where(logmass_gridpoints<massstep_threshold,
                                   0.0, massstep_value)

        if snr_model=='AH18S':
            ssnrfunc = ssnr_ah18_smooth
        elif snr_model=='AH18PW':
            ssnrfunc = ssnr_ah18_piecewise
        else:
            raise RuntimeError(
                f"No sSNR model known with the name {snr_model}")

        snrmax = ssnrfunc(_LOGSSFR_MAX) * np.power(10., _LOGMASS_MAX)
        snrmin = ssnrfunc(_LOGSSFR_MAX) * np.power(10., _LOGMASS_MIN)

        for lssfr in logssfr_gridpoints:
            ssnr = ssnrfunc(lssfr)
            for imass in range(len(logmass_gridpoints)):
                logmass = logmass_gridpoints[imass]
                if logmass > _LOGMASS_MIN:
                    wgt = np.float32(ssnr * np.power(10., logmass) / snrmax)
                else:
                    wgt = np.float32(snrmin/snrmax)
                wgtmap_str += "WGT: {:8.3f} {:8.3f} {:10.7f} {:6.2f}\n".format(
                    lssfr, logmass_gridpoints[imass], wgt, massstep_values[imass]
                )

        wgtmap_str += '\n\n'
        return(wgtmap_str)

    # TODO : make HOSTLIB header creation functions into more useful
    #  properties using the setter  to convey user input ?
    @property
    def HOSTLIB_DOCSTRING(self):
        docstring = f"""
DOCUMENTATION:
    PURPOSE:    Galaxy library for Roman sims with SNANA
    INTENT:     Nominal
    USAGE_KEY:  HOSTLIB_FILE
    USAGE_CODE: snlc_sim.exe        
    NOTES:
    - {len(self.galdatatable)} entries
    - includes 13 eazy-spectral coefficients (COEFF_SPECBASIS_XX) to construct host spectrum
    - galaxy properties (logsfr, logmass, logssfr) are derived from real 3DHST galaxies
    - generated using the catalogs.py module from github.com/srodney/romansims
    VERSIONS:
    - DATE:  {datetime.today().strftime('%Y-%m-%d')}
    - AUTHORS: Roman SN SIT
DOCUMENTATION_END:
    
"""
        return docstring

    @property
    def HOSTLIB_COLUMN_HDRLINE(self):
        headerstring = " ".join(
            ["VARNAMES:", "GALID", "RA_GAL", "DEC_GAL", "ZTRUE", "ZERR",
             "logsfr", 'logmass', 'logssfr',
             'n0_sersic', 're_sersic', 'a', 'b', 'kron_radius', 'FWHM',
             'R_obs', 'Z_obs', 'Y_obs', 'J_obs', 'H_obs', 'W_obs', 'F_obs'] + \
            [f'COEFF_SPECBASIS{i:02d}' for i in range(13)] + ["\n"])
        return headerstring

    def write_wgtmap(self, filename, overwrite=False):
        """Write out a weightmap indicating the probability of any given
        galaxy hosting a Type Ia SN, based on the star formation rate and mass
        of the galaxy.   The output is a stand-alone text file in the
        SNANA WGTMAP format.
        """
        if not overwrite:
            if os.path.exists(filename):
                print(f"{filename} exists. Use overwrite=True to overwrite.")
                return

        fout = open(filename, mode='w')
        fout.write(self.HOSTLIB_DOCSTRING)
        wgtmap_block = self.mk_wgtmap_block()
        fout.write(wgtmap_block)
        return

    def write_hostlib(self, filename, include_weightmap=False,
                      default_zerr=0.001, overwrite=False):
        """Write out the catalog as a text file in the SNANA Hostlib format.

        Parameters
        ----------
        include_weightmap :: bool  - Indicates whether to include a text block
           at the top of the file that provides the map for weighting galaxies
           to define the probability of hosting a SNIa.  Recommended to keep
           this as False and make a separate stand-alone WGTMAP using the
           write_wgtmap() function.
        """
        if not overwrite:
            if os.path.exists(filename):
                print(f"{filename} exists. Use overwrite=True to overwrite.")
                return

        fout = open(filename, mode='w')
        fout.write(self.HOSTLIB_DOCSTRING)

        if include_weightmap:
            wgtmap_block = self.mk_wgtmap_block()
            fout.write(wgtmap_block)

        fout.write(self.HOSTLIB_COLUMN_HDRLINE)
        for galdataline in self.galdatatable:
            outline = f"GAL: {galdataline['id_sim']:11d} " + \
                      f"{galdataline['ra']:11.5f} {galdataline['dec']:11.5f}" + \
                      f"{galdataline['z']:8.4f} {default_zerr:6.4f}" +\
                      f"{galdataline['logsfr']:8.4f} "+ \
                      f"{galdataline['logmass']:8.4f} "+ \
                      f"{galdataline['logssfr']:8.4f} "+ \
                      f"{galdataline['n0_sersic']:6.2f} "+ \
                      f"{galdataline['re_sersic']:6.2f} "+ \
                      f"{galdataline['a']:6.2f} "+ \
                      f"{galdataline['b']:6.2f} "+ \
                      f"{galdataline['kron_radius']:6.2f} "+ \
                      f"{galdataline['fwhm']:6.2f} "+ \
                      f"{galdataline['R_obs']:6.2f} "+ \
                      f"{galdataline['Z_obs']:6.2f} "+ \
                      f"{galdataline['Y_obs']:6.2f} "+ \
                      f"{galdataline['J_obs']:6.2f} "+ \
                      f"{galdataline['H_obs']:6.2f} "+ \
                      f"{galdataline['W_obs']:6.2f} "+ \
                      f"{galdataline['F_obs']:6.2f} "

            for i in range(0,13,1):
                specbasiscoeff = galdataline[f'COEFF_SPECBASIS{i:02d}']
                if specbasiscoeff != 0:
                    outline += f" {galdataline[f'COEFF_SPECBASIS{i:02d}']:13.6e}"
                else:
                    outline += f" {int(galdataline[f'COEFF_SPECBASIS{i:02d}']):13d}"
            outline += "\n"
            fout.write(outline)
        fout.close()
        return

class CatalogBasedRedshiftSim(GalaxyCatalog):
    """Class for projecting redshift completeness from an input
    galaxy catalog.
    """

    def __init__(self):
        self.postsurvey = False
        self.galaxies = None
        self.snhosts = None
        return

    def read_galaxy_catalog(self, filename):
        """Read in a catalog of galaxy properties

        Parameters
        ----------
        filename : str
          full path to the file containing galaxy properties (e.g. Mass, SFR,
          magnitudes, etc.).  May be a SNANA HOSTLIB file, or any formtat that
          can be auto-parsed by astropy.table.Table.read()
        """
        # TODO: check if it is a hostlib without try/except
        try :
            self.galaxies = table.Table.read(filename)
        except:
            try:
                hostlib = SNANAHostLib(filename)
                self.galaxies = hostlib.galdatatable
            except:
                raise RuntimeError(
                    f"Can't read in {filename}. "
                    "It may not be a valid hostlib or astropy-readable table.")

        self.galaxies_df = self.galaxies.to_pandas()

        return


    def assign_snhost_prob(self, snr_model='AH18PW',
                           logmasscolname='logmass',
                           logsfrcolname='logsfr',
                           verbose=True):
        """Add a column to the 'galaxies' catalog that gives the relative
        probability for each galaxy hosting a SN in any given observer-frame
        year.  This is computed based on the predicted SN rate (number of SN
        explosions per observer-frame year) of each galaxy, adopting the
        specified SN rate model.

        Parameters
        ----------
        snr_model : str
           'A+B' : SNR = A*M + B*SFR   (Scannapieco & Bildsten 2005)
           'AH18S' : the smooth logarithmic sSFR model (Andersen & Hjorth 2018)
           'AH18PW' : the piecewise sSFR model (Andersen & Hjorth 2018)

        logmasscolname : str
           name of column in the galaxies Table containing the log10(Mass)

        logsfrcolname : str
           name of column in the galaxies Table containing the
           log10(StarFormationRate)

        verbose : bool
            Set to True to print messages.
        """
        if self.galaxies is None:
            print("No 'galaxies' catalog loaded. Use 'read_galaxy_catalog()'")

        if snr_model.lower()=='a+b':
            # Note: adopting the A and B values from Andersen & Hjorth 2018
            # but dividing by 1e-4 (so the SNR below actually counts the number
            # of SN explodiing per 10000 yrs)
            A = 4.66 * 1e-10
            B = 4.88
            snr = A * 10 ** self.galaxies[logmasscolname] + B * 10 ** self.galaxies[logsfrcolname]
            # divide by the total snr to get relative probabilities
            snr /= np.nanmax(snr)
            snrcolname = 'snr_A+B'
            snrcol = table.Column(data=snr, name='snr_A+B')
        elif snr_model.lower() == 'ah18s':
            logssfr = self.galaxies[logsfrcolname] - self.galaxies[logmasscolname]
            ssnr = ssnr_ah18_smooth(logssfr)
            snr = ssnr * 10 ** self.galaxies[logmasscolname]
            snr /= np.nanmax(snr)
            snrcolname = 'snr_AH18_smooth'
            snrcol = table.Column(data=snr, name=snrcolname)
        elif snr_model.lower() == 'ah18pw':
            logssfr = self.galaxies[logsfrcolname] - self.galaxies[logmasscolname]
            ssnr = ssnr_ah18_piecewise(logssfr)
            snr = ssnr * 10 ** self.galaxies[logmasscolname]
            snr /= np.nanmax(snr)
            snrcolname = 'snr_AH18_piecewise'
        else:
            raise RuntimeError(r"{snr_model} is not a know SN rate model.")

        snrcol = table.Column(data=snr, name=snrcolname)
        if snrcolname in self.galaxies.colnames:
            self.galaxies[snrcolname] = snr
        else:
            self.galaxies.add_column(snrcol)
        if verbose:
            print(f"Added/updated relative SN rate column using {snr_model} model")
        return


    def pick_host_galaxies(self, nsn, snrcolname='snr_AH18_piecewise',
                           replace=False, verbose=True):
        """Do a random draw to assign 'nsn' supernovae to galaxies in the
        galaxies catalog, based on the (pre-defined) relative SN rates.

        TODO: (Alternatively, read in a SNANA output file (.dump file maybe?)
        that has already run a survey simulation and picked host galaxies.)

        Parameters
        ----------
        replace
        nsn : int
          number of SN to assign to host galaxies

        snrcolname : str
           name of the column in the galaxies catalog that gives the relative
           SN rate (or 'weight') for each galaxy.  This may be created by the
           assign_snhost_prob() method.

        replace : bool
           Whether to sample with replacement.  If True, a galaxy may host
           more than one SN. If False, then assign no more than one SN to
           each galaxy (requires nsn<len(galaxies))
        """
        if ~replace and nsn > len(self.galaxies):
            raise RuntimeError(
                r'Picking hosts without replacement, but Nsn > len(galaxies)')

        # Pick SN host galaxies
        galindices = np.arange(len(self.galaxies))
        psnhost = self.galaxies[snrcolname]/np.sum(self.galaxies[snrcolname])
        snindices = np.random.choice(
            galindices, nsn, replace=replace, p=psnhost)

        # Add a boolean 'host' column to the galaxies catalog
        ishost = np.zeros(len(self.galaxies), dtype=bool)
        ishost[snindices] = True
        hostcol = table.Column(name='host', data=ishost)
        if 'host' in self.galaxies.colnames:
            self.galaxies['host'] = hostcol
        else:
            self.galaxies.add_column(hostcol, index=1)

        self.snhosts = self.galaxies[ishost]
        self.snhosts_df = self.galaxies_df[ishost]

        # TODO: Alternate approach:  read in a SNANA output file (.dump file
        #  maybe?) that has already run a survey simulation and picked hosts.

        if verbose:
            print(f"Assigned {nsn} SNe to hosts using {snrcolname} probabilities.")
        return



    def apply_specz_completeness_map(self, filename,
                                     defining_columns_galcat,
                                     defining_columns_speczmap,
                                     efficiency_columns_speczmap,
                                     fill_value = np.nan
                                     ):
        """Read in a 'map' for spectroscopic redshift completeness, which
        maps from one or more galaxy properties (mag, SFR, z...) onto a
        probability of getting a spec-z.

        Preferred format of the input file is a .ecsv file, but anything
        that astropy.table can read is OK in principle.

        Then apply the specz completeness map to the catalog
        of host galaxy properties (already read in) to define exactly which
        of the galaxies gets a redshift.

        If the method 'pick_host_galaxies' has already been run
        (so the flag postsurvey == True), then only galaxies defined as SN
        hosts are assigned a redshift.

        Parameters
        ----------
        filename : str
           path to astropy-readable file

        defining_columns_galcat : listlike
           list of strings specifying the column names in the galaxy catalog
           (self.galaxies) for parameters that are used to define the specz
           efficiency (e.g. if this is a SFR-based specz map then this may
           be ['logSFR'])

        defining_columns_speczmap : listlike, same length as above
           list of strings specifying the corresponding column names in the
           specz map file (given by 'filename').  Must be the same length as
           defining_columns_galcat, giving corresponding column names in the
           same order.

        efficiency_columns_speczmap : listlike, same length as above
           list of column names giving the specz
           efficiency (or completeness fraction) for each row in the specz
           map file.
        """
        if (len(defining_columns_galcat)!=len(defining_columns_speczmap) or
            len(defining_columns_galcat)!=len(efficiency_columns_speczmap)):
            raise RuntimeError(
                'You must specify the same number of columns from the '
                'galaxy catalog and the specz efficiency catalog.')

        # TODO : make a masked array to remove NaNs ? ?
        speczmap = table.Table.read(filename)

        # TODO : build a separate interpolating function for each of
        #  the possible input parameters ?
        interpolatordict = {}
        for i in range(len(defining_columns_galcat)):
            colname_galcat = defining_columns_galcat[i]
            xobs = self.galaxies[colname_galcat]
            colname_param = defining_columns_speczmap[i]
            x = speczmap[colname_param]
            colname_efficiency = efficiency_columns_speczmap[i]
            y = speczmap[colname_efficiency]
            interpolator = scinterp.interp1d(
                x, y, bounds_error=False, fill_value=fill_value)
            interpolatordict[colname_galcat] = interpolator

        return(interpolatordict)

    def make_photoz_accuracy_map(self):
        """For every galaxy in the catalog of galaxy properties, apply a
        photo-z function that defines the 'measured' photo-z value and
        uncertainty (photoz pdf).  Includes catastrophic outliers.
        """
        pass

    def report_redshift_completeness(self):
        """Produce a report of the overall redshift completeness, accuracy
        and precision, based on multiple spectroscopic 'filters' and the
        random assignment of photo-z values.
        """
        pass

    def make_sfrmass_figure(self, catalog='galaxies', snhostoverlay=True,
                            **kwargs):
        """ Make some plots of the simulation input catalog properties """
        fig = plt.figure(figsize=[8,4])
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2, sharex=ax1)
        ax3 = fig.add_subplot(1,3,3)
        self.plot_mass_vs_z(ax=ax1, catalog=catalog, snhostoverlay=snhostoverlay, **kwargs)
        self.plot_sfr_vs_z(ax=ax2,  catalog=catalog, snhostoverlay=snhostoverlay, **kwargs)
        self.plot_sfr_vs_mass(ax=ax3,  catalog=catalog, snhostoverlay=snhostoverlay, **kwargs)
        ax1.text(0.05,0.95,'(a)',size='large',color='w',transform=ax1.transAxes)
        ax2.text(0.05,0.95,'(b)',size='large',color='w',transform=ax2.transAxes)
        ax3.text(0.05,0.95,'(c)',size='large',color='w',transform=ax3.transAxes)

        plt.tight_layout()
        return

    def plot_mass_vs_z(self, catalog='galaxies', plotstyle='hexbin', ax=None,
                       snhostoverlay=False, **kwargs):
        if catalog=='galaxies':
            plotcat = self.galaxies
            #plotdf = self.galaxies_df
        elif catalog=='snhosts':
            plotcat = self.snhosts
            #plotdf = self.snhosts_df
        else:
            raise RuntimeError(f"{catalog} is not allowed. Use 'galaxies' or 'snhosts'")

        if ax is None:
            ax = plt.gca()
        if plotstyle=='hexbin':
            z = plotcat['z']
            logm = plotcat['logmass']
            ax.hexbin(z, logm, extent=[0,6,4,12], bins='log', zorder=1, **kwargs)
        if snhostoverlay:
            sns.set_style("dark")
            z_sn = self.snhosts_df['z']
            logm_sn = self.snhosts_df['logmass']
            sns.kdeplot(x=z_sn, y=logm_sn, zorder=2, ax=ax, color='w', levels=5)

        ax.set_xlim(0, 6)
        ax.set_ylim(6, 12)
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Stellar Mass: log$_{10}$(M / M$_{\odot}$)')
        plt.tight_layout()
        return

    def plot_sfr_vs_z(self, catalog='galaxies', ax=None, snhostoverlay=False, **kwargs):
        if catalog=='galaxies':
            plotcat = self.galaxies
        elif catalog=='snhosts':
            plotcat = self.snhosts
        else:
            raise RuntimeError(f"{catalog} is not allowed. Use 'galaxies' or 'snhosts'")

        if ax is None:
            ax = plt.gca()
        z = plotcat['z']
        logsfr = plotcat['logsfr']
        ax.hexbin(z, logsfr,  extent=[0,6,-4,4], bins='log', zorder=1, **kwargs)

        if snhostoverlay:
            sns.set_style("dark")
            z_sn = self.snhosts_df['z']
            logsfr_sn = self.snhosts_df['logsfr']
            sns.kdeplot(x=z_sn, y=logsfr_sn, zorder=2, ax=ax, color='w', levels=5)

        ax.set_xlabel('Redshift')
        ax.set_xlim(0, 6)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Star Formation Rate: log$_{10}$(SFR / [M$_{\odot}$ yr$^{-1}$])')
        plt.tight_layout()
        return

    def plot_sfr_vs_mass(self, catalog='galaxies', ax=None, snhostoverlay=False, **kwargs):
        if catalog=='galaxies':
            plotcat = self.galaxies
        elif catalog=='snhosts':
            plotcat = self.snhosts

        if ax is None:
            ax = plt.gca()
        logmass = plotcat['logmass']
        logsfr = plotcat['logsfr']
        ax.hexbin(x=logmass, y=logsfr, extent=[4,12,-4,4], bins='log', zorder=1, **kwargs)

        if snhostoverlay:
            sns.set_style("dark")
            logm_sn = self.snhosts_df['logmass']
            logsfr_sn = self.snhosts_df['logsfr']
            sns.kdeplot(logm_sn, logsfr_sn, zorder=2, ax=ax, color='w', levels=5)

        ax.set_xlim(6, 12)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('Stellar Mass: log$_{10}$(M/M$_{\odot}$)')
        ax.set_ylabel('Star Formation Rate: log$_{10}$(SFR/[M$_{\odot}$ yr$^{-1}$])')
        plt.tight_layout()
        return



def ssnr_ah18_smooth(logssfr):
    """ Returns the Type Ia specific SN rate per Tyr
    (number of SN Ia exploding per 10^12 yr per solar mass)
    for a galaxy, using the model of Andersen & Hjorth 2018, which is based
    on the specific star formation rate, given as log10(SSFR).
    """
    a = (1.5)*1e-13 # (1.12)*1e-13
    b = 0.5 # 0.73
    k = 0.4 # 0.49
    ssfr0 = 1.7e-10# 1.665e-10
    # logssfr0 = -9.778585762157661    # log10(ssfr0)
    ssfr = np.power(10.,logssfr)
    ssnr = (a + (a/k) * np.log10(ssfr/ssfr0 + b)) * 1e12
    #ssnr = np.max(ssnr, 0.7)
    return(ssnr)


def ssnr_ah18_piecewise(logssfr):
    """ Returns the Type Ia specific SN rate per Tyr
    (number of SN Ia exploding per 10^12 yr per solar mass)
    for a galaxy, using the piecwise linear model
    of Andersen & Hjorth 2018, which is based
    on the specific star formation rate, given as log10(SSFR).
    """
    # Note that the alpha scaling parameter
    # has been multiplied by 1e12 to get units of Tyr-1
    alpha = (1.12)* 1e5
    beta = 0.586
    ssfr2 = 1.01e-11
    ssfr1 = 1.04e-9
    S1 = np.power(ssfr1, beta)
    S2 = np.power(ssfr2, beta)
    if not np.iterable(logssfr):
        logssfr = np.array([logssfr])
    ssfr = np.power(10.,logssfr)
    ssnr = alpha * np.where(ssfr<=ssfr2, S2,
                            np.where(ssfr>=ssfr1, S1,
                                     np.power(ssfr, beta)))
    if len(ssnr)==1:
        ssnr = ssnr[0]
    return(ssnr)

def plot_ssnr_ah18():
    """Make a plot of the Specific SN Rate model from Andersen & Hjorth 2018"""
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=[4,4])
    ax = fig.gca()

    lssfr_testvals = np.linspace(-13., -5., 1000)
    ssnr_smooth = ssnr_ah18_smooth(lssfr_testvals)*1e-12
    ssnr_piecewise = ssnr_ah18_piecewise(lssfr_testvals)*1e-12

    ax.semilogy(lssfr_testvals, ssnr_smooth, ls='-', marker=' ')
    ax.semilogy(lssfr_testvals, ssnr_piecewise, ls='-.', color='r', marker=' ')

    ax.axvline(-8, ls='--', color='0.5')

    ax.set_xlabel("log10(sSFR)")
    ax.set_ylabel("specific SNR")
    plt.tight_layout()
    return
