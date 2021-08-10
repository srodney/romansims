import numpy as np
import fitsio as fio
from astropy.time import Time
from matplotlib import pyplot as plt, cm, patches
import galsim
import galsim.wfirst as wfirst
import os

cmap = cm.get_cmap('nipy_spectral')

__DATADIR__ = os.path.abspath('../data/imagesims')
__TILINGPATTERNFITS__ = os.path.join(__DATADIR__, 'akari_5deg_tiling.fits')


def get_wcs(ra,dec,pa,date):
    return wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=ra*galsim.degrees, \
                                                            dec=dec*galsim.degrees),
                            PA          = pa*galsim.degrees,
                            date        = Time(date,format='mjd').datetime,
                            SCAs        = np.arange(18).astype(int)+1,
                            PA_is_FPA   = True
                            )

def make_dither_pattern(ax=None):
    dither = fio.FITS(__TILINGPATTERNFITS__)[-1].read()

    ra_min = 999
    dec_min = 999
    ra_max = -999
    dec_max = -999
    sca_pos = {}
    for i in range(len(dither)):
        d = dither[i]
        wcs = get_wcs(d['ra'],d['dec'],d['pa'],d['date'])
        sca_list = {}
        for sca in range(1,19):
            corners = {}
            tmp = wcs[sca].toWorld(galsim.PositionI(0,4096))
            if tmp.ra/galsim.degrees<ra_min:
                ra_min = tmp.ra/galsim.degrees
            if tmp.ra/galsim.degrees>ra_max:
                ra_max = tmp.ra/galsim.degrees
            if tmp.dec/galsim.degrees<dec_min:
                dec_min = tmp.dec/galsim.degrees
            if tmp.dec/galsim.degrees>dec_max:
                dec_max = tmp.dec/galsim.degrees
            corners['ul'] = [tmp.ra/galsim.degrees,tmp.dec/galsim.degrees]
            tmp = wcs[sca].toWorld(galsim.PositionI(0,0))
            corners['ll'] = [tmp.ra/galsim.degrees,tmp.dec/galsim.degrees]
            tmp = wcs[sca].toWorld(galsim.PositionI(4096,4096))
            corners['ur'] = [tmp.ra/galsim.degrees,tmp.dec/galsim.degrees]
            tmp = wcs[sca].toWorld(galsim.PositionI(4096,0))
            corners['lr'] = [tmp.ra/galsim.degrees,tmp.dec/galsim.degrees]
            sca_list[sca] = corners
        sca_pos[i] = sca_list

    if ax is None:
        ax = plt.axes(xlim=(ra_min-1, ra_max+1), ylim=(dec_min-1, dec_max+1))
    return sca_pos, ax


def focal_plane(i, sca_pos):
    polys = []
    for sca in range(1,19):
        polys.append(np.array([sca_pos[i][sca]['ul'], sca_pos[i][sca]['ur'], sca_pos[i][sca]['lr'], sca_pos[i][sca]['ll']]))
    return polys

def plot_single_patch(polys, ax=None, facecolor=None, alpha=0.5):
    if ax is None:
        ax = plt.gca()
    #ax.patches = []
    #polys = focal_plane(i)
    patches = [ plt.Polygon(polys[p], facecolor=facecolor, edgecolor='k', alpha=alpha) for p in range(18)]
    for p in range(18):
        ax.add_patch(patches[p])
    return

def make_tiling_figure():
    fig = plt.figure(figsize=[6,6], dpi=100)
    ax = fig.add_subplot(xlim=[68.5, 73.5], ylim=[-54.5,-51.5])
    sca_pos, ax = make_dither_pattern(ax)

    for i in range(19):
        rgba = cmap((i%16.+1)/17.)
        alpha=0.4
        if i>15:
            rgba='None'
            alpha=0.5
        #print((i%16.)/16.)
        polys = focal_plane(i, sca_pos)
        plot_single_patch(polys, ax = ax, facecolor=rgba, alpha=alpha)
        if i==4:
            arrow = patches.FancyArrowPatch(
                (72.5, -52.4), (71.1, -52),
                connectionstyle="arc3,rad=.5",
                arrowstyle="Fancy, head_length=15, head_width=15, tail_width=10",
                color=rgba, ec='k', alpha=0.7
            )
            ax.add_patch(arrow)
        if i==10:
            arrow = patches.FancyArrowPatch(
                (71, -53.6), (69.8, -53.7),
                connectionstyle="arc3,rad=-.5",
                arrowstyle="Fancy, head_length=15, head_width=15, tail_width=10",
                color=rgba, ec='k', alpha=0.7
            )
            ax.add_patch(arrow)

    ax.text(72.3, -53.75, 'Start', fontweight='bold', color='w', size=15, ha='center', va='center')
    ax.text(69.7, -52.2, 'Finish', fontweight='bold', color='w', size=15, ha='center', va='center')
    ax.tick_params(axis='both', direction='inout', labelsize=13)
    ax.set_xlabel('Right Ascension', fontsize=14)
    ax.set_ylabel('Declination', fontsize=14)
    plt.tight_layout()
    return


