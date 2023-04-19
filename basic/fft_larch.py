from numpy import (pi, arange, zeros, ones, sin, cos,
                   exp, log, sqrt, where, interp, linspace)
# from numpy.fft import fft, ifft
from scipy.fftpack import fft, ifft
from scipy.special import i0 as bessel_i0


sqrtpi=np.sqrt(np.pi)
FT_WINDOWS = ('Kaiser-Bessel', 'Hanning', 'Parzen', 'Welch', 'Gaussian', 'Sine')
FT_WINDOWS_SHORT = tuple([a[:3].lower() for a in FT_WINDOWS])
mass_e = 9.1e-31;
hbar = 6.626e-34/(2*np.pi);


def xftf_fast(chi, nfft=2048, kstep=0.05, **kws):
    cchi = zeros(nfft, dtype='complex128')
    cchi[0:len(chi)] = chi
    return (kstep / sqrtpi) * fft(cchi)[:int(nfft/2)]



def ftwindow(x, xmin=None, xmax=None, dx=1, dx2=None,
             window='hanning', **kws):
    """
    create a Fourier transform window array.
    Parameters:
    -------------
      x:        1-d array array to build window on.
      xmin:     starting x for FT Window
      xmax:     ending x for FT Window
      dx:       tapering parameter for FT Window
      dx2:      second tapering parameter for FT Window (=dx)
      window:   name of window type
    Returns:
    ----------
    1-d window array.
    Notes:
    -------
    Valid Window names:
        hanning              cosine-squared taper
        parzen               linear taper
        welch                quadratic taper
        gaussian             Gaussian (normal) function window
        sine                 sine function window
        kaiser               Kaiser-Bessel function-derived window
    """
    if window is None:
        window = FT_WINDOWS_SHORT[0]
    nam = window.strip().lower()[:3]
    if nam not in FT_WINDOWS_SHORT:
        raise RuntimeError("invalid window name %s" % window)

    dx1 = dx
    if dx2 is None:  dx2 = dx1
    if xmin is None: xmin = min(x)
    if xmax is None: xmax = max(x)

    xstep = (x[-1] - x[0]) / (len(x)-1)
    xeps  = 1.e-4 * xstep
    x1 = max(min(x), xmin - dx1/2.0)
    x2 = xmin + dx1/2.0  + xeps
    x3 = xmax - dx2/2.0  - xeps
    x4 = min(max(x), xmax + dx2/2.0)

    if nam == 'fha':
        if dx1 < 0: dx1 = 0
        if dx2 > 1: dx2 = 1
        x2 = x1 + xeps + dx1*(xmax-xmin)/2.0
        x3 = x4 - xeps - dx2*(xmax-xmin)/2.0
    elif nam == 'gau':
        dx1 = max(dx1, xeps)

    def asint(val): return int((val+xeps)/xstep)
    i1, i2, i3, i4 = asint(x1), asint(x2), asint(x3), asint(x4)
    i1, i2 = max(0, i1), max(0, i2)
    i3, i4 = min(len(x)-1, i3), min(len(x)-1, i4)
    if i2 == i1: i1 = max(0, i2-1)
    if i4 == i3: i3 = max(i2, i4-1)
    x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
    if x1 == x2: x2 = x2+xeps
    if x3 == x4: x4 = x4+xeps
    # initial window
    fwin =  zeros(len(x))
    if i3 > i2:
        fwin[i2:i3] = ones(i3-i2)

    # now finish making window
    if nam in ('han', 'fha'):
        fwin[i1:i2+1] = sin((pi/2)*(x[i1:i2+1]-x1) / (x2-x1))**2
        fwin[i3:i4+1] = cos((pi/2)*(x[i3:i4+1]-x3) / (x4-x3))**2
    elif nam == 'par':
        fwin[i1:i2+1] =     (x[i1:i2+1]-x1) / (x2-x1)
        fwin[i3:i4+1] = 1 - (x[i3:i4+1]-x3) / (x4-x3)
    elif nam == 'wel':
        fwin[i1:i2+1] = 1 - ((x[i1:i2+1]-x2) / (x2-x1))**2
        fwin[i3:i4+1] = 1 - ((x[i3:i4+1]-x3) / (x4-x3))**2
    elif nam  in ('kai', 'bes'):
        cen  = (x4+x1)/2
        wid  = (x4-x1)/2
        arg  = 1 - (x-cen)**2 / (wid**2)
        arg[where(arg<0)] = 0
        if nam == 'bes': # 'bes' : ifeffit 1.0 implementation of kaiser-bessel
            fwin = bessel_i0(dx* sqrt(arg)) / bessel_i0(dx)
            fwin[where(x<=x1)] = 0
            fwin[where(x>=x4)] = 0
        else: # better version
            scale = max(1.e-10, bessel_i0(dx)-1)
            fwin = (bessel_i0(dx * sqrt(arg)) - 1) / scale
    elif nam == 'sin':
        fwin[i1:i4+1] = sin(pi*(x4-x[i1:i4+1]) / (x4-x1))
    elif nam == 'gau':
        cen  = (x4+x1)/2
        fwin =  exp(-(((x - cen)**2)/(2*dx1*dx1)))
    return fwin


def xftf_prep(k, chi, kmin=0, kmax=20, kweight=2, dk=1, dk2=None,
                window='kaiser', nfft=2048, kstep=0.05):
    """
    calculate weighted chi(k) on uniform grid of len=nfft, and the
    ft window.
    Returns weighted chi, window function which can easily be multiplied
    and used in xftf_fast.
    """
    if dk2 is None: dk2 = dk
    npts = int(1.01 + max(k)/kstep)
    k_max = max(max(k), kmax+dk2)
    k_   = kstep * np.arange(int(1.01+k_max/kstep), dtype='float64')
    chi_ = interp(k_, k, chi)
    win  = ftwindow(k_, xmin=kmin, xmax=kmax, dx=dk, dx2=dk2, window=window)
    return ((chi_[:npts] *k_[:npts]**kweight), win[:npts])