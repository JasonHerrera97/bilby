#!/usr/bin/env python
import h5py

from astropy.coordinates import SkyCoord
from astropy import constants as const, units as u
import bilby
from gwpy.timeseries import TimeSeries
import numpy as np


outdir = 'bibly_5.445_0_followup_trueloc'
label = 'bilby_5.445_0_followup_trueloc'
bilby.core.utils.setup_logger(outdir=outdir, label=label,
                              log_level='info')
logger = bilby.core.utils.logger

roll_off = 0.4  # Roll off duration of tukey window in seconds

# Injection parameters
# Inject the following signal and sample on the redshift
chirp_mass =  1.43
mass_ratio = 0.833 
a_1 = 0.
a_2 = 0.
tilt_1 = 0.
tilt_2 = 0.
phi_12 = 0.
luminosity_distance = 100
theta_jn = 0.1
phi_jl = 0.
psi=2.659
phase=1.3

# These are two of the really important variables 
ra = 5.445
dec = 0.0

trigger_time = 1264069376 
sampling_frequency = 4096
duration = 320
start_time = trigger_time - duration

injection_parameters = dict(
    chirp_mass=chirp_mass, mass_ratio=mass_ratio, a_1=a_1, a_2=a_2,
    tilt_1=tilt_1, tilt_2=tilt_2, theta_jn=theta_jn,
    luminosity_distance=luminosity_distance, phi_jl=phi_jl,
    psi=psi, phase=phase, geocent_time=trigger_time, phi_12=phi_12,
    ra=ra, dec=dec
)

waveform_arguments = dict(waveform_approximant='TaylorF2ThreePointFivePN',
                          reference_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments
)

# load PSD files for O5
psd_filenames = {
    'H1': 'aligo_O4high_extrapolated.txt',
    'L1': 'aligo_O4high_extrapolated.txt',
    'V1': 'avirgo_O4high_NEW.txt'
}

ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1", "V1"]:
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    freq, asd = np.loadtxt(psd_filenames[det], unpack=True)
    psd = asd**2
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freq, psd_array=psd
    )
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time
    )
    ifo_list.append(ifo)

ifo_list.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator
)

logger.info("Finished Injecting signal")
logger.info("Saving IFO data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# createprior; sample in chirp_mass and mass_ratio
prior_dictionary = dict(
    chirp_mass=bilby.gw.prior.Uniform(name='chirp_mass', minimum=1.40, maximum=1.46),
    mass_ratio=bilby.gw.prior.Uniform(name='mass_ratio', minimum=0.25, maximum=1),
    mass_1=bilby.gw.prior.Constraint(name='mass_1', minimum=1.0, maximum=3.0),
    mass_2=bilby.gw.prior.Constraint(name='mass_2', minimum=1.0, maximum=3.0),
    a_1=bilby.gw.prior.Uniform(name='a_1', minimum=0, maximum=0.05,
                               latex_label='$a_1$', unit=None, boundary=None),
    a_2=bilby.gw.prior.Uniform(name='a_2', minimum=0, maximum=0.05,
                               latex_label='$a_2$', unit=None, boundary=None),
    tilt_1=bilby.core.prior.DeltaFunction(peak=0.0),
    tilt_2=bilby.core.prior.DeltaFunction(peak=0.0),
    phi_12=bilby.core.prior.DeltaFunction(peak=0.0),
    phi_jl=bilby.gw.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi,
                                  boundary='periodic', latex_label='$\\phi_{JL}$', unit=None),
    luminosity_distance=bilby.gw.prior.UniformComovingVolume(name='luminosity_distance',
                                                             minimum=0, maximum=500, latex_label='$d_L$',
                                                             unit='Mpc', boundary=None),
    dec=bilby.core.prior.DeltaFunction(peak=0.0),
    ra=bilby.core.prior.DeltaFunction(peak=5.445),
    theta_jn=bilby.prior.Sine(name='theta_jn', latex_label='$\\theta_{JN}$',
                              unit=None, minimum=0, maximum=np.pi, boundary=None),
    psi=bilby.gw.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic',
                               latex_label='$\\psi$', unit=None)
)

priors = bilby.gw.prior.BBHPriorDict(dictionary=prior_dictionary)

# set a small margin on time of arrival 
priors['geocent_time'] = bilby.core.prior.DeltaFunction(
    peak=trigger_time
)

# Initialise the likelihood
# Ricky change: set time_marginalization to True
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifo_list, waveform_generator=waveform_generator,
    time_marginalization=False, phase_marginalization=True,
    distance_marginalization=False, priors=priors
)

logger.info("Calling dynesty...")

# dynesty options
# ricky change: you should consider what you want to do to the npool variable
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', dlogz=0.1,
    walks=100, check_point_delta_t=15000, npool=64, outdir=outdir, label=label,
    nlive=1000, n_effective=1000, injection_parameters=injection_parameters
)
