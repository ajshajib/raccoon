# -*- coding: utf-8 -*-

__author__ = "ajshajib"

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import huber
from scipy.linalg import lstsq
from astropy.stats import sigma_clip
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm.notebook import tqdm

from .util import Util
from matplotlib.gridspec import GridSpec


class WiggleCleaner(object):
    """WiggleCleaner class for cleaning modulation/wiggles in spectral data."""

    RED = "#e41a1c"
    BLUE = "#377eb8"
    GREEN = "#4daf4a"
    PURPLE = "#984ea3"
    ORANGE = "#ff7f00"
    GREY = "#999999"

    def __init__(
        self,
        wavelengths,
        datacube,
        noise_cube,
        gaps=None,
        symmetric_sharpening=False,
        asymmetric_sharpening=False,
        continuum_diff_polynomial_order=2,
    ):
        """Initialize the WiggleCleaner object.

        :param wavelengths: Wavelengths
        :type wavelengths: list or np.ndarray
        :param datacube: 3D data cube containing the spectral data
        :type datacube: np.ndarray
        :param noise_cube: 3D data cube containing the noise associated with the
            spectral data
        :type noise_cube: np.ndarray
        :param gaps: Gaps
        :type gaps: list
        :param symmetric_sharpening: If True, apply symmetric sharpening
        :type symmetric_sharpening: bool
        :param asymmetric_sharpening: If True, apply asymmetric sharpening
        :type asymmetric_sharpening: bool
        :param continuum_diff_polynomial_order: Continuum polynomial order for the
            difference between the single spaxel and aperture spectra
        :type continuum_diff_polynomial_order: int
        :return: None
        :rtype: None
        """
        self._wavelengths = np.array(wavelengths)
        self._datacube = datacube
        self._noise_cube = noise_cube
        self._gaps = np.array(gaps)
        self._n_amplitude = -1
        self._n_frequency = -1
        self._continuum_diff_polynomial_order = continuum_diff_polynomial_order

        self._symmetric_sharpening = symmetric_sharpening
        self._asymmetric_sharpening = asymmetric_sharpening

        self._amplitude_spline = None
        self._frequency_spline = None

        self._use_huber_loss = False
        self._huber_delta = 1.35

        self._outlier_rejection_method = None

        # Initialize mask for handling gaps and outliers
        if gaps is None:
            self._gaps = []
            self._gap_mask = np.ones_like(self._wavelengths)
        else:
            self.set_gaps(gaps)

        self._outlier_mask = np.ones_like(self._wavelengths)

    def set_gaps(self, gaps):
        """Set the gaps to be ignored during the fitting process.

        :param gaps: List of wavelength ranges to be ignored
        :type gaps: list of tuples
        :return: None
        :rtype: None
        """
        self._gaps = gaps
        gap_mask = np.ones_like(self._wavelengths)
        for g in self._gaps:
            mask = (self._wavelengths > g[0]) & (self._wavelengths < g[1])
            gap_mask[mask] = 0
        self._gap_mask = np.array(gap_mask)

    @property
    def symmetric_sharpening(self):
        """Get the symmetric sharpening flag.

        :return: Symmetric sharpening flag
        :rtype: bool
        """
        return self._symmetric_sharpening

    @symmetric_sharpening.setter
    def symmetric_sharpening(self, value):
        """Set the symmetric sharpening flag.

        :param value: Symmetric sharpening flag
        :type value: bool
        :return: None
        :rtype: None
        """
        self._symmetric_sharpening = value

    @property
    def asymmetric_sharpening(self):
        """Get the asymmetric sharpening flag.

        :return: Asymmetric sharpening flag
        :rtype: bool
        """
        return self._asymmetric_sharpening

    @asymmetric_sharpening.setter
    def asymmetric_sharpening(self, value):
        """Set the asymmetric sharpening flag.

        :param value: Asymmetric sharpening flag
        :type value: bool
        :return: None
        :rtype: None
        """
        self._asymmetric_sharpening = value

    @property
    def scaled_w(self):
        """Scaled wavelengths.

        :return: Scaled wavelengths
        :rtype: np.ndarray
        """
        return self.scale_wavelengths_negative1_to_1(self._wavelengths)

    def wiggle_func(self, xs, amplitude_params, frequency_params, phi, a_1=0, a_2=0):
        """Get the wiggle function.

        :param xs: Scaled wavelengths
        :type xs: np.ndarray
        :param frequency_params: Frequency parameters
        :type frequency_params: np.ndarray
        :param amplitude_params: Amplitude parameters
        :type amplitude_params: np.ndarray
        :param phi: Phase
        :type phi: float
        :return: Wiggle function
        :rtype: np.ndarray
        """
        amplitude_spline = deepcopy(self._amplitude_spline)
        frequency_spline = deepcopy(self._frequency_spline)

        amplitude_spline.c = amplitude_params
        frequency_spline.c = frequency_params

        amplitude = amplitude_spline(xs)
        frequency = frequency_spline(xs)

        wave_function = (
            np.sin(frequency * xs + phi)
            + a_1 * (np.sin(frequency * xs + phi) ** 2)  # asymmetric sharpness
            + a_2 * np.sin(3 * (frequency * xs + phi))  # sharpness
        )

        return 1.0 + amplitude * wave_function

    def scale_wavelengths_negative1_to_1(self, w):
        """Scale the wavelengths to -1 to 1.

        :param w: Wavelengths
        :type w: np.ndarray
        :return: Scaled wavelengths
        :rtype: np.ndarray
        """
        return (w - self._wavelengths[0]) / (
            self._wavelengths[-1] - self._wavelengths[0]
        ) * 2 - 1

    def scale_wavelengths_to_0_1(self, w):
        """Scale the wavelengths to 0 to 1.

        :param w: Wavelengths
        :type w: np.ndarray
        :return: Scaled wavelengths
        :rtype: np.ndarray
        """
        return (w - self._wavelengths[0]) / (
            self._wavelengths[-1] - self._wavelengths[0]
        )

    def wiggle_model(self, params):
        """Get the wiggle model given the parameters.

        :param params: Parameters
        :type params: np.ndarray
        :return: Model
        :rtype: np.ndarray
        """
        n_amplitude, n_frequency = self.configure_polynomial_ns()

        amplitude_params, frequency_params, phi_0 = self.split_params(
            params, n_amplitude, n_frequency
        )

        a_1, a_2 = self.get_a1_a2(params)

        wiggle_model = self.wiggle_func(
            self.scaled_w,
            amplitude_params,
            frequency_params,
            phi_0,
            a_1=a_1,
            a_2=a_2,
        )
        return wiggle_model

    def get_a1_a2(self, params):
        """Get the a_1 and a_2 parameters for asymmetric and symmetric sharpening from
        the params array.

        :param params: Parameters
        :type params: np.ndarray
        :return: a_1 and a_2 parameters
        :rtype: Tuple[float, float]
        """
        n_amplitude, n_frequency = self.configure_polynomial_ns()
        if self._asymmetric_sharpening and not self._symmetric_sharpening:
            a_1 = params[n_amplitude + n_frequency + 5]
            a_2 = 0
        elif self._symmetric_sharpening and not self._asymmetric_sharpening:
            a_1 = 0
            a_2 = params[n_amplitude + n_frequency + 5]
        elif self._symmetric_sharpening and self._asymmetric_sharpening:
            a_1 = params[n_amplitude + n_frequency + 5]
            a_2 = params[n_amplitude + n_frequency + 6]
        else:
            a_1 = 0
            a_2 = 0
        return a_1, a_2

    def get_exponent_param(self, params):
        """Get the exponent parameter for the wavelength term in the power-law component
        of the continuum.

        :param params: Parameters
        :type params: np.ndarray
        :return: Exponent parameter
        :rtype: float
        """
        n_amplitude, n_frequency = self.configure_polynomial_ns()
        add_more = 0
        if self._asymmetric_sharpening and not self._symmetric_sharpening:
            add_more = 1
        elif self._symmetric_sharpening and not self._asymmetric_sharpening:
            add_more = 1
        elif self._symmetric_sharpening and self._asymmetric_sharpening:
            add_more = 2

        return params[n_amplitude + n_frequency + add_more + 5]

    def split_params(self, params, n_amplitude=None, n_frequency=None):
        """Split the parameters. Opposite of the set_params function.

        :param params: Parameters
        :type params: np.ndarray
        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :return: amplitude parameters, frequency parameters, and phi_0
        :rtype: Tuple
        """
        n_amplitude, n_frequency = self.configure_polynomial_ns(
            n_amplitude, n_frequency
        )

        amplitude_params = params[: n_amplitude + 2]
        frequency_params = params[n_amplitude + 2 : n_amplitude + n_frequency + 4]
        phi_0 = params[n_amplitude + n_frequency + 4]

        return amplitude_params, frequency_params, phi_0

    def set_params(
        self,
        amplitude_params,
        frequency_params,
        phi_0,
        n_amplitude=None,
        n_frequency=None,
    ):
        """Set the parameters. Opposite function of the split_params function.

        :param amplitude_params: Amplitude parameters
        :type amplitude_params: np.ndarray
        :param frequency_params: Frequency parameters
        :type frequency_params: np.ndarray
        :param phi_0: Phase
        :type phi_0: float
        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :return: Parameters
        :rtype: np.ndarray
        """
        if n_amplitude is None:
            n_amplitude = self._n_amplitude
        if n_frequency is None:
            n_frequency = self._n_frequency

        params = np.concatenate([amplitude_params, frequency_params, np.array([phi_0])])

        return params

    def configure_polynomial_ns(self, n_amplitude=None, n_frequency=None):
        """Configure the number of parameters.

        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :return: Number of amplitude and frequency parameters
        :rtype: Tuple
        """
        if n_frequency is None:
            n_frequency = self._n_frequency
        else:
            if n_frequency < 2:
                raise ValueError("n_frequency must be at least 2")
            self._n_frequency = n_frequency

        if n_amplitude is None:
            n_amplitude = self._n_amplitude
        else:
            if n_amplitude < 2:
                raise ValueError("n_amplitude must be at least 2")
            self._n_amplitude = n_amplitude

        return n_amplitude, n_frequency

    def model_full_fit(
        self,
        params,
        x,
        y,
        aperture_radius,
        annulus_outer_radius=0,
        annulus_inner_radius=0,
    ):
        """Compute the full model fit.

        :param params: Parameters
        :type params: np.ndarray
        :param x: x coordinates
        :type x: np.ndarray
        :param y: y coordinates
        :type y: np.ndarray
        :param aperture_radius: Aperture radius
        :type aperture_radius: float
        :param annulus_outer_radius: Outer radius of the annulus
        :type annulus_outer_radius: float
        :param annulus_inner_radius: Inner radius of the annulus
        :type annulus_inner_radius: float
        :return: Full model, spectra, and total noise
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        (
            spectra,
            noise,
            aperture_spectra,
            aperture_noise,
            annulus_spectra,
            annulus_noise,
        ) = self.get_spectra_set(
            x, y, aperture_radius, annulus_outer_radius, annulus_inner_radius
        )

        wavelengths = self.scale_wavelengths_negative1_to_1(self._wavelengths)

        # fit c_1 * aperture_spectra + c_3 * wavelengths**a + \sum_i c_i * wavelengths**i + c_N * annulus_spectra
        # given non-linear parameter a, treat all c_i parameters as linear parameters and derive them using linear inversion
        wiggle_model = self.wiggle_model(params)

        A = self.get_design_matrix(
            aperture_spectra,
            annulus_spectra,
            annulus_outer_radius,
            self.get_exponent_param(params),
            wavelengths,
        )

        A *= wiggle_model[:, np.newaxis]

        # Solve the linear least squares problem with Tikhonov regularization (ridge regression)
        coefficients = self.solve_ridge_regression(A, spectra)

        full_model = A @ coefficients

        fractional_variance = (noise / spectra) ** 2 + (
            aperture_noise / aperture_spectra
        ) ** 2
        if annulus_outer_radius > 0:
            fractional_variance += (annulus_noise / annulus_spectra) ** 2

        model_uncertainty = np.sqrt(fractional_variance * full_model**2)

        return full_model, model_uncertainty, spectra, noise

    def get_design_matrix(
        self,
        aperture_spectra,
        annulus_spectra,
        annulus_outer_radius,
        a,
        wavelengths,
    ):
        """Get the design matrix for the full model fit.

        :param a: Non-linear parameter, the exponent for the wavelength term in power-
            law component of the continuum
        :type a: float
        :param annulus_outer_radius: Outer radius of the annulus
        :type annulus_outer_radius: float
        :param aperture_spectra: Aperture spectra
        :type aperture_spectra: np.ndarray
        :param annulus_spectra: Annulus spectra
        :type annulus_spectra: np.ndarray
        :param wavelengths: Wavelengths
        :type wavelengths: np.ndarray
        :return: Design matrix
        :rtype: np.ndarray
        """
        A = np.column_stack(
            [
                aperture_spectra,
                np.ones_like(wavelengths),
            ]
        )
        for i in range(1, self._continuum_diff_polynomial_order + 1):
            A = np.column_stack(
                [
                    A,
                    wavelengths**i,
                ]
            )

        A = np.column_stack(
            [
                A,
                self.scale_wavelengths_to_0_1(self._wavelengths) ** np.abs(a),
            ]
        )

        if annulus_outer_radius > 0:
            # Add the annulus spectra to the design matrix
            A = np.column_stack([A, annulus_spectra])

        return A

    def residual_vector_full_fit(
        self,
        params,
        x,
        y,
        aperture_radius,
        annulus_outer_radius=0,
        annulus_inner_radius=0,
    ):
        """Compute the residual vector for the full model fit.

        :param params: Parameters
        :type params: np.ndarray
        :param x: x coordinates
        :type x: np.ndarray
        :param y: t coordinates
        :type y: np.ndarray
        :param aperture_radius: Aperture radius
        :type aperture_radius: float
        :param annulus_outer_radius: Outer radius of the annulus
        :type annulus_outer_radius: float
        :param annulus_inner_radius: Inner radius of the annulus
        :type annulus_inner_radius: float
        :return: Residual vector
        :rtype: np.ndarray
        """
        full_model, total_uncertainty, spectra, _ = self.model_full_fit(
            params,
            x,
            y,
            aperture_radius,
            annulus_outer_radius,
            annulus_inner_radius,
        )

        residual = (full_model - spectra) / total_uncertainty

        residual = residual * self._gap_mask * self._outlier_mask

        if self._use_huber_loss:
            huber_loss = huber(self._huber_delta, residual)
            residual = np.sqrt(np.abs(2 * huber_loss)) * np.sign(residual)

        return residual

    def residual_vector(self, params, wiggle_signal, wiggle_noise):
        """Get the residual vector.

        :param params: Parameters
        :type params: np.ndarray
        :param wiggle_signal: wiggle_signal
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle_noise
        :type wiggle_noise: np.ndarray
        :return: Residual vector
        :rtype: np.ndarray
        """
        model = self.wiggle_model(params)
        total_noise = wiggle_noise
        residual = (model - wiggle_signal) / total_noise
        residual = residual * self._gap_mask * self._outlier_mask
        if self._use_huber_loss:
            huber_loss = huber(self._huber_delta, residual)
            residual = np.sqrt(np.abs(2 * huber_loss)) * np.sign(residual)
        return residual

    def cost_function(self, params, wiggle_signal, wiggle_noise):
        """Compute the cost function (sum of squared residuals).

        :param params: Parameters
        :type params: np.ndarray
        :param wiggle_signal: wiggle data
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle_noise
        :type wiggle_noise: np.ndarray
        :return: cost
        :rtype: float
        """
        residual = self.residual_vector(params, wiggle_signal, wiggle_noise)
        cost = np.sum(residual**2)

        return cost

    def get_residual_func(self, wiggle_signal, wiggle_noise):
        """Get the residual function.

        :param wiggle_signal: wiggle data
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle noise
        :type wiggle_noise: np.ndarray
        :return: Residual function
        :rtype: Callable
        """

        def residual_func(params):
            return self.residual_vector(params, wiggle_signal, wiggle_noise)

        return residual_func

    def get_residual_func_phase_only(self, init_params, wiggle_signal, wiggle_noise):
        """Get the residual function with phase only.

        :param params: Parameters
        :type params: np.ndarray
        :param init_params: Initial parameters
        :type init_params: np.ndarray
        :param wiggle_signal: wiggle_signal
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle noise
        :type wiggle_noise: np.ndarray
        :return: Residual function
        :rtype: Callable
        """
        amplitude_params, _, _ = self.split_params(init_params)

        def residual_func(params):
            new_params = self.set_params(
                amplitude_params,
                params[:-1],
                params[-1],
            )
            return self.residual_vector(new_params, wiggle_signal, wiggle_noise)

        return residual_func

    def fit_wiggle(
        self,
        x,
        y,
        aperture_radius=4,
        annulus_outer_radius=5,
        annulus_inner_radius=3,
        n_amplitude=None,
        n_frequency=None,
        specified_noise_level=0,
        init_peak_detection_proximity_threshold=200,
        do_interim_fit_phase_only=False,
        extract_covariance=True,
        outlier_rejection_method=None,
        fdr_alpha=0.01,
        fdr_outlier_max_fraction=0.1,
        use_huber_loss=False,
        huber_delta=1.35,
        sigma_clip_sigma=5,
        sigma_clip_max_iterations=5,
        symmetric_sharpening=False,
        asymmetric_sharpening=False,
        plot=True,
        verbose=True,
        fit_full_model=True,
        save_figure_dir=None,
    ):
        """Fit the wiggle data.

        :param wiggle_signal: Wiggle data
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle noise
        :type wiggle_noise: np.ndarray
        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :param specified_noise_level: User-defined noise level to be used instead of the
            actual noise. Set to 0 to disable.
        :type specified_noise_level: float
        :param init_peak_detection_proximity_threshold: Proximity lower limit in
            Angstrom for initial identifaction of peaks and troughs
        :type init_peak_detection_proximity_threshold: float
        :param plot: If True, plot the results
        :type plot: bool
        :param verbose: If True, print the results
        :type verbose: bool
        :param do_interim_fit_phase_only: If True, do an interim fit with phase only
        :type do_interim_fit_phase_only: bool
        :param outlier_rejection_method: Outlier rejection method, "fdr" or
            "sigma_clip", set None to disable
        :type outlier_rejection_method: str
        :param fdr_alpha: False discovery rate (FDR) correction threshold, smaller value
            will reject less outliers
        :type fdr_alpha: float
        :param fdr_outlier_max_fraction: Maximum fraction of outliers to reject using
            FDR
        :type fdr_outlier_max_fraction: float
        :param use_huber_loss: If True, use Huber loss function for outlier rejection with FDR
        :type use_huber_loss: bool
        :param huber_delta: Delta for Huber loss function
        :type huber_delta: float
        :param sigma_clip_sigma: Sigma threshold for sigma clipping
        :type sigma_clip_sigma: float
        :param sigma_clip_max_iterations: Number of sigma clip iterations
        :type sigma_clip_max_iterations: int
        :param extract_uncertainty: If True, extract the uncertainties
        :type extract_uncertainty: bool
        :return: Fitted parameters
        :rtype: np.ndarray
        """
        assert (
            fdr_outlier_max_fraction < 1
        ), "fdr_outlier_max_fraction must be less than 1"

        wiggle_signal, wiggle_noise = self.get_wiggle_signal(
            x, y, aperture_radius, annulus_outer_radius, annulus_inner_radius
        )

        self._outlier_rejection_method = outlier_rejection_method
        self._use_huber_loss = use_huber_loss
        self._huber_delta = huber_delta
        self._symmetric_sharpening = symmetric_sharpening
        self._asymmetric_sharpening = asymmetric_sharpening

        self._outlier_mask = np.ones_like(self._wavelengths)

        n_amplitude, n_frequency = self.configure_polynomial_ns(
            n_amplitude, n_frequency
        )

        wiggle_noise = self.configure_noise(
            wiggle_signal, wiggle_noise, specified_noise_level
        )

        amplitude_spline, frequency_spline, init_phi_0 = Util.get_init_params_spline(
            wiggle_signal,
            self.scaled_w,
            n_amplitude=n_amplitude,
            n_frequency=n_frequency,
            init_peak_detection_proximity_threshold=init_peak_detection_proximity_threshold
            / np.mean(np.diff(self._wavelengths)),
            plot=False,
        )

        self._amplitude_spline = deepcopy(amplitude_spline)
        init_amplitude_params = deepcopy(amplitude_spline.c)
        self._frequency_spline = deepcopy(frequency_spline)
        init_frequency_params = deepcopy(frequency_spline.c)

        wiggle_signal = np.array(wiggle_signal)
        wiggle_noise = np.array(wiggle_noise)

        x0 = self.set_params(
            init_amplitude_params,
            init_frequency_params,
            init_phi_0,
            n_amplitude=n_amplitude,
            n_frequency=n_frequency,
        )

        # first do an interim fit to solve for initial phase only
        if do_interim_fit_phase_only:
            result = least_squares(
                self.get_residual_func_phase_only(x0, wiggle_signal, wiggle_noise),
                np.concatenate([init_frequency_params, x0]),
            )
            interim_frquency_params = result.x[:-1]
            interim_phi_0 = result.x[-1]

            x0 = self.set_params(
                init_amplitude_params,
                interim_frquency_params,
                interim_phi_0,
                n_amplitude=n_amplitude,
                n_frequency=n_frequency,
            )

        # Add parameters for asymmetric and symmetric sharpening
        if self._symmetric_sharpening and self._asymmetric_sharpening:
            x0 = np.concatenate([x0, np.array([0, 0])])
        elif self._symmetric_sharpening or self._asymmetric_sharpening:
            x0 = np.concatenate([x0, np.array([0])])

        # fitting with the extracted wiggle signal in any case, as the fitted parameters
        # will be used as initial parameters when fitting the full spectra
        if outlier_rejection_method != "fdr":
            self._use_huber_loss = False

        # initial "robust" regression (using Huber loss)
        # here, fitting the wiggle signal that was extracted from the spectra using the template(s)
        result = least_squares(
            self.get_residual_func(wiggle_signal, wiggle_noise),
            x0,
        )

        if self._outlier_rejection_method is not None:
            residual = self.residual_vector(result.x, wiggle_signal, wiggle_noise)

            clipped_pixels = self.reject_outliers(
                residual,
                num_params=len(result.x),
                fdr_alpha=fdr_alpha,
                fdr_outlier_max_fraction=fdr_outlier_max_fraction,
                sigma_clip_sigma=sigma_clip_sigma,
                sigma_clip_max_iterations=sigma_clip_max_iterations,
            )

            self._outlier_mask[clipped_pixels] = 0

            self._use_huber_loss = False

            result = least_squares(
                self.get_residual_func(wiggle_signal, wiggle_noise),
                result.x,
            )

        result_params = result.x

        if fit_full_model:
            # Add a parameter for the exponent of the wavelength term in the power-law component of the continuum
            x0 = np.concatenate(
                [
                    result_params,
                    np.array([1.5]),
                ]
            )

            # Reset outlier mask for full fit
            self._outlier_mask = np.ones_like(self._wavelengths)
            # Turning on Huber loss again for the full fit if user_huber_loss is True
            if outlier_rejection_method == "fdr" and use_huber_loss:
                self._use_huber_loss = True

            result = least_squares(
                self.residual_vector_full_fit,
                x0,
                args=(
                    x,
                    y,
                    aperture_radius,
                    annulus_outer_radius,
                    annulus_inner_radius,
                ),
            )

            result_params = result.x[:-1]

            if self._outlier_rejection_method is not None:
                residual = self.residual_vector_full_fit(
                    result.x,
                    x,
                    y,
                    aperture_radius,
                    annulus_outer_radius,
                    annulus_inner_radius,
                )

                clipped_pixels = self.reject_outliers(
                    residual,
                    num_params=len(result.x),
                    fdr_alpha=fdr_alpha,
                    fdr_outlier_max_fraction=fdr_outlier_max_fraction,
                    sigma_clip_sigma=sigma_clip_sigma,
                    sigma_clip_max_iterations=sigma_clip_max_iterations,
                )

                self._outlier_mask[clipped_pixels] = 0

                # Turn off Huber loss for the final fit
                self._use_huber_loss = False

                result = least_squares(
                    self.get_residual_func(wiggle_signal, wiggle_noise),
                    result.x,
                )

            # Update wiggle signal based on the full fit
            bestfit_model, model_uncertainty, spectra, noise = self.model_full_fit(
                result.x,
                x,
                y,
                aperture_radius,
                annulus_outer_radius,
                annulus_inner_radius,
            )

            wiggle_model = self.wiggle_model(result_params)
            wiggle_signal = spectra / (bestfit_model / wiggle_model)
            wiggle_noise = noise / (bestfit_model / wiggle_model)

            if plot:
                self.plot_full_fit_model(
                    bestfit_model,
                    model_uncertainty,
                    spectra,
                    noise,
                    wiggle_model,
                    save_figure_info={"x": x, "y": y, "dir": save_figure_dir},
                )

        if extract_covariance:
            residuals = result.fun
            jacobian = result.jac

            # Get number of observations (m) and parameters (n)
            m, n = jacobian.shape

            # Check degrees of freedom
            assert (
                m > n
            ), "Number of observations must exceed number of parameters to estimate uncertainty."

            # Calculate residual sum of squares and variance estimate
            sum_of_squared_residuals = np.sum(residuals**2)
            dof = m - n
            sigma_squared = sum_of_squared_residuals / dof

            # Compute covariance matrix using pseudoinverse for stability
            cov_matrix = sigma_squared * np.linalg.pinv(jacobian.T @ jacobian)

            if fit_full_model:
                cov_matrix = cov_matrix[:-1, :-1]
        else:
            cov_matrix = None

        if verbose:
            print(
                "Cost (from fitting extracted wiggle signal): ",
                self.cost_function(result_params, wiggle_signal, wiggle_noise),
            )
            if fit_full_model:
                print(
                    "Cost (from fitting full spectra): ",
                    np.sum(
                        self.residual_vector_full_fit(
                            result.x,
                            x,
                            y,
                            aperture_radius,
                            annulus_outer_radius,
                            annulus_inner_radius,
                        )
                        ** 2
                    ),
                )

        if plot:
            self.plot_wiggle_model(
                wiggle_signal,
                wiggle_noise,
                result_params,
                cov_matrix=cov_matrix,
                save_figure_info={"x": x, "y": y, "dir": save_figure_dir},
            )

        return result_params, cov_matrix, wiggle_signal, wiggle_noise

    def plot_full_fit_model(
        self,
        bestfit_model,
        model_uncertainty,
        spectra,
        noise,
        wiggle_model=None,
        save_figure_info={"dir": None, "x": None, "y": None},
    ):
        """Plot the full fit model.

        :param bestfit_model: Best-fit model
        :type bestfit_model: np.ndarray
        :param model_uncertainty: Model uncertainty
        :type model_uncertainty: np.ndarray
        :param spectra: Spectra
        :type spectra: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :return: None
        :rtype: None
        """

        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(4, 1, height_ratios=[3, 0.05, 1, 0.05], hspace=0)
        ax = fig.add_subplot(gs[0])
        ax.plot(
            self._wavelengths,
            bestfit_model,
            label="Best-fit model",
            lw=1,
            c=self.ORANGE,
        )
        ax.fill_between(
            self._wavelengths,
            bestfit_model - model_uncertainty,
            bestfit_model + model_uncertainty,
            lw=1,
            color=self.ORANGE,
            alpha=0.3,
        )
        ax.errorbar(
            self._wavelengths,
            spectra,
            yerr=noise,
            markersize=2,
            label="Data",
            lw=1,
            c=self.BLUE,
        )
        if wiggle_model is not None:
            ax.plot(
                self._wavelengths,
                spectra / wiggle_model,
                label="Corrected spectra",
                lw=1,
                ls="--",
                c="k",
            )
        ax.legend()
        ax.tick_params(labelbottom=False)  # Hide x tick labels on top axis
        ax.set_ylabel("Flux")
        ax.set_title("Full model fit including wiggles")

        # Residuals subplot
        ax2 = fig.add_subplot(gs[2], sharex=ax)
        residuals = bestfit_model - spectra
        ax2.errorbar(
            self._wavelengths,
            residuals,
            yerr=noise,
            # label="Residuals",
            lw=1,
            c=self.GREEN,
        )
        ax2.axhline(0, color="black", lw=0.5, ls="--")
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel(r"Wavelengths ($\rm\AA$)")
        # ax2.legend()

        # plot vertical shaded regions for gaps
        for g in self._gaps:
            ax.axvspan(g[0], g[1], color="gray", alpha=0.3, lw=0)
            ax2.axvspan(g[0], g[1], color="gray", alpha=0.3, lw=0)

        # Remove space between axes
        plt.subplots_adjust(hspace=0)

        ax.set_xlim(
            self._wavelengths[0],
            self._wavelengths[-1],
        )
        ax2.set_xlim(
            self._wavelengths[0],
            self._wavelengths[-1],
        )

        if save_figure_info["dir"] is not None:
            plt.savefig(
                f"{save_figure_info['dir']}/wiggle_full_fit_{save_figure_info['x']}_{save_figure_info['y']}.png",
                bbox_inches="tight",
            )
        plt.show()

    def reject_outliers(
        self,
        residual,
        num_params=0,
        fdr_alpha=0.01,
        fdr_outlier_max_fraction=0.1,
        sigma_clip_sigma=5,
        sigma_clip_max_iterations=5,
    ):
        """Reject outliers using the selected method.

        :param residual: Residuals
        :type residual: np.ndarray
        :param num_params: Number of parameters
        :type num_params: int
        :param fdr_q: FDR correction threshold
        :type fdr_q: float
        :param fdr_outlier_max_fraction: Maximum fraction of outliers to reject
        :type fdr_outlier_max_fraction: float
        :param huber_delta: Delta for Huber loss function
        :type huber_delta: float
        :param sigma: Sigma threshold for sigma clipping
        :type sigma: float
        :param sigma_clip_max_iterations: Maximum number of iterations for sigma
            clipping
        :type sigma_clip_max_iterations: int
        :return: Indices of outliers
        :rtype: np.ndarray
        """
        outlier_pixels = np.zeros_like(residual, dtype=bool)

        if self._outlier_rejection_method == "sigma_clip":
            clipped = sigma_clip(
                residual,
                sigma=sigma_clip_sigma,
                maxiters=sigma_clip_max_iterations,
                masked=True,
            )
            outlier_pixels = clipped.mask
        elif self._outlier_rejection_method == "fdr":
            # standardized_residuals = residuals / mad(residuals)

            # Compute two-tailed p-values (assuming ~t-distribution)
            # Degrees of freedom approximation (n - p - 1)
            df = len(residual) - num_params - 1
            p_values = 2 * (1 - stats.t.cdf(np.abs(residual), df))

            # Apply FDR correction (Benjamini-Hochberg)
            reject, corrected_p = fdrcorrection(p_values, alpha=fdr_alpha)

            # Limit rejection to a maximum of 30% of the data points with the highest p-values
            max_reject = int(fdr_outlier_max_fraction * len(residual))
            sorted_indices = np.argsort(p_values)
            top_indices = sorted_indices[:max_reject]
            outlier_pixels = np.zeros_like(residual, dtype=bool)
            outlier_pixels[top_indices] = reject[top_indices]
        else:
            raise ValueError(
                f"Unrecognized outlier rejection_method: {self._outlier_rejection_method}"
            )

        return outlier_pixels

    def configure_noise(self, wiggle_signal, wiggle_noise, specified_noise_level):
        """Configure the noise.

        :param wiggle_signal: wiggle_signal
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle noise
        :type wiggle_noise: np.ndarray
        :param specified_noise_level: User-defined noise level to be used instead of the
            actual noise. Set to 0 to disable.
        :type specified_noise_level: float
        :return: modified wiggle noise
        :rtype: np.ndarray
        """
        if wiggle_noise is None and specified_noise_level == 0:
            raise ValueError(
                "Noise level not set! Either provide the noise or set the specified_noise_level."
            )
        if specified_noise_level > 0:
            wiggle_noise = np.ones_like(wiggle_signal) * specified_noise_level
        return wiggle_noise

    def plot_wiggle_model(
        self,
        wiggle_signal,
        wiggle_noise,
        result_params,
        cov_matrix=None,
        num_samples_uncertainty_region=1000,
        save_figure_info={"dir": None, "x": None, "y": None},
    ):
        """Plot the model.

        :param wiggle_signal: wiggle_signal
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle noise
        :type wiggle_noise: np.ndarray
        :param result_params: Fitted parameters
        :type result_params: np.ndarray
        :param cov_matrix: Covariance matrix
        :type cov_matrix: np.ndarray
        :param num_samples_uncertainty_region: Number of samples for uncertainty region
        :type num_samples_uncertainty_region: int
        :return: None
        :rtype: None
        """
        fig = plt.figure(figsize=(12, 2.5))
        ax = fig.add_subplot(111)

        ax.errorbar(
            self._wavelengths[(self._outlier_mask == 1) & (self._gap_mask == 1)],
            wiggle_signal[(self._outlier_mask == 1) & (self._gap_mask == 1)],
            yerr=wiggle_noise[(self._outlier_mask == 1) & (self._gap_mask == 1)],
            label="Fitted points",
            ls="None",
            marker="o",
            markersize=3,
            alpha=0.3,
            c=self.BLUE,
        )
        ax.errorbar(
            self._wavelengths[(self._outlier_mask == 0) | (self._gap_mask == 0)],
            wiggle_signal[(self._outlier_mask == 0) | (self._gap_mask == 0)],
            yerr=wiggle_noise[(self._outlier_mask == 0) | (self._gap_mask == 0)],
            label="Rejected outliers",
            ls="None",
            marker="o",
            markersize=3,
            alpha=0.4,
            c=self.GREY,
        )

        model = self.wiggle_model(result_params)

        line_color = self.ORANGE

        ax.plot(
            self._wavelengths,
            model,
            label="Model",  # n_param({len(result_params)})",
            lw=1,
            c=line_color,
            zorder=10,
        )

        if cov_matrix is not None:
            model_uncertainty = self.get_model_uncertainty(
                result_params, cov_matrix, num_samples_uncertainty_region
            )

            ax.fill_between(
                self._wavelengths,
                model - model_uncertainty,
                model + model_uncertainty,
                color=line_color,
                alpha=0.4,
            )

        for g in self._gaps:
            ax.axvspan(g[0], g[1], color=self.GREY, alpha=0.1)

        ax.set_xlabel(r"Wavelengths ($\rm\AA$)")
        ax.set_ylabel("Wiggle model")
        ax.legend(loc="lower left", ncol=3)
        ax.set_ylim(
            np.min(wiggle_signal) * 0.7,
            np.max(wiggle_signal) * 1.02,
        )
        # ax.set_ylim(np.min(wiggle_signal) * 0.95, np.max(wiggle_signal) * 1.05)

        delta_lambda = self._wavelengths[1] - self._wavelengths[0]
        ax.set_xlim(
            self._wavelengths[0] - delta_lambda * 2,
            self._wavelengths[-1] + delta_lambda * 2,
        )
        if save_figure_info["dir"] is not None:
            fig.savefig(
                f"{save_figure_info['dir']}/wiggle_model_{save_figure_info['x']}_{save_figure_info['y']}.png",
                bbox_inches="tight",
            )
        plt.show()

    def get_model_uncertainty(
        self, result_params, cov_matrix, num_samples_uncertainty_region=1000
    ):
        """Get the model uncertainty.

        :param result_params: Fitted parameters
        :type result_params: np.ndarray
        :param cov_matrix: Covariance matrix
        :type cov_matrix: np.ndarray
        :param num_samples_uncertainty_region: Number of samples for uncertainty region
        :type num_samples_uncertainty_region: int
        :return: Model uncertainty at each wavelength
        :rtype: np.ndarray
        """
        models = []
        for i in range(num_samples_uncertainty_region):
            sampled_params = np.random.multivariate_normal(
                result_params, cov_matrix, 1
            )[0]
            models.append(self.wiggle_model(sampled_params))

        models = np.array(models)

        model_up, model_down = np.percentile(models, [16, 84], axis=0)
        model_uncertainty = (model_up - model_down) / 2
        return model_uncertainty

    def get_wiggle_signal(
        self,
        x,
        y,
        aperture_radius=4,
        annulus_outer_radius=0,
        annulus_inner_radius=0,
        plot=False,
    ):
        """Get the wiggle signal for preliminary assessments.

        :param spaxel_x: Spaxel x
        :type spaxel_x: int
        :param spaxel_y: Spaxel y
        :type spaxel_y: int
        :param aperture: aperture size to sum the spectra to average out the wiggles
        :type aperture: int
        :return: Modulation wiggle_signal and noise
        :rtype: Tuple of np.ndarray
        """
        (
            spectra,
            noise,
            aperture_spectra,
            aperture_noise,
            annulus_spectra,
            annulus_noise,
        ) = self.get_spectra_set(
            x, y, aperture_radius, annulus_outer_radius, annulus_inner_radius
        )

        wavelengths = self.scale_wavelengths_negative1_to_1(self._wavelengths)

        # fit c_1 * aperture_spectra + c_3 * wavelengths**a + \sum_i c_i * wavelengths**i + c_N * annulus_spectra
        # given non-linear parameter a, treat all c_i parameters as linear parameters and derive them using linear inversion
        def model(a):
            # Construct the design matrix for the current 'a'
            A = self.get_design_matrix(
                aperture_spectra,
                annulus_spectra,
                annulus_outer_radius,
                a,
                wavelengths,
            )

            coefficients = self.solve_ridge_regression(A, spectra)

            return A @ coefficients, coefficients

        def residual_vector(a):
            model_spectra, _ = model(a)
            return (model_spectra - spectra) / noise

        result = least_squares(residual_vector, 0.5, bounds=(0, 6))

        best_model, _ = model(result.x)

        model_noise_fraction = (aperture_noise / aperture_spectra) ** 2
        if annulus_outer_radius > annulus_inner_radius:
            model_noise_fraction += (annulus_noise / annulus_spectra) ** 2

        wiggle_signal = spectra / best_model
        wiggle_noise = (
            np.sqrt((noise / spectra) ** 2 + model_noise_fraction) * wiggle_signal
        )

        # replace non-positive noise with minimum non-negative value
        min_positive_noise = np.nanmin(wiggle_noise[wiggle_noise >= 0])
        wiggle_noise[wiggle_noise <= 0] = min_positive_noise

        return wiggle_signal, wiggle_noise

    @staticmethod
    def solve_ridge_regression(A, spectra):
        """Solve the ridge regression problem.

        :param A: Design matrix
        :type A: np.ndarray
        :param spectra: Target spectra
        :type spectra: np.ndarray
        :return: Coefficients
        :rtype: np.ndarray
        """
        # Normalize columns to unit length
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0  # Avoid division by zero
        A_normalized = A / norms

        # Solve with Ridge Regression (Tikhonov regularization)
        alpha = 1e-10  # Small regularization strength
        B = np.vstack([A_normalized, np.sqrt(alpha) * np.eye(A_normalized.shape[1])])
        target_extended = np.concatenate([spectra, np.zeros(A_normalized.shape[1])])
        coef_normalized, _, _, _ = lstsq(B, target_extended)

        # Rescale coefficients to original units
        coefficients = coef_normalized / norms

        return coefficients

    def get_spectra_set(
        self, x, y, aperture_radius, annulus_outer_radius, annulus_inner_radius
    ):
        """Get the spectra set.

        :param x: x coordinate
        :type x: int
        :param y: y coordinate
        :type y: int
        :param aperture_radius: Aperture radius
        :type aperture_radius: int
        :param annulus_outer_radius: annulus outer radius
        :type annulus_outer_radius: int
        :param annulus_inner_radius: annulus inner radius
        :type annulus_inner_radius: int
        :return: spectra, noise, aperture_spectra, aperture_noise, annulus_spectra,
            annulus_noise
        :rtype: Tuple of np.ndarray
        """
        spectra = deepcopy(self._datacube[:, x, y])
        noise = deepcopy(self._noise_cube[:, x, y])

        # make circular mask around the pixel with radius s
        mask = np.zeros_like(self._datacube[0], dtype=bool)
        annulus_mask = np.zeros_like(self._datacube[0], dtype=bool)
        for i in range(x - 2 * aperture_radius, x + 2 * aperture_radius):
            for j in range(y - 2 * aperture_radius, y + 2 * aperture_radius):
                if (i - x) ** 2 + (j - y) ** 2 <= aperture_radius**2:
                    mask[i, j] = True

                if ((i - x) ** 2 + (j - y) ** 2 <= annulus_outer_radius**2) and (
                    (i - x) ** 2 + (j - y) ** 2 > annulus_inner_radius**2
                ):
                    annulus_mask[i, j] = True

        aperture_spectra = np.nansum(self._datacube[:, mask], axis=(1))
        aperture_noise = np.sqrt(np.nansum(self._noise_cube[:, mask] ** 2, axis=(1)))

        annulus_spectra = np.nansum(self._datacube[:, annulus_mask], axis=(1)) + 0
        annulus_noise = (
            np.sqrt(np.nansum(self._noise_cube[:, annulus_mask] ** 2, axis=(1))) + 0
        )

        return (
            spectra,
            noise,
            aperture_spectra,
            aperture_noise,
            annulus_spectra,
            annulus_noise,
        )

    def fit_wiggle_with_model_selection(
        self,
        x,
        y,
        aperture_radius=4,
        annulus_outer_radius=0,
        annulus_inner_radius=0,
        n_amplitude=10,
        n_frequency=1,
        min_n_amplitude=None,
        min_n_frequency=None,
        specified_noise_level=0,
        init_peak_detection_proximity_threshold=200,
        plot=False,
        selection_criteria="bic",
        min_selection_difference=None,
        extract_covariance=True,
        outlier_rejection_method=None,
        fdr_alpha=0.01,
        fdr_outlier_max_fraction=0.1,
        use_huber_loss=False,
        huber_delta=1.35,
        sigma_clip_sigma=5,
        sigma_clip_max_iterations=3,
        symmetric_sharpening=False,
        asymmetric_sharpening=False,
        fit_full_model=False,
    ):
        """Fit the wiggle signal with selecting amplitude polynomial order based on BIC.

        :param x: x coordinate
        :type x: int
        :param y: y coordinate
        :type y: int
        :param aperture_radius: Aperture radius
        :type aperture_radius: int
        :param annulus_outer_radius: Annulus outer radius
        :type annulus_outer_radius: int
        :param annulus_inner_radius: Annulus inner radius
        :type annulus_inner_radius: int
        :param n_amplitude: Maximum number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :param min_n_amplitude: Minimum number of amplitude parameters
        :type min_n_amplitude: int
        :param min_n_frequency: Minimum number of frequency parameters
        :type min_n_frequency: int
        :param specified_noise_level: Artificial noise level
        :type specified_noise_level: float
        :param init_peak_detection_proximity_threshold: Proximity lower limit in
            Angstrom for initial identifaction of peaks and troughs
        :type init_peak_detection_proximity_threshold: float
        :param plot: If True, plot the results
        :type plot: bool
        :param selection_criteria: Selection criteria, "bic" or "chi2"
        :type selection_criteria: str
        :param sigma_clip_sigma: Sigma clip threshold
        :type sigma_clip_sigma: float
        :param sigma_clip_max_iterations: Number of sigma clip iterations
        :type sigma_clip_max_iterations: int
        :param combine_bic_weighted: If True, combine the BIC weighted by the number of
            parameters
        :type combine_bic_weighted: bool
        :param extract_uncertainty: If True, extract the uncertainties
        :type extract_uncertainty: bool
        :param outlier_rejection_method: Outlier rejection method, "fdr" or
            "sigma_clip", set None to disable
        :type outlier_rejection_method: str
        :param huber_delta: Delta for Huber loss function
        :type huber_delta: float
        :param fdr_alpha: False discovery rate (FDR) correction threshold, smaller value
            will reject less outliers
        :type fdr_alpha: float
        :param fdr_outlier_max_fraction: Maximum fraction of outliers to reject using
            FDR
        :type fdr_outlier_max_fraction: float
        :param sigma_clip_sigma: Sigma threshold for sigma clipping
        :type sigma_clip_sigma: float
        :param sigma_clip_max_iterations: Number of sigma clip iterations
        :type sigma_clip_max_iterations: int
        :param extract_uncertainty: If True, extract the uncertainties
        :type extract_uncertainty: bool
        :param symmetric_sharpening: If True, use symmetric sharpening
        :type symmetric_sharpening: bool
        :param asymmetric_sharpening: If True, use asymmetric sharpening
        :type asymmetric_sharpening: bool
        :param selection_criteria: Selection criteria for model selection, "bic" or "chi2"
        :type selection_criteria: str
        :param min_selection_difference: Minimum difference between the best and the next best model selection criteria
        :type min_selection_difference: float
        :param verbose: If True, print the results
        :type verbose: bool
        :param plot: If True, plot the results
        :type plot: bool
        :return: Fitted parameters
        :rtype: np.ndarray
        """
        print(
            f"Computing {selection_criteria.upper()} for choices of n_amplitude and n_frequency..."
        )
        if min_n_amplitude is None:
            min_n_amplitude = n_amplitude
        elif min_n_amplitude < 2:
            raise ValueError("min_n_amplitude must be at least 2")

        if min_n_frequency is None:
            min_n_frequency = n_frequency
        elif min_n_frequency < 2:
            raise ValueError("min_n_frequency must be at least 2")

        best_metric = None
        for k in tqdm(range(min_n_frequency, n_frequency + 1)):
            for i in range(min_n_amplitude, n_amplitude + 1):
                result_params, cov_matrix, wiggle_signal, wiggle_noise = (
                    self.fit_wiggle(
                        x,
                        y,
                        aperture_radius=aperture_radius,
                        annulus_outer_radius=annulus_outer_radius,
                        annulus_inner_radius=annulus_inner_radius,
                        n_amplitude=i,
                        n_frequency=k,
                        specified_noise_level=specified_noise_level,
                        init_peak_detection_proximity_threshold=init_peak_detection_proximity_threshold,
                        plot=False,
                        extract_covariance=extract_covariance,
                        outlier_rejection_method=outlier_rejection_method,
                        use_huber_loss=use_huber_loss,
                        huber_delta=huber_delta,
                        fdr_alpha=fdr_alpha,
                        fdr_outlier_max_fraction=fdr_outlier_max_fraction,
                        sigma_clip_sigma=sigma_clip_sigma,
                        sigma_clip_max_iterations=sigma_clip_max_iterations,
                        symmetric_sharpening=symmetric_sharpening,
                        asymmetric_sharpening=asymmetric_sharpening,
                        fit_full_model=fit_full_model,
                        verbose=False,
                    )
                )

                fit_metric = self.get_model_selection_metric(
                    wiggle_signal,
                    wiggle_noise,
                    result_params,
                    selection_criteria=selection_criteria,
                )

                if min_selection_difference is None:
                    if selection_criteria == "bic":
                        min_selection_difference = 10
                    elif selection_criteria == "chi2":
                        min_selection_difference = 25

                if best_metric is None:
                    tqdm.write(
                        f"n_amplitude: {i}, n_frequency: {k}, {selection_criteria}: {fit_metric}"
                    )
                    best_n_amplitude = i
                    best_n_frequency = k
                    best_metric = fit_metric

                elif fit_metric < best_metric - min_selection_difference:
                    tqdm.write(
                        f"n_amplitude: {i}, n_frequency: {k}, {selection_criteria}: {fit_metric}"
                    )
                    best_n_amplitude = i
                    best_n_frequency = k
                    best_metric = fit_metric

        best_params, cov_matrix, wiggle_signal, wiggle_noise = self.fit_wiggle(
            x,
            y,
            aperture_radius=aperture_radius,
            annulus_outer_radius=annulus_outer_radius,
            annulus_inner_radius=annulus_inner_radius,
            n_amplitude=best_n_amplitude,
            n_frequency=best_n_frequency,
            specified_noise_level=specified_noise_level,
            init_peak_detection_proximity_threshold=init_peak_detection_proximity_threshold,
            extract_covariance=extract_covariance,
            outlier_rejection_method=outlier_rejection_method,
            use_huber_loss=use_huber_loss,
            huber_delta=huber_delta,
            fdr_alpha=fdr_alpha,
            fdr_outlier_max_fraction=fdr_outlier_max_fraction,
            sigma_clip_sigma=sigma_clip_sigma,
            sigma_clip_max_iterations=sigma_clip_max_iterations,
            symmetric_sharpening=symmetric_sharpening,
            asymmetric_sharpening=asymmetric_sharpening,
            plot=plot,
        )

        print("Best n_amplitude: ", best_n_amplitude)
        print("Best n_frequency: ", best_n_frequency)

        self._n_amplitude = best_n_amplitude
        self._n_frequency = best_n_frequency

        return best_params, cov_matrix

    def get_model_selection_metric(
        self, wiggle_signal, wiggle_noise, result_params, selection_criteria="bic"
    ):
        """
        Get the fit metric: BIC or chi^2.

        :param wiggle_signal: wiggle_signal
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle noise
        :type wiggle_noise: np.ndarray
        :param result_params: Fitted parameters
        :type result_params: np.ndarray
        :param selection_criteria: Selection criteria, "bic" or "chi2"
        :type selection_criteria: str
        :return: Fit metric
        :rtype: float
        """
        n_dof = np.sum(np.logical_or(self._gap_mask, self._outlier_mask))
        k = len(result_params)

        chi2 = self.cost_function(result_params, wiggle_signal, wiggle_noise)

        if selection_criteria == "bic":
            return chi2 + k * np.log(n_dof)
        elif selection_criteria == "chi2":
            return chi2
        else:
            raise ValueError(f"Invalid selection_criteria: {selection_criteria}")

    def is_wiggle_detected(
        self,
        wiggle_signal,
        wiggle_noise,
        result_params,
        sigma_threshold=5,
        variance_ratio_threshold=0.8,
        verbose=True,
    ):
        """Check if wiggle is detected.

        :param wiggle_signal: wiggle_signal
        :type wiggle_signal: np.ndarray
        :param wiggle_noise: wiggle_noise
        :type wiggle_noise: np.ndarray
        :param sigma_threshold: Sigma threshold
        :type sigma_threshold: float
        :param variance_ratio_threshold: Variance ratio threshold, set 0 to disable
        :type variance_ratio_threshold: float
        :param verbose: If True, print the results
        :type verbose: bool
        :return: True if wiggle is detected
        :rtype: bool
        """

        # residual for flat spectrum
        null_residual = wiggle_signal - np.ones_like(wiggle_signal)
        n_data = np.sum(self._gap_mask)

        total_noise = wiggle_noise

        chi2 = np.sum(
            (
                (null_residual**2 / total_noise**2) * self._gap_mask
            )  # * self._outlier_mask)
        )
        chi2_red = chi2 / n_data

        model = self.wiggle_model(result_params)
        model_residual = wiggle_signal - model
        chi2_model_red = (
            np.sum(
                model_residual**2
                / total_noise**2
                * self._gap_mask  # * self._outlier_mask
            )
            / n_data
        )

        sigma = np.sqrt(chi2_red - chi2_model_red)

        model_values = model[np.logical_and(self._gap_mask, self._outlier_mask)]
        residual_values = model_residual[
            np.logical_and(self._gap_mask, self._outlier_mask)
        ]
        model_variance = np.std(model_values)
        residual_variance = np.std(residual_values)

        if verbose:
            print(f"Wiggle detection sigma: {sigma:.2f}")
            print(
                f"Model variance: {model_variance:.4f}, residual variance: {residual_variance:.4f}, ratio: {model_variance / residual_variance:.4f}"
            )

        return (sigma > sigma_threshold) and (
            model_variance > variance_ratio_threshold * residual_variance
        )

    def clean_cube(
        self,
        wiggle_detection_sigma_threshold=3,
        wiggle_detection_variance_ratio_threshold=0.8,
        n_amplitude=10,
        n_frequency=1,
        min_n_amplitude=None,
        min_n_frequency=None,
        n_amplitude_for_detection=None,
        n_frequency_for_detection=None,
        specified_noise_level=0,
        cleaning_mask=None,
        init_peak_detection_proximity_threshold=200,
        aperture_radius=4,
        annulus_outer_radius=0,
        annulus_inner_radius=0,
        conserve_flux=True,
        outlier_rejection_method="fdr",
        fdr_alpha=0.01,
        fdr_outlier_max_fraction=0.1,
        use_huber_loss=False,
        huber_delta=1.35,
        sigma_clip_sigma=5,
        sigma_clip_max_iterations=5,
        extract_uncertainty=True,
        symmetric_sharpening=False,
        asymmetric_sharpening=False,
        selection_criteria="bic",
        min_selection_difference=None,
        num_samples_uncertainty_region=1000,
        verbose=True,
        plot=True,
        fit_full_model=True,
    ):
        """Clean the datacube.

        :param sigma_threshold: Sigma threshold
        :type sigma_threshold: float
        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :param min_n_amplitude: Minimum number of amplitude parameters
        :type min_n_amplitude: int
        :param min_n_frequency: Minimum number of frequency parameters
        :type min_n_frequency: int
        :param specified_noise_level: Artificial noise level in fraction of the spectrum
        :type specified_noise_level: float
        :param init_peak_detection_proximity_threshold: Proximity lower limit in Angstrom for initial identifaction of peaks and troughs
        :type init_peak_detection_proximity_threshold: float
        :param aperture_size: Aperture size. Spaxels at the edges with width less than this value will not be cleaned.
        :type aperture_size: int
        :param min_x: Minimum spaxel x to set lower limit for the cleaning process
        :type min_x: int
        :param max_x: Maximum spaxel x to set upper limit for the cleaning process
        :type max_x: int
        :param min_y: Minimum spaxel y to set lower limit for the cleaning process
        :type min_y: int
        :param max_y: Maximum spaxel y to set upper limit for the cleaning process
        :type max_y: int
        :param conserve_flux: If True, conserve flux in each spaxel
        :type conserve_flux: bool
        :param cleaning_mask: Mask for cleaning, if None, clean all spaxels
        :type cleaning_mask: np.ndarray
        :param outlier_rejection_method: Outlier rejection method, "fdr" or "sigma_clip", set None to disable
        :type outlier_rejection_method: str
        :param use_huber_loss: If True, use Huber loss function
        :type use_huber_loss: bool
        :param huber_delta: Delta for Huber loss function
        :type huber_delta: float
        :param fdr_alpha: False discovery rate (FDR) correction threshold, smaller value will reject less outliers
        :type fdr_alpha: float
        :param fdr_outlier_max_fraction: Maximum fraction of outliers to reject using FDR
        :type fdr_outlier_max_fraction: float
        :param sigma_clip_sigma: Sigma threshold for sigma clipping
        :type sigma_clip_sigma: float
        :param sigma_clip_max_iterations: Number of sigma clip iterations
        :type sigma_clip_max_iterations: int
        :param extract_uncertainty: If True, extract the uncertainties
        :type extract_uncertainty: bool
        :param symmetric_sharpening: If True, use symmetric sharpening
        :type symmetric_sharpening: bool
        :param asymmetric_sharpening: If True, use asymmetric sharpening
        :type asymmetric_sharpening: bool
        :param selection_criteria: Selection criteria for model selection, "bic" or "chi2"
        :type selection_criteria: str
        :param min_selection_difference: Minimum difference between the best and the next best model selection criteria
        :type min_selection_difference: float
        :param verbose: If True, print the results
        :type verbose: bool
        :param plot: If True, plot the results
        :type plot: bool
        :return: Cleaned datacube
        :rtype: np.ndarray
        :return: Cleaned datacube
        :rtype: np.ndarray
        """
        self.cleaned_datacube = np.copy(self._datacube)
        self.cleaned_noisecube = np.copy(self._noise_cube)

        if cleaning_mask is None:
            cleaning_mask = np.ones_like(self._datacube[0], dtype=bool)

        cleaned_mask = np.zeros_like(cleaning_mask, dtype=bool)

        n_amplitude, n_frequency = self.configure_polynomial_ns(
            n_amplitude, n_frequency
        )

        total_iterations = np.sum(cleaning_mask)
        with tqdm(total=total_iterations, desc="Cleaning spaxels") as pbar:
            for i in range(cleaning_mask.shape[0]):
                for j in range(cleaning_mask.shape[1]):
                    if not cleaning_mask[i, j]:
                        continue

                    if n_amplitude_for_detection is None:
                        n_amplitude_for_detection = n_amplitude
                    if n_frequency_for_detection is None:
                        n_frequency_for_detection = n_frequency

                    if verbose:
                        print("###########################")
                        print(f"Fitting spaxel: {i}, {j}")

                    result_params, cov_matrix, wiggle_signal, wiggle_noise = (
                        self.fit_wiggle(
                            i,
                            j,
                            aperture_radius=aperture_radius,
                            annulus_outer_radius=annulus_outer_radius,
                            annulus_inner_radius=annulus_inner_radius,
                            n_amplitude=n_amplitude_for_detection,
                            n_frequency=n_frequency_for_detection,
                            specified_noise_level=specified_noise_level,
                            init_peak_detection_proximity_threshold=init_peak_detection_proximity_threshold,
                            outlier_rejection_method=outlier_rejection_method,
                            use_huber_loss=use_huber_loss,
                            huber_delta=huber_delta,
                            fdr_alpha=fdr_alpha,
                            fdr_outlier_max_fraction=fdr_outlier_max_fraction,
                            extract_covariance=extract_uncertainty,
                            sigma_clip_sigma=sigma_clip_sigma,
                            sigma_clip_max_iterations=sigma_clip_max_iterations,
                            symmetric_sharpening=symmetric_sharpening,
                            asymmetric_sharpening=asymmetric_sharpening,
                            plot=False,
                            fit_full_model=False,  # do not fit the full model for detection
                        )
                    )

                    plt.show()

                    if self.is_wiggle_detected(
                        wiggle_signal,
                        wiggle_noise,
                        result_params,
                        sigma_threshold=wiggle_detection_sigma_threshold,
                        variance_ratio_threshold=wiggle_detection_variance_ratio_threshold,
                        verbose=verbose,
                    ):
                        if verbose:
                            print(f"Wiggle detected. Cleaning spaxel: {i}, {j}.")
                        cleaned_mask[i, j] = 1

                        if min_n_amplitude is None or min_n_frequency is None:
                            # performing model fitting with the specified n_amplitude and n_frequency,
                            # if they are not equal to n_amplitude_for_detection and n_frequency_for_detection
                            result_params, cov_matrix, _, _ = self.fit_wiggle(
                                i,
                                j,
                                aperture_radius=aperture_radius,
                                annulus_outer_radius=annulus_outer_radius,
                                annulus_inner_radius=annulus_inner_radius,
                                n_amplitude=n_amplitude,
                                n_frequency=n_frequency,
                                specified_noise_level=specified_noise_level,
                                init_peak_detection_proximity_threshold=init_peak_detection_proximity_threshold,
                                outlier_rejection_method=outlier_rejection_method,
                                use_huber_loss=use_huber_loss,
                                huber_delta=huber_delta,
                                fdr_alpha=fdr_alpha,
                                fdr_outlier_max_fraction=fdr_outlier_max_fraction,
                                extract_covariance=extract_uncertainty,
                                sigma_clip_sigma=sigma_clip_sigma,
                                sigma_clip_max_iterations=sigma_clip_max_iterations,
                                symmetric_sharpening=symmetric_sharpening,
                                asymmetric_sharpening=asymmetric_sharpening,
                                plot=plot,
                                fit_full_model=fit_full_model,
                            )
                        else:
                            # performing model selection as min_n_amplitude and min_n_frequency are set
                            result_params, cov_matrix = (
                                self.fit_wiggle_with_model_selection(
                                    i,
                                    j,
                                    aperture_radius=aperture_radius,
                                    annulus_outer_radius=annulus_outer_radius,
                                    annulus_inner_radius=annulus_inner_radius,
                                    n_amplitude=n_amplitude,
                                    n_frequency=n_frequency,
                                    min_n_amplitude=min_n_amplitude,
                                    min_n_frequency=min_n_frequency,
                                    specified_noise_level=specified_noise_level,
                                    init_peak_detection_proximity_threshold=init_peak_detection_proximity_threshold,
                                    outlier_rejection_method=outlier_rejection_method,
                                    use_huber_loss=use_huber_loss,
                                    huber_delta=huber_delta,
                                    fdr_alpha=fdr_alpha,
                                    fdr_outlier_max_fraction=fdr_outlier_max_fraction,
                                    extract_covariance=extract_uncertainty,
                                    sigma_clip_sigma=sigma_clip_sigma,
                                    sigma_clip_max_iterations=sigma_clip_max_iterations,
                                    symmetric_sharpening=symmetric_sharpening,
                                    asymmetric_sharpening=asymmetric_sharpening,
                                    plot=plot,
                                    fit_full_model=fit_full_model,
                                    selection_criteria=selection_criteria,
                                    min_selection_difference=min_selection_difference,
                                )
                            )

                        amplitude_params, frequency_params, phi = self.split_params(
                            result_params
                        )

                        wiggle_model = self.wiggle_func(
                            self.scaled_w,
                            amplitude_params,
                            frequency_params,
                            phi,
                        )

                        if cov_matrix is not None:
                            model_uncertainty = self.get_model_uncertainty(
                                result_params,
                                cov_matrix,
                                num_samples_uncertainty_region=num_samples_uncertainty_region,
                            )

                        integral = 1
                        integral_base = 1
                        if conserve_flux:
                            integral = np.trapezoid(
                                self._datacube[:, i, j] / wiggle_model,
                                self.scaled_w,
                            )
                            integral_base = np.trapezoid(
                                self._datacube[:, i, j], self.scaled_w
                            )

                        self.cleaned_datacube[:, i, j] = (
                            self._datacube[:, i, j]
                            / wiggle_model
                            / integral
                            * integral_base
                        )
                        self.cleaned_noisecube[:, i, j] = (
                            np.sqrt(
                                (self._noise_cube[:, i, j] / self._datacube[:, i, j])
                                ** 2
                                + (model_uncertainty / wiggle_model) ** 2
                            )
                            * self.cleaned_datacube[:, i, j]
                        )
                    else:
                        if verbose:
                            print(
                                f"No wiggle detected at spaxel: {i}, {j}, skipping..."
                            )

                    pbar.update(1)

        if verbose:
            print("Cleaning done!")

        return self.cleaned_datacube, self.cleaned_noisecube, cleaned_mask
