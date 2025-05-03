__author__ = "ajshajib"

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
from scipy.special import huber
from scipy.linalg import lstsq
from astropy.stats import sigma_clip
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

# from statsmodels.robust import mad
from tqdm.notebook import tqdm


from .util import Util

# from .util import polyval
# from .util import polyfit


class WiggleCleaner(object):

    def __init__(
        self,
        wavelengths,
        datacube,
        noise_cube,
        gaps=None,
        symmetric_sharpening=False,
        asymmetric_sharpening=False,
    ):
        """
        Initialize the WiggleCleaner object.

        :param wavelengths: Wavelengths
        :type wavelengths: list or np.ndarray
        :param datacube: 3D data cube containing the spectral data
        :type datacube: np.ndarray
        :param noise_cube: 3D data cube containing the noise associated with the spectral data
        :type noise_cube: np.ndarray
        :param gaps: Gaps
        :type gaps: list
        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :param symmetric_sharpening: If True, use sharpen symmetrically on both peaks and troughs
        :type symmetric_sharpening: bool
        :param asymmetric_sharpening: If True, use sharpen and smooth oppositely on peaks and troughs
        :type asymmetric_sharpening: bool
        :param use_huber_loss: If True, use Huber loss function
        :type use_huber_loss: bool
        :param huber_delta: Delta for Huber loss function
        :type huber_delta: float
        :param outlier_detection: Outlier detection method, "fdr" or "sigma_clip", set None to disable
        :type outlier_detection: str
        """
        self._wavelengths = np.array(wavelengths)
        self._datacube = datacube
        self._noise_cube = noise_cube
        self._gaps = np.array(gaps)
        self._n_amplitude = -1
        self._n_frequency = -1

        self._symmetric_sharpening = symmetric_sharpening
        self._asymmetric_sharpening = asymmetric_sharpening

        self._amplitude_spline = None
        self._frequency_spline = None

        self._include_scatter = False
        self._use_huber_loss = False
        self._huber_delta = 1.35

        self._outlier_rejection_method = None

        if gaps is None:
            self._gaps = []
            self._gap_mask = np.ones_like(self._wavelengths)
        else:
            self.set_gaps(gaps)

        self._outlier_mask = np.ones_like(self._wavelengths)

    def set_gaps(self, gaps):
        """
        Set the gaps to be ignored during the fitting process.

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
        """
        Get the symmetric sharpening flag.

        :return: Symmetric sharpening flag
        :rtype: bool
        """
        return self._symmetric_sharpening

    @symmetric_sharpening.setter
    def symmetric_sharpening(self, value):
        """
        Set the symmetric sharpening flag.

        :param value: Symmetric sharpening flag
        :type value: bool
        :return: None
        :rtype: None
        """
        self._symmetric_sharpening = value

    @property
    def asymmetric_sharpening(self):
        """
        Get the asymmetric sharpening flag.

        :return: Asymmetric sharpening flag
        :rtype: bool
        """
        return self._asymmetric_sharpening

    @asymmetric_sharpening.setter
    def asymmetric_sharpening(self, value):
        """
        Set the asymmetric sharpening flag.

        :param value: Asymmetric sharpening flag
        :type value: bool
        :return: None
        :rtype: None
        """
        self._asymmetric_sharpening = value

    @property
    def scaled_w(self):
        """
        Scaled wavelengths

        :return: Scaled wavelengths
        :rtype: np.ndarray
        """
        return self.scale_wavelengths_negative1_to_1(self._wavelengths)

    def wiggle_func(self, xs, amplitude_params, frequency_params, phi, k_1=0, k_2=0):
        """
        Get the wiggle function.

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
            + k_1 * (np.sin(frequency * xs + phi) ** 2)  # asymmetric sharpness
            + k_2 * np.sin(3 * (frequency * xs + phi))  # sharpness
        )

        return 1.0 + amplitude * wave_function

    def scale_wavelengths_negative1_to_1(self, w):
        """
        Scale the wavelengths to -1 to 1.

        :param w: Wavelengths
        :type w: np.ndarray
        :return: Scaled wavelengths
        :rtype: np.ndarray
        """
        return (w - self._wavelengths[0]) / (
            self._wavelengths[-1] - self._wavelengths[0]
        ) * 2 - 1

    def scale_wavelengths_to_0_1(self, w):
        """
        Scale the wavelengths to 0 to 1.

        :param w: Wavelengths
        :type w: np.ndarray
        :return: Scaled wavelengths
        :rtype: np.ndarray
        """
        return (w - self._wavelengths[0]) / (
            self._wavelengths[-1] - self._wavelengths[0]
        )

    def model(self, params):
        """
        Get the wiggle model given the parameters.

        :param params: Parameters
        :type params: np.ndarray
        :return: Model
        :rtype: np.ndarray
        """
        n_amplitude, n_frequency = self.configure_polynomial_ns()

        amplitude_params, frequency_params, phi_0 = self.split_params(
            params, n_amplitude, n_frequency
        )

        if self._asymmetric_sharpening and not self._symmetric_sharpening:
            k_1 = params[-1]
            k_2 = 0
        elif self._symmetric_sharpening and not self._asymmetric_sharpening:
            k_1 = 0
            k_2 = params[-1]
        elif self._symmetric_sharpening and self._asymmetric_sharpening:
            k_1 = params[-2]
            k_2 = params[-1]
        else:
            k_1 = 0
            k_2 = 0

        model = self.wiggle_func(
            self.scaled_w,
            amplitude_params,
            frequency_params,
            phi_0,
            k_1=k_1,
            k_2=k_2,
        )
        return model

    def split_params(self, params, n_amplitude=None, n_frequency=None):
        """
        Split the parameters. Opposite of the set_params function.

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
        """
        Set the parameters. Opposite function of the split_params function.

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
        """
        Configure the number of parameters.

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

    def residual_vector(self, params, curve, noise):
        """ "
        Get the residual vector.

        :param params: Parameters
        :type params: np.ndarray
        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :return: Residual vector
        :rtype: np.ndarray
        """
        model = self.model(params)

        residual = (model - curve) / noise
        residual = residual * self._gap_mask * self._outlier_mask

        if self._use_huber_loss:
            huber_loss = huber(self._huber_delta, residual)
            residual = np.sqrt(np.abs(2 * huber_loss)) * np.sign(residual)

        return residual

    def cost_function(self, params, curve, noise):
        """
        Get the cost.

        :param params: Parameters
        :type params: np.ndarray
        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :return: cost
        :rtype: float
        """
        residual = self.residual_vector(params, curve, noise)
        cost = np.sum(residual**2)

        return cost

    def get_residual_func(self, curve, noise):
        """
        Get the residual function.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :return: Residual function
        :rtype: Callable
        """

        def residual_func(params):
            return self.residual_vector(params, curve, noise)

        return residual_func

    def get_residual_func_phase_only(self, init_params, curve, noise):
        """
        Get the residual function with phase only.

        :param params: Parameters
        :type params: np.ndarray
        :param init_params: Initial parameters
        :type init_params: np.ndarray
        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
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
            return self.residual_vector(new_params, curve, noise)

        return residual_func

    def fit_curve(
        self,
        curve,
        noise=None,
        n_amplitude=None,
        n_frequency=None,
        specified_noise_level=0.005,
        proximity_threshold=200,
        do_interim_fit_phase_only=False,
        include_scatter=True,
        extract_covariance=True,
        outlier_rejection_method=None,
        use_huber_loss=False,
        huber_delta=1.35,
        fdr_alpha=0.01,
        fdr_outlier_max_fraction=0.3,
        sigma_clip=5,
        sigma_clip_max_iterations=5,
        plot=False,
        verbose=False,
    ):
        """
        Fit the curve.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :param specified_noise_level: User-defined noise level to be used instead of the actual noise. Set to 0 to disable.
        :type specified_noise_level: float
        :param proximity_threshold: Proximity lower limit in Angstrom for initial identifaction of peaks and troughs
        :type proximity_threshold: float
        :param plot: If True, plot the results
        :type plot: bool
        :param verbose: If True, print the results
        :type verbose: bool
        :param do_interim_fit_phase_only: If True, do an interim fit with phase only
        :type do_interim_fit_phase_only: bool
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
        :param sigma_clip: Sigma clip threshold
        :type sigma_clip: float
        :param sigma_clip_max_iterations: Number of sigma clip iterations
        :type sigma_clip_max_iterations: int
        :param extract_uncertainty: If True, extract the uncertainties
        :type extract_uncertainty: bool
        :return: Fitted parameters
        :rtype: np.ndarray
        """
        self._outlier_mask = np.ones_like(self._wavelengths)

        n_amplitude, n_frequency = self.configure_polynomial_ns(
            n_amplitude, n_frequency
        )

        noise = self.configure_noise(curve, noise, specified_noise_level)

        amplitude_spline, frequency_spline, init_phi_0 = Util.get_init_params_spline(
            curve,
            self.scaled_w,
            n_amplitude=n_amplitude,
            n_frequency=n_frequency,
            proximity_threshold=proximity_threshold
            / np.mean(np.diff(self._wavelengths)),
            plot=False,
        )

        self._amplitude_spline = deepcopy(amplitude_spline)
        init_amplitude_params = deepcopy(amplitude_spline.c)
        self._frequency_spline = deepcopy(frequency_spline)
        init_frequency_params = deepcopy(frequency_spline.c)

        curve = np.array(curve)
        noise = np.array(noise)

        x0 = self.set_params(
            init_amplitude_params,
            init_frequency_params,
            init_phi_0,
            n_amplitude=n_amplitude,
            n_frequency=n_frequency,
        )

        if do_interim_fit_phase_only:
            result = least_squares(
                self.get_residual_func_phase_only(x0, curve, noise),
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

        self._include_scatter = True
        self._outlier_rejection_method = outlier_rejection_method
        self._use_huber_loss = use_huber_loss
        self._huber_delta = huber_delta

        is_turn_off_huber_loss = False
        if outlier_rejection_method == "fdr":
            if not self._use_huber_loss:
                is_turn_off_huber_loss = True

            self._use_huber_loss = True

        result = least_squares(self.get_residual_func(curve, noise), x0)

        if self._outlier_rejection_method is not None:
            residual = self.residual_vector(result.x, curve, noise)

            clipped_pixels = self.reject_outliers(
                residual,
                num_params=len(result.x),
                fdr_alpha=fdr_alpha,
                fdr_outlier_max_fraction=fdr_outlier_max_fraction,
                sigma=sigma_clip,
                sigma_clip_max_iterations=sigma_clip_max_iterations,
            )

            self._outlier_mask[clipped_pixels] = 0

            if is_turn_off_huber_loss:
                self._use_huber_loss = False

            result = least_squares(
                self.get_residual_func(curve, noise),
                result.x,
            )

        # for i in range(sigma_clip_iterations):
        #     residual = np.abs(self.residual_vector(result.x, curve, noise))
        #     # Keep the top sigma_clip_fraction fraction of the residuals
        #     residuals = residual[residual > sigma_clip]
        #     if len(residuals) == 0:
        #         break
        #     threshold = np.percentile(residuals, 100 * (1 - sigma_clip_fraction))
        #     clipped_pixels = residual > threshold
        #     self._outlier_mask[clipped_pixels] = 0

        #     result = least_squares(
        #         self.get_residual_func(curve, noise),
        #         result.x,
        #     )

        result_params = result.x

        if extract_covariance:
            residuals = result.fun
            jacobian = result.jac

            # Get number of observations (m) and parameters (n)
            m, n = jacobian.shape

            # Check degrees of freedom
            if m <= n:
                raise ValueError(
                    "Number of observations must exceed number of parameters to estimate uncertainty."
                )

            # Calculate residual sum of squares and variance estimate
            sum_of_squared_residuals = np.sum(residuals**2)
            dof = m - n
            sigma_squared = sum_of_squared_residuals / dof

            # Compute covariance matrix using pseudoinverse for stability
            cov_matrix = sigma_squared * np.linalg.pinv(jacobian.T @ jacobian)
        else:
            cov_matrix = None

        if verbose:
            print("Cost: ", self.cost_function(result_params, curve, noise))

        if plot:
            self.plot_model(
                curve,
                noise,
                result_params,
                cov_matrix=cov_matrix,
            )

        return result_params, cov_matrix

    def reject_outliers(
        self,
        residual,
        num_params=0,
        fdr_alpha=0.01,
        fdr_outlier_max_fraction=0.3,
        sigma=5,
        sigma_clip_max_iterations=5,
    ):
        """
        Reject outliers using the selected method.

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
        :param sigma_clip_max_iterations: Maximum number of iterations for sigma clipping
        :type sigma_clip_max_iterations: int
        :return: Indices of outliers
        :rtype: np.ndarray
        """
        outlier_mask = np.zeros_like(residual, dtype=bool)

        if self._outlier_rejection_method == "sigma_clip":
            clipped = sigma_clip(
                residual,
                sigma=sigma,
                maxiters=sigma_clip_max_iterations,
                masked=True,
            )
            outlier_mask = clipped.mask
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
            sorted_indices = np.argsort(corrected_p)
            top_indices = sorted_indices[:max_reject]

            outlier_mask = np.zeros_like(residual, dtype=bool)
            outlier_mask[top_indices] = reject[top_indices]
        else:
            raise ValueError(
                f"Unrecognized outlier rejection_method: {self._outlier_rejection_method}"
            )

        return outlier_mask

    def configure_noise(self, curve, noise, specified_noise_level):
        """
        Configure the noise.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :param specified_noise_level: User-defined noise level to be used instead of the actual noise. Set to 0 to disable.
        :type specified_noise_level: float
        :return: Noise
        :rtype: np.ndarray
        """
        if noise is None and specified_noise_level == 0:
            raise ValueError(
                "Noise level not set! Either provide the noise or set the specified_noise_level."
            )
        if specified_noise_level > 0:
            noise = np.ones_like(curve) * specified_noise_level
        return noise

    def plot_model(
        self,
        curve,
        noise,
        result_params,
        cov_matrix=None,
        num_samples_uncertainty_region=1000,
    ):
        """
        Plot the model.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :param result_params: Fitted parameters
        :type result_params: np.ndarray
        :param cov_matrix: Covariance matrix
        :type cov_matrix: np.ndarray
        :param num_samples_uncertainty_region: Number of samples for uncertainty region
        :type num_samples_uncertainty_region: int
        :return: None
        :rtype: None
        """
        red = "#e41a1c"
        blue = "#377eb8"
        green = "#4daf4a"
        purple = "#984ea3"
        orange = "#ff7f00"
        grey = "#999999"

        plt.errorbar(
            self._wavelengths[(self._outlier_mask == 1) & (self._gap_mask == 1)],
            curve[(self._outlier_mask == 1) & (self._gap_mask == 1)],
            yerr=noise[(self._outlier_mask == 1) & (self._gap_mask == 1)],
            label="Fitted points",
            ls="None",
            marker="o",
            markersize=3,
            alpha=0.3,
            c=blue,
        )
        plt.errorbar(
            self._wavelengths[(self._outlier_mask == 0) | (self._gap_mask == 0)],
            curve[(self._outlier_mask == 0) | (self._gap_mask == 0)],
            yerr=noise[(self._outlier_mask == 0) | (self._gap_mask == 0)],
            label="Rejected outliers",
            ls="None",
            marker="o",
            markersize=3,
            alpha=0.4,
            c=grey,
        )

        model = self.model(result_params)
        plt.plot(self._wavelengths, model, label="Model", lw=1, c=orange)

        if cov_matrix is not None:
            models = []
            for i in range(num_samples_uncertainty_region):
                sampled_params = np.random.multivariate_normal(
                    result_params, cov_matrix, 1
                )[0]
                models.append(self.model(sampled_params))

            models = np.array(models)

            model_up, model_down = np.percentile(models, [16, 84], axis=0)
            model_uncertainty = (model_up - model_down) / 2
            plt.fill_between(
                self._wavelengths,
                model - model_uncertainty,
                model + model_uncertainty,
                color=orange,
                alpha=0.4,
            )

        for g in self._gaps:
            plt.axvspan(g[0], g[1], color="black", alpha=0.1)

        # if x0 is not None:
        #     plt.plot(
        #         self._wavelengths,
        #         self.model(x0),
        #         ls=":",
        #         label="Init",
        #         c=red,
        #     )

        plt.xlabel("Wavelengths")
        plt.ylabel("Modulation curve")
        plt.legend()
        plt.ylim(np.min(curve) * 0.9, np.max(curve) * 1.1)
        plt.show()

    def get_modulation_curve(
        self,
        x,
        y,
        aperture_size=4,
    ):
        """
        Get the modulation curve.

        :param spaxel_x: Spaxel x
        :type spaxel_x: int
        :param spaxel_y: Spaxel y
        :type spaxel_y: int
        :param aperture: aperture size to sum the spectra to average out the wiggles
        :type aperture: int
        :return: Modulation curve and noise
        :rtype: Tuple of np.ndarray
        """
        spectra = self._datacube[:, x, y]
        noise = self._noise_cube[:, x, y]
        wavelengths = self.scale_wavelengths_negative1_to_1(self._wavelengths)

        # make circular mask around the pixel with radius s
        mask = np.zeros_like(self._datacube[0], dtype=bool)

        for i in range(x - 2 * aperture_size, x + 2 * aperture_size):
            for j in range(y - 2 * aperture_size, y + 2 * aperture_size):
                if (i - x) ** 2 + (j - y) ** 2 <= aperture_size**2:
                    mask[i, j] = True

        aperture_spectra = np.nansum(self._datacube[:, mask], axis=(1))
        aperture_noise = np.sqrt(np.nansum(self._noise_cube[:, mask] ** 2, axis=(1)))

        # fit c_1 * aperture_spectra + c_2 * shell_spectra + c_3 * wavelengths**a + (c_4 * wavelengths**2 + c_5 * wavelengths + c_6)
        # given non-linear parameter a, treat all c_1 parameters as linear parameters and derive them using linear inversion

        def model(a):
            # Construct the design matrix for the current 'a'
            A = np.column_stack(
                [
                    aperture_spectra,
                    self.scale_wavelengths_to_0_1(self._wavelengths) ** a,
                    wavelengths**2,
                    wavelengths,
                    np.ones_like(wavelengths),
                ]
            )

            # Solve the linear least squares problem
            coefficients, _, _, _ = lstsq(A, spectra)

            return A @ coefficients, coefficients

        def residual(a):
            model_spectra, _ = model(a)
            return np.sum((model_spectra - spectra) ** 2)

        result = minimize_scalar(residual, bounds=[0, 6], method="bounded")
        best_model, _ = model(result.x)

        model_noise = aperture_noise / aperture_spectra * best_model

        curve = spectra / best_model
        curve_noise = (
            np.sqrt((noise / spectra) ** 2 + (model_noise / best_model) ** 2) * curve
        )

        # replace non-positive noise with minimum non-negative value
        min_positive_noise = np.nanmin(curve_noise[curve_noise > 0])
        curve_noise[curve_noise <= 0] = min_positive_noise

        # # normalize the curve
        # median = np.nanmedian(curve)
        # curve /= median
        # curve_noise /= median

        return curve, curve_noise

    def fit_curve_with_model_selection(
        self,
        curve,
        noise=None,
        n_amplitude=10,
        n_frequency=1,
        min_n_amplitude=None,
        min_n_frequency=None,
        specified_noise_level=0.005,
        proximity_threshold=200,
        plot=False,
        selection_criteria="bic",
        extract_uncertainty=True,
        outlier_rejection_method=None,
        huber_delta=1.35,
        sigma_clip=5,
        sigma_clip_max_iterations=3,
    ):
        """
        Fit the curve with selecting amplitude polynomial order based on BIC.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
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
        :param proximity_threshold: Proximity lower limit in Angstrom for initial identifaction of peaks and troughs
        :type proximity_threshold: float
        :param plot: If True, plot the results
        :type plot: bool
        :param selection_criteria: Selection criteria, "bic" or "chi2"
        :type selection_criteria: str
        :param sigma_clip: Sigma clip threshold
        :type sigma_clip: float
        :param sigma_clip_max_iterations: Number of sigma clip iterations
        :type sigma_clip_max_iterations: int
        :param combine_bic_weighted: If True, combine the BIC weighted by the number of parameters
        :type combine_bic_weighted: bool
        :param extract_uncertainty: If True, extract the uncertainties
        :type extract_uncertainty: bool
        :param outlier_rejection_method: Outlier rejection method, "fdr" or "sigma_clip", set None to disable
        :type outlier_rejection_method: str
        :param huber_delta: Delta for Huber loss function
        :type huber_delta: float
        :return: Fitted parameters
        :rtype: np.ndarray
        """
        print(
            f"Computing {selection_criteria} for choices of n_amplitude and n_frequency..."
        )
        if min_n_amplitude is None:
            min_n_amplitude = n_amplitude
        elif min_n_amplitude < 2:
            raise ValueError("min_n_amplitude must be at least 2")

        if min_n_frequency is None:
            min_n_frequency = n_frequency
        elif min_n_frequency < 2:
            raise ValueError("min_n_frequency must be at least 2")

        noise = self.configure_noise(curve, noise, specified_noise_level)

        best_metric = None
        for i in tqdm(range(n_amplitude, min_n_amplitude - 1, -1)):
            for k in range(n_frequency, min_n_frequency - 1, -1):
                result_params, cov_matrix = self.fit_curve(
                    curve,
                    noise,
                    n_amplitude=i,
                    n_frequency=k,
                    specified_noise_level=specified_noise_level,
                    proximity_threshold=proximity_threshold,
                    plot=False,
                    outlier_rejection_method=outlier_rejection_method,
                    huber_delta=huber_delta,
                    sigma_clip=sigma_clip,
                    sigma_clip_max_iterations=sigma_clip_max_iterations,
                    extract_covariance=extract_uncertainty,
                )

                fit_metric = self.get_model_selection_metric(
                    curve,
                    noise,
                    result_params,
                    selection_criteria=selection_criteria,
                )

                if best_metric is None:
                    tqdm.write(
                        f"n_amplitude: {i}, n_frequency: {k}, {selection_criteria}: {fit_metric}"
                    )
                    best_n_amplitude = i
                    best_n_frequency = k
                    best_metric = fit_metric

                elif fit_metric < best_metric:
                    tqdm.write(
                        f"n_amplitude: {i}, n_frequency: {k}, {selection_criteria}: {fit_metric}"
                    )
                    best_n_amplitude = i
                    best_n_frequency = k
                    best_metric = fit_metric

        best_params, cov_matrix = self.fit_curve(
            curve,
            noise,
            n_amplitude=best_n_amplitude,
            n_frequency=best_n_frequency,
            specified_noise_level=specified_noise_level,
            proximity_threshold=proximity_threshold,
            plot=False,
            outlier_rejection_method=outlier_rejection_method,
            huber_delta=huber_delta,
            sigma_clip=sigma_clip,
            sigma_clip_max_iterations=sigma_clip_max_iterations,
            extract_covariance=extract_uncertainty,
        )

        print("Best n_amplitude: ", best_n_amplitude)
        print("Best n_frequency: ", best_n_frequency)

        self._n_amplitude = best_n_amplitude
        self._n_frequency = best_n_frequency

        if plot:
            self.plot_model(
                curve,
                noise,
                best_params,
                cov_matrix=cov_matrix,
            )

        return best_params, cov_matrix

    def get_model_selection_metric(
        self, curve, noise, result_params, selection_criteria="bic"
    ):
        """
        Get the fit metric: BIC or chi^2.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :param result_params: Fitted parameters
        :type result_params: np.ndarray
        :param selection_criteria: Selection criteria, "bic" or "chi2"
        :type selection_criteria: str
        :return: Fit metric
        :rtype: float
        """
        n_dof = np.sum(self._gap_mask)
        k = len(result_params)

        chi2 = self.cost_function(result_params, curve, noise)

        if selection_criteria == "bic":
            return chi2 + k * np.log(n_dof)
        elif selection_criteria == "chi2":
            return chi2
        else:
            raise ValueError(f"Invalid selection_criteria: {selection_criteria}")

    def is_wiggle_detected(self, curve, noise, result_params, sigma_threshold=5):
        """
        Check if wiggle is detected.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :param sigma_threshold: Sigma threshold
        :type sigma_threshold: float
        :return: True if wiggle is detected
        :rtype: bool
        """
        residual = curve - np.ones_like(curve)
        n_data = len(curve)

        p = np.percentile(np.abs(residual), 97.5)
        indices = np.abs(residual) < p
        chi2 = np.sum(((residual**2 / noise**2) * self._gap_mask)[indices])
        chi2_red = chi2 / n_data

        residual = self.residual_vector(result_params, curve, noise)
        chi2_model_red = np.sum(residual[indices] ** 2) / n_data

        sigma = np.sqrt(chi2_red - chi2_model_red)

        return sigma > sigma_threshold

    def clean_cube(
        self,
        sigma_threshold=5,
        n_amplitude=10,
        n_frequency=1,
        min_n_amplitude=None,
        min_n_frequency=None,
        specified_noise_level=0.005,
        proximity_threshold=200,
        aperture_size=4,
        min_x=None,
        max_x=None,
        min_y=None,
        max_y=None,
        conserve_flux=True,
        verbose=True,
        plot=True,
    ):
        """
        Clean the datacube.

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
        :param specified_noise_level: Artificial noise level
        :type specified_noise_level: float
        :param proximity_threshold: Proximity lower limit in Angstrom for initial identifaction of peaks and troughs
        :type proximity_threshold: float
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

        n_amplitude, n_frequency = self.configure_polynomial_ns(
            n_amplitude, n_frequency
        )

        if min_x is None:
            min_x = aperture_size
        if max_x is None:
            max_x = self.cleaned_datacube.shape[1] - aperture_size
        if min_y is None:
            min_y = aperture_size
        if max_y is None:
            max_y = self.cleaned_datacube.shape[2] - aperture_size

        total_iterations = (max_x - min_x) * (max_y - min_y)
        with tqdm(total=total_iterations, desc="Cleaning spaxels") as pbar:
            for i in range(min_x, max_x):
                for j in range(min_y, max_y):
                    curve, noise = self.get_modulation_curve(
                        i, j, aperture_size=aperture_size
                    )

                    result_params = self.fit_curve(
                        curve,
                        noise,
                        n_amplitude=5,
                        n_frequency=3,
                        specified_noise_level=specified_noise_level,
                        proximity_threshold=proximity_threshold,
                        plot=False,
                    )

                    if self.is_wiggle_detected(
                        curve,
                        noise,
                        result_params,
                        sigma_threshold=sigma_threshold,
                    ):
                        print(f"Wiggle detected. Cleaning spaxel: {i}, {j}.")

                        if min_n_amplitude is None or min_n_frequency is None:
                            result_params = self.fit_curve(
                                curve,
                                noise,
                                n_amplitude=n_amplitude,
                                n_frequency=n_frequency,
                                specified_noise_level=specified_noise_level,
                                proximity_threshold=proximity_threshold,
                                plot=plot,
                            )
                        else:
                            result_params = self.fit_curve_with_best_bic(
                                curve,
                                noise,
                                n_amplitude=n_amplitude,
                                n_frequency=n_frequency,
                                min_n_amplitude=min_n_amplitude,
                                min_n_frequency=min_n_frequency,
                                specified_noise_level=specified_noise_level,
                                proximity_threshold=proximity_threshold,
                                plot=plot,
                            )

                        amplitude_params, frequency_params, _, phi = self.split_params(
                            result_params
                        )

                        wave_model = self.wiggle_func(
                            self.scaled_w,
                            amplitude_params,
                            frequency_params,
                            phi,
                        )
                        integral = 1
                        integral_base = 1
                        if conserve_flux:
                            integral = np.trapz(
                                self._datacube[:, i, j] / wave_model,
                                self.scaled_w,
                            )
                            integral_base = np.trapz(
                                self._datacube[:, i, j], self.scaled_w
                            )

                        self.cleaned_datacube[:, i, j] = (
                            self._datacube[:, i, j]
                            / wave_model
                            / integral
                            * integral_base
                        )
                        self.cleaned_noisecube[:, i, j] = (
                            self._noise_cube[:, i, j]
                            / wave_model
                            / integral
                            * integral_base
                        )

                        if plot:
                            plt.plot(
                                self._wavelengths,
                                self._datacube[:, i, j],
                                label="Input",
                            )
                            plt.plot(
                                self._wavelengths,
                                self.cleaned_datacube[:, i, j],
                                label="Cleaned",
                            )
                            plt.xlabel("Wavelengths")
                            plt.ylabel("Flux")
                            plt.title(f"Spaxel {i}, {j}")
                            plt.legend()
                            plt.show()
                        # else:
                        #     if verbose:
                        #         print(f"No wiggle detected at spaxel: {i}, {j}, skipping...")

                    pbar.update(1)

        if verbose:
            print("Cleaning done!")

        return self.cleaned_datacube, self.cleaned_noisecube
