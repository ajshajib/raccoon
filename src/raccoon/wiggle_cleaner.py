__author__ = "ajshajib"

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm

from .util import Util
from .util import polyval
from .util import polyfit


class WiggleCleaner(object):

    def __init__(
        self,
        wavelengths,
        datacube,
        noise_cube,
        gaps,
        n_amplitude=9,
        n_frequency=9,
        n_offset=9,
        symmetric_sharpenning=False,
        asymmetric_sharpenning=False,
    ):
        """
        Initialize the WiggleCleaner object.

        :param wavelengths: Wavelengths
        :type wavelengths: list or np.ndarray
        :param gaps: Gaps
        :type gaps: list
        :param n_amplitude: Number of amplitude parameters
        :type n_amplitude: int
        :param n_frequency: Number of frequency parameters
        :type n_frequency: int
        :param n_offset: Number of offset parameters
        :type n_offset: int
        """
        self._wavelengths = np.array(wavelengths)
        self._datacube = datacube
        self._noise_cube = noise_cube
        self._gaps = np.array(gaps)
        self._n_amplitude = n_amplitude
        self._n_frequency = n_frequency
        self._n_offset = n_offset

        self._symmetric_sharpenning = symmetric_sharpenning
        self._asymmetric_sharpenning = asymmetric_sharpenning

        gap_mask = np.ones_like(wavelengths)
        for g in self._gaps:
            mask = (wavelengths > g[0]) & (wavelengths < g[1])
            gap_mask[mask] = 0
        self._gap_mask = np.array(gap_mask)

    @property
    def scaled_w(self):
        """
        Scaled wavelengths

        :return: Scaled wavelengths
        :rtype: np.ndarray
        """
        return self.scale_wavelengths_to_1_m1(self._wavelengths)

    def wiggle_func(
        self, xs, frequency_params, amplitude_params, offset_params, phi, k_1=0, k_2=0
    ):
        """
        Get the wiggle function.

        :param xs: Scaled wavelengths
        :type xs: np.ndarray
        :param frequency_params: Frequency parameters
        :type frequency_params: np.ndarray
        :param amplitude_params: Amplitude parameters
        :type amplitude_params: np.ndarray
        :param offset_params: Offset parameters
        :type offset_params: np.ndarray
        :param phi: Phase
        :type phi: float
        :return: Wiggle function
        :rtype: np.ndarray
        """
        frequency = polyval(frequency_params, xs)
        amplitude = np.abs(polyval(amplitude_params, xs))
        offset = polyval(offset_params, xs)

        wave_function = (
            np.sin(frequency * xs + phi)
            + k_1 * (np.sin(frequency * xs + phi) ** 2)  # asymmetric sharpness
            + k_2 * np.sin(3 * (frequency * xs + phi))  # sharpness
        )

        return 1.0 + amplitude * wave_function + offset

    def scale_wavelengths_to_1_m1(self, w):
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

    def model(self, params):
        """
        Get the wiggle model given the parameters.

        :param params: Parameters
        :type params: np.ndarray
        :return: Model
        :rtype: np.ndarray
        """
        n_amplitude = self._n_amplitude
        n_frequency = self._n_frequency
        n_offset = self._n_offset

        frequency_params = params[: n_frequency + 1]
        phi_0 = params[n_frequency + 1]
        amplitude_params = params[n_frequency + 2 : n_amplitude + n_frequency + 3]
        offset_params = params[
            n_amplitude + n_frequency + 3 : n_amplitude + n_frequency + n_offset + 4
        ]
        if self._asymmetric_sharpenning and not self._asymmetric_sharpenning:
            k_1 = params[-1]
            k_2 = 0
        elif self._symmetric_sharpenning and not self._asymmetric_sharpenning:
            k_1 = 0
            k_2 = params[-1]
        elif self._symmetric_sharpenning and self._asymmetric_sharpenning:
            k_1 = params[-2]
            k_2 = params[-1]
        else:
            k_1 = 0
            k_2 = 0

        model = self.wiggle_func(
            self.scaled_w,
            frequency_params,
            amplitude_params,
            offset_params,
            phi_0,
            k_1=k_1,
            k_2=k_2,
        )
        return model

    def loss_vector(self, params, curve, noise=None):
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
        residual = residual * self._gap_mask

        return residual

    def loss(self, params, curve, noise):
        """
        Get the loss.

        :param params: Parameters
        :type params: np.ndarray
        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :return: Loss
        :rtype: float
        """
        return np.sum(self.loss_vector(params, curve, noise) ** 2)

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
            return self.loss_vector(params, curve, noise)

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

        def residual_func(params):
            new_params = np.concatenate([params, init_params[self._n_frequency + 2 :]])
            return self.loss_vector(new_params, curve, noise)

        return residual_func

    def fit_curve(
        self,
        curve,
        noise=None,
        n_amplitude=None,
        n_frequency=None,
        n_offset=None,
        artificial_noise_level=0.005,
        proximity_threshold=200,
        plot=False,
        plot_amplitude_offset=False,
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
        :param n_offset: Number of offset parameters
        :type n_offset: int
        :param artificial_noise_level: Artificial noise level. Set to 0 to disable.
        :type artificial_noise_level: float
        :param proximity_threshold: Proximity lower limit in Angstrom for initial identifaction of peaks and troughs
        :type proximity_threshold: float
        :param plot: If True, plot the results
        :type plot: bool
        :param plot_amplitude_offset: If True, plot the amplitude and offset
        :type plot_amplitude_offset: bool
        :param verbose: If True, print the results
        :type verbose: bool
        :return: Fitted parameters
        :rtype: np.ndarray
        """
        if n_frequency is None:
            n_frequency = self._n_frequency
        else:
            self._n_frequency = n_frequency

        if n_amplitude is None:
            n_amplitude = self._n_amplitude
        else:
            self._n_amplitude = n_amplitude

        if n_offset is None:
            n_offset = self._n_offset
        else:
            self._n_offset = n_offset

        if artificial_noise_level > 0:
            noise = np.ones_like(curve) * artificial_noise_level

        init_frequency_params, init_amplitude_params, init_offset_params, init_phi = (
            Util.get_init_params(
                curve,
                self.scaled_w,
                n_amplitude=n_amplitude,
                n_offset=n_offset,
                n_frequency=n_frequency,
                proximity_threshold=proximity_threshold
                / np.mean(np.diff(self._wavelengths)),
            )
        )

        curve = np.array(curve)
        noise = np.array(noise)

        x0 = np.concatenate(
            [
                init_frequency_params,
                np.array([init_phi]),
                init_amplitude_params,
                init_offset_params,
            ]
        )
        if self._symmetric_sharpenning and self._asymmetric_sharpenning:
            x0 = np.concatenate([x0, np.array([0, 0])])
        elif self._symmetric_sharpenning or self._asymmetric_sharpenning:
            x0 = np.concatenate([x0, np.array([0])])

        res = least_squares(
            self.get_residual_func_phase_only(x0, curve, noise), x0[: n_frequency + 2]
        )
        res_params = res.x

        x0[: n_frequency + 2] = res_params
        res = least_squares(self.get_residual_func(curve, noise), x0)
        res_params = res.x

        if verbose:
            print("Loss: ", self.loss(res_params, curve, noise))

        if plot:
            self.plot_model(
                curve,
                noise,
                n_amplitude,
                n_frequency,
                n_offset,
                res_params,
                plot_amplitude_offset,
                x0,
            )

        return res_params

    def plot_model(
        self,
        curve,
        noise,
        n_amplitude,
        n_frequency,
        n_offset,
        res_params,
        plot_amplitude_offset=False,
        x0=None,
    ):
        red = "#e41a1c"
        blue = "#377eb8"
        green = "#4daf4a"
        purple = "#984ea3"
        orange = "#ff7f00"
        yellow = "#ffff33"

        plt.errorbar(
            self._wavelengths,
            curve,
            yerr=noise,
            label="Input",
            ls="None",
            marker="o",
            markersize=2,
            alpha=0.2,
            c=blue,
        )
        plt.plot(
            self._wavelengths,
            curve,
            label="Input",
            ls="None",
            marker="o",
            markersize=2,
            c=blue,
            alpha=0.6,
        )
        plt.plot(
            self._wavelengths, self.model(res_params), label="Model", lw=2, c=orange
        )
        for g in self._gaps:
            plt.axvspan(g[0], g[1], color="black", alpha=0.1)

        if plot_amplitude_offset:
            offset_params = res_params[
                n_frequency + n_amplitude + 3 : n_frequency + n_amplitude + n_offset + 4
            ]
            plt.plot(
                self._wavelengths,
                1 + polyval(offset_params, self.scaled_w),
                label="Offset",
                c=green,
            )
            amplitude_params = res_params[
                n_frequency + 2 : n_frequency + n_amplitude + 3
            ]
            plt.plot(
                self._wavelengths,
                1.0
                + np.abs(polyval(amplitude_params, self.scaled_w))
                + polyval(offset_params, self.scaled_w),
                label="Amplitude",
                c=red,
            )

        if x0 is not None:
            plt.plot(
                self._wavelengths,
                self.model(x0),
                ls=":",
                label="Init",
                c=purple,
            )

        plt.xlabel("Wavelengths")
        plt.ylabel("Curve")
        plt.legend()
        plt.ylim(np.min(curve) * 0.9, np.max(curve) * 1.1)
        plt.show()

    def get_modulaion_curve(self, x, y, aperture=4):
        """
        Get the modulation curve.

        :param spaxel_x: Spaxel x
        :type spaxel_x: int
        :param spaxel_y: Spaxel y
        :type spaxel_y: int
        :param aperture: aperture size to sum the spectra to average out the wiggles
        :type aperture: int
        :return: Modulation curve
        :rtype: np.ndarray
        """
        spectra = self._datacube[:, x, y]
        noise = self._noise_cube[:, x, y]

        # make circular mask around the pixel with radius s
        mask = np.zeros_like(self._datacube[0], dtype=bool)

        for i in range(x - 2 * aperture, x + 2 * aperture):
            for j in range(y - 2 * aperture, y + 2 * aperture):
                if (i - x) ** 2 + (j - y) ** 2 <= aperture**2:
                    mask[i, j] = True

        aperture_spectra = np.nansum(self._datacube[:, mask], axis=(1))
        aperture_spectra /= np.nanmax(aperture_spectra) / np.nanmax(
            self._datacube[:, x, y]
        )
        aperture_noise = np.sqrt(np.nansum(self._noise_cube[:, mask] ** 2, axis=(1)))
        aperture_noise /= np.nanmax(aperture_noise) / np.nanmax(noise)

        ratio = spectra / aperture_spectra
        ratio_noise = np.abs(
            np.sqrt((noise / spectra) ** 2 + (aperture_noise / aperture_spectra) ** 2)
            * ratio
        )

        return ratio, ratio_noise

    def fit_curve_with_best_bic(
        self,
        curve,
        noise=None,
        n_amplitude=10,
        n_frequency=1,
        n_offset=7,
        min_n_amplitude=None,
        min_n_frequency=None,
        min_n_offset=None,
        artificial_noise_level=0.005,
        proximity_threshold=200,
        plot=False,
        plot_amplitude_offset=False,
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
        :param n_offset: Number of offset parameters
        :type n_offset: int
        :param min_n_amplitude: Minimum number of amplitude parameters
        :type min_n_amplitude: int
        :param min_n_frequency: Minimum number of frequency parameters
        :type min_n_frequency: int
        :param min_n_offset: Minimum number of offset parameters
        :type min_n_offset: int
        :param artificial_noise_level: Artificial noise level
        :type artificial_noise_level: float
        :param proximity_threshold: Proximity lower limit in Angstrom for initial identifaction of peaks and troughs
        :type proximity_threshold: float
        :param plot: If True, plot the results
        :type plot: bool
        :param plot_amplitude_offset: If True, plot the amplitude and offset
        :type plot_amplitude_offset: bool
        :return: Fitted parameters
        :rtype: np.ndarray
        """
        print("Computing BIC for choices of n_amplitude...")
        if min_n_amplitude is None:
            min_n_amplitude = n_amplitude
        if min_n_offset is None:
            min_n_offset = n_offset
        if min_n_frequency is None:
            min_n_frequency = n_frequency

        best_bic = None
        for i in tqdm(range(n_amplitude, min_n_amplitude - 1, -1)):
            for j in range(n_offset, min_n_offset - 1, -1):
                for k in range(n_frequency, min_n_frequency - 1, -1):
                    res_params = self.fit_curve(
                        curve,
                        noise,
                        n_amplitude=i,
                        n_frequency=k,
                        n_offset=j,
                        artificial_noise_level=artificial_noise_level,
                        proximity_threshold=proximity_threshold,
                        plot=False,
                        plot_amplitude_offset=False,
                    )

                    bic = self.get_bic(curve, noise, res_params)

                    if best_bic is None:
                        print(
                            f"n_amplitude: {i}, n_offset: {j}, n_frequency: {k}, BIC: {bic}"
                        )
                        best_n_amplitude = i
                        best_n_offset = j
                        best_n_frequency = k
                        best_bic = bic
                        best_params = res_params
                    elif bic < best_bic:
                        print(
                            f"n_amplitude: {i}, n_offset: {j}, n_frequency: {k}, BIC: {bic}"
                        )
                        best_n_amplitude = i
                        best_n_offset = j
                        best_n_frequency = k
                        best_bic = bic
                        best_params = res_params

        print("Best n_amplitude: ", best_n_amplitude)
        print("Best n_offset: ", best_n_offset)
        print("Best n_frequency: ", best_n_frequency)

        self._n_amplitude = best_n_amplitude
        self._n_offset = best_n_offset
        self._n_frequency = best_n_frequency

        if plot:
            self.plot_model(
                curve,
                noise,
                best_n_amplitude,
                n_frequency,
                n_offset,
                best_params,
                plot_amplitude_offset,
            )

        return best_params

    def get_bic(self, curve, noise, res_params):
        """
        Get the BIC.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :param res_params: Fitted parameters
        :type res_params: np.ndarray
        :return: BIC
        :rtype: float
        """
        n = len(curve)
        k = len(res_params)
        loss = self.loss(res_params, curve, noise)
        return n * np.log(loss) + k * np.log(n)

    def is_wiggle_detected(
        self, curve, noise, res_params, n_offset=5, sigma_threshold=5
    ):
        """
        Check if wiggle is detected.

        :param curve: Curve
        :type curve: np.ndarray
        :param noise: Noise
        :type noise: np.ndarray
        :param n_offset: Number of offset parameters
        :type n_offset: int
        :param sigma_threshold: Sigma threshold
        :type sigma_threshold: float
        :return: True if wiggle is detected
        :rtype: bool
        """
        coeffs = polyfit(
            self.scaled_w,
            curve,
            n_offset,
            w=1 / noise,
        )

        n = len(curve)
        residual = curve - polyval(coeffs, self.scaled_w)

        p = np.percentile(np.abs(residual), 97.5)
        indices = np.abs(residual) < p
        chi2 = np.sum(((residual**2 / noise**2) * self._gap_mask)[indices])
        chi2_red = chi2 / n

        residual = self.loss_vector(res_params, curve, noise)
        chi2_model_red = np.sum(residual[indices] ** 2) / n

        sigma = np.sqrt(chi2_red - chi2_model_red)
        print(sigma, chi2_red - chi2_model_red)
        return sigma > sigma_threshold
