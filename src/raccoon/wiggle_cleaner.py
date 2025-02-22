import jax.numpy as jnp
import jax
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from jaxopt import LevenbergMarquardt

from .util import Util


class WiggleCleaner(object):

    def __init__(
        self,
        wavelengths,
        gaps,
        n_amplitude=9,
        n_frequency=9,
        n_offset=9,
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
        self._wavelengths = jnp.array(wavelengths)
        self._gaps = jnp.array(gaps)
        self._n_amplitude = n_amplitude
        self._n_frequency = n_frequency
        self._n_offset = n_offset

        gap_mask = np.ones_like(wavelengths)
        for g in self._gaps:
            mask = (wavelengths > g[0]) & (wavelengths < g[1])
            gap_mask[mask] = 0
        self._gap_mask = jnp.array(gap_mask)

    @property
    def scaled_w(self):
        """
        Scaled wavelengths

        :return: Scaled wavelengths
        :rtype: jax.numpy.array
        """
        return self.scale_wavelengths_to_1_m1(self._wavelengths)

    def wiggle_func(self, xs, frequency_params, amplitude_params, offset_params, phi):
        """
        Get the wiggle function.

        :param xs: Scaled wavelengths
        :type xs: jax.numpy.array
        :param frequency_params: Frequency parameters
        :type frequency_params: jax.numpy.array
        :param amplitude_params: Amplitude parameters
        :type amplitude_params: jax.numpy.array
        :param offset_params: Offset parameters
        :type offset_params: jax.numpy.array
        :param phi: Phase
        :type phi: float
        :return: Wiggle function
        :rtype: jax.numpy.array
        """
        frequency = jnp.abs(jnp.polyval(frequency_params, xs))
        amplitude = jnp.abs(jnp.polyval(amplitude_params, xs))
        offset = jnp.polyval(offset_params, xs)

        return 1.0 + amplitude * jnp.sin(frequency * xs + phi) + offset

    def scale_wavelengths_to_1_m1(self, w):
        """
        Scale the wavelengths to -1 to 1.

        :param w: Wavelengths
        :type w: jax.numpy.array
        :return: Scaled wavelengths
        :rtype: jax.numpy.array
        """
        return (w - self._wavelengths[0]) / (
            self._wavelengths[-1] - self._wavelengths[0]
        ) * 2 - 1

    def model(self, params):
        """
        Get the wiggle model given the parameters.

        :param params: Parameters
        :type params: jax.numpy.array
        :return: Model
        :rtype: jax.numpy.array
        """
        n_amplitude = self._n_amplitude
        n_frequency = self._n_frequency
        n_offset = self._n_offset

        frequency_params = params[: n_frequency + 1]
        phi_0 = params[n_frequency + 1]
        amplitude_params = params[n_frequency + 2 : n_amplitude + n_frequency + 3]
        offset_params = params[-n_offset - 1 :]

        model = self.wiggle_func(
            self.scaled_w, frequency_params, amplitude_params, offset_params, phi_0
        )
        return model

    def loss_vector(self, params, curve, noise=None):
        """ "
        Get the residual vector.

        :param params: Parameters
        :type params: jax.numpy.array
        :param curve: Curve
        :type curve: jax.numpy.array
        :param noise: Noise
        :type noise: jax.numpy.array
        :return: Residual vector
        :rtype: jax.numpy.array
        """
        model = self.model(params)

        residual = (model - curve) / noise
        residual = residual * self._gap_mask

        return residual

    def loss(self, params, curve, noise):
        """
        Get the loss.

        :param params: Parameters
        :type params: jax.numpy.array
        :param curve: Curve
        :type curve: jax.numpy.array
        :param noise: Noise
        :type noise: jax.numpy.array
        :return: Loss
        :rtype: float
        """
        return jnp.sum(self.loss_vector(params, curve, noise) ** 2)

    def get_residual_func(self, curve, noise):
        """
        Get the residual function.

        :param curve: Curve
        :type curve: jax.numpy.array
        :param noise: Noise
        :type noise: jax.numpy.array
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
        :type params: jax.numpy.array
        :param init_params: Initial parameters
        :type init_params: jax.numpy.array
        :param curve: Curve
        :type curve: jax.numpy.array
        :param noise: Noise
        :type noise: jax.numpy.array
        :return: Residual function
        :rtype: Callable
        """

        def residual_func(params):
            new_params = jnp.concatenate([params, init_params[self._n_frequency + 2 :]])
            new_params
            return self.loss_vector(new_params, curve, noise)

        return residual_func

    def fit_curve(
        self,
        curve,
        noise=None,
        n_amplitude=None,
        n_frequency=None,
        n_offset=None,
        verbose=False,
        proximity_threshold=200,
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
        :return: Fitted parameters
        :rtype: jax.numpy.array
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

        init_frequency_params, init_amplitude_params, init_offset_params, init_phi = (
            Util.get_init_params(
                curve,
                self.scaled_w,
                n_frequency=n_frequency,
                n_amplitude=n_amplitude,
                n_offset=n_offset,
                proximity_threshold=proximity_threshold
                / np.mean(np.diff(self._wavelengths)),
            )
        )

        curve = jnp.array(curve)
        noise = jnp.array(noise)

        # print(
        #     init_amplitude_params, init_frequency_params, init_offset_params, init_phi
        # )
        x0 = np.concatenate(
            [
                init_frequency_params,
                np.array([init_phi]),
                init_amplitude_params,
                init_offset_params,
            ]
        )
        x0 = jnp.array(x0)

        solver = LevenbergMarquardt(
            self.get_residual_func_phase_only(x0, curve, noise), xtol=1e-6
        )
        res = solver.run(x0[: n_frequency + 2])
        res_params = res.params

        x0 = x0.at[: n_frequency + 2].set(res_params)
        solver = LevenbergMarquardt(self.get_residual_func(curve, noise), xtol=1e-6)
        res = solver.run(x0)
        res_params = res.params

        if verbose:
            # plt.plot(
            #     self._wavelengths, jnp.polyval(init_amplitude_params, self.scaled_w)
            # )
            # plt.plot(
            #     self._wavelengths, jnp.polyval(init_frequency_params, self.scaled_w)
            # )
            # # plt.plot(self._wavelengths, jnp.polyval(init_offset_params, self.scaled_w))
            # plt.show()

            plt.plot(
                self._wavelengths,
                curve,
                label="Input",
                ls="None",
                marker="o",
                markersize=2,
            )
            plt.plot(self._wavelengths, self.model(res_params), label="Model", lw=2)
            for g in self._gaps:
                plt.axvspan(g[0], g[1], color="black", alpha=0.1)
            offset_params = res_params[-n_offset - 1 :]
            plt.plot(
                self._wavelengths,
                1 + jnp.polyval(offset_params, self.scaled_w),
                label="Offset",
            )
            amplitude_params = res_params[
                n_frequency + 2 : n_amplitude + n_frequency + 3
            ]
            plt.plot(
                self._wavelengths,
                1
                + jnp.abs(jnp.polyval(amplitude_params, self.scaled_w))
                + jnp.polyval(offset_params, self.scaled_w),
                label="Amplitude",
            )
            plt.plot(
                self._wavelengths,
                self.model(x0),
                ls=":",
                label="Init",
            )
            plt.xlabel("Wavelengths")
            plt.ylabel("Curve")
            plt.legend()
            plt.show()
        return res_params
