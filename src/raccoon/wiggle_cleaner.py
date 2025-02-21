import jax.numpy as jnp
from jax import grad, jit
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from jax.example_libraries import optimizers
from jax.scipy.optimize import minimize
import jax
import optax


class WiggleCleaner(object):

    def __init__(
        self,
        wavelengths,
        gaps,
        n_amplitude=9,
        n_frequency=9,
        n_offset=9,
    ):
        self._wavelengths = jnp.array(wavelengths)
        self._gaps = gaps
        self._n_amplitude = n_amplitude
        self._n_frequency = n_frequency
        self._n_offset = n_offset

    @property
    def scaled_w(self):
        return self.scale_wavelengths_to_1_m1(self._wavelengths)

    def wiggle_func(self, xs, amplitude_params, frequency_params, offset_params, phi):
        amplitude = jnp.abs(jnp.polyval(amplitude_params, xs))
        frequency = jnp.abs(jnp.polyval(frequency_params, xs))
        offset = jnp.polyval(offset_params, xs)
        return 1 + amplitude * jnp.sin(frequency * xs + phi) + offset

    def scale_wavelengths_to_1_m1(self, w):
        return (w - self._wavelengths[0]) / (
            self._wavelengths[-1] - self._wavelengths[0]
        ) * 2 - 1

    def find_extrema(self, curve, proximity_threshold=200, is_peak=True):
        proximity_threshold /= jnp.mean(jnp.diff(self._wavelengths))
        smooth_curve = self.smooth_curve(curve)
        val = -2 if is_peak else 2
        extrema = jnp.where(jnp.diff(jnp.sign(jnp.diff(smooth_curve))) == val)[0]
        # delete_indices = []
        keep_indices = []
        for i in range(len(extrema) - 1):
            if extrema[i + 1] - extrema[i] > proximity_threshold:
                keep_indices.append(i + 1)
        extrema = extrema[jnp.array(keep_indices)]
        return extrema

    def smooth_curve(self, curve):
        return savgol_filter(savgol_filter(curve, 31, 3), 31, 3)

    def find_init_peaks_troughs_mids(self, curve, proximity_threshold=200):
        peaks = self.find_extrema(curve, proximity_threshold, is_peak=True)
        troughs = self.find_extrema(curve, proximity_threshold, is_peak=False)
        all_extrema = []
        if peaks[0] < troughs[0]:
            for p, t in zip(peaks, troughs):
                all_extrema.append(p)
                all_extrema.append(t)
            if len(peaks) > len(troughs):
                all_extrema.append(peaks[-1])
        else:
            for p, t in zip(peaks, troughs):
                all_extrema.append(t)
                all_extrema.append(p)
            if len(troughs) > len(peaks):
                all_extrema.append(troughs[-1])
        smooth_curve = self.smooth_curve(curve)
        midpoints = []
        if peaks[0] < troughs[0]:
            iterator = [(a, b) for a, b in zip(peaks, troughs)] + [
                (a, b) for a, b in zip(troughs, peaks[1:])
            ]
        else:
            iterator = [(a, b) for a, b in zip(troughs, peaks)] + [
                (a, b) for a, b in zip(peaks, troughs[1:])
            ]
        for a, b in iterator:
            if a > b:
                a, b = b, a
            a_value = smooth_curve[a]
            b_value = smooth_curve[b]
            mid_value = (a_value + b_value) / 2
            slice = smooth_curve[a:b]
            print(slice - mid_value)
            mid_index = jnp.argmin(jnp.abs(slice - mid_value))
            midpoints.append(a + mid_index)
        midpoints = jnp.array(midpoints)
        return peaks, troughs, midpoints

    def get_init_params(self, curve, n_amplitude=None, n_frequency=None, n_offset=None):
        if n_amplitude is None:
            n_amplitude = self._n_amplitude
        if n_frequency is None:
            n_frequency = self._n_frequency
        if n_offset is None:
            n_offset = self._n_offset

        peaks, troughs, midpoints = self.find_init_peaks_troughs_mids(curve)
        sw = self.scaled_w

        smooth_curve = self.smooth_curve(curve)

        extrema = jnp.concatenate((peaks, troughs))
        extrema = jnp.sort(extrema)
        extrema_values = smooth_curve[extrema]
        extrema_sw = sw[extrema]

        amplitude_params = jnp.polyfit(
            extrema_sw, jnp.abs(extrema_values - 1), n_amplitude
        )

        midpoint_values = smooth_curve[midpoints]
        midpoint_sw = sw[midpoints]
        offset_params = jnp.polyfit(midpoint_sw, midpoint_values - 1, n_offset)

        modulation_k = []
        for i in range(len(extrema)):
            if i == 0:
                modulation_k.append((extrema_sw[1] - extrema_sw[0]) * 2)
            elif i == len(extrema) - 1:
                modulation_k.append((extrema_sw[-1] - extrema_sw[-2]) * 2)
            else:
                modulation_k.append(extrema_sw[i + 1] - extrema_sw[i - 1])
        modulation_frequency = 2 * jnp.pi / jnp.array(modulation_k)
        frequency_params = jnp.polyfit(extrema_sw, modulation_frequency, n_frequency)

        peak_0 = sw[peaks[0]]
        init_phi = ((peak_0 - sw[0]) / modulation_k[0] / 4) * jnp.pi / 2

        return amplitude_params, frequency_params, offset_params, init_phi

    def model(self, params):
        n_amplitude = self._n_amplitude
        n_frequency = self._n_frequency
        n_offset = self._n_offset
        amplitude_params = params[: n_amplitude + 1]
        frequency_params = params[n_amplitude + 1 : n_amplitude + n_frequency + 2]
        offset_params = params[-n_offset - 1 : -1]
        phi_0 = params[-1]
        model = self.wiggle_func(
            self.scaled_w, amplitude_params, frequency_params, offset_params, phi_0
        )
        return model

    def loss_vector(self, params, curve, noise=None):
        model = self.model(params)
        if noise is not None:
            residual = (curve - model) / noise
        else:
            residual = curve - model
        for g in self._gaps:
            residual = residual.at[
                (self._wavelengths > g[0]) & (self._wavelengths < g[1])
            ].set(0)
        return residual

    def loss(self, params, curve, noise=None):
        return jnp.sum(self.loss_vector(params, curve, noise) ** 2)

    def fit_curve(
        self,
        curve,
        noise=None,
        n_amplitude=None,
        n_frequency=None,
        n_offset=None,
        verbose=False,
        iterations=1000,
    ):
        if n_amplitude is None:
            n_amplitude = self._n_amplitude
        else:
            self._n_amplitude = n_amplitude
        if n_frequency is None:
            n_frequency = self._n_frequency
        else:
            self._n_frequency = n_frequency
        if n_offset is None:
            n_offset = self._n_offset
        else:
            self._n_offset = n_offset

        init_amplitude_params, init_frequency_params, init_offset_params, init_phi = (
            self.get_init_params(curve, n_amplitude, n_frequency, n_offset)
        )
        # init_frequency_params = jnp.array(
        #     [3.5820956, 1.7547786, -15.777637, 39.377594]
        # )[-n_frequency - 1 :]
        x0 = jnp.concatenate(
            [
                init_amplitude_params,
                init_frequency_params,
                init_offset_params,
                jnp.array([init_phi]),
            ]
        )
        print(
            init_amplitude_params, init_frequency_params, init_offset_params, init_phi
        )

        solver = optax.adam(learning_rate=0.003)
        params = x0

        opt_state = solver.init(params)

        for i in range(iterations):
            grad = jax.grad(self.loss)(params, curve, noise)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            if i % 100 == 0:
                print(f"Loss: {self.loss(params, curve, noise)}")
        # use optax.adam to optimize the loss function
        # opt = optax.adam(1e-3)
        # opt_state = opt.init(x0)
        # opt_update = jit(opt.update)
        # opt_params = x0
        # for i in range(1000):
        #     loss = self.loss(opt_params, curve, noise)
        #     grads = grad(self.loss)(opt_params, curve, noise)
        #     opt_state, opt_params = opt_update(grads, opt_state, opt_params)
        #     if i % 100 == 0:
        #         print(f"Loss: {loss}")

        res_params = params

        if verbose:
            # plt.plot(
            #     self._wavelengths, jnp.polyval(init_amplitude_params, self.scaled_w)
            # )
            plt.plot(
                self._wavelengths, jnp.polyval(init_frequency_params, self.scaled_w)
            )
            # plt.plot(self._wavelengths, jnp.polyval(init_offset_params, self.scaled_w))
            plt.show()

            plt.plot(self._wavelengths, curve, label="Input")
            plt.plot(self._wavelengths, self.model(res_params), label="Model")
            for g in self._gaps:
                plt.axvspan(g[0], g[1], color="black", alpha=0.1)
            offset_params = res_params[-n_offset - 1 : -1]
            plt.plot(
                self._wavelengths,
                1 + jnp.polyval(offset_params, self.scaled_w),
                label="Offset",
            )
            amplitude_params = res_params[: n_amplitude + 1]
            plt.plot(
                self._wavelengths,
                1 + jnp.abs(jnp.polyval(amplitude_params, self.scaled_w)),
                label="Amplitude",
            )
            plt.plot(
                self._wavelengths,
                self.wiggle_func(
                    self.scaled_w,
                    init_amplitude_params,
                    init_frequency_params,
                    init_offset_params,
                    init_phi,
                ),
                label="Init",
            )
            plt.xlabel("Wavelengths")
            plt.ylabel("Curve")
            plt.legend()
            plt.show()
        return res_params
