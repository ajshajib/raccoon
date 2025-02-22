import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


class Util(object):

    @classmethod
    def find_extrema(cls, curve, proximity_threshold=50, is_peak=True):
        """
        Find peaks or troughs of a curve

        :param curve: 1D numpy array
        :type curve: np.ndarray
        :param proximity_threshold: Minimum distance between two extrema in pixels
        :type proximity_threshold: int
        :param is_peak: If True, find peaks, otherwise find troughs
        :type is_peak: bool
        :return: Indices of the extrema
        :rtype: np.ndarray
        """
        smooth_curve = cls.smooth_curve(curve)

        val = -2 if is_peak else 2
        extrema = np.where(np.diff(np.sign(np.diff(smooth_curve))) == val)[0]

        while np.any(np.diff(extrema) < proximity_threshold):
            close_index = np.where(np.diff(extrema) < proximity_threshold)[0][0]
            extrema = np.delete(extrema, close_index + 1)

        return extrema

    @staticmethod
    def smooth_curve(curve):
        """
        Smooth a curve using Savitzky-Golay filter

        :param curve: 1D numpy array
        :type curve: np.ndarray
        :return: Smoothed curve
        :rtype: np.ndarray
        """
        return savgol_filter(savgol_filter(curve, 31, 3), 31, 3)

    @classmethod
    def find_init_peaks_troughs_mids(cls, curve, proximity_threshold=50):
        """
        Find initial peaks, troughs, and midpoints of a curve

        :param curve: 1D numpy array
        :type curve: np.ndarray
        :param proximity_threshold: Minimum distance between two extrema in pixels
        :type proximity_threshold: int
        :return: Peaks, troughs, and midpoints
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        peaks = cls.find_extrema(curve, proximity_threshold, is_peak=True)
        troughs = cls.find_extrema(curve, proximity_threshold, is_peak=False)
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

        smooth_curve = cls.smooth_curve(curve)

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

            mid_index = np.argmin(np.abs(slice - mid_value))
            midpoints.append(a + mid_index)

        midpoints = np.array(midpoints)

        return peaks, troughs, midpoints

    @classmethod
    def get_init_params(
        cls,
        curve,
        scaled_wavelengths,
        n_frequency=1,
        n_amplitude=2,
        n_offset=7,
        proximity_threshold=50,
        plot=False,
    ):
        """
        Get initial parameters for the curve fitting

        :param curve: 1D numpy array
        :type curve: np.ndarray
        :param n_amplitude: Degree of the polynomial for the amplitude
        :type n_amplitude: int
        :param n_frequency: Degree of the polynomial for the frequency
        :type n_frequency: int
        :param n_offset: Degree of the polynomial for the offset
        :type n_offset: int
        :param plot: If True, plot the initial parameters
        :type plot: bool
        :return: Initial amplitude, frequency, offset, and phase
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
        """
        peaks, troughs, midpoints = cls.find_init_peaks_troughs_mids(
            curve, proximity_threshold=proximity_threshold
        )

        smooth_curve = cls.smooth_curve(curve)

        extrema = np.concatenate((peaks, troughs))
        extrema = np.sort(extrema)
        extrema_values = smooth_curve[extrema]
        extrema_sw = scaled_wavelengths[extrema]

        # offset polyomial
        midpoint_values = smooth_curve[midpoints]
        midpoint_sw = scaled_wavelengths[midpoints]
        n_init_fit = min(2, n_offset)
        offset_params = np.polyfit(midpoint_sw, midpoint_values - 1, n_init_fit)
        if n_offset > n_init_fit:
            offset_params = np.concatenate(
                (np.zeros(n_offset - n_init_fit), offset_params)
            )

        # amplitude polynomial
        n_init_fit = min(1, n_amplitude)
        amplitude_params = np.polyfit(
            extrema_sw,
            np.abs(
                extrema_values
                - np.polyval(offset_params, scaled_wavelengths)[extrema]
                - 1
            ),
            n_init_fit,
        )
        if n_amplitude > n_init_fit:
            amplitude_params = np.concatenate(
                (np.zeros(n_amplitude - n_init_fit), amplitude_params)
            )
        # plt.plot(
        #     extrema_sw,
        #     np.abs(
        #         extrema_values
        #         - np.polyval(offset_params, scaled_wavelengths)[extrema]
        #         - 1
        #     ),
        # )
        # plt.plot(extrema_sw, np.polyval(amplitude_params, extrema_sw))
        # plt.show()

        # frequency polynomial
        modulation_k = []
        for i in range(len(extrema)):
            if i == 0:
                modulation_k.append((extrema_sw[1] - extrema_sw[0]) * 2)
            elif i == len(extrema) - 1:
                modulation_k.append((extrema_sw[-1] - extrema_sw[-2]) * 2)
            else:
                modulation_k.append(extrema_sw[i + 1] - extrema_sw[i - 1])
        modulation_frequency = 2 * np.pi / np.array(modulation_k)
        frequency_params = np.polyfit(extrema_sw, modulation_frequency, n_frequency)

        peak_0 = scaled_wavelengths[peaks[0]]
        init_phi = ((peak_0 - scaled_wavelengths[0]) / modulation_k[0] / 4) * np.pi / 2

        if plot:
            plt.plot(scaled_wavelengths, curve, label="Input")
            plt.plot(scaled_wavelengths, smooth_curve, label="Smoothed")
            plt.scatter(scaled_wavelengths[peaks], smooth_curve[peaks], c="r")
            plt.scatter(scaled_wavelengths[troughs], smooth_curve[troughs], c="b")
            plt.scatter(midpoint_sw, midpoint_values, c="g")
            plt.plot(
                scaled_wavelengths,
                1 + np.polyval(amplitude_params, scaled_wavelengths),
                label="Amplitude",
            )
            plt.plot(
                scaled_wavelengths,
                1 + np.polyval(offset_params, scaled_wavelengths),
                label="Offset",
            )
            plt.legend()
            plt.show()

        return frequency_params, amplitude_params, offset_params, init_phi
