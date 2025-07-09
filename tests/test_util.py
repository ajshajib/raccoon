#!/usr/bin/env python

"""Tests for `raccoon.util` module."""

import numpy.testing as npt
import numpy as np

from raccoon import util


class TestUtil:

    def setup_method(self):
        self.util = util.Util()

    def teardown_method(self):
        pass

    def test_polyval(self):
        assert util.polyval([1, 2, 3], 1) == 6

    def test_polyfit(self):
        xs = np.linspace(0, 50, 500)
        curve = np.sin(2 * np.pi * xs / 10)
        coeffs = util.polyfit(xs, curve, 3)
        assert len(coeffs) == 4

    def test_find_extrema(self):
        xs = np.linspace(0, 50, 500)
        curve = np.sin(2 * np.pi * xs / 10)
        peaks = self.util.find_extrema(curve)
        troughs = self.util.find_extrema(curve, is_peak=False)
        npt.assert_array_equal(peaks, np.array([23, 124, 224, 323, 423]))
        npt.assert_array_equal(troughs, np.array([74, 174, 273, 373, 474]))

    def test_smooth_curve(self):
        xs = np.linspace(0, 50, 500)
        curve = np.sin(2 * np.pi * xs / 10)
        smoothed_curve = self.util.smooth_curve(curve)
        assert len(smoothed_curve) == len(curve)

    def test_lighter_smooth_curve(self):
        xs = np.linspace(0, 50, 500)
        curve = np.sin(2 * np.pi * xs / 10)
        smoothed_curve = self.util.lighter_smooth_curve(curve)
        assert len(smoothed_curve) == len(curve)

    def test_find_init_peaks_troughs_mids(self):
        xs = np.linspace(0, 50, 500)
        curve = np.sin(2 * np.pi * xs / 10)
        peaks, troughs, midpoints, all_extrema = self.util.find_init_peaks_troughs_mids(
            curve
        )
        npt.assert_array_equal(peaks, np.array([23, 124, 224, 323, 423]))
        npt.assert_array_equal(troughs, np.array([74, 174, 273, 373, 474]))
        npt.assert_array_equal(
            midpoints, np.array([50, 150, 249, 349, 449, 100, 200, 299, 399])
        )
        npt.assert_array_equal(
            all_extrema, np.array([23, 74, 124, 174, 224, 273, 323, 373, 423, 474])
        )

    def test_get_linear_freq_coeffs_from_extrema(self):
        extrema = np.array([23, 74, 124, 174, 224, 273, 323, 373, 423, 474])
        xs = np.linspace(0, 50, 500)
        coeffs = self.util.get_linear_freq_coeffs_from_extrema(extrema, xs)
        print(coeffs)
        assert len(coeffs) == 2
        npt.assert_array_almost_equal(coeffs, np.array([0, 0.1]), decimal=3)

    def test_fit_sine_function_to_extrema(self):
        extrema_positions = np.array([23, 74, 124, 174, 224, 273, 323, 373, 423, 474])
        extrema_vals = np.sin(2 * np.pi * extrema_positions / 10)
        is_peak = np.array(
            [True, False, True, False, True, False, True, False, True, False]
        )
        n_amplitude = 2
        n_offset = 3
        n_frequency = 1
        amplitude_coeffs, offset_coeffs, frequency_coeffs, phi_0 = (
            self.util.fit_sine_function_to_extrema(
                extrema_positions,
                extrema_vals,
                is_peak,
                n_amplitude,
                n_offset,
                n_frequency,
            )
        )
        print(amplitude_coeffs)
        print(offset_coeffs)
        print(frequency_coeffs)
        print(phi_0)
        assert len(amplitude_coeffs) == 3
        assert len(offset_coeffs) == 3
        assert len(frequency_coeffs) == 3
        assert len(phi_0) == 3
        npt.assert_array_almost_equal(amplitude_coeffs, np.array([1, 0, 0]), decimal=4)
        npt.assert_array_almost_equal(offset_coeffs, np.array([0, 0, 0]), decimal=4)
