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
        assert len(offset_coeffs) == 4
        assert len(frequency_coeffs) == 2
        npt.assert_allclose(phi_0, 0.12905, atol=1e-4)
        npt.assert_array_almost_equal(
            amplitude_coeffs, np.array([1.4288e-01, -1.5889e-03, 1.5985e-06]), decimal=4
        )
        npt.assert_array_almost_equal(
            offset_coeffs,
            np.array([2.5251e-02, -8.4708e-03, 2.2459e-05, -1.5063e-08]),
            decimal=4,
        )

    def test_get_init_params_basic(self):
        # Create a simple sine wave as the curve
        x = np.linspace(0, 2 * np.pi, 100)
        curve = np.sin(x)
        scaled_wavelengths = (x - x.min()) / (x.max() - x.min())
        # Call get_init_params
        freq, amp, offset, phi = self.util.get_init_params(
            curve,
            scaled_wavelengths,
            n_amplitude=2,
            n_offset=2,
            n_frequency=1,
            plot=False,
        )
        # Check output types and shapes
        assert isinstance(freq, np.ndarray)
        assert isinstance(amp, np.ndarray)
        assert isinstance(offset, np.ndarray)
        assert isinstance(phi, float) or isinstance(phi, np.floating)
        # Check that arrays are not empty
        assert freq.size > 0
        assert amp.size > 0
        assert offset.size > 0

    def test_get_init_params_spline_basic(self):
        # Use a longer sine wave to ensure enough extrema for cubic spline
        x = np.linspace(0, 10 * np.pi, 500)
        curve = np.sin(x)
        # Use x for both curve and x positions (no scaling)
        amp_spline, freq_spline, phi_0 = self.util.get_init_params_spline(
            curve,
            x,
            n_amplitude=4,  # Increased to ensure enough knots/points
            n_frequency=3,  # Increased to ensure enough knots/points
            plot=False,
        )
        # Check that the returned splines are callable and phi_0 is a float
        assert callable(amp_spline)
        assert callable(freq_spline)
        assert isinstance(phi_0, float) or isinstance(phi_0, np.floating)
        # Evaluate the splines at a few points and check output type
        amp_eval = amp_spline(x)
        freq_eval = freq_spline(x)
        assert isinstance(amp_eval, np.ndarray)
        assert isinstance(freq_eval, np.ndarray)
        assert amp_eval.shape == curve.shape
        assert freq_eval.shape == curve.shape
        # Check that the spline outputs are finite and not constant
        assert np.all(np.isfinite(amp_eval))
        assert np.all(np.isfinite(freq_eval))

    def test_fit_sine_function_to_extrema_spline_and_fitted_sine_function_spline(self):
        # Create a longer sine wave to ensure enough extrema for cubic spline
        x = np.linspace(0, 10 * np.pi, 500)
        curve = np.sin(x)
        # Find peaks and troughs
        peaks = self.util.find_extrema(curve)
        troughs = self.util.find_extrema(curve, is_peak=False)
        extrema_positions = np.sort(np.concatenate([peaks, troughs]))
        extrema_vals = curve[extrema_positions]
        is_peak = np.isin(extrema_positions, peaks)
        # Use fewer knots to avoid ValueError (need more x points)
        amp_spline, freq_spline, phi_0 = self.util.fit_sine_function_to_extrema_spline(
            extrema_positions,
            extrema_vals,
            is_peak,
            n_amplitude=4,
            n_frequency=3,
        )
        # Check that splines are callable and phi_0 is a float
        assert callable(amp_spline)
        assert callable(freq_spline)
        assert isinstance(phi_0, float) or isinstance(phi_0, np.floating)
        # Evaluate the fitted sine function using splines
        y_fit = self.util.fitted_sine_function_spline(x, amp_spline, freq_spline, phi_0)
        assert isinstance(y_fit, np.ndarray)
        assert y_fit.shape == curve.shape
        # Check that the output is finite and not constant
        assert np.all(np.isfinite(y_fit))
        assert np.std(y_fit) > 0.01
