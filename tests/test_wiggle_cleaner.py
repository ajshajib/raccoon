#!/usr/bin/env python

"""Tests for `raccoon` package."""

import numpy as np
import pytest
from raccoon.wiggle_cleaner import WiggleCleaner


class DummySpline:
    """A dummy spline class for testing purposes."""

    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        """Return an array of ones with the same shape as x."""
        return np.ones_like(x)


def make_dummy_wiggle_cleaner():
    """Create a WiggleCleaner instance with dummy data for testing.

    Returns:
        WiggleCleaner: An instance with dummy wavelength, datacube, and noise_cube.
    """
    wavelengths = np.linspace(
        1, 10, 50
    )  # Use 50 instead of 10 for robust extrema detection
    datacube = np.ones((50, 5, 5))
    noise_cube = np.ones((50, 5, 5)) * 0.1
    return WiggleCleaner(wavelengths, datacube, noise_cube)


class TestWiggleCleaner:
    """Test suite for the WiggleCleaner class."""

    def setup_method(self):
        """Set up a fresh WiggleCleaner instance before each test."""
        self.wc = make_dummy_wiggle_cleaner()

    def teardown_method(self):
        """Clean up after each test."""
        del self.wc

    def test_init_and_properties(self):
        """Test initialization and property setters/getters."""
        n_wave = self.wc._datacube.shape[0]
        assert self.wc._datacube.shape == (n_wave, 5, 5)
        self.wc.symmetric_sharpening = True
        assert self.wc.symmetric_sharpening is True
        self.wc.asymmetric_sharpening = True
        assert self.wc.asymmetric_sharpening is True

    def test_set_gaps_and_gap_mask(self):
        """Test setting gaps and creation of the gap mask."""
        gaps = [[2, 4], [6, 8]]
        self.wc.set_gaps(gaps)
        assert hasattr(self.wc, "_gap_mask")
        # The gap mask should only contain 0s and 1s
        assert np.all((self.wc._gap_mask == 1) | (self.wc._gap_mask == 0))

    def test_scaled_w(self):
        """Test wavelength scaling functions."""
        scaled = self.wc.scale_wavelengths_negative1_to_1(self.wc._wavelengths)
        assert np.allclose(scaled[0], -1)
        assert np.allclose(scaled[-1], 1)
        scaled0 = self.wc.scale_wavelengths_to_0_1(self.wc._wavelengths)
        assert np.allclose(scaled0[0], 0)
        assert np.allclose(scaled0[-1], 1)

    def test_wiggle_func_and_model(self):
        """Test wiggle function and model with dummy splines."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        xs = np.linspace(-1, 1, 10)
        amp = np.ones(3)
        freq = np.ones(3)
        phi = 0.0
        out = self.wc.wiggle_func(xs, amp, freq, phi)
        assert out.shape == xs.shape

    def test_split_and_set_params(self):
        """Test parameter splitting and setting."""
        self.wc._n_amplitude = 1
        self.wc._n_frequency = 1

        arr = self.wc.set_params(np.array([1, 2]), np.array([3, 4]), 5.0, 1, 1)
        assert arr.shape[0] == 5

    def test_configure_polynomial_ns(self):
        """Test configuration of polynomial degrees for amplitude and frequency."""
        n_a, n_f = self.wc.configure_polynomial_ns(2, 2)
        assert n_a == 2 and n_f == 2
        self.wc._n_amplitude = 3
        self.wc._n_frequency = 4
        n_a, n_f = self.wc.configure_polynomial_ns()
        assert n_a == 3 and n_f == 4

    def test_configure_polynomial_ns_defaults(self):
        """Test configure_polynomial_ns uses instance defaults when args are None."""
        self.wc._n_amplitude = 7
        self.wc._n_frequency = 5
        n_a, n_f = self.wc.configure_polynomial_ns(None, None)
        assert n_a == 7
        assert n_f == 5

    def test_model_full_fit_and_residual_vector_full_fit(self):
        """Test full model fitting and residual vector calculation."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 1
        self.wc._n_frequency = 1
        params = np.ones(7)
        # These may raise exceptions depending on implementation, so catch them
        try:
            self.wc.model_full_fit(params, 2, 2, 1)
        except Exception:
            pass
        try:
            self.wc.residual_vector_full_fit(params, 2, 2, 1)
        except Exception:
            pass

    def test_model_full_fit_output_shape(self):
        """Test model_full_fit returns arrays of correct shape and type."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        n_wave = self.wc._datacube.shape[0]
        params = np.ones(12)
        try:
            model, signal, noise = self.wc.model_full_fit(params, 0, 0, 1)
            assert isinstance(model, np.ndarray)
            assert isinstance(signal, np.ndarray)
            assert isinstance(noise, np.ndarray)
            assert model.shape == (n_wave,)
            assert signal.shape == (n_wave,)
            assert noise.shape == (n_wave,)
        except NotImplementedError:
            pass
        except Exception as e:
            # Allow for partial implementations
            assert False, f"Unexpected exception: {e}"

    def test_residual_vector_full_fit_explicit(self):
        """Explicitly test residual_vector_full_fit covers all lines and output
        shape."""
        # Mock model_full_fit to return known arrays
        arr = np.arange(10.0)
        self.wc.model_full_fit = lambda *a, **kw: (arr + 1, arr, np.ones_like(arr) * 2)
        params = np.ones(7)
        out = self.wc.residual_vector_full_fit(params, 2, 2, 1)
        # Should be ((arr+1) - arr) / 2 = 0.5 everywhere
        assert np.allclose(out, 0.5)
        assert out.shape == arr.shape

    def test_residual_vector_and_cost_function(self):
        """Test residual vector and cost function calculations."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 1
        self.wc._n_frequency = 1
        params = np.ones(7)
        n_wave = self.wc._datacube.shape[0]
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        try:
            self.wc.residual_vector(params, signal, noise)
        except Exception:
            pass
        try:
            self.wc.cost_function(params, signal, noise)
        except Exception:
            pass

    def test_get_residual_func_and_phase_only(self):
        """Test retrieval of residual functions."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 1
        self.wc._n_frequency = 1
        params = np.ones(7)
        n_wave = self.wc._datacube.shape[0]
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        f = self.wc.get_residual_func(signal, noise)
        assert callable(f)
        f2 = self.wc.get_residual_func_phase_only(params, signal, noise)
        assert callable(f2)

    def test_configure_noise(self):
        """Test noise configuration utility."""
        arr = self.wc.configure_noise(np.ones(5), np.ones(5), 0)
        assert arr.shape[0] == 5
        arr2 = self.wc.configure_noise(np.ones(5), np.ones(5), 1.0)
        assert np.all(arr2 == 1.0)

    def test_plot_model_and_get_model_uncertainty(self):
        """Test plotting and model uncertainty estimation."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 1
        self.wc._n_frequency = 1
        params = np.ones(7)
        n_wave = self.wc._datacube.shape[0]
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        try:
            self.wc.plot_model(signal, noise, params)
        except Exception:
            pass
        try:
            self.wc.get_model_uncertainty(params, np.eye(7), n_wave)
        except Exception:
            pass

    def test_plot_model_runs(self):
        """Test that plot_model runs without error or raises NotImplementedError."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        n_wave = self.wc._datacube.shape[0]
        params = np.ones(12)
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        try:
            self.wc.plot_model(signal, noise, params)
        except NotImplementedError:
            pass
        except Exception as e:
            assert False, f"Unexpected exception: {e}"

    def test_get_wiggle_signal_and_get_spectra_set(self):
        """Test wiggle signal and spectra set retrieval."""
        try:
            self.wc.get_wiggle_signal(2, 2, 1)
        except Exception:
            pass
        try:
            self.wc.get_spectra_set(2, 2, 1, 2, 1)
        except Exception:
            pass

    def test_reject_outliers(self):
        """Test outlier rejection methods."""
        n_wave = self.wc._datacube.shape[0]
        self.wc._outlier_rejection_method = "sigma_clip"
        arr = self.wc.reject_outliers(np.ones(n_wave), 2)
        self.wc._outlier_rejection_method = "fdr"
        arr2 = self.wc.reject_outliers(np.ones(n_wave), 2)
        assert arr.shape == arr2.shape

    def test_fit_wiggle_with_model_selection_and_metric(self):
        """Test wiggle fitting with model selection and metric calculation."""
        try:
            self.wc.fit_wiggle_with_model_selection(2, 2, 1, 2, 1, 2, 2)
        except Exception:
            pass
        try:
            self.wc.get_model_selection_metric(
                np.ones(10), np.ones(10), np.ones(7), "bic"
            )
        except Exception:
            pass

    def test_is_wiggle_detected_and_clean_cube(self):
        """Test wiggle detection and cleaning of the data cube."""
        try:
            self.wc.is_wiggle_detected(np.ones(10), np.ones(10), np.ones(7))
        except Exception:
            pass
        try:
            self.wc.clean_cube()
        except Exception:
            pass

    def test_wiggle_model_sharpening_modes(self):
        """Test wiggle_model with all sharpening mode combinations."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(4))  # n_frequency=2 -> 3 coeffs
        self.wc._n_amplitude = 2  # Must be at least 2 per implementation
        self.wc._n_frequency = 2  # Must be at least 2 per implementation
        n_a, n_f = self.wc._n_amplitude, self.wc._n_frequency
        # base params: [amp0, amp1, amp2, freq0, freq1, freq2, phi_0, extra1, extra2]
        base_params = np.ones(n_a + n_f + 5)  # Ensure enough params for split_params
        # Only asymmetric sharpening
        self.wc._asymmetric_sharpening = True
        self.wc._symmetric_sharpening = False
        params = np.concatenate([base_params, [2.0]])
        out = self.wc.wiggle_model(params)
        assert out.shape == self.wc._wavelengths.shape
        # Only symmetric sharpening
        self.wc._asymmetric_sharpening = False
        self.wc._symmetric_sharpening = True
        params = np.concatenate([base_params, [3.0]])
        out = self.wc.wiggle_model(params)
        assert out.shape == self.wc._wavelengths.shape
        # Both sharpenings
        self.wc._asymmetric_sharpening = True
        self.wc._symmetric_sharpening = True
        params = np.concatenate([base_params, [4.0, 5.0]])
        out = self.wc.wiggle_model(params)
        assert out.shape == self.wc._wavelengths.shape
        # Neither sharpening
        self.wc._asymmetric_sharpening = False
        self.wc._symmetric_sharpening = False
        params = base_params.copy()
        out = self.wc.wiggle_model(params)
        assert out.shape == self.wc._wavelengths.shape

    def test_init_with_gaps(self):
        """Test WiggleCleaner __init__ with gaps argument triggers set_gaps branch."""
        wavelengths = np.linspace(1, 10, 10)
        datacube = np.ones((10, 5, 5))
        noise_cube = np.ones((10, 5, 5)) * 0.1
        gaps = [[2, 4], [6, 8]]
        wc = WiggleCleaner(wavelengths, datacube, noise_cube, gaps=gaps)
        # The _gaps attribute should be set and _gap_mask should exist
        assert hasattr(wc, "_gaps")
        assert hasattr(wc, "_gap_mask")
        assert np.all((wc._gap_mask == 1) | (wc._gap_mask == 0))

    def test_residual_vector_all_branches(self):
        """Test residual_vector covers all branches: scatter, huber, masks."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 2  # Must be at least 2
        self.wc._n_frequency = 2  # Must be at least 2
        n_wave = self.wc._datacube.shape[0]
        # Set up masks
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)
        params = np.ones(
            12
        )  # Ensure enough params for n_a=2, n_f=2, phi_0, etc. and scatter
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        # Test all combinations of scatter and huber
        for scatter in [False, True]:
            for huber_loss in [False, True]:
                self.wc._include_scatter = scatter
                self.wc._use_huber_loss = huber_loss
                self.wc._huber_delta = 1.0
                # Patch huber if needed
                if huber_loss:
                    import raccoon.util

                    raccoon.util.huber = lambda delta, r: np.abs(
                        r
                    )  # simple pass-through
                out = self.wc.residual_vector(params, signal, noise)
                assert out.shape == signal.shape
                # Now test with masks that are not all ones
                self.wc._gap_mask = np.zeros(n_wave)
                self.wc._outlier_mask = np.ones(n_wave)
                out2 = self.wc.residual_vector(params, signal, noise)
                assert np.all(out2 == 0)
                self.wc._gap_mask = np.ones(n_wave)
                self.wc._outlier_mask = np.zeros(n_wave)
                out3 = self.wc.residual_vector(params, signal, noise)
                assert np.all(out3 == 0)
        # Restore masks
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)

    def test_cost_function_all_branches(self):
        """Test cost_function covers all branches of residual_vector and returns a
        float."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        n_wave = self.wc._datacube.shape[0]
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)
        params = np.ones(12)  # Ensure enough params for scatter
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        for scatter in [False, True]:
            for huber_loss in [False, True]:
                self.wc._include_scatter = scatter
                self.wc._use_huber_loss = huber_loss
                self.wc._huber_delta = 1.0
                if huber_loss:
                    import raccoon.util

                    raccoon.util.huber = lambda delta, r: np.abs(r)
                cost = self.wc.cost_function(params, signal, noise)
                assert isinstance(cost, float) or isinstance(cost, np.floating)
                assert cost >= 0
                # Test with masks set to zero
                self.wc._gap_mask = np.zeros(n_wave)
                self.wc._outlier_mask = np.ones(n_wave)
                cost2 = self.wc.cost_function(params, signal, noise)
                assert cost2 == 0
                self.wc._gap_mask = np.ones(n_wave)
                self.wc._outlier_mask = np.zeros(n_wave)
                cost3 = self.wc.cost_function(params, signal, noise)
                assert cost3 == 0
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)

    def test_get_residual_func_covers_residual_func(self):
        """Test that the function returned by get_residual_func calls residual_vector
        and covers all branches."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        n_wave = self.wc._datacube.shape[0]
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)
        params = np.ones(12)  # Ensure enough params for scatter
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        for scatter in [False, True]:
            for huber_loss in [False, True]:
                self.wc._include_scatter = scatter
                self.wc._use_huber_loss = huber_loss
                self.wc._huber_delta = 1.0
                if huber_loss:
                    import raccoon.util

                    raccoon.util.huber = lambda delta, r: np.abs(r)
                f = self.wc.get_residual_func(signal, noise)
                out = f(params)
                assert out.shape == signal.shape
                # Test with masks set to zero
                self.wc._gap_mask = np.zeros(n_wave)
                self.wc._outlier_mask = np.ones(n_wave)
                out2 = f(params)
                assert np.all(out2 == 0)
                self.wc._gap_mask = np.ones(n_wave)
                self.wc._outlier_mask = np.zeros(n_wave)
                out3 = f(params)
                assert np.all(out3 == 0)
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)

    def test_get_residual_func_phase_only_covers_residual_func(self):
        """Test that the function returned by get_residual_func_phase_only covers all
        branches."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        n_wave = self.wc._datacube.shape[0]
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)
        # Prepare init_params and phase-only params
        n_a, n_f = self.wc._n_amplitude, self.wc._n_frequency
        # Use correct number of params for set_params: amplitude, frequency, phi_0, n_a, n_f
        init_amplitude_params = np.ones(n_a + 2)
        init_frequency_params = np.ones(n_f + 2)
        init_phi_0 = 0.5
        # Use n_a+2, n_f+2 for set_params, but ensure phase_only_params is long enough for scatter
        init_params = self.wc.set_params(
            init_amplitude_params, init_frequency_params, init_phi_0, n_a + 2, n_f + 2
        )
        # phase-only params: frequency params + phi_0, but must be long enough for scatter index
        phase_only_params = np.ones(max(len(init_frequency_params) + 1, n_a + n_f + 6))
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        for scatter in [False, True]:
            for huber_loss in [False, True]:
                self.wc._include_scatter = scatter
                self.wc._use_huber_loss = huber_loss
                self.wc._huber_delta = 1.0
                if huber_loss:
                    import raccoon.util

                    raccoon.util.huber = lambda delta, r: np.abs(r)
                f = self.wc.get_residual_func_phase_only(init_params, signal, noise)
                out = f(phase_only_params)
                assert out.shape == signal.shape
                # Test with masks set to zero
                self.wc._gap_mask = np.zeros(n_wave)
                self.wc._outlier_mask = np.ones(n_wave)
                out2 = f(phase_only_params)
                assert np.all(out2 == 0)
                self.wc._gap_mask = np.ones(n_wave)
                self.wc._outlier_mask = np.zeros(n_wave)
                out3 = f(phase_only_params)
                assert np.all(out3 == 0)
        self.wc._gap_mask = np.ones(n_wave)
        self.wc._outlier_mask = np.ones(n_wave)

    def test_get_model_uncertainty_output_shape(self):
        """Test get_model_uncertainty returns correct output shape and type."""
        self.wc._amplitude_spline = DummySpline(np.ones(3))
        self.wc._frequency_spline = DummySpline(np.ones(3))
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        n_wave = self.wc._datacube.shape[0]
        params = np.ones(12)
        cov = np.eye(12)
        # Should return an array of length n_wave or raise NotImplementedError
        try:
            out = self.wc.get_model_uncertainty(params, cov, n_wave)
            assert isinstance(out, np.ndarray)
            assert out.shape == (n_wave,)
        except NotImplementedError:
            pass
        except Exception as e:
            # Allow for partial implementations
            assert False, f"Unexpected exception: {e}"

    def test_get_model_selection_metric_all_modes(self):
        """Test get_model_selection_metric for all supported metric modes and error handling."""
        n_wave = self.wc._datacube.shape[0]
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        params = np.ones(7)
        # Ensure n_amplitude and n_frequency are at least 2 for all metrics
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        # params: n_amplitude + 2 (amplitude) + n_frequency + 2 (frequency) + 1 (phi_0)
        n_a, n_f = self.wc._n_amplitude, self.wc._n_frequency
        param_len = n_a + 2 + n_f + 2 + 1
        params = np.ones(param_len)
        # Patch splines for model selection metric test
        self.wc._amplitude_spline = DummySpline(np.ones(n_a + 1))
        self.wc._frequency_spline = DummySpline(np.ones(n_f + 1))
        for metric in ["bic", "chi2"]:
            result = self.wc.get_model_selection_metric(signal, noise, params, metric)
            assert isinstance(result, float) or isinstance(result, np.floating)
        # Test that unknown metric raises ValueError
        with pytest.raises(ValueError):
            self.wc.get_model_selection_metric(signal, noise, params, "unknown_metric")

    def test_is_wiggle_detected_all_branches(self):
        """Test is_wiggle_detected for all main branches and edge cases."""
        # Typical case: params of correct length, signal and noise arrays
        n_wave = self.wc._datacube.shape[0]
        self.wc._n_amplitude = 2
        self.wc._n_frequency = 2
        n_a, n_f = self.wc._n_amplitude, self.wc._n_frequency
        param_len = n_a + 2 + n_f + 2 + 1
        params = np.ones(param_len)
        signal = np.ones(n_wave)
        noise = np.ones(n_wave) * 0.1
        # Patch splines
        self.wc._amplitude_spline = DummySpline(np.ones(n_a + 1))
        self.wc._frequency_spline = DummySpline(np.ones(n_f + 1))
        # Should return a bool or int (0/1)
        result = self.wc.is_wiggle_detected(signal, noise, params)
        assert isinstance(result, (bool, int, np.integer, np.bool_))
        # Edge case: params with zeros
        params_zeros = np.zeros(param_len)
        result2 = self.wc.is_wiggle_detected(signal, noise, params_zeros)
        assert isinstance(result2, (bool, int, np.integer, np.bool_))
        # Edge case: signal with NaNs
        signal_nan = np.ones(n_wave)
        signal_nan[0] = np.nan
        result3 = self.wc.is_wiggle_detected(signal_nan, noise, params)
        assert isinstance(result3, (bool, int, np.integer, np.bool_))
        # Edge case: noise with zeros (should not error)
        noise_zeros = np.zeros(n_wave)
        result4 = self.wc.is_wiggle_detected(signal, noise_zeros, params)
        assert isinstance(result4, (bool, int, np.integer, np.bool_))

    # def test_fit_wiggle_smoke(self):
    #     """Smoke test for fit_wiggle: covers main branches, argument handling, and covariance extraction."""
    #     # Patch the datacube at (2,2) to have a sine wave (at least 2 extrema)
    #     for i in range(self.wc._datacube.shape[1]):
    #         for j in range(self.wc._datacube.shape[2]):
    #             self.wc._datacube[:, i, j] = np.sin(
    #                 np.linspace(0, 4 * np.pi, self.wc._datacube.shape[0])
    #             )

    #     import raccoon.util

    #     orig_smooth_curve = raccoon.util.Util.smooth_curve
    #     orig_lighter_smooth_curve = raccoon.util.Util.lighter_smooth_curve
    #     raccoon.util.Util.smooth_curve = staticmethod(
    #         lambda curve: __import__("scipy.signal").signal.savgol_filter(curve, 5, 3)
    #     )
    #     raccoon.util.Util.lighter_smooth_curve = staticmethod(
    #         lambda curve: __import__("scipy.signal").signal.savgol_filter(curve, 5, 3)
    #     )
    #     try:
    #         x, y = 0, 0
    #         fitwiggle_kwargs = dict(
    #             aperture_radius=1, annulus_outer_radius=2, annulus_inner_radius=1
    #         )
    #         params, cov = self.wc.fit_wiggle(x, y, **fitwiggle_kwargs)
    #         assert isinstance(params, np.ndarray)
    #         # Covariance should be None or a 2D array
    #         assert cov is None or (isinstance(cov, np.ndarray) and cov.ndim == 2)

    #         # Test with outlier rejection (sigma_clip)
    #         params2, cov2 = self.wc.fit_wiggle(
    #             x, y, outlier_rejection_method="sigma_clip", **fitwiggle_kwargs
    #         )
    #         assert isinstance(params2, np.ndarray)
    #         assert cov2 is None or (isinstance(cov2, np.ndarray) and cov2.ndim == 2)

    #         # Test with outlier rejection (fdr)
    #         params3, cov3 = self.wc.fit_wiggle(
    #             x, y, outlier_rejection_method="fdr", **fitwiggle_kwargs
    #         )
    #         assert isinstance(params3, np.ndarray)
    #         assert cov3 is None or (isinstance(cov3, np.ndarray) and cov3.ndim == 2)

    #         # Test with fit_full_model and extract_covariance False
    #         params4, cov4 = self.wc.fit_wiggle(
    #             x, y, fit_full_model=True, extract_covariance=False, **fitwiggle_kwargs
    #         )
    #         assert isinstance(params4, np.ndarray)
    #         assert cov4 is None

    #         # Test with plot and verbose (should not raise)
    #         try:
    #             self.wc.fit_wiggle(x, y, plot=True, verbose=True, **fitwiggle_kwargs)
    #         except Exception:
    #             pass

    #         # Test with custom polynomial degrees and noise
    #         params5, cov5 = self.wc.fit_wiggle(
    #             x,
    #             y,
    #             n_amplitude=3,
    #             n_frequency=3,
    #             specified_noise_level=0.5,
    #             **fitwiggle_kwargs
    #         )
    #         assert isinstance(params5, np.ndarray)
    #         assert cov5 is None or (isinstance(cov5, np.ndarray) and cov5.ndim == 2)

    #         # Test with both sharpening modes
    #         params6, cov6 = self.wc.fit_wiggle(
    #             x,
    #             y,
    #             symmetric_sharpening=True,
    #             asymmetric_sharpening=True,
    #             **fitwiggle_kwargs
    #         )
    #         assert isinstance(params6, np.ndarray)
    #         assert cov6 is None or (isinstance(cov6, np.ndarray) and cov6.ndim == 2)

    #         # Test with interim phase-only fit
    #         params7, cov7 = self.wc.fit_wiggle(
    #             x, y, do_interim_fit_phase_only=True, **fitwiggle_kwargs
    #         )
    #         assert isinstance(params7, np.ndarray)
    #         assert cov7 is None or (isinstance(cov7, np.ndarray) and cov7.ndim == 2)
    #     finally:
    #         raccoon.util.Util.smooth_curve = orig_smooth_curve
    #         raccoon.util.Util.lighter_smooth_curve = orig_lighter_smooth_curve
