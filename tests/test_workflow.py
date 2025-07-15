import numpy as np
from astropy.io import fits
from raccoon import WiggleCleaner
import pytest


class DummySpline:
    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        return np.ones_like(x)


class TestWorkflow:
    @classmethod
    def setup_class(cls):
        datacube_file = "example/example_data.fits"
        cls.data_cube, cls.header = fits.getdata(datacube_file, header=True)
        cls.noise_cube = fits.getdata(datacube_file, ext=1)
        cls.wavelengths = (
            cls.header["CRVAL3"]
            + cls.header["CDELT3"] * np.arange(cls.header["NAXIS3"])
        ) * 1e4
        cls.quasar_x, cls.quasar_y = 16, 16
        cls.wcleaner = WiggleCleaner(
            cls.wavelengths,
            cls.data_cube,
            cls.noise_cube,
            continuum_diff_polynomial_order=2,
            symmetric_sharpening=True,
            asymmetric_sharpening=True,
        )

    def test_data_shapes(self):
        assert self.data_cube.shape == self.noise_cube.shape
        assert self.data_cube.shape[0] == self.wavelengths.shape[0]

    def test_get_wiggle_signal_apertures(self):
        for s in np.arange(1, 3, 1):
            wiggle_signal, wiggle_noise = self.wcleaner.get_wiggle_signal(
                self.quasar_x,
                self.quasar_y,
                aperture_radius=s,
                annulus_outer_radius=s + 1,
                annulus_inner_radius=s - 1,
            )
            assert wiggle_signal.shape == self.wavelengths.shape
            assert wiggle_noise.shape == self.wavelengths.shape

    def test_get_wiggle_signal_annuli(self):
        for i in range(0, 2, 1):
            wiggle_signal, wiggle_noise = self.wcleaner.get_wiggle_signal(
                self.quasar_x,
                self.quasar_y,
                aperture_radius=4,
                annulus_outer_radius=4 - i + 2,
                annulus_inner_radius=4 - i,
            )
            assert wiggle_signal.shape == self.wavelengths.shape
            assert wiggle_noise.shape == self.wavelengths.shape

    def test_get_wiggle_signal_inner_annuli(self):
        for i in range(0, 2, 1):
            wiggle_signal, wiggle_noise = self.wcleaner.get_wiggle_signal(
                self.quasar_x,
                self.quasar_y,
                aperture_radius=4,
                annulus_outer_radius=5,
                annulus_inner_radius=5 - i,
            )
            assert wiggle_signal.shape == self.wavelengths.shape
            assert wiggle_noise.shape == self.wavelengths.shape

    def test_fit_wiggle(self):
        aperture_radius = 4
        annulus_outer_radius = 5
        annulus_inner_radius = 3
        # Use a gap to trigger ax.axvspan and ax2.axvspan coverage
        self.wcleaner.set_gaps([(self.wavelengths[10], self.wavelengths[20])])
        result_params = self.wcleaner.fit_wiggle(
            x=self.quasar_x,
            y=self.quasar_y,
            aperture_radius=aperture_radius,
            annulus_outer_radius=annulus_outer_radius,
            annulus_inner_radius=annulus_inner_radius,
            n_amplitude=8,
            n_frequency=5,
            init_peak_detection_proximity_threshold=30,
            use_huber_loss=True,
            outlier_rejection_method="fdr",
            fdr_alpha=0.05,
            fdr_outlier_max_fraction=0.2,
            extract_covariance=True,
            fit_full_model=True,
            verbose=True,
            plot=True,
            save_figure_dir="./",
        )
        assert isinstance(result_params, tuple)
        assert isinstance(result_params[0], np.ndarray)
        # Reset gaps for other tests
        self.wcleaner.set_gaps([])

    def test_fit_wiggle_with_sharpening(self):
        """Test that fit_wiggle covers sharpening parameter branches."""
        # Patch Util.get_init_params_spline to always return finite, positive arrays
        from raccoon import util as raccoon_util

        def dummy_get_init_params_spline(
            wiggle_signal,
            scaled_w,
            n_amplitude,
            n_frequency,
            init_peak_detection_proximity_threshold,
            plot,
        ):
            # Return dummy splines and positive params
            c_len = n_amplitude + 2
            f_len = n_frequency + 2
            amp_spline = DummySpline(np.ones(c_len))
            freq_spline = DummySpline(np.ones(f_len))
            phi_0 = 0.0
            return amp_spline, freq_spline, phi_0

        old_get_init_params_spline = raccoon_util.Util.get_init_params_spline
        raccoon_util.Util.get_init_params_spline = staticmethod(
            dummy_get_init_params_spline
        )

        try:
            aperture_radius = 4
            annulus_outer_radius = 5
            annulus_inner_radius = 3

            # symmetric sharpening only
            result_params = self.wcleaner.fit_wiggle(
                x=self.quasar_x,
                y=self.quasar_y,
                aperture_radius=aperture_radius,
                annulus_outer_radius=annulus_outer_radius,
                annulus_inner_radius=annulus_inner_radius,
                plot=False,
                n_amplitude=8,
                n_frequency=5,
                init_peak_detection_proximity_threshold=30,
                verbose=False,
                use_huber_loss=False,
                outlier_rejection_method="fdr",
                fdr_alpha=0.05,
                fdr_outlier_max_fraction=0.2,
                extract_covariance=True,
                fit_full_model=True,
                symmetric_sharpening=True,
                asymmetric_sharpening=False,
            )
            assert isinstance(result_params, tuple)
            assert isinstance(result_params[0], np.ndarray)

            # asymmetric sharpening only
            result_params = self.wcleaner.fit_wiggle(
                x=self.quasar_x,
                y=self.quasar_y,
                aperture_radius=aperture_radius,
                annulus_outer_radius=annulus_outer_radius,
                annulus_inner_radius=annulus_inner_radius,
                plot=False,
                n_amplitude=8,
                n_frequency=5,
                init_peak_detection_proximity_threshold=30,
                verbose=False,
                use_huber_loss=False,
                outlier_rejection_method="fdr",
                fdr_alpha=0.05,
                fdr_outlier_max_fraction=0.2,
                extract_covariance=True,
                fit_full_model=True,
                asymmetric_sharpening=True,
            )
            assert isinstance(result_params, tuple)
            assert isinstance(result_params[0], np.ndarray)

            # both symmetric and asymmetric sharpening
            result_params = self.wcleaner.fit_wiggle(
                x=self.quasar_x,
                y=self.quasar_y,
                aperture_radius=aperture_radius,
                annulus_outer_radius=annulus_outer_radius,
                annulus_inner_radius=annulus_inner_radius,
                plot=False,
                n_amplitude=8,
                n_frequency=5,
                init_peak_detection_proximity_threshold=30,
                verbose=False,
                use_huber_loss=False,
                outlier_rejection_method="fdr",
                fdr_alpha=0.05,
                fdr_outlier_max_fraction=0.2,
                extract_covariance=True,
                fit_full_model=True,
                asymmetric_sharpening=True,
                symmetric_sharpening=True,
            )
            assert isinstance(result_params, tuple)
            assert isinstance(result_params[0], np.ndarray)
        finally:
            raccoon_util.Util.get_init_params_spline = old_get_init_params_spline

    def test_fit_wiggle_with_model_selection(self):
        aperture_radius = 4

        # with bic
        self.wcleaner.fit_wiggle_with_model_selection(
            15,  # quasar_x
            15,  # quasar_y
            aperture_radius=aperture_radius,
            annulus_outer_radius=aperture_radius + 1,
            annulus_inner_radius=aperture_radius - 1,
            plot=True,
            n_amplitude=5,
            n_frequency=3,
            min_n_amplitude=5,
            min_n_frequency=2,
            selection_criteria="bic",
            init_peak_detection_proximity_threshold=200,
            outlier_rejection_method="fdr",
            use_huber_loss=False,
            fdr_alpha=0.01,
            fdr_outlier_max_fraction=0.10,
            extract_covariance=True,
            fit_full_model=True,
        )

        # with chi2
        self.wcleaner.fit_wiggle_with_model_selection(
            15,  # quasar_x
            15,  # quasar_y
            aperture_radius=aperture_radius,
            annulus_outer_radius=aperture_radius + 1,
            annulus_inner_radius=aperture_radius - 1,
            plot=False,
            n_amplitude=5,
            n_frequency=3,
            min_n_amplitude=5,
            min_n_frequency=2,
            selection_criteria="chi2",
            init_peak_detection_proximity_threshold=200,
            outlier_rejection_method="fdr",
            use_huber_loss=False,
            fdr_alpha=0.01,
            fdr_outlier_max_fraction=0.10,
            extract_covariance=True,
            fit_full_model=True,
        )

        # with sigma_clip
        self.wcleaner.fit_wiggle_with_model_selection(
            15,  # quasar_x
            15,  # quasar_y
            aperture_radius=aperture_radius,
            annulus_outer_radius=aperture_radius + 1,
            annulus_inner_radius=aperture_radius - 1,
            plot=False,
            n_amplitude=5,
            n_frequency=3,
            min_n_amplitude=5,
            min_n_frequency=2,
            selection_criteria="bic",
            init_peak_detection_proximity_threshold=200,
            outlier_rejection_method="sigma_clip",
            sigma_clip_sigma=3,
            sigma_clip_max_iterations=2,
            use_huber_loss=True,
            fdr_alpha=0.01,
            fdr_outlier_max_fraction=0.10,
            extract_covariance=True,
            fit_full_model=True,
        )

        with pytest.raises(ValueError):
            self.wcleaner.fit_wiggle_with_model_selection(
                15,  # quasar_x
                15,  # quasar_y
                aperture_radius=aperture_radius,
                annulus_outer_radius=aperture_radius + 1,
                annulus_inner_radius=aperture_radius - 1,
                plot=False,
                n_amplitude=5,
                n_frequency=3,
                min_n_amplitude=5,
                min_n_frequency=1,
                selection_criteria="bic",
                init_peak_detection_proximity_threshold=200,
                outlier_rejection_method="sigma_clip",
                sigma_clip_sigma=3,
                sigma_clip_max_iterations=2,
                use_huber_loss=True,
                fdr_alpha=0.01,
                fdr_outlier_max_fraction=0.10,
                extract_covariance=True,
                fit_full_model=True,
            )

        with pytest.raises(ValueError):
            self.wcleaner.fit_wiggle_with_model_selection(
                15,  # quasar_x
                15,  # quasar_y
                aperture_radius=aperture_radius,
                annulus_outer_radius=aperture_radius + 1,
                annulus_inner_radius=aperture_radius - 1,
                plot=False,
                n_amplitude=5,
                n_frequency=3,
                min_n_amplitude=1,
                min_n_frequency=2,
                selection_criteria="bic",
                init_peak_detection_proximity_threshold=200,
                outlier_rejection_method="sigma_clip",
                sigma_clip_sigma=3,
                sigma_clip_max_iterations=2,
                use_huber_loss=True,
                fdr_alpha=0.01,
                fdr_outlier_max_fraction=0.10,
                extract_covariance=True,
                fit_full_model=True,
            )

    def test_clean_cube(self):
        aperture_radius = 4
        annulus_outer_radius = 5
        annulus_inner_radius = 3
        mask = np.zeros_like(self.data_cube[0], dtype=bool)
        center_x = self.quasar_x
        center_y = self.quasar_y
        radius = 1
        for i in range(center_x - 2 * radius, center_x + 2 * radius):
            for j in range(center_y - 2 * radius, center_y + 2 * radius):
                if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius**2:
                    mask[i, j] = True

        cleaned_cube, cleaned_noise_cube, cleaned_map = self.wcleaner.clean_cube(
            wiggle_detection_sigma_threshold=5.0,
            wiggle_detection_variance_ratio_threshold=0.2,
            n_amplitude=10,
            n_frequency=7,
            fit_full_model=True,
            min_n_amplitude=None,
            min_n_frequency=None,
            cleaning_mask=mask,
            init_peak_detection_proximity_threshold=200,
            aperture_radius=aperture_radius,
            annulus_outer_radius=annulus_outer_radius,
            annulus_inner_radius=annulus_inner_radius,
            plot=False,
            verbose=True,
            extract_uncertainty=True,
            outlier_rejection_method="fdr",
            use_huber_loss=False,
            fdr_alpha=0.01,
            fdr_outlier_max_fraction=0.15,
            sigma_clip_sigma=3,
            sigma_clip_max_iterations=20,
            num_samples_uncertainty_region=1000,
        )
        assert cleaned_cube.shape == self.data_cube.shape
        assert cleaned_noise_cube.shape == self.data_cube.shape
        assert cleaned_map.shape == self.data_cube[0].shape

    def test_clean_cube_no_wiggle_detection(self):
        """Test clean_cube with too high wiggle detection criteria."""
        aperture_radius = 4
        annulus_outer_radius = 5
        annulus_inner_radius = 3
        mask = np.zeros_like(self.data_cube[0], dtype=bool)
        center_x = self.quasar_x
        center_y = self.quasar_y
        radius = 1
        for i in range(center_x - 2 * radius, center_x + 2 * radius):
            for j in range(center_y - 2 * radius, center_y + 2 * radius):
                if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius**2:
                    mask[i, j] = True

        cleaned_cube, cleaned_noise_cube, cleaned_map = self.wcleaner.clean_cube(
            wiggle_detection_sigma_threshold=1000.0,
            wiggle_detection_variance_ratio_threshold=1.0,
            n_amplitude=10,
            n_frequency=7,
            fit_full_model=True,
            min_n_amplitude=None,
            min_n_frequency=None,
            cleaning_mask=mask,
            init_peak_detection_proximity_threshold=200,
            aperture_radius=aperture_radius,
            annulus_outer_radius=annulus_outer_radius,
            annulus_inner_radius=annulus_inner_radius,
            plot=False,
            verbose=True,
            extract_uncertainty=True,
            outlier_rejection_method="fdr",
            use_huber_loss=False,
            fdr_alpha=0.01,
            fdr_outlier_max_fraction=0.15,
            sigma_clip_sigma=3,
            sigma_clip_max_iterations=20,
            num_samples_uncertainty_region=1000,
        )
        assert cleaned_cube.shape == self.data_cube.shape
        assert cleaned_noise_cube.shape == self.data_cube.shape
        assert cleaned_map.shape == self.data_cube[0].shape

    def test_clean_cube_with_min_n_amplitude_and_min_n_frequency(self):
        """Test clean_cube with min_n_amplitude and min_n_frequency."""
        aperture_radius = 4
        annulus_outer_radius = 5
        annulus_inner_radius = 3
        mask = np.zeros_like(self.data_cube[0], dtype=bool)
        center_x = self.quasar_x
        center_y = self.quasar_y
        radius = 1
        for i in range(center_x - 2 * radius, center_x + 2 * radius):
            for j in range(center_y - 2 * radius, center_y + 2 * radius):
                if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius**2:
                    mask[i, j] = True

        cleaned_cube, cleaned_noise_cube, cleaned_map = self.wcleaner.clean_cube(
            wiggle_detection_sigma_threshold=5.0,
            wiggle_detection_variance_ratio_threshold=0.2,
            n_amplitude=10,
            n_frequency=7,
            fit_full_model=True,
            min_n_amplitude=9,
            min_n_frequency=7,
            cleaning_mask=mask,
            init_peak_detection_proximity_threshold=200,
            aperture_radius=aperture_radius,
            annulus_outer_radius=annulus_outer_radius,
            annulus_inner_radius=annulus_inner_radius,
            plot=False,
            verbose=True,
            extract_uncertainty=True,
            outlier_rejection_method="fdr",
            use_huber_loss=False,
            fdr_alpha=0.01,
            fdr_outlier_max_fraction=0.15,
            sigma_clip_sigma=3,
            sigma_clip_max_iterations=20,
            num_samples_uncertainty_region=1000,
        )
        assert cleaned_cube.shape == self.data_cube.shape
        assert cleaned_noise_cube.shape == self.data_cube.shape
        assert cleaned_map.shape == self.data_cube[0].shape

    def test_clean_cube_with_wiggle_detection_thresholds(self):
        """Test clean_cube with wiggle_detection_sigma_threshold and
        wiggle_detection_variance_ratio_threshold."""
        aperture_radius = 4
        annulus_outer_radius = 5
        annulus_inner_radius = 3
        mask = np.zeros_like(self.data_cube[0], dtype=bool)
        center_x = self.quasar_x
        center_y = self.quasar_y
        radius = 1
        for i in range(center_x - 2 * radius, center_x + 2 * radius):
            for j in range(center_y - 2 * radius, center_y + 2 * radius):
                if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius**2:
                    mask[i, j] = True

        # when n_amplitude_for_detection is not equal to n_amplitude
        cleaned_cube, cleaned_noise_cube, cleaned_map = self.wcleaner.clean_cube(
            wiggle_detection_sigma_threshold=5.0,
            wiggle_detection_variance_ratio_threshold=0.2,
            n_amplitude=10,
            n_frequency=7,
            n_amplitude_for_detection=8,
            # n_frequency_for_detection=4,
            fit_full_model=True,
            min_n_amplitude=None,
            min_n_frequency=None,
            cleaning_mask=mask,
            init_peak_detection_proximity_threshold=200,
            aperture_radius=aperture_radius,
            annulus_outer_radius=annulus_outer_radius,
            annulus_inner_radius=annulus_inner_radius,
            plot=False,
            verbose=True,
            extract_uncertainty=True,
            outlier_rejection_method="fdr",
            use_huber_loss=False,
            fdr_alpha=0.01,
            fdr_outlier_max_fraction=0.15,
            sigma_clip_sigma=3,
            sigma_clip_max_iterations=20,
            num_samples_uncertainty_region=1000,
        )
        assert cleaned_cube.shape == self.data_cube.shape
        assert cleaned_noise_cube.shape == self.data_cube.shape
        assert cleaned_map.shape == self.data_cube[0].shape
