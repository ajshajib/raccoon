---
title: 'raccoon: A Python package for removing wiggle artifacts in the JWST NIRSpec integral field spectroscopy'
tags:
  - Python
  - astronomy
  - JWST
  - NIRSpec
  - spectroscopy
  - data reduction
  - IFU
authors:
  - name: Anowar J. Shajib
    orcid: 0000-0002-5558-888X
    affiliation: "1, 2, 3"
affiliations:
 - name: Department of Astronomy & Astrophysics, University of Chicago, Chicago, IL 60637, USA
   index: 1
   ror: 024mw5h28
 - name: Kavli Institute for Cosmological Physics, University of Chicago, Chicago, IL 60637, USA
   index: 2
 - name: Center for Astronomy, Space Science and Astrophysics, Independent University, Bangladesh, Dhaka 1229, Bangladesh
   index: 3
date: 14 July 2025
bibliography: paper.bib

---

# Summary

`raccoon` is a Python package for removing resampling noise — commonly referred to as "wiggles" — from spaxel-level spectra in datacubes obtained from JWST's Near Infrared Spectrograph's (NIRSpec) integral field spectroscopy (IFS) mode. These wiggles arise as artifacts during resampling of the 2D raw data into 3D datacubes, due to the point spread function (PSF) being undersampled. The standard JWST data reduction pipeline does not correct for this noise. The wiggle artifacts can significantly degrade the scientific usability of the data, particularly at the pixel level, undermining the exquisite spatial resolution of NIRSpec. `raccoon` provides an empirical correction by modeling and removing these artifacts, thereby restoring the fidelity of the extracted spectra. `raccoon` forward-models the wiggles as a chirp function impacting one or more template spectra that are directly fit to the original data across the entire wavelength range. The best-fit wiggle model is then used to clean the data while propagating the associated uncertainties.

# Statement of need

The JWST NIRSpec's IFS mode [@Boker23] enables spatially resolved infrared spectroscopy of astronomical sources with an unprecedented combination of signal-to-noise ratio, redshift coverage, and spatial resolution. However, it suffers from heavy undersampling of the point-spread function, leading to resampling noise that manifests as low-frequency sinusoidal artifacts known as "wiggles." These artifacts can significantly distort the overall spectral shape, bias line measurements, and compromise kinematic analyses at the single-spaxel level, thereby limiting the scientific potential of the NIRSpec IFU data. `raccoon` provides a user-friendly, robust, and computationally efficient solution to identify and remove these artifacts, enabling precise studies of galaxy kinematics and early-Universe phenomena that would otherwise be hindered by the presence of wiggles. `raccoon` has already been used in scientific publications to facilitate robust measurements of stellar kinematics from JWST NIRSpec spectra [@TDCOSMO25; @Shajib25].

# Functionality

Since the wiggles in the single-spaxel spectra in the reduced JWST NIRSpec datacube are artifacts or resampling noise arising due to undersampling the PSF, dithering can, in principle, mitigate the issue of PSF undersampling and thus the wiggles. However, the commonly adopted 4-point dither pattern is still insufficient to recover the optimal sampling and completely eliminate the wiggles in the rectified datacube [@Law23]. The wiggles are washed out in the spectra summed from multiple spaxels within a sufficiently large aperture (illustrated in \autoref{wiggled-in-spectra}). This provides a basis for making an empirical correction for the wiggles by comparing the single-spaxel spectrum to the one summed within an aperture around it. This principle was also used by previous corrective algorithms, such as in @Perna23 and in the Python routine `WICKED` [@Dumont25].

![Illustration of the wiggles in the single-spaxel spectrum (blue), which is a manifestation of the resampling noise in the standard-pipeline-reduced datacube due to PSF undersampling. The red spectrum shows the aperture-summed spectrum within a 4-spaxel radius, normalized to match its maximum to that of the single-spaxel one. The illustrated data are of an active galactic nucleus (AGN) from @Perna23.](./single_spaxel_vs_aperture_summed.png){#wiggled-in-spectra}

In `raccoon`, the wiggle is modeled as a sinusoidal chirp function
$$ W(\lambda) = 1 + A(\lambda) \left[ \sin \phi(\lambda) + a_1 \sin^2 \phi (\lambda) + a_2 \sin(3\phi (\lambda)) \right], \label{wiggle} $$
where $A(\lambda)$ is the wavelength-dependent amplitude and $\phi (\lambda) = \lambda\,k(\lambda) + \phi_0$ is the phase that depends on a wavelength-dependent wavenumber $k(\lambda)$. The wiggle's peaks and troughs can be asymmetrically and symmetrically sharpened (or de-sharpened) by varying the $a_1$ and $a_2$ parameters, respectively.

Given this parametrization for the wiggle, a single-spaxel spectrum $D(\lambda)$ is modeled with
$$ M (\lambda) = W(\lambda) \, T(\lambda), \label{model} $$
where $T(\lambda)$ is a template for the correct spectrum devoid of the wiggles. In `raccoon`, this template is constructed based on the circular-aperture-summed spectrum $C(\lambda)$. The template can also optionally include a spectrum $S(\lambda)$ summed from a shell or annulus centered on the corresponding spaxel, following @Dumont25. Including the shell-summed spectrum in the template can account for changes in the line shape between the single-spaxel spectrum and the aperture-summed one, for example, in the case of lines broadened by stellar kinematics that can vary across the galaxy [@Dumont25]. The aperture radius and the inner and outer radii of the shell are to be adjusted by the user, as the appropriate values can depend on the source morphology and the astronomical scene. The template is constructed as
$$ T(\lambda) = c_1\,C(\lambda) + c_2 \, S(\lambda) + c_3 \,\lambda^b + \sum_{n=0}^N p_n\, \lambda^n ,  \label{template} $$
where $c_1,c_2, c_3, p_0, \dots, p_N$ are linear coefficients. Here, the power-law plus polynomial term (i.e., $c_3 \,\lambda^b + \sum_{n=0}^N p_n\, \lambda^n$) on the right-hand side models the difference in the continuum between the single-spaxel spectrum and $c_1\,C(\lambda) + c_2 \, S(\lambda)$.

The functions $A(\lambda)$ and $k(\lambda)$ are modeled with B-splines, with the number of knots adjustable by the user. `raccoon` provides functionality for the user to determine the appropriate number of knots using model selection criteria such as the Bayesian information criterion (BIC) or the minimum *a posteriori* chi-squared metric ($\chi^2_{\rm MAP}$). The best-fit values for the linear coefficients (i.e., $c_1, c_2, c_3, p_0, \dots, p_N$) and non-linear parameters (i.e., $a_1$, $a_2$, $b$ and the coefficients of the B-spline basis functions) are determined by minimizing the chi-squared quantity
$$ \chi^2 = \sum_{i} \frac{(D_i - M_i)^2}{\sigma_i^2}, \label{chisquare} $$
where the index $i$ runs across the wavelength pixels and $\sigma_{i}$ is the associated noise level. \autoref{full-fit-example} illustrates an example of the fitted model to a given spectrum (of an AGN). In this example, `raccoon`'s robust performance is demonstrated in fitting the given spectrum while modeling the wiggle signal (\autoref{wiggle-model-example}). The user can mask out regions of the spectrum that have strong features potentially impacting the quality of the fit, or optionally adopt outlier rejection in the fit using the false discovery rate method [@Benjamini95] or sigma-clipping.

![Modeling of the full spectrum (blue in the top panel) based on the template spectra and the wiggles impacting it. The illustrated spectrum is the same one from \autoref{wiggled-in-spectra}. The best-fit model is shown in orange and the wiggle-corrected spectrum is shown in black. The bottom panel illustrates the residuals (green) between the original data and the best-fit model.](wiggle_full_fit_16_16.png){#full-fit-example}

![Illustration of the extracted and modeled wiggle signal. The illustrated data points represent $D(\lambda)/M(\lambda)$ based on the best-fit $M(\lambda)$, and the orange line illustrates the best-fit wiggle model $W(\lambda)$. The grey points are rejected outliers using the false discovery rate method, and the blue points mark the wavelengths where the data were fit to the model.](wiggle_model_16_16.png){#wiggle-model-example}

`raccoon` enables users to efficiently process multiple spaxels in a datacube, either across the entire field or within a specified spatial region. A user-defined detection threshold for the wiggle signal can be set, ensuring that corrections are only applied to spectra where significant wiggles are present.

`raccoon` offers several advantages over previously available scripts and routines. Earlier tools [@Perna23; @Dumont25] extract the wiggle signal by subtracting the single-spaxel spectrum from a template spectrum (similar to or the same as that in Eq. \ref{template}) fitted to the data without accounting for wiggles. This approach can result in the loss of a fraction of the wiggle signal during numerical fitting. In contrast, `raccoon` fits the template spectrum and the parametric wiggle model $W(\lambda)$ simultaneously to the data, eliminating the need for an intermediate extraction step and preserving the full wiggle signal. While previous scripts perform fitting separately in multiple wavelength slices, `raccoon` fits across the entire wavelength range at once. This makes `raccoon` more robust to gaps in the fitted spectrum, whether excluded by manual masking or outlier rejection. Another important distinction is that earlier tools apply an additive correction, whereas `raccoon` uses a multiplicative correction. Since both approaches derive the correction empirically from the data, this difference should not affect performance in practice. However, as the wiggles manifest as a multiplicative effect on the spectra [@Law23], `raccoon`'s implementation is physically motivated. Additionally, `raccoon` is uniquely installable via the `pip` command, allowing more flexible import into Python scripts or notebooks and making it more user-friendly and portable than its peers.

# Acknowledgements

The author acknowledges helpful discussions with Michele Cappellari, Frédéric Courbin, David Law, and Tommaso Treu. AJS received support from NASA through STScI grants JWST-GO-2974 and HST-GO-16773. This work makes use of the `Astropy`, `NumPy`, `SciPy`, and `Matplotlib` packages.

# References