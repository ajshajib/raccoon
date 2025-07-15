---
title: 'raccoon: A package to correct for wiggles in the JWST NIRSpec integral field spectroscopy'
tags:
  - Python
  - astronomy
  - JWST
  - spectroscopy
  - data reduction
  - IFU
authors:
  - name: Anowar J. Shajib
    orcid: 0000-0002-5558-888X
    equal-contrib: true
    affiliation: "1, 2, 3"
affiliations:
 - name: Department  of  Astronomy  \&  Astrophysics,  University  of Chicago, Chicago, IL 60637, USA
   index: 1
   ror: 024mw5h28
 - name: Kavli Institute for Cosmological Physics, University of Chicago, Chicago, IL 60637, USA
   index: 2
 - name: Center for Astronomy, Space Science and Astrophysics, Independent University, Bangladesh, Dhaka 1229, Bangladesh
   index: 3
date: 13 July 2025
bibliography: paper.bib

---

# Summary

`raccoon` is a Python package designed to correct resampling noise — called as "wiggles" — in the reduced spectra from the JWST Near Infrared Spectrograph's (NIRSpec) integral field spectroscopy (IFS) mode. These wiggles arise as artifacts during the resampling of the 2D raw data, affected by undersampling of the point spread function (PSF), into 3D datacubes. The standard JWST data reduction pipeline does not correct for this noise. The wiggle artifacts can significantly degrade the scientific usability of the data at the pixel level, undermining the exquisite spatial resolution of NIRSpec. `raccoon` provides an empirical correction by modeling and removing these artifacts, thereby restoring the fidelity of the extracted spectra. Whereas previously available tools for this purpose first extract a "wiggle signal" from the data to then model it through fast Fourier transforms, `raccoon` models the wiggles as a sinusoidal chirp function impacting a template spectra that is directly fit to the original data, without an intermediate extraction step. As a result, `raccoon` robustly captures the global characteristics of the wiggles while avoiding potential impacts from local imperfections in an extracted wiggle signal.


# Statement of need

The JWST NIRSpec's IFS mode [@Boker23] enables spatially resolved infrared spectroscopy of astronomical sources with an unprecedented combination of signal to noise ratio, redshift, and spatial resolution. However, it suffers from heavy under-sampling of the point-spread function, leading to resampling noise that manifests as low-frequency sinusoidal artifacts known as “wiggles.” These artifacts can significantly distort the overall spectral shape, bias line measurements, and compromise kinematic analyses at the single-pixel level, thereby limiting the scientific potential of the NIRSpec IFU data. `raccoon` provides a user-friendly, robust, and computationally efficient solution to identify and remove these artifacts, enabling precise studies of galaxy kinematics and early universe phenomena that would otherwise be hindered by the presence of the wiggles. `raccoon` has already been used in scientific publications that robustly measured stellar kinematics from the JWST NIRSpec spectra [@TDCOSMO25; @Shajib25].

<!-- The JWST NIRSpec IFU enables spatially resolved spectroscopy of astronomical sources, but the resampling process can introduce systematic, high-frequency artifacts in the extracted spectra. These artifacts, if uncorrected, can bias measurements of emission/absorption lines, continuum shapes, and derived physical properties. Existing reduction pipelines do not fully address this issue, and manual correction is labor-intensive and non-reproducible. `raccoon` fills this gap by providing a flexible, open-source, and well-tested solution for the astronomical community. -->

# Functionality

Since the wiggles in the single-spaxel spectra in the reduced JWST NIRSpec datacube are artifacts due to resampling noise due to PSF undersampling, dithering can be used in principle to mitigate the issue of PSF undersampling and thus the wiggles. However, the typically adopted 4-point dither pattern is still insufficient to recover the optimal sampling and completely eliminate the wiggles in the rectified datacube [@Law23]. The wiggles are washed out in the spectra summed from multiple spaxels within the sufficiently large aperture (illustrated in Fig. \ref{wiggled-in-spectra}). This provides a basis for making an empirical correction for the wiggles by comparing the single-spaxel spectra to the one summed within an aperture around it. This principle was also used by prevous corrective algorithms used by @Perna23 and in the Python routine `WICKED` [@Dumont25].

![Illustration of the wiggles in the single-spaxel spectrum (blue), which is a manifestation of the resampling noise in the standard-pipeline-reduced datacube due to PSF undersampling. The orange spectrum shows the aperture-summed spectrum within a 4-spaxel radius. The illustrated data is a of a quasar from @Perna23.](./single_spaxel_vs_aperture_summed.png){#wiggled-in-spectra}

In `raccoon`, the wiggle is modeled as a sinusoidal chirp function
$$
	W(\lambda) = 1 + A(\lambda) \left[ \sin (\phi_\lambda) + k_1 \sin^2 (\phi_\lambda) + k_2 \sin(3\phi_\lambda) \right],
$$
where $A(\lambda)$ is the wavelength-dependent amplitude and $\phi_\lambda = \lambda\,k(\lambda) + \phi_0$ is wavelength-depedent phase term. The wiggle's peaks and troughs can be asymmetrically and symetercialy sharpened (or, de-sharpened) by freely varying the $k_1$ and $k_2$ parameters, respectively.

Given this model for the wiggle, a single-spaxel spectrum $D(\lambda)$ is modeled with
$$
	M (\lambda) = W(\lambda) \, T(\lambda),
$$
where $T(\lambda)$ is a template for the correct spectrum devoid of the wiggles. In `raccoon`, this template is constructed based on the circular aperture-summed spectra $C(\lambda)$. The can also optionally include a spectra $S(\lambda)$ summed from a shell or annulus centered on the corresponding spaxel, following @Dumont25. Including the shell-summed spectra in the template can account for changes in the line shape between the single spaxel spectrum and the aperture-summed one, for example, in the case of lines broadened by stellar kinematics that can vary across the galaxy [@Dumont25]. The aperture radius and the inner and outer radii of the shell are to be adjusted by the user, as the appropriate values for them depend on the source morphology and the astronomical scene. The template is constructed as
$$
    T(\lambda) = c_1\,C(\lambda) + c_2 \, S(\lambda) + c_3 \,\lambda^b + \sum_{n=0}^N p_n\, \lambda^n ,
$$
where $c_1,c_2, c_3, p_0, \dots, p_N$ are linear coefficients. Here, the power-law plus polynominal $c_3 \,\lambda^b + \sum_{n=0}^N p_n\, \lambda^n$ in the right-hand side models the difference in the continuum between the single-spaxel spectra and $c_1\,C(\lambda) + c_2 \, S(\lambda)$. The best-fit values for the linear coefficients $c_1,c_2, c_3, p_0, \dots, p_N$ and non-linear parameters

The functions $A(\lambda)$ and $\phi_\lambda$ are modeled with B-splines with the number of knots adjustable by the user. `raccoon` provides functionalities for the user to determine the appropriate number of knots using model selection criteria such as the Bayesian information criterion (BIC) or minimum *a posteriori* chi-squared metric ($\chi^2_{\rm MAP}$).  The best-fit values for the linear coefficients $c_1,c_2, c_3, p_0, \dots, p_N$ and non-linear parameters ($b$ and the coefficients of the B-spline basis functions) are determined by minimizing the $\chi^2$ function
$$
\chi^2 = \sum_{i} \frac{(D_i - M_i)^2}{\sigma_i^2},
$$
where the index $i$ runs across the wavelength pixels and $\sigma_{\rm i}$ is the associated noise level. Figure \ref{full-fit-example} illustrate and example of the fitted model to a given spectrum (of an active galactic nucleas). In this example, `raccoon` robust performance is demonstrated in fitting the given spectrum while modeling the wiggle signal (Figure \ref{wiggle-model-example}). The user can mask out regions of spectrum that has strong features potentially impacting the quality of th fit, or optionally adopt an outlier rejection in the fit using the false discovery method [@Benjamini95] or sigma-clipping.

![Modeling of the full spectrum (blue in top panel) based on the template spectra an the wiggles impacting it. The illustrated spectrum is the same one from Fig \ref{wiggle_in_spectrum}. The best-fit model is shown in orange and the wiggle-corrected spectrum is shown in black. The bottom panel illustrates the residual (green) between the original data and the best-fit model.](wiggle_full_fit_16_16.png){#full-fit-example}

![Illustration of the extracted and modeled wiggle signal. The illustrated data points represent $D(\lambda)/M(\lambda)$ based on the best-fit $M(\lambda)$ and the orange line illustrate the best fit wiggle model $W(\lambda)$. The grey points are rejected outliers using the false discovery rate method, and the blue points mark the wavelengths where the data were fit to the model.](wiggle_model_16_16.png){#wiggle-model-example}


`raccoon` additionally allows the user to easily loops through multiple spaxels within the datacube, optionally within a user-specified region. The user can also set a detection threshold for the wiggle signal before making a correction on the given spectra.

There are several advantages of `raccoon` over previously available scripts or tools. Both previous tools [@Perna23; @Dumont25] first extract the wiggle signal by comparing the single-spaxel spectrum with the template spectra and then model the low-frequency behavior of the wiggles through a fast Fourier transformation. `raccoon` fits the model including the wiggles directly to the data, bypassing the need for an intermediate extraction step. Furthermore, `raccoon` employs a single parametric model that is continuous through the fitted wavelength range. This combination of differences makes `raccoon` less susceptible to local imperfections (potentially present in an extracted wiggle signal) or gaps in the wavelength range, either excluded through masking of strong spectral features or through outlier rejection. Another notable difference is that previously available tools makes an additive correction, whereas `raccoon` makes a multiplicative one. However, since both cases obtain the necessary corrrection factor or term empirically from the data, this difference should not put any of them at an disadvantage at a practical level. However, the wiggles manifest as a multiplicative effect on the extracted spectra [@Law23], which is the reason for `raccoon` to model them as such. Furthermore, uniquely among its peers, `raccoon` is installable through the `pip` command and thus it is more user-friendly in its portability and flexibility of use.


# Acknowledgements

We acknowledge helpful discussions with ... This work makes use of the `Astropy`, `NumPy`, `SciPy`, and `Matplotlib` packages. AJS
received support from NASA through STScI grants JWST-GO-2974 and HST-
GO-16773.

# References