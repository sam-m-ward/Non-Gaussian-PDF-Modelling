# Non-Gaussian-PDF-Modelling
*Non-Gaussian pdfs: how to model them, simulate from them, and include them as Bayesian Priors*

This repo project contains methods for modelling Non-Gaussian pdfs.

## Motivation

Functions describing a population are often selected from a well-behaved, well-understood family of distributions (e.g. Gaussian, exponential, uniform, truncated Gaussian, Cauchy, Chi-squared etc.). However, pdfs obtained in real-world scenarios are sometimes messy, and poorly described by these well-behaved functions; in these scenarios, methods that can parameterise a 'no-name' distribution are extremely valuable. This repo is inspired by Ben Goodrich's presentation here at StanCon2020: https://www.youtube.com/watch?v=_wfZSvasLFk

## Methodology

The methodology is simple and proceeds like so: any distribution has a CDF, which is a monotonically increasing function from 0 to 1. The shape of the CDF is characterised by quantiles, i.e. summing up the non-Gaussian pdf probability from x=-infty to x=infty. With these quantile coordinates in hand, you can flip the x and y axes, so that the range u=0 to u=1 is on the x-axis, and the ICDF is on the y-axis (the inverse cumulative distribution function). 

The magic is then to take draws from a uniform distribution u\~U(0,1), and plug these u-values into the ICDF. The draws x=ICDF(u) where u\~U(0,1) are equivalent to draws from the original non-Gaussian pdf: x\~pdf(x). Therefore, one need only derive a parametric model for the (relatively smooth) monotically increasing ICDF function, and you now have access to a no-name distribution in your computer (just as you would a Gaussian, uniform, exponential etc.). 

The accuracy of your new ICDF approximation depends on the accuracy of the parametric model, which in turn depends on two things: the number of quantiles you compute, and how reliable your quantiles are; the latter depends on the number of data points that form your original non-Gaussian pdf (i.e. how reflective your no-name distribution is of the 'true' underlying pdf.) 

In this repo, I use monotonically increasing cubic splines to interpolate the ICDF quantiles, and build a parametric model.

## Application

I explore two use-cases for non-Gaussian pdf sampling:
1. Building a distribution from a histogram comprised of ~100's of real-world data points (distance errors from fits to SN Ia light curves); [**see use_case_1.ipynb**](https://github.com/sam-m-ward/Non-Gaussian-PDF-Modelling/blob/main/use_case_1.ipynb)
2. Implementing non-Gaussian (photometric redshift) pdfs into a hierarchical Bayesian model (BayeSN); [**see use_case_2.ipynb**](https://github.com/sam-m-ward/Non-Gaussian-PDF-Modelling/blob/main/use_case_2.ipynb)
