# Non-Gaussian-PDF-Modelling
Non-Gaussian pdfs: how to model them, simulate from them, and include them as Bayesian Priors

This repo contains methods for modelling Non-Gaussian pdfs.

Functions describing a population are often selected from a well-behaved, well-understood family of distributions (e.g. Gaussian, exponential, uniform, truncated Gaussian, Cauchy, Chi-squared etc.). However, pdfs obtained in real-world scenarios are sometimes messy, and poorly described by these well-behaved functions; in these scenarios, methods that can parameterise a 'no-name' distribution are extremely valuable. This repo is inspired by Ben Goodrich's presentation here at StanCon2020: https://www.youtube.com/watch?v=_wfZSvasLFk

The methodology is simple and proceeds like so: any distribution has a CDF, which is a monotonically increasing function. The shape of the CDF can be characterised by quantiles, i.e. summing up the non-Gaussian pdf probability from x=-infty to x=infty. With these quantile coordinates in hand, you can flip the x and y axes, and you have the range u=0 to u=1 on the x-axis, and the ICDF on the y-axis, i.e. the inverse cumulative distribution function. 

The magic is then to take draws from a uniform distribution u\~U(0,1), and plug these u-values into the ICDF. The draws x=ICDF(u) where u\~U(0,1) are equivalent to draws from the original non-Gaussian pdf, x\~pdf(x). Therefore, one need only derive a parametric model for the (relatively smooth) monotically increasing ICDF function, and you now have access to a no-name distribution in your computer (just as you would a Gaussian, uniform, exponential etc.). 


The accuracy of your new ICDF approximation depends on the accuracy of the parametric model, which in turn depends on two things: the number of quantiles you compute, and how reliable your quantiles are; the latter depends on the number of data points that form your original non-Gaussian pdf (i.e. how reflective your no-name distribution is of the 'true' underlying pdf.)
