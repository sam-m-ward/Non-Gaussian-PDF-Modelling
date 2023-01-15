import matplotlib.pyplot as pl
import numpy as np
import scipy.interpolate

def finish_plot(xlabel,ylabel,title,FS,show=True):
	'''
	Simple function to finish off matplotlib plots
	'''
	if title!='':
		pl.title(title,fontsize=FS)
	pl.xlabel(xlabel,fontsize=FS)
	pl.ylabel(ylabel,fontsize=FS)
	pl.tick_params(labelsize=FS)
	pl.tight_layout()
	if show:
		pl.show()


def get_Cheby_nodes(Np):
	'''
	Get Chebyshev Nodes

	Function to get Chebyshev node (https://en.wikipedia.org/wiki/Chebyshev_nodes).
	Natural way of choosing the locations of quantiles on the range (0,1)

	Parameters
	----------
	Np: integer
		number of quantiles

	Returns
	----------
	nodes: np.array
		locations of quantiles in (0,1)
	'''
	assert type(Np)==int
	k     = np.arange(1,Np+1)[::-1]
	nn    = (2*k-1)/(2*Np)
	nodes = 0.5*(1+np.cos(nn*np.pi))
	return nodes

class nonGauss:
	def __init__(self,samples):
		sorted_samples = list(samples)
		sorted_samples.sort()
		self.samples = sorted_samples

	def plot_samples(self,Nbins=None,alpha=0.5,xlabel='Distance Errors',ylabel='Number',FS=18,show=True,create_new_plot=True):
		'''
		Plot Samples

		Function to plot histogram of samples, i.e. the pdf we wish to approximate

		Parameters
		----------
		Various aesthetic choices, self-explanatory

		Returns
		----------
		None, displays matplotlib plot
		'''
		if create_new_plot:
			pl.figure(figsize=(9.6,7.2))
		if Nbins is not None:
			pl.hist(self.samples,alpha=alpha,bins=Nbins)
		else:
			pl.hist(self.samples,alpha=alpha)
		pl.annotate(r'$N_{points}=%s$'%(len(self.samples)),xy=(0.7,0.7),xycoords='axes fraction',fontsize=FS+2,weight='bold')
		finish_plot(xlabel,ylabel,'No-Name Distribution to Model/Simulate From',FS,show)
		self.hist_ylims = pl.gca().get_ylim()



	def get_ICDF_coords(self,Nquantiles=15,xmin=None,xmax=None,quantiles=None):
		'''
		Get ICDF Coordinates

		Method to get the Nquantiles Coordinates that characterise the CDF of the samples

		Parameters
		----------
		Nquantiles: int (optional; default=15)
			the number of coordinates

		xmin,xmax: floats (optional; default=None)
			truncations on the CDF x-values

		quantiles: list (optional; default=None)
			option to manually input your own set of quantiles

		Returns
		----------
			self.Nquantiles      = Nquantiles
			self.xmin            = xmin
			self.xmax            = xmax
			self.uquantiles      = quantiles
			self.icdf_uquantiles = icdf
		'''
		assert type(Nquantiles)==int

		#Option to set boundaries on the pdf function (e.g. instead of x=-inft to infty, can truncate)
		if xmin==None:
			xmin=np.amin(self.samples)
		if xmax==None:
			xmax=np.amax(self.samples)

		#Get quantiles using CDF
		CDF = np.arange(1,len(self.samples)+1)/len(self.samples)
		if quantiles is None:
			quantiles = [0]+list(get_Cheby_nodes(Nquantiles-2))+[1]

		#Get values of x at each CDF quantile value (i.e. the other coordinate in the coordinate pair)
		icdf = [self.samples[np.argmin(np.abs(CDF-q))] for q in quantiles]


		#Assign class values
		self.Nquantiles      = Nquantiles
		self.xmin            = xmin
		self.xmax            = xmax
		self.uquantiles      = np.asarray(quantiles)#The y-values of the CDF==the x-values of the ICDF
		self.icdf_uquantiles = np.asarray(icdf)	  #The x-values of the CDF==the y-values of the ICDF

	def get_ICDF_parametric_model(self,Npoints=10000):
		'''
		Get ICDF Parametric Model

		Use the quantile coordinates + Monotic Cubic Interpolation to Get ICDF Parametric Model

		Parameters
		----------
		Npoints: int (optional; default=10000)
			Number of points in parametric model

		Returns
		----------
			self.ucont  = np.asarray(ucont)
			self.icdf   = np.asarray(icdf_cont)
		'''
		assert type(Npoints)==int

		ucont      = np.linspace(0,1,10000)
		base_f     = scipy.interpolate.PchipInterpolator(self.uquantiles,self.icdf_uquantiles)#Monotonic Cubic Spline
		icdf_cont  = base_f(ucont)

		self.ucont  = np.asarray(ucont)
		self.icdf   = np.asarray(icdf_cont)

	def plot_CDF(self,FS=18,show=True,create_new_plot=True,rescale=False):
		'''
		Plot up the CDF quantiles and continuous function
		'''
		if create_new_plot:
			pl.figure(figsize=(9.7,7.2))
		fac = {False:1,True:self.hist_ylims[1]}[rescale]
		pl.plot(self.icdf,self.ucont*fac,c='C0')
		pl.scatter(self.icdf_uquantiles,self.uquantiles*fac,c='C1')
		finish_plot('x','CDF','',FS,show)

	def plot_ICDF(self,FS=18,show=True,create_new_plot=True):
		'''
		Plot up the ICDF quantiles and continuous function
		'''
		if create_new_plot:
			pl.figure(figsize=(9.7,7.2))
		pl.plot(self.ucont,self.icdf,c='C0')
		pl.scatter(self.uquantiles,self.icdf_uquantiles,c='C1')
		finish_plot('u','ICDF','Parametric ICDF from Quantiles+Monotonic Cubic Spline',FS,show)
