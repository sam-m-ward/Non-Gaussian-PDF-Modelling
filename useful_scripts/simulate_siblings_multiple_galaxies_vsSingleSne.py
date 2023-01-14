import numpy as np
import scipy.stats
from scipy.stats import truncnorm
import matplotlib.pyplot as pl
import pickle
import os
import copy
from SNmodel import bayesn_model, io
from glob import glob
from astropy.cosmology import FlatLambdaCDM
import scipy.interpolate
import random
#print ('16 and 84 corresponds to 1 sigma')
#print ('2.3 and 97.7 corresponds to 2 sigma')
#err=1/0
def convert_usample_to_indices(u_samples,xcont):
	#u_samples = np.random.uniform(0,1,Nsims)
	u_indices = []
	for u in u_samples:
		curve = abs(xcont-u)
		index = list(curve).index(np.amin(curve))
		u_indices.append(index)
	return u_indices
def get_Cheby_nodes(Np):
	n  = Np#number of points
	k  = np.arange(1,n+1)[::-1]
	nn = (2*k-1)/(2*n)
	nodes = 0.5*(1+np.cos(nn*np.pi))
	return nodes
def get_ICDF_from_samples(samples,Nquantiles=13,zquantiles=None,zphotmin=None,zphotmax=None):
	if zphotmin==None:
		zphotmin=np.amin(samples)
	if zphotmax==None:
		zphotmax=np.amax(samples)
	#zz  = np.linspace(zphotmin,zphotmax,10000)
	#CDF = #np.cumsum(samples)/sum(samples)
	CDF = np.arange(1,len(samples)+1)/len(samples)
	if zquantiles is None:
		zquantiles = [0]+list(get_Cheby_nodes(Nquantiles))+[1]
	zicdf      = []
	for zq in zquantiles:
		curve  = np.abs(CDF-zq)
		zindex = list(curve).index(np.amin(curve))
		zicdf.append(samples[zindex])
	return zquantiles,zicdf
def Gaussian2(mean,sigma,free): return ((2*np.pi*sigma**2)**-0.5)*np.exp(-0.5*((mean-free)/sigma)**2)
def get_KDE(x,Nx,L,U,smoothing=1):#Create mean samples, useful for creating smoothed KDE encompassing error bars,Y is data,x is free xtest
	N,sigma = len(x),np.std(np.asarray(x))
	bw = ((4*sigma**5)/(3*N))**(1/5)
	bw*=smoothing
	#L,U = 0,np.inf
	MIN,MAX  = np.amin(x)-5*bw,np.amax(x)+5*bw
	xtest = np.linspace(zphotmin,zphotmax,Nx)
	#xtest    = np.linspace(MIN,MAX,Nx)
	activate_lo_bound,activate_hi_bound=False,False
	if L>np.amin(x)-5*bw:
		activate_lo_bound=True
	if U<np.amax(x)+5*bw:
		activate_hi_bound=True

	if not activate_lo_bound and not activate_hi_bound:
		#Original KDE no reflections required
		KDE   = np.zeros(len(xtest))
		for ix,xt in enumerate(x):
			KDE += Gaussian2(xt,bw,xtest)/len(x)

	else:
		stepsize     = xtest[1]-xtest[0]
		lower,upper  = max(L,MIN),min(U,MAX)
		Nsteps       = int(round((upper-lower)/stepsize))
		new_stepsize = (upper-lower)/Nsteps#Now we have roughly the same stepsize as before, but xtest passes precisely between L and U bounded support (or just previous xmin, xmax)
		Nr_new       = int(round((MAX-upper)/new_stepsize))
		Nl_new       = Nx-Nsteps-Nr_new-1
		new_xmin,new_xmax = lower-Nl_new*new_stepsize,upper+Nr_new*new_stepsize
		new_xtest         = np.linspace(new_xmin,new_xmax,Nx)#This new xtest goes through L and U precisely by adopting new stepsize and new xmin and xmax
		new_xtest         = np.array([round(x,10) for x in new_xtest])#Gets round annoying rounding errors e.g. -1e-16==0
		Lindex = np.where(new_xtest==round(lower,10))[0][0]
		Rindex = np.where(new_xtest==round(upper,10))[0][0]

		KDE   = np.zeros(len(new_xtest))
		for ix,xt in enumerate(x):
			KDE += Gaussian2(xt,bw,new_xtest)/len(x)
			if activate_lo_bound:
				KDE+= Gaussian2(2*L-xt,bw,new_xtest)/len(x)
			if activate_hi_bound:
				KDE+= Gaussian2(2*U-xt,bw,new_xtest)/len(x)

		xtest = new_xtest[Lindex:Rindex+1]
		KDE   = KDE[Lindex:Rindex+1]

	return xtest,KDE
def get_Mega_dict(files,Keys_of_interest,mos = ['M20','T21'],checkmo=False,sigmapec=150e3):
	dummy_dict = {'mean':[],'std':[]}
	PATH_meta = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/bayesn-data/lcs/meta/'
	Mega_dict = {key:copy.deepcopy(dummy_dict) for key in Keys_of_interest}
	if checkmo:
		new_files = []
		for file in files:
			for mmm in mos:
				PATHext = PATH_meta+mmm+'_training_set_meta.txt'
				with open(PATHext,'r') as f:
					SNe = f.read().splitlines()
				for SN in SNe:
					items  = SN.split()
					snname = items[0]
					if snname in file:
						new_files.append(file)
		files=new_files

	for file in files:
		x = np.load(file).item()
		for key in Keys_of_interest:
			go=True
			try:
				chain = x[key]
			except:
				if key=='muprime':
					chain = x['mu']+x['delM']
				elif key=='fitting_unc':
					go=False
					mean = np.average(x['mu'])
					std  = ((np.std(x['mu']))**2-(np.std(x['delM']))**2)**0.5
				elif key=='muext' or key=='zcmb' or key=='SN':
					go = False
					for mmm in mos:
						PATHext = PATH_meta+mmm+'_training_set_meta.txt'
						with open(PATHext,'r') as f:
							SNe = f.read().splitlines()
						for SN in SNe:
							items  = SN.split()
							snname = items[0]
							if snname in file:
								if key=='SN':
									mean,std=snname,snname
								ii,jj = 1,2
								if mmm=='T21':
									ii,jj= 2,3
								zcmb,zcmberr = float(items[ii]),float(items[jj])
								if key=='zcmb':
									mean,std = zcmb,zcmberr
								cosmo  = FlatLambdaCDM(H0=73.24, Om0=0.28)
								MUCDM  = cosmo.distmod(zcmb).value
								#mu_err_squared = ((5/(zcmb*np.log(10)))**2)*((150e3/3e8)**2 + zcmberr**2)
								mu_err_squared = ((5/(zcmb*np.log(10)))**2)*((sigmapec/3e8)**2 + zcmberr**2)
								mu_err = mu_err_squared**0.5
								if key=='muext':
									#if mu_err>0.4:
									#	print (snname,zcmb,zcmberr,mu_err)
									mean,std = MUCDM,mu_err
								break

				else:
					print ('Failed')
					err=1/0
			if go:
				mean,std = np.mean(chain),np.std(chain)
			Mega_dict[key]['mean'].append(mean)
			Mega_dict[key]['std'].append(std)

	return Mega_dict
def Gaussian2(mean,sigma,free): return ((2*np.pi*sigma**2)**-0.5)*np.exp(-0.5*((mean-free)/sigma)**2)
def get_KDE(x,Nx,L,U,smoothing=1):#Create mean samples, useful for creating smoothed KDE encompassing error bars,Y is data,x is free xtest
	N,sigma = len(x),np.std(np.asarray(x))
	bw = ((4*sigma**5)/(3*N))**(1/5)
	bw*=smoothing
	#L,U = 0,np.inf
	MIN,MAX  = np.amin(x)-5*bw,np.amax(x)+5*bw
	xtest    = np.linspace(MIN,MAX,Nx)
	########################################
	#Original KDE here, no reflections required
	#KDE   = np.zeros(len(xtest))
	#for ix,xt in enumerate(x):
	#    KDE += Gaussian(xt,bw,xtest)/len(x)
	########################################
	#https://uk.mathworks.com/help/stats/ksdensity.html
	activate_lo_bound,activate_hi_bound=False,False
	if L>np.amin(x)-5*bw:
		activate_lo_bound=True
	if U<np.amax(x)+5*bw:
		activate_hi_bound=True

	if not activate_lo_bound and not activate_hi_bound:
		#Original KDE no reflections required
		KDE   = np.zeros(len(xtest))
		for ix,xt in enumerate(x):
			KDE += Gaussian2(xt,bw,xtest)/len(x)

	else:
		stepsize     = xtest[1]-xtest[0]
		lower,upper  = max(L,MIN),min(U,MAX)
		Nsteps       = int(round((upper-lower)/stepsize))
		new_stepsize = (upper-lower)/Nsteps#Now we have roughly the same stepsize as before, but xtest passes precisely between L and U bounded support (or just previous xmin, xmax)
		Nr_new       = int(round((MAX-upper)/new_stepsize))
		Nl_new       = Nx-Nsteps-Nr_new-1
		new_xmin,new_xmax = lower-Nl_new*new_stepsize,upper+Nr_new*new_stepsize
		new_xtest         = np.linspace(new_xmin,new_xmax,Nx)#This new xtest goes through L and U precisely by adopting new stepsize and new xmin and xmax
		new_xtest         = np.array([round(x,10) for x in new_xtest])#Gets round annoying rounding errors e.g. -1e-16==0
		#print (float(round(lower,10)),float(round(upper,10)))
		#print (new_xtest)
		Lindex = np.where(new_xtest==round(lower,10))[0][0]
		Rindex = np.where(new_xtest==round(upper,10))[0][0]
		#pl.figure()
		#pl.scatter(new_xtest,np.zeros(Nx))
		#pl.scatter(lower,0.002)
		#pl.scatter(upper,0.002)
		#pl.show()

		KDE   = np.zeros(len(new_xtest))
		for ix,xt in enumerate(x):
			KDE += Gaussian2(xt,bw,new_xtest)/len(x)
			if activate_lo_bound:
				KDE+= Gaussian2(2*L-xt,bw,new_xtest)/len(x)
			if activate_hi_bound:
				KDE+= Gaussian2(2*U-xt,bw,new_xtest)/len(x)

		xtest = new_xtest[Lindex:Rindex+1]
		KDE   = KDE[Lindex:Rindex+1]

	return xtest,KDE
def find_conf_interval_samples_count(CHAIN,x,conf,meanormode,m_loc,m_loc_index):
	lbub,sigloup,sigmalowupp = [],[],[]
	chain = copy.deepcopy(CHAIN)
	chain.sort()
	#############################
	if meanormode=='mean':
		confs        = [(1-conf)/2,(1+conf)/2]
	if meanormode=='mode':#e.g. peaks at boundary on LHS
		confs        = [0,conf]
		if m_loc_index==len(x)-1:#e.g. peaks at boundary on RHS
			confs    = [1,1-conf]
	#############################
	conf_indices = [int((len(chain)-1)*c) for c in confs]
	for ii,ic in enumerate(conf_indices):
		curve = abs(x-chain[ic])
		index = (curve.tolist()).index(np.amin(np.asarray(curve)))
		if meanormode=='mode' and index<10:#Cheat way of enforcing mode at zero, often index will be closer to 1 than zero this is small fix to make graphs easier
			index=0
		if meanormode=='mode' and index>len(x)-11:#Cheat way of enforcing mode at len-1, often index will be closer to len-2 than len-1 this is small fix to make graphs easier
			index=len(x)-1
		lbub.append(x[index])
		sigloup.append(index)
		sigmalowupp.append((m_loc-x[index])*(-1)**(ii))

	return lbub[0],lbub[1],sigmalowupp[0],sigmalowupp[1],sigloup[0],sigloup[1]
def get_conf_interval(chain,conf,x,KDE,meanormode):
	######################################################################
	if meanormode=='mode':
		mm          = np.amax(KDE)
		m_loc_index = (KDE.tolist()).index(mm)
		m_loc       = x[m_loc_index]
	######################################################################
	if meanormode=='mean':
		m_loc       = np.average(chain)#sum(x*KDE)/sum(KDE)
		curve       = abs(x-m_loc)
		m_loc_index = (curve.tolist()).index(np.amin(np.asarray(curve)))
		m_loc       = x[m_loc_index]
	#####################################################################
	lb,ub,sigma_lower,sigma_upper,siglo,sigup = find_conf_interval_samples_count(chain,x,conf,meanormode,m_loc,m_loc_index)
	return lb,ub,sigma_lower,sigma_upper,siglo,sigup,KDE[m_loc_index],m_loc,m_loc_index

def simulate_Dg(Nsiblings,sigmaRel,sigmafit):
	mu_plus_dM_common = np.random.uniform(25,50)


	dM_Rel_s     = np.random.normal(0,sigmaRel,Nsiblings)
	E_fit_s      = np.random.normal(0,sigmafit,Nsiblings)

	mu_hat_s    = mu_plus_dM_common+dM_Rel_s+E_fit_s
	return mu_hat_s

def Gaussian(x,mu,sigma):
	num = np.exp(-0.5*( ( (x-mu)/sigma)**2) )
	den = sigma*((2*np.pi)**0.5)
	return num/den

def simulate_G_single_SNe(G,sigmaRel,sigerr):
	HRs = np.random.normal(0,(sigmaRel**2+sigerr**2)**0.5,G)
	return HRs

def get_single_SNe_posterior(G_DICT,G):
	'''
	sigma0_posterior = []
	for isGr in range(len(G_DICT['posterior'][0])):
		p = 1
		for s in range(G):
			mini_p = G_DICT['posterior'][s][isGr]#Gaussian(HRs[s],0,(sigR**2+fit_samples[s]**2+ext_samples[s]**2)**0.5)
			p     *= mini_p
		sigma0_posterior.append(p)
	return np.asarray(sigma0_posterior)/np.amax(sigma0_posterior)#Divide by max to avoid small numbers
	'''
	##############################
	sigma0_posterior_2 = 1
	for s in range(G):
		sigma0_posterior_2 *= np.asarray(G_DICT['posterior'][s])/np.amax(np.asarray(G_DICT['posterior'][s]))#Divide to prevent small/large numbers
	return sigma0_posterior_2/np.amax(sigma0_posterior_2)#Divide to prevent small/large numbers
	#pl.figure()
	#pl.plot(sigma0_posterior_2/np.amax(sigma0_posterior_2))
	#pl.plot(sigma0_posterior/np.amax(sigma0_posterior))
	#pl.show()
	#########################

def get_single_SN_single_SNe_posterior(HR,sigRs,fit_sample,ext_sample):
	def mini_posterior(HR,sigR,fit_sample,ext_sample):
		return Gaussian(HR,0,(sigR**2+fit_sample**2+ext_sample**2)**0.5)
	posterior = np.array([mini_posterior(HR,sigR,fit_sample,ext_sample) for sigR in sigRs])
	return posterior



def get_fit_ext_samples(sigFit_icdf_cont,sigExt_icdf_cont,xcont,G):
	u_samples        = np.random.uniform(0,1,G)
	u_indices        = convert_usample_to_indices(u_samples,xcont)
	fit_samples      = np.array([sigFit_icdf_cont[index] for index in u_indices])
	ext_samples      = np.array([sigExt_icdf_cont[index] for index in u_indices])
	return fit_samples,ext_samples

def get_CDF(G_DICT,G): return np.cumsum(get_multi_gal_posterior(G_DICT,G))/sum(get_multi_gal_posterior(G_DICT,G))

def get_CDF_sing(G_DICT,G):
	#print (get_single_SNe_posterior(G_DICT,G), len(get_single_SNe_posterior(G_DICT,G)),len(get_single_SNe_posterior(G_DICT,G)[0]))
	#print (sum(get_single_SNe_posterior(G_DICT,G)))
	return np.cumsum(get_single_SNe_posterior(G_DICT,G))/sum(get_single_SNe_posterior(G_DICT,G))
#def get_CDF(prior_sigma0s, posterior_probability):
#	CDF = []
#	Norm = sum(posterior_probability)
#	for i,p in enumerate(posterior_probability):
#		CDF.append(sum(posterior_probability[:i+1])/Norm)
#	return np.asarray(CDF)

def get_X_percent(prior_sigma0s,CDF,percent):
	curve      = abs(CDF-percent)
	index      = (curve.tolist()).index(np.amin(curve))
	sigma_X    = prior_sigma0s[index]
	#CDF_height = CDF[index]
	return sigma_X#,CDF_height,int(index)

def get_single_galaxy_posterior(mu_hat_s,sigma_fit_s,sigRs):
	def get_posterior(mu_hat_s,sigma_fit_s,sigR):
		ws  = 1/(sigma_fit_s**2 + sigR**2)
		muw = sum(ws*mu_hat_s)/sum(ws)
		p   = 1
		for mu_hat,sigma_fit in zip(mu_hat_s,sigma_fit_s):
			#print (scipy.stats.norm(muw,(sigR**2+sigma_fit**2)**0.5).pdf(mu_hat) - Gaussian(muw,(sigR**2+sigma_fit**2)**0.5,mu_hat))
			#p*=scipy.stats.norm(muw,(sigR**2+sigma_fit**2)**0.5).pdf(mu_hat)
			p*=Gaussian(muw,mu_hat,(sigR**2+sigma_fit**2)**0.5)
		p *= 1/((sum(ws))**0.5)
		return p

	posterior = np.array([get_posterior(mu_hat_s,sigma_fit_s,sigR) for sigR in sigRs])
	return posterior

def get_multi_gal_posterior(G_DICT,G):
	Total_Posterior=1
	for p in G_DICT['posterior'][:G]:#Up to G means you only take the first G galaxies
		Total_Posterior*=p/np.amax(p)#Divide to avoid small numbers
	return Total_Posterior/np.amax(Total_Posterior)#Divide to avoid small numbers


def plot_distances(sigmaRel,G_DICT):
	pl.figure()
	pl.title(r'$\sigma_{\rm{Rel}}$'+f'={sigmaRel}',fontsize=FS)
	mmm = 0
	for mupair,sigmafitpair in zip(G_DICT['mu_hat_s'],G_DICT['sigma_fit_s']):
		pl.errorbar(mmm-0.05,mupair[0]-np.average(mupair),yerr=sigmafitpair[0],marker='o',c=f'C{mmm}')
		pl.errorbar(mmm+0.05,mupair[1]-np.average(mupair),yerr=sigmafitpair[1],marker='o',c=f'C{mmm}')
		mmm+=1
	pl.tick_params(labelsize=FS)
	pl.plot([0,len(G_DICT['mu_hat_s'])],[0,0],color='black')
	pl.show()

def plus_posterior(posterior):
	lower_half = np.zeros(Ndec-1)
	return np.concatenate((lower_half,posterior))
def minus_posterior(posterior):
	lower_half = posterior[1:][::-1]
	return np.concatenate((lower_half,np.array([posterior[0]]),np.zeros(Ndec-1)))

def get_FILENAME(G,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR):
	FILENAME= f'G{G}_sigRel{sigmaRel}_fittingmean{SIGMAFIT[0]}_fittingstd{SIGMAFIT[1]}_Nsibs{Nsiblings}_Nsims{Nsims}_Ndec{Ndec}_prior{PRIOR}'
	return FILENAME
##############################################################
##############################################################
##############################################################
FS    = 18
NSIBLINGs = np.array([2])
Gmax = np.array([1000])
Gs   = np.array([10,25,50,100,200,400,800,800])
#Gs   = np.array([10,25,50,100,200,400,800])#,800])
SIGMARELs = np.array([0.005,0.025,0.05,0.075,0.1])#,0.125,0.15])
'''
SIGMAFITs = {'Avelino':[0.038,0.011],
			 'Avelino_RVfree':[0.049,0.014],
			 'Foundation':[0.066,0.012],
			 'Foundation_RVfree':[0.091,0.029],
			 'Avelino_Opt':[0.062,0.019],
			 'Avelino_Opt_RVfree':[0.100,0.037],
			 'LSST':[],
			 'LSST_RVfree':[]}
'''
SIGMAFITs = {'Avelino':[0.04,0.01]}
#SIGMAFITs = {'Avelino_Opt_RVfree':[0.1,0.04]}
#SIGMAFITs  = {'Foundation':[0.07,0.01]}
#SIGMAFITs = {'Foundation_RVfree':[0.09,0.03]}
#SIGMAFITs = {'Avelino':[0.04,0.01],'Avelino_Opt_RVfree':[0.1,0.04]}
#SIGMAFITs = {'LSST':[0.33,0.17]}
#SIGMAFITs = {'LSST_freetmax':[0.33,0.17]}
#SIGMAFITs = {'LSST_snrcut1':[0.33,0.17]}
#SIGMAFITs = {'LSST_snrcut0':[0.33,0.17]}

'''
SIGMAFITs = {'Avelino':[0.04,0.01],
			 'Avelino_RVfree':[0.05,0.015],
			 'Foundation':[0.07,0.01],
			 'Foundation_RVfree':[0.09,0.03],
			 'Avelino_Opt':[0.06,0.02],
			 'Avelino_Opt_RVfree':[0.1,0.04],
			 'LSST':[],
			 'LSST_RVfree':[]}
#'''

PRIORS = np.array([1])
##############################################################
##############################################################
##############################################################
Nsims = 1000
Ndec  = 1000#number of sigR places for posterior evaluation

#Nsims = 1000
#Ndec  = 1000#0#number of sigR places for posterior evaluation

'''#Trial Runs
Gmax = np.array([50])
Gs   = np.array([10,20,50])
SIGMARELs = np.array([0.005,0.05,0.1])
Nsims = 100#0
Ndec  = 1000#number of sigR places for posterior evaluation
#'''

##############################################################
##############################################################
#sigmapec = 225e3#Based on sigma0sigmapec W22 HF fits
Avelinolcs = io.read_sn_sample_file("M20_training_set", metafile="M20_training_set_meta")
Fndlcs     = io.read_sn_sample_file("T21_training_set", metafile="T21_training_set_meta")

sigmapec   = 150e3#Based on sigma0sigmapec W22 HF fits
fitextsamples = {}
for SIGMAFIT in SIGMAFITs:
	if 'RVfree' in SIGMAFIT:
		if 'Opt' in SIGMAFIT:
			PATH = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/X22_fits_210610_135216_M22setOptOnly_RVfree/'
		else:
			PATH = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/X22_fits_210610_135216_RVfree/'
	else:
		if 'Opt' in SIGMAFIT:
			PATH = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/X22_fits_210610_135216_M22setOptOnly/'
		if 'LSST' in SIGMAFIT:
			PATH = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/SW_SNIa_2kLCs/SW_SNIa_2kLCs_Samples/'
		else:
			PATH = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/X22_fits_210610_135216/'


	#files = glob(PATH+'/*.npy')
	##########################################################################
	#New section purely for LSST SIMS
	if 'LSST' in SIGMAFIT:
		files = glob(PATH+'/*meta_tmax*.npy')
		#files = glob(PATH+'/*fit_tmax*.npy')
		files = [f for f in files if 'snrcut' not in f]
		#files = [f for f in files if 'snrcut1' in f]
		#print ('LEN LSST FILES:',len(files))
		########################
		datnames = []
		ddict    = {}
		ddpath   = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/SW_SNIa_2kLCs/SW_SNIa_2kLCs_Dats/'
		ffiles   = os.listdir(path='%s.'%ddpath)
		for ffile in ffiles:
			filename, file_extension = os.path.splitext(ffile)
			if file_extension=='.DATSMW':
				datnames.append(filename)
		for file in files:
			for datname in datnames:
				if datname in file:
					ddict[file] = datname
		########################
		sigmaFit_dist,sigmaExt_dist = [],[]
		for file in files:
			samples  = np.load(file).item()
			sigmafit = (np.var(samples['mu'])-np.var(samples['delM']))**0.5
			if sigmafit<1.2:
				sigmaFit_dist.append(sigmafit)
				sn, lc    = io.read_snana_lcfile(ddpath+ddict[file]+'.DATSMW')
				sigmaExt_dist.append(  (5/(lc.meta['REDSHIFT_CMB']*np.log(10)))*( ((sigmapec/3e8)**2 + 0.01**2)**0.5) )
		sigmaFit_dist.sort()
		sigmaExt_dist.sort()
	##########################################################################
	else:
		files = glob(PATH+'/*.npy')
		Keys_of_interest = ['SN','muext','zcmb','fitting_unc']#'muprime'
		#Mega_dict    = get_Mega_dict(files,Keys_of_interest,sigmapec=sigmapec)
		if 'Avelino' in SIGMAFIT:
			   Mega_dict  = get_Mega_dict(files,Keys_of_interest,mos=['M20'],checkmo=True,sigmapec=sigmapec)
		elif 'Foundation' in SIGMAFIT:
			   Mega_dict = get_Mega_dict(files,Keys_of_interest,mos=['T21'],checkmo=True,sigmapec=sigmapec)
		sigmaFit_dist = Mega_dict['fitting_unc']['std']
		sigmaFit_dist.sort()
		sigmaExt_dist = Mega_dict['muext']['std']
		sigmaExt_dist.sort()
	print (SIGMAFIT)
	print (np.amin(sigmaFit_dist),np.amax(sigmaFit_dist))
	print (np.amin(sigmaExt_dist),np.amax(sigmaExt_dist))

	sigFit_quantiles,sigFit_icdf = get_ICDF_from_samples(sigmaFit_dist)
	sigExt_quantiles,sigExt_icdf = get_ICDF_from_samples(sigmaExt_dist)

	xcont  = np.linspace(0,1,10000)

	base_f = scipy.interpolate.PchipInterpolator(sigFit_quantiles,sigFit_icdf)
	sigFit_icdf_cont = base_f(xcont)
	base_f = scipy.interpolate.PchipInterpolator(sigExt_quantiles,sigExt_icdf)
	sigExt_icdf_cont = base_f(xcont)

	fitextsamples[SIGMAFIT] = {'fit':sigFit_icdf_cont,
							  'ext':sigExt_icdf_cont}

	'''
	u_samples = np.random.uniform(0,1,Nsims)
	u_indices = convert_usample_to_indices(u_samples,xcont)
	fit_samples = np.array([sigFit_icdf_cont[index] for index in u_indices])
	ext_samples = np.array([sigExt_icdf_cont[index] for index in u_indices])

	sFits,sFit_pdf = get_KDE(fit_samples,1000,0,np.inf,smoothing=1.5)
	sExts,sExt_pdf = get_KDE(ext_samples,1000,0,np.inf,smoothing=1.5)
	pl.figure()
	pl.scatter(sigFit_quantiles,sigFit_icdf,color='b',label='sigmaFit')
	pl.scatter(sigExt_quantiles,sigExt_icdf,color='r',label='sigmaExt')
	pl.plot(xcont,sigFit_icdf_cont,c='b')
	pl.plot(xcont,sigExt_icdf_cont,c='r')
	pl.legend()
	pl.show()

	pl.figure()
	pl.hist(sigmaFit_dist,color='b',label='sigmaFit',alpha=0.4)
	pl.plot(sFits,sFit_pdf,c='b')
	pl.hist(sigmaExt_dist,color='r',label='sigmaExt',alpha=0.4)
	pl.plot(sExts,sExt_pdf,c='r')
	pl.legend()
	pl.show()
	'''


##############################################################
##############################################################
#overlay_posteriors = True
overlay_posteriors = False
Gs       = Gs[:-1]
PATH     = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/NSiblingsRel_SingleSNe/'
SAVEPATH =  PATH+'PKLS/'+f'Nsims{Nsims}/'
intervals_SAVEPATH =  PATH+'Sigma_Intervals/'+f'Nsims{Nsims}/'
if not os.path.exists(SAVEPATH):
	os.mkdir(SAVEPATH)

print ('###'*10)
print ('Creating Sigma Rel Posteriors')
for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
	for isGr,sigmaRel in enumerate(SIGMARELs):
		for iG,G in enumerate(Gmax):
			for ipr,PRIOR in enumerate(PRIORS):
				sigRs      = np.linspace(0,PRIOR,Ndec)
				#prior_sigR = 1/PRIOR#dont need prior as its just a uniform prior
				for iNs,Nsiblings in enumerate(NSIBLINGs):
					SIGMAFIT = SIGMAFITs[SIGMAFITkey]
					if SIGMAFIT!=[]:
						print (f'Simulating G={G}, sigmaRel={sigmaRel}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
						FILENAME        = get_FILENAME(G,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
						FILENAME_single = 'SINGLE_SNE_'+FILENAME
						SAVEPATH_mini        = SAVEPATH+FILENAME+'/'
						#sigmafitdist  = scipy.stats.truncnorm((0 - SIGMAFIT[0])/SIGMAFIT[1], (np.inf - SIGMAFIT[0]) / SIGMAFIT[1], loc=SIGMAFIT[0], scale=SIGMAFIT[1])
						if not os.path.exists(SAVEPATH_mini):
							os.mkdir(SAVEPATH_mini)
						for isim in range(Nsims):
							filename        = FILENAME       +f'isim{isim}'
							filename_single = FILENAME_single+f'isim{isim}'
							if not os.path.exists(SAVEPATH_mini+filename+'.pkl') or not os.path.exists(SAVEPATH_mini+filename_single+'.pkl'):
								#print (f'Creating filename: {filename}')
								if (isim+1)%10==0:
									print (f'Sim{isim+1}/{Nsims}')
								G_DICT_SING = {'posterior':[]}
								G_DICT_SIBS = { 'mu_hat_s':[],'sigma_fit_s':[],'posterior':[]}

								sigFit_icdf_cont = fitextsamples[SIGMAFITkey]['fit']
								sigExt_icdf_cont = fitextsamples[SIGMAFITkey]['ext']
								fit_samples,ext_samples = get_fit_ext_samples(sigFit_icdf_cont,sigExt_icdf_cont,xcont,G)
								###################################################################################
								#SINGLE SNE
								HRs                            = simulate_G_single_SNe(G,sigmaRel,(fit_samples**2+ext_samples**2)**0.5)
								G_DICT_SING['HRs']             = HRs
								G_DICT_SING['sigma_fit_s']     = fit_samples
								G_DICT_SING['sigma_ext_s']     = ext_samples
								for s,HR in enumerate(HRs):
									posterior = get_single_SN_single_SNe_posterior(HR,sigRs,fit_samples[s],ext_samples[s])
									G_DICT_SING['posterior'].append(posterior)
								sigma0_posterior = get_single_SNe_posterior(G_DICT_SING,G)#Here G is Gmax
								G_DICT_SING['Total_Posterior'] = sigma0_posterior
								###################################################################################
								#SIBLINGS SNE
								for g in range(G):
									sigma_fit_s          = np.array([fit_samples[g] for _ in range(Nsiblings)])
									mu_hat_s             = simulate_Dg(Nsiblings,sigmaRel,fit_samples[g])#Here sigmafit is the same for both siblings, and same as single SN
									posterior            = get_single_galaxy_posterior(mu_hat_s,sigma_fit_s,sigRs)
									G_DICT_SIBS['mu_hat_s'].append(mu_hat_s)
									G_DICT_SIBS['sigma_fit_s'].append(sigma_fit_s)
									G_DICT_SIBS['posterior'].append(posterior)
								G_DICT_SIBS['Total_Posterior'] = get_multi_gal_posterior(G_DICT_SIBS,G)#Here G=Gmax[0]==1000
								###################################################################################
								#plot_distances(sigmaRel,G_DICT_SIBS)
								with open(SAVEPATH_mini+filename+'.pkl','wb') as f:
									#print (f'Saving {filename}')
									pickle.dump(G_DICT_SIBS, f)
								with open(SAVEPATH_mini+filename_single+'.pkl','wb') as f:
									#print (f'Saving {filename}')
									pickle.dump(G_DICT_SING, f)
							else:
								#print (f'Already created {filename}')
								continue

####################################################################
'''
pab_SAVEPATH = PATH+'pab/'+f'Nsims{Nsims}/'
if not os.path.exists(pab_SAVEPATH):
	os.mkdir(pab_SAVEPATH)
print ('###'*10)
print ('Creating pabs')
for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
	for isGr,sigmaRel in enumerate(SIGMARELs):
		SIGMARELs2 = SIGMARELs[isGr+1:]
		for isGr2,sigmaRel2 in enumerate(SIGMARELs2):
			for iG,G in enumerate(Gs):
				for ipr,PRIOR in enumerate(PRIORS):
					sigRs      = np.linspace(0,PRIOR,Ndec)
					negsigRs   = np.linspace(-PRIOR,0,Ndec)
					totsigRs   = np.concatenate((negsigRs[:-1],sigRs))
					for iNs,Nsiblings in enumerate(NSIBLINGs):
						SIGMAFIT = SIGMAFITs[SIGMAFITkey]
						if SIGMAFIT!=[]:
							print (f'Loading G={G}, sigmaRel={sigmaRel}<sigmaRel2={sigmaRel2}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
							FILENAMEtot    = get_FILENAME(G,str(sigmaRel)+'vs'+str(sigmaRel2),SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
							if not os.path.exists(pab_SAVEPATH+FILENAMEtot+'_SIBS.npy') or not os.path.exists(pab_SAVEPATH+FILENAMEtot+'_SING.npy'):
								FILENAME       = get_FILENAME(Gmax[0],sigmaRel, SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
								FILENAME2      = get_FILENAME(Gmax[0],sigmaRel2,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
								psibs,psings = [],[]
								for isim in range(Nsims):
									if (isim+1)%200==0:
										print (f'{isim+1}/{Nsims}')
									filename         =               FILENAME +f'isim{isim}'
									filename_single  = 'SINGLE_SNE_'+FILENAME +f'isim{isim}'
									filename2        =               FILENAME2 +f'isim{isim}'
									filename_single2 = 'SINGLE_SNE_'+FILENAME2 +f'isim{isim}'
									with open(SAVEPATH+FILENAME+'/'+filename+'.pkl','rb') as f:
										G_DICT_SIBS = pickle.load(f)
									with open(SAVEPATH+FILENAME+'/'+filename_single+'.pkl','rb') as f:
										G_DICT_SING = pickle.load(f)
									with open(SAVEPATH+FILENAME2+'/'+filename2+'.pkl','rb') as f:
										G_DICT_SIBS2 = pickle.load(f)
									with open(SAVEPATH+FILENAME2+'/'+filename_single2+'.pkl','rb') as f:
										G_DICT_SING2 = pickle.load(f)


									sibs_posterior =  plus_posterior(get_multi_gal_posterior(G_DICT_SIBS,G))
									sing_posterior = plus_posterior(get_single_SNe_posterior(G_DICT_SING,G))


									sibs_posterior2 =  minus_posterior(get_multi_gal_posterior(G_DICT_SIBS2,G))
									sing_posterior2 = minus_posterior(get_single_SNe_posterior(G_DICT_SING2,G))

									mm_sib  = max(np.concatenate((sibs_posterior,sibs_posterior2)))
									mm_sing = max(np.concatenate((sing_posterior,sing_posterior2)))
									#Apply rescaling for the cases where posteriors are too large meaning convolution tends to infinity, stabilise by rescaling! Gives identical results
									pab_sibs = np.convolve(sibs_posterior/mm_sib,sibs_posterior2/mm_sib,mode='same')
									pab_sing = np.convolve(sing_posterior/mm_sing,sing_posterior2/mm_sing,mode='same')
									#print (sigmaRel,sigmaRel2,mm_sib,mm_sing,pab_sibs, pab_sing, sum(pab_sibs), sum(pab_sing))

									pless0_sibs = sum(pab_sibs[:Ndec])/sum(pab_sibs)
									pless0_sing = sum(pab_sing[:Ndec])/sum(pab_sing)
									psibs.append(pless0_sibs)
									psings.append(pless0_sing)
									#print (pless0_sing)#pless0_sibs)

									###############
									#pab_sing_scaled = np.convolve(sing_posterior/mm,sing_posterior2/mm,mode='same')
									#pless0_sing_scaled = sum(pab_sing_scaled[:Ndec])/sum(pab_sing_scaled)
									#print (pless0_sing_scaled-pless0_sing,pless0_sing_scaled,pless0_sing)
									###############
									#pl.figure()
									#pl.plot(totsigRs,sibs_posterior)
									#pl.plot(totsigRs,sibs_posterior2)
									#pl.plot(totsigRs,pab_sibs,c='g')
									#pl.plot(totsigRs,sing_posterior)
									#pl.plot(totsigRs,sing_posterior2)
									#pl.plot(totsigRs,pab_sing,c='g')
									#pl.plot(totsigRs,sing_posterior/mm)
									#pl.plot(totsigRs,sing_posterior2/mm)
									#pl.plot(totsigRs,pab_sing_scaled,c='r')
									#pl.show()
								print (f'{sigmaRel}<{sigmaRel2},G={G},  Sibs:{np.average(psibs):.3}+/-{np.std(psibs):.3}')
								print (f'{sigmaRel}<{sigmaRel2},G={G},Single:{np.average(psings):.3}+/-{np.std(psings):.3}')
								np.save(pab_SAVEPATH+FILENAMEtot+'_SIBS',np.asarray(psibs))
								np.save(pab_SAVEPATH+FILENAMEtot+'_SING',np.asarray(psings))
#'''
####################################################################

if not os.path.exists(intervals_SAVEPATH):
	os.mkdir(intervals_SAVEPATH)
intervals  = np.array([0.023,0.05,0.16,0.50,0.68,0.84,0.95,0.977])#the bounds we wish to probe
intervals2 = np.array([0.023,0.05,0.16,0.50,0.68,0.84,0.95,0.977])#the intervals on those particular bounds
print ('###'*10)
print ('Creating Sigmas')
for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
	for isGr,sigmaRel in enumerate(SIGMARELs):
		for iG,G in enumerate(Gs):
			for ipr,PRIOR in enumerate(PRIORS):
				sigRs      = np.linspace(0,PRIOR,Ndec)
				for iNs,Nsiblings in enumerate(NSIBLINGs):
					SIGMAFIT = SIGMAFITs[SIGMAFITkey]
					if SIGMAFIT!=[]:
						print (f'Loading G={G}, sigmaRel={sigmaRel}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
						FILENAME       = get_FILENAME(Gmax[0],sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
						FILENAME2      = get_FILENAME(G,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
						SAVEPATH_mini  = intervals_SAVEPATH+FILENAME2+'/'
						if not os.path.exists(SAVEPATH_mini):
							os.mkdir(SAVEPATH_mini)
						if not os.path.exists(SAVEPATH_mini+FILENAME2+'_SIGMAS_SIBLINGS.pkl') or not os.path.exists(SAVEPATH_mini+FILENAME2+'_SIGMAS_SINGLE.pkl'):
							sigmas_sibs   = {x:{} for x in intervals}
							sigmas_single = {x:{} for x in intervals}
							go=True
							for isim in range(Nsims):
								filename        =               FILENAME +f'isim{isim}'
								filename_single = 'SINGLE_SNE_'+FILENAME +f'isim{isim}'
								try:
								#for _____ in range(1):
									with open(SAVEPATH+FILENAME+'/'+filename+'.pkl','rb') as f:
										G_DICT_SIBS = pickle.load(f)
									with open(SAVEPATH+FILENAME+'/'+filename_single+'.pkl','rb') as f:
										G_DICT_SING = pickle.load(f)
									for Xcred in intervals:
										#sigma_X = get_X_percent(sigRs,get_CDF(G_DICT,Gmax[0]),Xcred)
										sigma_X_siblings = get_X_percent(sigRs,     get_CDF(G_DICT_SIBS,G),Xcred)#Here we only take the first G galaxies from set of Gmax
										sigma_X_single   = get_X_percent(sigRs,get_CDF_sing(G_DICT_SING,G),Xcred)#Here we only take the first G galaxies from set of Gmax

										try:
											sigmas_sibs[Xcred][sigma_X_siblings].append(isim)
										except KeyError:
											sigmas_sibs[Xcred][sigma_X_siblings] = [isim]
										try:
											sigmas_single[Xcred][sigma_X_single].append(isim)
										except KeyError:
											sigmas_single[Xcred][sigma_X_single] = [isim]
								except Exception as e:
									print (f'Not created yet: {filename}', e)
									go=False
							if go:
								with open(SAVEPATH_mini+FILENAME2+'_SIGMAS_SIBLINGS.pkl','wb') as f:
									print (f'Saving {FILENAME2}_SIGMAS siblings')
									pickle.dump(sigmas_sibs,f)
								with open(SAVEPATH_mini+FILENAME2+'_SIGMAS_SINGLE.pkl','wb') as f:
									print (f'Saving {FILENAME2}_SIGMAS single')
									pickle.dump(sigmas_single,f)


print ('###'*10)
print ('Creating Sigma Intervals')
for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
	for isGr,sigmaRel in enumerate(SIGMARELs):
		for iG,G in enumerate(Gs):
			for ipr,PRIOR in enumerate(PRIORS):
				for iNs,Nsiblings in enumerate(NSIBLINGs):
					SIGMAFIT = SIGMAFITs[SIGMAFITkey]
					if SIGMAFIT!=[]:
						print (f'Loading G={G}, sigmaRel={sigmaRel}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
						FILENAME      = get_FILENAME(G,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
						SAVEPATH_mini = intervals_SAVEPATH+FILENAME+'/'
						if not os.path.exists(SAVEPATH_mini):
							os.mkdir(SAVEPATH_mini)
						if not os.path.exists(SAVEPATH_mini+FILENAME+'_INTERVALS_SIBLINGS.pkl') or not os.path.exists(SAVEPATH_mini+FILENAME+'_INTERVALS_SINGLE.pkl'):
							if os.path.exists(SAVEPATH_mini+FILENAME+'_SIGMAS_SIBLINGS.pkl') and os.path.exists(SAVEPATH_mini+FILENAME+'_SIGMAS_SINGLE.pkl'):
								with open(SAVEPATH_mini+FILENAME+'_SIGMAS_SIBLINGS.pkl','rb') as f:
									sigmas_sibs   = pickle.load(f)
								with open(SAVEPATH_mini+FILENAME+'_SIGMAS_SINGLE.pkl','rb') as f:
									sigmas_single = pickle.load(f)
								for irun,sigmas in enumerate([sigmas_sibs,sigmas_single]):
									sigma_interval_map = {interval:{} for interval in intervals}
									for Xcred in intervals:
										sigma_Xs_list = list(sigmas[Xcred].keys())
										################################################
										sigma_Xs_list = [x for x in sigma_Xs_list for _ in range(len(sigmas[Xcred][x]))]#Here we repeat the values for the number of times it appears
										################################################
										sigma_Xs_list.sort()
										Xcreds = {Y:sigma_Xs_list[int( (len(sigma_Xs_list)-1)*Y) ] for Y in intervals2}
										sigma_interval_map[Xcred]=Xcreds
									if irun==0:
										with open(SAVEPATH_mini+FILENAME+'_INTERVALS_SIBLINGS.pkl','wb') as f:
											print (f'Saving {FILENAME}_INTERVALS_SIBLINGS intervals')
											pickle.dump(sigma_interval_map,f)
									elif irun==1:
										with open(SAVEPATH_mini+FILENAME+'_INTERVALS_SINGLE.pkl','wb') as f:
											print (f'Saving {FILENAME}_INTERVALS_SINGLE intervals')
											pickle.dump(sigma_interval_map,f)
							else:
								print (f'Not created sigmas yet: {FILENAME}_SIGMAS.pkl')


PATH     = '/data/smw92/mandel/bayesn_github/bayesn-pre-release/NSiblingsRel_SingleSNe/'
PICSAVEPATH =  PATH+'Pics/'+f'Nsims{Nsims}/'
if not os.path.exists(PICSAVEPATH):
	os.mkdir(PICSAVEPATH)
####################################

sss    = 30
Capsize= 4
dd     = 0.00171
alph   = 0.5
ddlist = (np.arange(0,len(Gs),1) - int(len(Gs)/2) + (0.5-(len(Gs)/2)%1))*dd*2
APPENDERS = ['SIBLINGS']#,'SINGLE']
print ('###'*10)
print ('Plotting Sigma Intervals')
ffextra = 0.75
for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
	pl.figure(figsize=(12,5))
	for iapp,APPENDER in enumerate(APPENDERS):
		for isGr,sigmaRel in enumerate(SIGMARELs):
			for iG,G in enumerate(Gs):
				for ipr,PRIOR in enumerate(PRIORS):
					for iNs,Nsiblings in enumerate(NSIBLINGs):
						SIGMAFIT = SIGMAFITs[SIGMAFITkey]
						if SIGMAFIT!=[]:
							#print (f'Loading G={G}, sigmaRel={sigmaRel}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
							FILENAME      = get_FILENAME(G,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
							SAVEPATH_mini = intervals_SAVEPATH+FILENAME+'/'
							if os.path.exists(SAVEPATH_mini+FILENAME+f'_INTERVALS_{APPENDER}.pkl'):
								with open(SAVEPATH_mini+FILENAME+f'_INTERVALS_{APPENDER}.pkl','rb') as f:
									#print (f'Loading {FILENAME}_INTERVALS intervals')
									sigma_interval_map = pickle.load(f)
								XCOORD = sigmaRel + ddlist[iG]+iapp*dd*ffextra -0.005*(sigmaRel==0.005)
								ccc = f'C{iG}'
								mmarker = ['o','x'][iapp]
								sigma_023,sigma_16,sigma_50,sigma_84,sigma_977 = sigma_interval_map[0.023][0.50],sigma_interval_map[0.16][0.50],sigma_interval_map[0.50][0.50],sigma_interval_map[0.84][0.50],sigma_interval_map[0.977][0.50]
								pl.errorbar(XCOORD,sigma_50-sigmaRel,yerr = [[sigma_50-sigma_16],[sigma_84-sigma_50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=1)
								pl.errorbar(XCOORD,sigma_50-sigmaRel,yerr = [[sigma_50-sigma_023],[sigma_977-sigma_50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=0.3)
								pl.scatter(XCOORD,sigma_50-sigmaRel,marker=mmarker,s=sss,c=ccc,label=r'$G = %s$'%{G}*(isGr==0)*(iapp==0))
			ddlist = list(ddlist)+[ddlist[iG]+dd*ffextra]
			#print (ddlist[-1]+dd*ffextra,sigmaRel+ddlist[iG]+iapp*dd*ffextra,XCOORD)
			pl.plot([sigmaRel+ddlist[0]-0.005*(sigmaRel==0.005),sigmaRel+ddlist[-1]-0.005*(sigmaRel==0.005)], [0-sigmaRel,0-sigmaRel], color='black',linewidth=2,linestyle='--')
	#pl.legend(fontsize=FS,bbox_to_anchor=(1.005,1.035),loc='upper left')
	if 'SIBLINGS' in APPENDERS:
		pl.scatter(XCOORD,100,marker='o',s=sss,c='black',label=f'{Nsiblings} Siblings')
	if 'SINGLE' in APPENDERS:
		pl.scatter(XCOORD,100,marker='x',s=sss,c='black',label='Single SNe')

	pl.legend(fontsize=FS,bbox_to_anchor=(0,1.4),loc='upper left',ncol=4)
	f1,f2,f3,f4 = 2,1,0.5,0.2

	ddlist[0]=ddlist[0]-0.005*(0.005 in SIGMARELs)
	pl.plot([SIGMARELs[0]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[0.03,0.03],linewidth=1,color='black',alpha=alph*f4)
	pl.plot([SIGMARELs[0]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[0.02,0.02],linewidth=1,color='black',alpha=alph*f3)
	pl.plot([SIGMARELs[0]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[0.01,0.01],linewidth=1,color='black',alpha=alph*f2)
	pl.plot([SIGMARELs[0]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[0,0],linewidth=2,color='black',alpha=alph*f1)
	ddlist[0]=ddlist[0]+0.005*(0.005 in SIGMARELs)
	try:
		pl.plot([SIGMARELs[1]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[-0.01,-0.01],linewidth=1,color='black',alpha=alph*f2)
		pl.plot([SIGMARELs[1]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[-0.02,-0.02],linewidth=1,color='black',alpha=alph*f3)
	except:
		donothing=0
	try:
		pl.plot([SIGMARELs[2]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[-0.03,-0.03],linewidth=1,color='black',alpha=alph*f4)
	except:
		try:
			pl.plot([SIGMARELs[1]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[-0.03,-0.03],linewidth=1,color='black',alpha=alph*f4)
		except:
			pl.plot([SIGMARELs[0]+ddlist[0],SIGMARELs[-1]+ddlist[-1]],[-0.03,-0.03],linewidth=1,color='black',alpha=alph*f4)
	pl.tick_params(labelsize=FS)
	pl.xlabel(r'$\sigma_{\rm{Rel, True}}$',fontsize=FS)
	pl.ylabel(r'$p(\sigma_{\rm{Rel}}|\mathbf{\mathcal{D}}) - \sigma_{\rm{Rel, True}}$',fontsize=FS)
	pl.xticks(SIGMARELs)
	pl.ylim([-0.1,0.15])
	pl.tight_layout()
	FILENAME      = get_FILENAME(Gs,SIGMARELs,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
	pl.savefig(PICSAVEPATH+f'SigmaIntervals_'+FILENAME+'.png')
	pl.show()



####################################
'''
print ('###'*10)
print ('Plots of pab')
dG     = 5
sss    = 30
Capsize= 4
def get_paLb_err(Nsims,a,b):
    aa  = copy.deepcopy(a)
    bb  = copy.deepcopy(b)
    LEN = len(aa)
    Ps = []
    for i in range(Nsims):
        random.shuffle(aa)
        random.shuffle(bb)
        p   = [float(ai<bi) for ai,bi in zip(aa,bb)]
        Ps.append(sum(p)/LEN)
    return Ps
for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
	for isGr,sigmaRel in enumerate(SIGMARELs):
		SIGMARELs2 = SIGMARELs[isGr+1:]
		for isGr2,sigmaRel2 in enumerate(SIGMARELs2):
			pl.figure()
			pl.title(f'{SIGMAFITkey}; '+r'$p(\sigma_{Rel,1}<\sigma_{Rel,2})$'+'\n'+r'$\sigma_{Rel,1}=%s; \sigma_{Rel,2}=%s$'%(sigmaRel,sigmaRel2),fontsize=FS)
			for iG,G in enumerate(Gs):
				psibs,psings = [],[]
				for ipr,PRIOR in enumerate(PRIORS):
					sigRs      = np.linspace(0,PRIOR,Ndec)
					negsigRs   = np.linspace(-PRIOR,0,Ndec)
					totsigRs   = np.concatenate((negsigRs[:-1],sigRs))
					for iNs,Nsiblings in enumerate(NSIBLINGs):
						SIGMAFIT = SIGMAFITs[SIGMAFITkey]
						if SIGMAFIT!=[]:
							print (f'Loading G={G}, sigmaRel={sigmaRel}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
							FILENAMEtot    = get_FILENAME(G,str(sigmaRel)+'vs'+str(sigmaRel2),SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
							psibs  = np.load(pab_SAVEPATH+FILENAMEtot+'_SIBS.npy')
							psings = np.load(pab_SAVEPATH+FILENAMEtot+'_SING.npy')

							poutperform = sum([float(ai>bi) for ai,bi in zip(psibs,psings)])/len(psibs)
							poutperformerr = np.std(get_paLb_err(1000,psibs,psings))

							psibs.sort()
							psings.sort()

							psib023,psib16,psib50,psib84,psib977      = psibs[int((len(psibs)-1)*0.023)],psibs[int((len(psibs)-1)*0.16)],psibs[int((len(psibs)-1)*0.5)],psibs[int((len(psibs)-1)*0.84)],psibs[int((len(psibs)-1)*0.977)]
							psing023,psing16,psing50,psing84,psing977 = psings[int((len(psings)-1)*0.023)],psings[int((len(psings)-1)*0.16)],psings[int((len(psings)-1)*0.5)],psings[int((len(psings)-1)*0.84)],psings[int((len(psings)-1)*0.977)]

							ccc = f'C{iG}'
							#x1 = G-dG
							#x2 = G+dG
							dG = 0.1
							x1 = G*np.exp(-dG)
							x2 = G*np.exp(dG)
							pl.errorbar(x1,np.average(psibs),yerr = [[psib50-psib16],[psib84-psib50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=1)
							pl.errorbar(x1,np.average(psibs),yerr = [[psib50-psib023],[psib977-psib50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=0.3)
							pl.scatter(x1,np.average(psibs),marker='o',s=sss,c=ccc)


							pl.errorbar(x2,np.average(psings),yerr = [[psing50-psing16],[psing84-psing50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=1)
							pl.errorbar(x2,np.average(psings),yerr = [[psing50-psing023],[psing977-psing50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=0.3)
							pl.scatter(x2,np.average(psings),marker='x',s=sss,c=ccc)

							pl.errorbar(G,poutperform,yerr=poutperformerr,marker='s',markersize=5,c='grey')

							print (f'{sigmaRel}<{sigmaRel2},G={G},Sibs:{np.average(psibs):.3}+/-{np.std(psibs):.3}')
							print (f'{sigmaRel}<{sigmaRel2},G={G},Single:{np.average(psings):.3}+/-{np.std(psings):.3}')
			pl.scatter(Gs[0],-1,marker='o',s=sss,c='black',label='2 Siblings')
			pl.scatter(Gs[0],-1,marker='s',s=sss,c='grey',label='p(Sibs. > Sing.)')
			pl.scatter(Gs[0],-1,marker='x',s=sss,c='black',label='Single SNe')
			pl.legend(fontsize=FS-2,ncol=2)#,bbox_to_anchor=(0,1))
			pl.xscale('log')
			pl.ylim([0,1])
			pl.xticks(Gs)
			pl.gca().set_xticklabels(Gs)
			pl.xlabel('G',fontsize=FS)
			#pl.ylabel(r'$p(\sigma_{Rel,1}<\sigma_{Rel,2})$',fontsize=FS)
			pl.ylabel(r'$p$',fontsize=FS)
			pl.tick_params(labelsize=FS)
			xmin,xmax = pl.gca().get_xlim()[:]
			pl.plot([xmin,xmax],[0.5,0.5],color='black',alpha=0.5)
			pl.plot([xmin,xmax],[0.68,0.68],color='black',alpha=1)
			pl.plot([xmin,xmax],[0.95,0.95],color='black',alpha=1)
			pl.plot([xmin,xmax],[0.997,0.997],color='black',alpha=1,linewidth=2)
			dy=0.06
			pl.annotate('50%',xy=(6,0.5-dy), fontsize=12.5,weight='bold',color='black',alpha=0.5)
			pl.annotate('68%',xy=(6,0.68-dy),fontsize=12.5,weight='bold',color='black',alpha=1)
			pl.annotate('95%',xy=(6,0.95-dy),fontsize=12.5,weight='bold',color='black',alpha=1)
			pl.xlim([xmin,xmax])
			pl.tight_layout()
			FILENAMEtot  = get_FILENAME(Gs,str(sigmaRel)+'vs'+str(sigmaRel2),SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
			pl.savefig(PICSAVEPATH+f'pab_'+FILENAMEtot+'.png')
			#pl.show()
#'''
####################################
#intervals  = np.array([0.84,0.95])#the bounds we wish to probe
#intervals  = np.array([0.05,0.95])#the bounds we wish to probe
intervals  = np.array([0.95])
for interval in intervals:
	for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
		for isGr,sigmaRel in enumerate(SIGMARELs[:1]):
			pl.figure()
			pl.title(r'$\sigma_{%s}$ Bounds for $\sigma_{\rm{Rel}}=$'%(int(interval*100))+f'{sigmaRel}',fontsize=FS)
			maxy = 0
			for iapp,APPENDER in enumerate(APPENDERS):
			#for isGr,sigmaRel in enumerate(SIGMARELs):
				for iG,G in enumerate(Gs):
					for ipr,PRIOR in enumerate(PRIORS):
						for iNs,Nsiblings in enumerate(NSIBLINGs):
							SIGMAFIT = SIGMAFITs[SIGMAFITkey]
							if SIGMAFIT!=[]:
								#print (f'Loading G={G}, sigmaRel={sigmaRel}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
								FILENAME      = get_FILENAME(G,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
								SAVEPATH_mini = intervals_SAVEPATH+FILENAME+'/'
								if os.path.exists(SAVEPATH_mini+FILENAME+f'_SIGMAS_{APPENDER}.pkl'):
									with open(SAVEPATH_mini+FILENAME+f'_SIGMAS_{APPENDER}.pkl','rb') as f:
										sigmas=pickle.load(f)
									sigma_Xs = list(sigmas[interval].keys())
									xtest,KDE = get_KDE(sigma_Xs,1000,0,1)
									lb,ub,sigma_lower,sigma_upper,siglo,sigup,Km,m_loc,m_loc_index = get_conf_interval(sigma_Xs,0.68,xtest,KDE,'mean')
									ccc=f'C{iG}'
									lstyle = ['-','--'][iapp]
									pl.plot(xtest,KDE,c=ccc,label=r'$G = %s$'%{G}*(iapp==0),linestyle=lstyle) #*(isGr==0))
									pl.plot([m_loc,m_loc],[0,KDE[m_loc_index]],c=ccc,alpha=0.3,linestyle=lstyle)
									pl.fill_between(xtest[siglo:sigup],np.zeros(sigup-siglo),KDE[siglo:sigup],color=ccc,alpha=0.3)
									maxy = max(maxy,np.amax(KDE))
			FILENAME = get_FILENAME(Gs,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
			pl.plot([sigmaRel,sigmaRel],[0,maxy],c='black',linestyle='--')

			pl.plot([-10,-9],[0,1],c='black',label=f'{Nsiblings} Siblings',linestyle='-')
			pl.plot([-10,-9],[0,1],c='black',label=f'Single SNe',linestyle='--')

			#pl.plot([0,0],[0,maxy],c='black',linewidth=5)
			pl.legend(fontsize=FS)#,bbox_to_anchor=(0,1.4),loc='upper left',ncol=4)
			pl.tick_params(labelsize=FS)
			pl.xlabel(r'$\sigma_{%s}$'%(int(interval*100)),fontsize=FS)
			pl.ylabel(r'Density',fontsize=FS)
			pl.yticks([])
			#pl.xticks(SIGMARELs)
			#pl.xlim([max(X[0],0),min(X[1],0.2)])
			pl.ylim([0,maxy])
			if interval>=0.5:
				Xx = max(0,sigmaRel-0.01)
				pl.xlim([Xx,Xx+0.2])
				################################################################
				X = pl.gca().get_xlim()
				notches = np.arange(sigmaRel,X[1]+0.1,0.01)
				pl.plot([sigmaRel,X[1]],[maxy,maxy],c='black',linewidth=2)#,linestyle='--')
				for inn,notch in enumerate(notches):
					dy=5
					pl.plot([notch,notch],[maxy-dy,maxy],c='black')
					if inn%5==0:
						pl.plot([notch,notch],[maxy-dy*1.5,maxy],c='black',linewidth=2)
			else:
				pl.xlim([0,sigmaRel+0.01])
				notches = np.arange(sigmaRel,-0.1,-0.01)
				pl.plot([0,sigmaRel],[maxy,maxy],c='black',linewidth=2)#,linestyle='--')
				for inn,notch in enumerate(notches):
					dy=5
					pl.plot([notch,notch],[maxy-dy,maxy],c='black')
					if inn%5==0:
						pl.plot([notch,notch],[maxy-dy*1.5,maxy],c='black',linewidth=2)
			################################################################
			pl.tight_layout()
			pl.savefig(PICSAVEPATH+f'95Bound_'+FILENAME+'.png')
			#pl.show()
#'''
#################################################################################
def get_indices(sigmas,Q):
	indices_list = np.zeros((Nsims,Nsims))
	for ix,x in enumerate(list(sigmas[Q].keys())):
		#print (Q,x,sigmas[Q][x][:])
		indices_list[ix,:len(sigmas[Q][x])] = sigmas[Q][x][:]
	return indices_list
def get_sigma_A_Bs(sigmas,A,B):
	sigmaAs,sigmaBs = list(sigmas[A].keys()),list(sigmas[B].keys())
	indicesA_list = get_indices(sigmas,A)
	indicesB_list = get_indices(sigmas,B)
	sigma_A_Bs    = np.zeros(Nsims)
	sigma_A_Bs[0] = sigmaBs[0]-sigmaAs[0]
	for isim in range(1,Nsims):
		Alocx = np.where(indicesA_list==isim)[0][0]#,Alocy =np.where(indicesA_list==isim)[1][0]
		Blocx = np.where(indicesB_list==isim)[0][0]#,Blocy =np.where(indicesB_list==isim)[1][0]
		sigma_A_Bs[isim] = (sigmaBs[Blocx]-sigmaAs[Alocx])/2
	return sigma_A_Bs
#################################################################################

#A,B   = 0.016,0.84
#Width = 1
#A,B   = 0.05,0.95
#Width = 2
SETS = {1:[0.16,0.84],2:[0.023,0.977]}
#SETS = {1:[0.16,0.84],2:[0.05,0.95]}
for Width in SETS:
	A,B = SETS[Width][:]
	for isFt, SIGMAFITkey in enumerate(SIGMAFITs):
		for isGr,sigmaRel in enumerate(SIGMARELs[1:]):
			pl.figure()
			if Width==1:
				pl.title(r'$(\sigma_{%s}-\sigma_{%s})/2$ Widths for $\sigma_{\rm{Rel}}=$'%(int(B*100),int(A*100))+f'{sigmaRel}',fontsize=FS)
			if Width==2:
				pl.title(r'$(\sigma_{%s}-\sigma_{%s})/2$ Widths for $\sigma_{\rm{Rel}}=$'%(int(B*1000)/10,int(A*1000)/10)+f'{sigmaRel}',fontsize=FS)
			maxy = 0
			for iapp,APPENDER in enumerate(APPENDERS):
				for iG,G in enumerate(Gs):
					for ipr,PRIOR in enumerate(PRIORS):
						for iNs,Nsiblings in enumerate(NSIBLINGs):
							SIGMAFIT = SIGMAFITs[SIGMAFITkey]
							if SIGMAFIT!=[]:
								#print (f'Loading G={G}, sigmaRel={sigmaRel}, sigmaFit = N({SIGMAFIT[0]},{SIGMAFIT[1]})')
								FILENAME      = get_FILENAME(G,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
								SAVEPATH_mini = intervals_SAVEPATH+FILENAME+'/'
								if os.path.exists(SAVEPATH_mini+FILENAME+f'_SIGMAS_{APPENDER}.pkl'):
									with open(SAVEPATH_mini+FILENAME+f'_SIGMAS_{APPENDER}.pkl','rb') as f:
										sigmas=pickle.load(f)
									sigma_A_Bs = get_sigma_A_Bs(sigmas,A,B)
									xtest,KDE  = get_KDE(sigma_A_Bs,1000,0,1,smoothing=3)
									lb,ub,sigma_lower,sigma_upper,siglo,sigup,Km,m_loc,m_loc_index = get_conf_interval(sigma_A_Bs,0.68,xtest,KDE,'mean')
									ccc=f'C{iG}'
									lstyle = ['-','--'][iapp]
									pl.plot(xtest,KDE,c=ccc,label=r'$G = %s$'%{G}*(iapp==0),linestyle=lstyle)
									pl.plot([m_loc,m_loc],[0,KDE[m_loc_index]],c=ccc,alpha=0.3,linestyle=lstyle)
									pl.fill_between(xtest[siglo:sigup],np.zeros(sigup-siglo),KDE[siglo:sigup],color=ccc,alpha=0.3)
									maxy = max(maxy,np.amax(KDE))
			FILENAME      = get_FILENAME(Gs,sigmaRel,SIGMAFIT,Nsiblings,Nsims,Ndec,PRIOR)
			pl.plot([-10,-9],[0,1],c='black',label=f'{Nsiblings} Siblings',linestyle='-')
			pl.plot([-10,-9],[0,1],c='black',label=f'Single SNe',linestyle='--')

			pl.plot([0.01,0.01],[0,maxy],c='black')#,linestyle='--')
			pl.plot([0.02,0.02],[0,maxy],c='black',linestyle='--')
			pl.plot([0.03,0.03],[0,maxy],c='black',linestyle=':')
			#pl.plot([0,0],[0,maxy],c='black',linewidth=5)
			pl.legend(fontsize=FS)#,bbox_to_anchor=(0,1.4),loc='upper left',ncol=4)
			pl.tick_params(labelsize=FS)
			#pl.xlabel(r'$\sigma_{%s}-\sigma_{%s}$'%(int(B*100),int(A*100)),fontsize=FS)
			pl.xlabel(f'Posterior {Width}'+r'$\sigma$' + ' Width',fontsize=FS)
			pl.ylabel(r'Density',fontsize=FS)
			pl.yticks([])
			#pl.xticks(SIGMARELs)
			#pl.xlim([max(X[0],0),min(X[1],0.2)])
			pl.ylim([0,maxy])
			#Xx = max(0,sigmaRel-0.01)
			pl.xlim([0,0.05])
			pl.tight_layout()
			pl.savefig(PICSAVEPATH+f'Widths_{A}_{B}'+FILENAME+'.png')
			#pl.show()
