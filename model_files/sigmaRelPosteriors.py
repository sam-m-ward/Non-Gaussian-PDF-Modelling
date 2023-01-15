import numpy as np
import os, pickle
import matplotlib.pyplot as pl

def get_FILENAME(NAME,G,sigmaRel,Nsiblings,Nsims,Ndec,PRIOR):
	FILENAME= f'{NAME}_G{G}_sigRel{sigmaRel}_Nsibs{Nsiblings}_Nsims{Nsims}_Ndec{Ndec}_prior{PRIOR}'
	return FILENAME

def get_sim_samples(icdf,ucont,G,Nsiblings):
	sim_samples = {}
	for g in range(G):
		u_samples        = np.random.uniform(0,1,Nsiblings)
		u_indices        = [np.argmin(np.abs(ucont-u)) for u in u_samples]
		sim_samples[g]   = np.array([icdf[index] for index in u_indices])
	return sim_samples

def simulate_Dg(Nsiblings,sigmaRel,sigmafit):
	mu_plus_dM_common = np.random.uniform(25,50)
	dM_Rel_s          = np.random.normal(0,sigmaRel,Nsiblings)
	E_fit_s           = np.random.normal(0,sigmafit)
	mu_hat_s          = mu_plus_dM_common+dM_Rel_s+E_fit_s
	return mu_hat_s

def Gaussian(x,mu,sigma):
	num = np.exp(-0.5*( ( (x-mu)/sigma)**2) )
	den = sigma*((2*np.pi)**0.5)
	return num/den

def get_single_galaxy_posterior(mu_hat_s,sigma_fit_s,sigRs):
	def get_posterior(mu_hat_s,sigma_fit_s,sigR):
		ws  = 1/(sigma_fit_s**2 + sigR**2)
		muw = sum(ws*mu_hat_s)/sum(ws)
		p   = 1
		for mu_hat,sigma_fit in zip(mu_hat_s,sigma_fit_s):
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

def get_CDF(G_DICT,G): return np.cumsum(get_multi_gal_posterior(G_DICT,G))/sum(get_multi_gal_posterior(G_DICT,G))



class sigmaRelPosteriors:
	def __init__(self,choices):
		self.choices=choices
		globals().update(choices)
		SAVEPATH = PRODUCTPATH + NAME + '/' ; intervals_SAVEPATH =  INTERVALSPATH + NAME + '/'
		for x in [PRODUCTPATH,INTERVALSPATH,SAVEPATH,intervals_SAVEPATH]:
			if not os.path.exists(x):
				os.mkdir(x)
		self.SAVEPATH = SAVEPATH
		self.intervals_SAVEPATH = intervals_SAVEPATH

	def create_posterior_files(self,pdf):
		for isGr,sigmaRel in enumerate(SIGMARELs):
			for ipr,PRIOR in enumerate(PRIORS):
				sigRs      = np.linspace(0,PRIOR,Ndec)
				for iNs,Nsiblings in enumerate(NSIBLINGs):
					FILENAME        = get_FILENAME(NAME,Gmax,sigmaRel,Nsiblings,Nsims,Ndec,PRIOR)

					for isim in range(Nsims):
						G_DICT_SIBS = {'mu_hat_s':[],'sigma_fit_s':[],'posterior':[]}
						filename    = self.SAVEPATH+FILENAME+f'isim{isim}.pkl'
						if not os.path.exists(filename):
							if (isim+1)%10==0:
								print (f'Sim{isim+1}/{Nsims}')
							sim_samples = get_sim_samples(pdf.icdf,pdf.ucont,Gmax,Nsiblings)
							for g in range(Gmax):
								sigma_fit_s          = sim_samples[g]
								mu_hat_s             = simulate_Dg(Nsiblings,sigmaRel,sim_samples[g])#Here sigmafit is the same for both siblings, and same as single SN
								posterior            = get_single_galaxy_posterior(mu_hat_s,sigma_fit_s,sigRs)
								G_DICT_SIBS['mu_hat_s'].append(mu_hat_s)
								G_DICT_SIBS['sigma_fit_s'].append(sigma_fit_s)
								G_DICT_SIBS['posterior'].append(posterior)
							G_DICT_SIBS['Total_Posterior'] = get_multi_gal_posterior(G_DICT_SIBS,Gmax)
							with open(filename,'wb') as f:
								pickle.dump(G_DICT_SIBS,f)
						else:
							continue


		if not os.path.exists(self.intervals_SAVEPATH):
			os.mkdir(self.intervals_SAVEPATH)
		intervals  = np.array([0.023,0.05,0.16,0.50,0.68,0.84,0.95,0.977])#the bounds we wish to probe
		intervals2 = np.array([0.023,0.05,0.16,0.50,0.68,0.84,0.95,0.977])#the intervals on those particular bounds
		print ('###'*10)
		print ('Creating Sigmas')
		for isGr,sigmaRel in enumerate(SIGMARELs):
			for iG,G in enumerate(Gs):
				for ipr,PRIOR in enumerate(PRIORS):
					sigRs      = np.linspace(0,PRIOR,Ndec)
					for iNs,Nsiblings in enumerate(NSIBLINGs):
						print (f'Loading G={G}, sigmaRel={sigmaRel}')
						FILENAME_Gdict       = get_FILENAME(NAME,Gmax,sigmaRel,Nsiblings,Nsims,Ndec,PRIOR)
						FILENAME_intervals   = get_FILENAME(NAME,G,sigmaRel,Nsiblings,Nsims,Ndec,PRIOR)
						SAVEPATH_mini  = self.intervals_SAVEPATH+FILENAME_intervals+'/'
						if not os.path.exists(SAVEPATH_mini):
							os.mkdir(SAVEPATH_mini)
						if not os.path.exists(SAVEPATH_mini+'sigmas.pkl'):
							sigmas_sibs   = {x:{} for x in intervals}
							go=True
							for isim in range(Nsims):
								filename  =  FILENAME_Gdict +f'isim{isim}'
								try:
									with open(self.SAVEPATH+filename+'.pkl','rb') as f:
										G_DICT_SIBS = pickle.load(f)
									for Xcred in intervals:
										sigma_X_siblings = sigRs[np.argmin(np.abs(get_CDF(G_DICT_SIBS,G)-Xcred))]#Here we only take the first G galaxies from set of Gmax
										try:
											sigmas_sibs[Xcred][sigma_X_siblings].append(isim)
										except KeyError:
											sigmas_sibs[Xcred][sigma_X_siblings] = [isim]
								except Exception as e:
									print (f'Not created yet: {filename}', e)
									go=False
							if go:
								with open(SAVEPATH_mini+'sigmas.pkl','wb') as f:
									print (f'Saving {FILENAME_intervals}/sigmas.pkl siblings')
									pickle.dump(sigmas_sibs,f)




		print ('###'*10)
		print ('Creating Sigma Intervals')
		for isGr,sigmaRel in enumerate(SIGMARELs):
			for iG,G in enumerate(Gs):
				for ipr,PRIOR in enumerate(PRIORS):
					for iNs,Nsiblings in enumerate(NSIBLINGs):
						print (f'Loading G={G}, sigmaRel={sigmaRel}')
						FILENAME      = get_FILENAME(NAME,G,sigmaRel,Nsiblings,Nsims,Ndec,PRIOR)
						SAVEPATH_mini = self.intervals_SAVEPATH+FILENAME+'/'
						if not os.path.exists(SAVEPATH_mini+'intervals.pkl'):
							if os.path.exists(SAVEPATH_mini+'sigmas.pkl'):
								with open(SAVEPATH_mini+'sigmas.pkl','rb') as f:
									sigmas_sibs   = pickle.load(f)
								for irun,sigmas in enumerate([sigmas_sibs]):
									sigma_interval_map = {interval:{} for interval in intervals}
									for Xcred in intervals:
										sigma_Xs_list = list(sigmas[Xcred].keys())
										################################################
										sigma_Xs_list = [x for x in sigma_Xs_list for _ in range(len(sigmas[Xcred][x]))]#Here we repeat the values for the number of times it appears
										################################################
										sigma_Xs_list.sort()
										Xcreds = {Y:sigma_Xs_list[int( (len(sigma_Xs_list)-1)*Y) ] for Y in intervals2}
										sigma_interval_map[Xcred]=Xcreds
									with open(SAVEPATH_mini+'intervals.pkl','wb') as f:
										print (f'Saving {FILENAME}/intervals.pkl')
										pickle.dump(sigma_interval_map,f)
							else:
								print (f'Not created sigmas yet: {FILENAME}/sigmas.pkl')
	def plot_intervals(self):
		sss    = 30 ; Capsize= 4 ; dd     = 0.00171 ; alph   = 0.5 ; ffextra = 0.75 ; FS = 18
		ddlist = (np.arange(0,len(Gs),1) - int(len(Gs)/2) + (0.5-(len(Gs)/2)%1))*dd*2
		pl.figure(figsize=(12,5))
		for isGr,sigmaRel in enumerate(SIGMARELs):
			for iG,G in enumerate(Gs):
				for ipr,PRIOR in enumerate(PRIORS):
					for iNs,Nsiblings in enumerate(NSIBLINGs):
						FILENAME      = get_FILENAME(NAME,G,sigmaRel,Nsiblings,Nsims,Ndec,PRIOR)
						SAVEPATH_mini = self.intervals_SAVEPATH+FILENAME+'/'
						if os.path.exists(SAVEPATH_mini+f'intervals.pkl'):
							with open(SAVEPATH_mini+f'intervals.pkl','rb') as f:
								sigma_interval_map = pickle.load(f)
							XCOORD = sigmaRel + ddlist[iG]-0.005*(sigmaRel==0.005)
							ccc = f'C{iG}' ; mmarker = 'o'
							sigma_023,sigma_16,sigma_50,sigma_84,sigma_977 = sigma_interval_map[0.023][0.50],sigma_interval_map[0.16][0.50],sigma_interval_map[0.50][0.50],sigma_interval_map[0.84][0.50],sigma_interval_map[0.977][0.50]
							pl.errorbar(XCOORD,sigma_50-sigmaRel,yerr = [[sigma_50-sigma_16],[sigma_84-sigma_50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=1)
							pl.errorbar(XCOORD,sigma_50-sigmaRel,yerr = [[sigma_50-sigma_023],[sigma_977-sigma_50]],marker=None,markersize=5,c=ccc,capsize=Capsize,alpha=0.3)
							pl.scatter(XCOORD,sigma_50-sigmaRel,marker=mmarker,s=sss,c=ccc,label=r'$G = %s$'%{G}*(isGr==0))
			ddlist = list(ddlist)+[ddlist[iG]+dd*ffextra]
			pl.plot([sigmaRel+ddlist[0]-0.005*(sigmaRel==0.005),sigmaRel+ddlist[-1]-0.005*(sigmaRel==0.005)], [0-sigmaRel,0-sigmaRel], color='black',linewidth=2,linestyle='--')
		pl.scatter(XCOORD,100,marker='o',s=sss,c='black',label=f'{Nsiblings} Siblings')
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
		FILENAME      = get_FILENAME(NAME,Gs,SIGMARELs,Nsiblings,Nsims,Ndec,PRIOR)
		#pl.savefig(PICSAVEPATH+f'SigmaIntervals_'+FILENAME+'.png')
		pl.show()
