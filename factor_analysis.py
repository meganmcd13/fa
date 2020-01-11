import numpy as np
import sklearn.model_selection as ms

class factor_analysis:


	def __init__(self,model_type='fa',min_var=0.01):
		if model_type.lower()!='ppca' and model_type.lower()!='fa':
			raise ValueError('model_type should be "fa" or "ppca"')
		self.model_type = model_type
		self.min_var = min_var


	def train(self,X,zDim,tol=1e-8,max_iter=int(1e8),verbose=False,rand_seed=None):
		# set random seed
		if not(rand_seed is None):
			np.random.seed(rand_seed)

		# some useful parameters
		N,D = X.shape
		mu = X.mean(axis=0)
		cX = X - mu
		covX = np.cov(cX.T,bias=True)
		var_floor = self.min_var*np.diag(covX)
		Iz = np.identity(zDim)
		Ix = np.identity(D)
		const = D*np.log(2*np.pi)

		# check that the data matrix is full rank
		if np.linalg.matrix_rank(covX)==D:
			scale = np.exp(2/D*np.sum(np.log(np.diag(np.linalg.cholesky(covX)))))
		else:
			raise np.linalg.LinAlgError('Covariance matrix is low rank')
		
		# initialize fa parameters (L and Ph)
		Ph = np.diag(covX)
		# return mu and Ph if zDim=0
		if zDim==0:
			L = np.random.randn(D,zDim)
			fa_params = {'mu':mu,'L':L,'Ph':Ph,'zDim':zDim}
			self.fa_params = fa_params
			z,LL = self.estep(X)
			return LL
		else:
			L = np.random.randn(D,zDim) * np.sqrt(scale/zDim)

		# fit fa model
		LL = []
		for i in range(max_iter):
			# E-step
			iPh = np.diag(1/Ph)
			iPhL = iPh.dot(L)
			iSig = iPh - iPhL.dot(np.linalg.inv(Iz+(L.T).dot(iPhL))).dot(iPhL.T)
			iSigL = iSig.dot(L)
			cov_iSigL = covX.dot(iSigL)
			E_zz = Iz - (L.T).dot(iSigL) + (iSigL.T).dot(cov_iSigL)
			
			# compute log likelihood
			logDet = 2*np.sum(np.log(np.diag(np.linalg.cholesky(iSig))))
			curr_LL = -N/2 * (const - logDet + np.trace(iSig.dot(covX)))
			LL.append(curr_LL)
			if verbose:
				print('EM iteration ',i,', LL={:.2f}'.format(curr_LL))

			# M-step
			L = cov_iSigL.dot(np.linalg.inv(E_zz))
			Ph = np.diag(covX) - np.diag(cov_iSigL.dot(L.T))

			# set noise values to be the same if ppca
			if self.model_type.lower()=='ppca':
				Ph = np.mean(Ph) * np.ones(Ph.shape)

			# make sure values are above noise floor
			Ph = np.maximum(Ph,self.min_var)

			# check convergence
			if i<=1:
				LLbase = curr_LL
			elif curr_LL<LL[-2]:
				print('EM violoation')
			elif (LL[-1]-LLbase)<((1+tol)*(LL[-2]-LLbase)):
				break

		# store model parameters in dict
		fa_params = {'mu':mu,'L':L,'Ph':Ph,'zDim':zDim}
		self.fa_params = fa_params

		tmp,curr_LL = self.estep(X)
		LL.append(curr_LL)

		return np.array(LL)


	def get_params(self):
		return self.fa_params

	def set_params(self,fa_params):
		self.fa_params = fa_params


	def estep(self,X):
		mu = self.fa_params['mu']
		L = self.fa_params['L']
		Ph = self.fa_params['Ph']
		zDim = self.fa_params['zDim']

		N,xDim = X.shape
		cX = X - mu
		covX = (1/N) * ((cX.T).dot(cX))
		Iz = np.eye(zDim)

		est_cov = L.dot(L.T) + np.diag(Ph)
		iSig = np.linalg.inv(est_cov)

		# compute posterior on latents
		z_mu = (L.T).dot(iSig).dot(cX.T)
		z_cov = Iz - (L.T).dot(iSig).dot(L)

		# compute LL
		const = xDim*np.log(2*np.pi)
		logDet = 2*np.sum(np.log(np.diag(np.linalg.cholesky(iSig))))
		LL = -N/2 * (const - logDet + np.trace(iSig.dot(covX)))

		# return posterior and LL
		z = { 'z_mu':z_mu.T, 'z_cov':z_cov }
		return z,LL


	def orthogonalize(self,z_mu):
		N,zDim = z_mu.shape

		L = self.fa_params['L']
		u,s,vt = np.linalg.svd(L,full_matrices=False)
		Lorth = u
		TT = np.diag(s).dot(vt)
		z_orth = (TT.dot(z_mu.T)).T

		return z_orth, Lorth


	def crossvalidate(self,X,zDim_list=np.linspace(0,10,11),n_folds=10,verbose=True,rand_seed=None):
		N,D = X.shape

		# make sure z dims are integers
		z_list = zDim_list.astype(int)

		# create k-fold iterator
		if verbose:
			print('Crossvalidating FA model to choose # of dims...')
		cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

		# iterate through train/test splits
		i = 0
		LLs = np.zeros([n_folds,len(z_list)])
		for train_idx,test_idx in cv_kfold.split(X):
			if verbose:
				print('   Fold ',i+1,' of ',n_folds,'...')

			X_train,X_test = X[train_idx], X[test_idx]
			# iterate through each zDim
			for j in range(len(z_list)):
				tmp = factor_analysis(self.model_type,self.min_var)
				tmp.train(X_train,z_list[j])
				z,curr_LL = tmp.estep(X_test)
				LLs[i,j] = curr_LL
			i = i+1
		
		sum_LLs = LLs.sum(axis=0)

		# find the best # of z dimensions and train FA model
		max_idx = np.argmax(sum_LLs)
		self.train(X,z_list[max_idx])

		return sum_LLs, z_list, sum_LLs[max_idx], z_list[max_idx]

