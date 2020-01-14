import numpy as np
from AuxFuncs import cellMatMult,cellTransMatMult

def lsr1tr(g_old, g,x_old, x, SY, SS, YY, Sc, Yc, ind, lqn_nbupdate, lqn_start, lqn_end, Hdiag, itr, idebug, delta, Bs):
	xs = np.empty(0)
	if(itr==1):
		normg = np.linalg.norm(g)
		normg = max(1.0e-6, normg)

		s = -(delta/normg)*g
		Bs = (1/Hdiag)*s
	else:
		[SY,SS,YY,Sc,Yc,ind,lqn_nbupdate,lqn_start,lqn_end,Hdiag,skipped] = lsr1_updateSY(g-g_old,x-x_old,SY,SS,YY,Sc,Yc,ind,lqn_nbupdate,lqn_start,lqn_end,Hdiag,Bs, delta, g)
		if (skipped and idebug):
			print('At Iteration = {}, \t Skipping the LSR1 Update \n'.format(itr))

    	gamma = 1/Hdiag
    	if(lqn_nbupdate>0):
    		[_,s,Bs,_,_,_,gamma] = lsr1tr_obs(g,SY,SS,YY,Sc,Yc,ind,delta,gamma)
    		Hdiag = 1/gamma
    	else:
			normg = np.linalg.norm(g)
			normg = max(1.0e-6, normg)
			s = -(delta/normg)*g
			Bs = gamma*s

	np.append(xs,s)
	iterses = 1
	return [xs,iterses, SY, SS, YY,Sc,ind,lqn_nbupdate,lqn_start,lqn_end,Hdiag,Bs]

def lsr1tr_obs(g,SY, SS,YY, Sc, Yc, indS, delta, gamma):
	maxiter = 100
	tol = 1e-10
	try:
		gammaIn = gamma
		A = np.tril(SY) + np.tril(SY,-1)
		B = SS
		eABmin = min(np.eig(A,B))

		if(gamma >-eABmin or gamma==1):
			if (eABmin > 0):
				gamma = max(0.5*eABmin, 1e-6)
			else:
				gamma = min(1.5*eABmin,-1e-6)
			print('gamma={}, eABmin={}, gammaNew={}\n'.format(gammaIn, eABmin, gamma))
	except:
		gamma = gammaIn

	invM = np.tril(SY) + np.transpose(np.tril(SY,-1)) - gamma*SS
	invM = (invM +np.transpose(invM))/2
	PsiPsi = YY - gamma *(SY +np.transpose(SY) + gamma**2*SS)

	R = np.linalg.cholesky(PsiPsi)
	RMR = R*(invM*np.linalg.inv(np.transpose(R)))
	RMR = (RMR +np.transpose(RMR))/2
	U,D = np.linalg.eig(RMR)
	U = np.matrix(U)
	D = np.matrix(D)
	diag = np.diag(D)
	D = np.sort(diag)
	indD = np.argsort(diag)
	U = U[:,indD]
	sizeD = np.size(D)
	Lambda_one = D + gamma*np.ones(sizeD)
	Lambda = np.append(Lambda_one,gamma)
	Lambda = Lambda*(np.absolute(Lambda)>tol)
	lambda_min = np.min([Lambda[1], gamma])

	RU = R*np.linalg.inv(U)
	Psig = cellTransMatMult(Yc, indS, g) - gamma*cellTransMatMult(Sc, indS, g)
	g_parallel = np.transpose(RU)*Psig

	a_kp2 = np.sqrt(np.absoute(np.transpose(g)*g-np.transpose(g_parallel)*g_parallel))
	if(a_kp2<tol):
		a_kp2 = 0

	a_j = np.append(g_parallel,a_kp2)

	if(lambda_min>0 and np.linalg.norm(a_j/Lambda)<=delta):
		sigmaStar = 0
		pStar = ComputeSBySMW(gamma, g, Psing,Sc,Yc, indS,gamma, invM, PsiPsi)
	elif(lambda_min<=0 and phiBar_f(-lambda_min,delta, Lambda, delta,Lambda, a_j)>0):
		Psi = cell2mat(Yc[indS]) - gamma*cell2mat(Sc[indS])
		sigmaStar = -lambda_min
		P_parallel = Psi*RU
		index_pseudo = find(np.absolute(Lambda+sigmaStar)>tol)
		v = np.zeros(sizeD+1,1)
		v[index_pseudo] = a_j[index_pseudo]/(Lamda[index_pseudo]+sigmaStar)
		if(np.absolute(gamma + sigmaStar)<tol):
			pStar = -P_parallel*v[0:sizeD-1]
		else:
			pStar = -P_parallel*v[0:sizeD-1] + (1/(gamma+sigmaStar))*(Psi*(PsiPsi*np.linalg.inv(Psig))) - (g / (gamma+sigmaStar))

		if(lambda_min<0):
			alpha = np.sqrt(delta^2-np.transpose(pStar)*pStar)
			pHatStar = pStar
		
			if(np.absolute(lambda_min-Lambda[1])<tol):
				zstar = (1/np.linalg.norm(P_parallel[:,0]))*alpha*P_parallel[:,0]
			else:
				e = np.zeros(np.size(g,0),0)
				found = 0
				for i in range(sizeD):
					e[i] = 1
					u_min = e-P_parallel*np.transpose(P_parallel[i,:])
					if (np.norm(u_min)>tol):
						found =1
						break
					e[i] =0
				if(found==0):
					e[m+1] = 1
					u_min = e - P_parallel*P_parallel[i,:]
				u_min = u_min/np.linalg.norm(u_min)
				zstar = alpha*u_min

			pStar = pHatStar + zstar
		else:
			if(lambda_min>0):
				sigmaStar = Newton(0,maxiter,tol,delta,Lambda,a_j)
			else:
				sigmaHat = max(a_j/delta - Lambda)
				if (sigmaHat>-lambda_min):
					sigmaStar = Newton(sigmaHat, maxiter,tol,delta,Lambda,a_j)
				else:
					sigmaStar = Newton(-lambda_min, maxiter,tol,delta,Lambda,a_j)

		pStar = ComputeSBySMW(gamma+sigmaStar,g,Psig,Sc, Yc,indS, gamma,invM, PsiPsi)

	PsipStar = cellTransMatMult(Yc, indS,pStar) - gamma*cellTransMatMult(Sc,indS,pStar)
	tmp = invM* np.linalg.inv(PsipStar)
	Psitmp = cellMatMult(Yc, indS,tmp) - gamma*cellMatMult(Sc,indS,pStar)
	tmp = invM*np.linalg.inv(PsipStar)
	Psitmp = cellMatMult(Yc,indS,tmp) - gamma*cellMatMult(Sc,indS,tmp)
	BpStar = gamma*pStar + Psitmp
	if(show>1):
		opt1 = np.linalg.norm(BpStar+sigmaStar*pStar + g)
		opt2 = sigmaStar*np.linalg.norm(delta-np.linalg.norm(pStar))
		spd_check = lambda_min + sigmaStar
		if(show==2):
			print('Optimality condition #1: {}'.format(opt1))
			print('Optimality condition #2: {}'.format(opt2))
			print('lambda_min+sigma*:{}, lam:{}, sig={}'.format(spd_check, lambda_min,sigmaStar))
			print('\n')
	else:
		opt1 = []
		opt2 = []
		spd_check = []
		phiBar_check = []

	return [sigmaStar,pStar,BpStar,opt1,opt2,spd_check, gamma]

def ComputeSBySMW(tauStar,g,Psig, Sc,Yc,indS,gamma,invM, PsiPsi):
	vw = tauStar^2*invM + tauStar*PsiPsi
	tmp = vw*np.linalg.inv(Psig)
	Psitmp = cellMatMult(Yc, indS, tmp)-gamma*cellMatMult(Sc,indS, tmp)
	pStar = -g/tauStar + Psitmp
	return pStar

def phiBar_f(sugma,delta,D,a_j):
	m = np.size(a_j,0)
	D = D + sigma*np.matrix(np.ones(m,1))
	eps_tol = 1e-10
	if (np.sum(np.absolute(a_j))>0 or np.sum(np.absolute(np.diag(D)))<eps_tol):
		pnorm2 = 0
		for i in range(m):
			if (np.absolute(a_j[i]) > eps_tol and (np.absolute(D[i]) > eps_tol)):
				phiBar = -1/delta
				return phiBar
			elif (np.absolute(a_j[i])>eps_tol and np.absolute(D[i])>eps_tol):
				pnorm2 = pnorm2 + (a_j[i]/D[i])^2

		phiBar = np.sqrt(1/pnorm2) - 1/delta
		
	return phiBar

def phiBar_fg(sigma, delta, D, a_j):
	m = np.size(a_j,0)
	D = D + simga*np.ones(m,1)
	eps_tol = 1e-10
	phiBar_g = 0
	if(np.sum([np.absolute(i)<eps_tol for i in a_j]) > 0 or np.sum([np.absolute(d)<eps_tol for d in D])>0):
		pnorm2 = 0
		for i in range(m):
			if(np.sum(np.absolute(aj)>eps_tol for aj in a_j) and np.absolute(D[i])>eps_tol):
				phiBar = -1/delta
				phiBar_g = 1/sqrt(eps_tol)
				return [phiBar, phiBar_g]
			elif( np.absolute(a_j[i]) > eps_tol and np.absolute(D[i])>eps_tol):
				pnorm2 = pnorm2 + (a_j[i]/D[i])^2
				phiBar_g = phiBar_g + ((a_j[i])^2)/((D[i])^3)

		normP = np.sqrt(pnorm2)
		phiBar = 1/normP - 1/delta
		phiBar_g = phiBar_g/normP^3
		return [phiBar, phiBar_g]

	return [phiBar, phiBar_g]

def Newton(x0, maxIter, tol, delta, Lambda, a_j):
	x = x0
	k = 0
	[f, g] = phiBar_fg(x,delta, Lambda, a_j)
	while(np.absolute(f)>eps and (k<maxIter)):
		x = x - f/g
		[f,g] = phiBar_fg(x, delta, Lambda, a_j)
		k = k+1
	return x 