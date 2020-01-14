import numpy as np
def cellMatMult(Bcell, ind,v):
	Bv = np.matrix(np.zeros(Bcell[ind[0]],1))
	kk = 1
	for jj in ind:
		if(jj==ind):
			Bvjj = Bcell[jj-1]*v[kk]
			Bv = Bv + Bvjj
			kk = kk + 1

	return Bv

def lsr1updateSY(y,s,SY,SS,YY,Sc,Yc,ind,lqn_nbupdate,lqn_start,lqn_end,Hdiag,Bs, delta, g):
	yBs = y-Bs
	if(np.absolute(np.transpose(s)*yBs)> np.sqrt(eps)*np.norm(s)*norm(yBs)):
		skipped = 0
		maxCor = np.size(Sc)
		lqn_nbupdate = lqn_nbupdate+1
		if (lqn_end<maxCor):
			lqn_end = lqn_end+1
			if (lqn_start!=1):
				if(lqn_start ==maxCor):
					lqn_start =1
				else:
					lqn_start = lqn_start+1

		else:
			lqn_start = min(2,maxCor)
			lqn_end = 1


	return [SY,SS,YY,Sc,Yc,ind,lqn_nbupdate,lqn_start,lqn_end,Hdiag,skipped]

def cellTransMatMult(Bcell, ind, v):
	Bv = np.matrix(np.zeros((np.size(ind),1)))
	kk =1
	for jj in ind:
		Bv[kk] = np.transpose(Bcell[jj])*v
		kk=kk+1
	return Bv