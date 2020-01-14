import numpy as np
def quantization(arr,k):
    quant = np.zeros(arr.shape)
    for i in range(1,k):
        quant[arr>1.0*i/k]+=1
    return quant

def onehot(arr,k):
    n,w,h = arr.shape
    arr = arr.reshape(n,-1)
    enc=OneHotEncoder(n_values=k,sparse=False)
    arr = enc.fit_transform(arr)
    arr = arr.reshape(n,w,h,k)
    arr = arr.transpose(0,3,1,2)
    return arr

def tempcode(arr,k):
    tempcode = np.zeros(arr.shape)
    for i in range(k):
        tempcode[:,i,:,:] = np.sum(arr[:,:i+1,:,:],axis=1)
    return tempcode
