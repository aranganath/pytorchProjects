import numpy as np
from sklearn.preprocessing import OneHotEncoder
class ThermometerEncoder():

    def __init__(self,arr,level):
       	self.data = arr
        self.level = level


    def quantization(self):
        quantVec = np.zeros(self.data.shape)
        for i in range(1,self.level):
            quantVec[self.data>1.0*i/self.level]+=1
        self.quantVec = quantVec
        return quantVec

    def OneHotEncode(self):

        n,w,h = self.quantVec.shape
        data = self.quantVec.reshape(n,-1)
        trm = OneHotEncoder(categorical_features = [0])
        OneHotEncoded = trm.fit(data)
        OneHotEncoded = trm.transform(OneHotEncoded).toarray()
        self.OneHotEncoded = OneHotEncoded
	#Do the encoding here
        return OneHotEncoded

    def ThermEncode(self):
        return ThermEncoded

    def ThermDecode(self, Thermdata):
        return ThermDecoded
