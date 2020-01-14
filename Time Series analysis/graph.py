from numpy import array
from keras.models import Sequential
from keras.layers import Dense

X= array([[10,20,30],[20,30,40],[30,40,50],[40,50,60]])
y = array([40,50,60,70])

model = Sequential()
model.add(Dense(100,activation='relu', input_dim=3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X,y,epochs=2000,verbose=0)
x_input = array([50,60,70])
x_input=x_input.reshape((1,3))
yhat = model.predict(x_input,verbose=0)
print(yhat)