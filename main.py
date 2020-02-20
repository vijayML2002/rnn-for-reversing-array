

#importing the required lib
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Input
from keras.optimizers import Adam


#creating dataset for training
x = []
y = []

no_ex = 10000 #the number of training examples
len_seq = 10  #sequence length

for i in range(no_ex):
  inp = np.random.randint(0,50,size=(1,len_seq))[0]
  out = inp[::-1]

  x.append(inp)
  y.append(out)


#converting the input and output list to array
x = np.array(x)
y = np.array(y)


#printing out shape of input and output
print('X SHAPE -- {}'.format(x.shape))
print('Y SHAPE -- {}'.format(y.shape))


#printing out input and output
for i in range(5):
  print('INPUT   - {}'.format(x[i]))
  print('OUTPUT  - {}'.format(y[i]))
  print()


#changing the input dimension for training
x = np.expand_dims(x , axis = 2)

print('X SHAPE  -  {}'.format(x.shape))
print('Y SHAPE  -  {}'.format(y.shape))


#defining the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=x.shape[1:]))
model.add(tf.keras.layers.SimpleRNN(150,activation='relu'))
model.add(tf.keras.layers.Dense(len_seq,activation='relu'))
model.compile(optimizer = tf.keras.optimizers.Adam(0.001) , loss = 'mean_squared_error' , metrics = ['accuracy'])


#fiting or training the model
EPOCHS = 20
model.fit(x , y , epochs=EPOCHS)


#function for prediction
def predict(x,len_seq):
  val = x
  val = np.array(val)
  val = val.reshape(1,len_seq,1)
  
  ans = model.predict(val)[0]
  pred = []

  for i in range(len(ans)):
    pred.append(int(round(ans[i])))

  return pred


ans = predict([2,4,6,8,9,33,11,22,2,9],len_seq)
print('The predicted value -- {}'.format(ans))
