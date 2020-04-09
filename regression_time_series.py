import pandas as pd
import numpy as np
import random
import sys
from sklearn.linear_model import LinearRegression

arg_list = sys.argv
path = arg_list[1]

data = pd.read_csv(path, sep=';',dtype='unicode')
data
X = pd.DataFrame()
X['row_no'] = np.arange(len(data))
X['data'] = data['Global_active_power']

window_size = 60
X2 = np.array([0] * (window_size + 1))

count = 1
while(count<=50000):
  row = random.randrange(len(X) - window_size - 1)
  temp = X[row:row+window_size+1]['data'].tolist()
  if('?' not in temp):
    temp = [float(item) for item in temp]
    X2 = np.vstack((X2,temp)) 
    count = count + 1

X2 = X2[1:,:]
# print(X2)
# print(X2.shape)

X_train = X2[:,0:60]
y_train = X2[:,60]

model = LinearRegression()
model.fit(X_train, y_train)

X1 = X
X1 = np.array(X['data'])
count = 0
predictions = []
for i in range(len(data)):
  temp = X1[i]
  if(temp == '?'):
    if(i < window_size):
      pred1 = np.nanmean()
      X1[i] = pred1
    else:
      temp1 = np.array(X1[i-60:i].tolist())
      temp1 = temp1.astype('float32')
      test = temp1.reshape((1,-1))
      pred1 = model.predict(test)
      predictions.append(pred1[0])
      X1[i] = pred1[0]
    count = count + 1

# print(predictions)
for i in predictions:
    print(i)
    
# print(len(predictions))