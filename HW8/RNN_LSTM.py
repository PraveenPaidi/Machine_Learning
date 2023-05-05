#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as dt
import yahoo_fin.stock_info as yf
import pandas as pd
import numpy as np
import torch
import time
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def dataprocess(newdata, lookback, Delta ):
    newdata = newdata.numpy()
    data = []
    for index in range(len(newdata) - lookback): 
        data.append(newdata[index: index + lookback])
    xdata = np.array(data); 
    y = xdata[1:,Delta-1] 
    x_train, x_test, y_train, y_test = train_test_split(xdata[0:len(xdata[:,0])-1],y,test_size=0.3,random_state=42)
    return [x_train, y_train, x_test, y_test]

def scaling(timer,price):
    x_train = np.zeros([len(price),len(timer)],dtype=float)
    for i,t in enumerate(timer):
        x_tr = scaler.fit_transform(price['close'].values.reshape(-1,1))
        x_train[:,i] = x_tr.reshape(1,-1)
    x_train = torch.from_numpy(x_train).float()
    return(x_train)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

timer = ['PBR']
input_dim = len(timer)
output_dim = 1
hidden_dim = 15
num_layers = 3
start = dt.datetime(2020, 1, 1)
end = dt.datetime(2022, 1, 1)
dataf = yf.get_data('PBR', start, end, interval='1d')
price = dataf[['close']]    
scaler = MinMaxScaler(feature_range=(-1, 1))
x_train_ = scaling(timer,price)
x_train, y_train, x_test, y_test = dataprocess(x_train_, 30, 1)
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


hist = np.zeros(400)
start_time = time.time()
lstm = []
error = 1 
t = 0

while (error > 0.01) and (t < 400):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_lstm)
    error = abs(loss.item())

    if loss.item() < 0.01:
        break

    hist[t] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    t += 1

print(f"Training completed in {time.time() - start_time:.2f} seconds.")

ypred_train = model(x_train).detach().numpy()
acc_train = r2_score(y_train_lstm, ypred_train) 
print('Train Accuracy' , acc_train)
ypred_test = model(x_test).detach().numpy()
acc_test = r2_score(y_test_lstm, ypred_test)
print('Test Accuracy' ,acc_test)

# Plot 
fig = plt.figure(figsize=(6, 4))
plt.plot(y_test_lstm, label='ground past')
plt.plot(ypred_test, label='predicted past')
plt.grid()
plt.xlabel('Past ')
plt.ylabel('Past scaled Price')
plt.legend()
plt.show()

start = dt.datetime(2022, 1, 1)
end = dt.datetime(2023, 1, 1)
# Use the trained model to predict future prices
dataf = yf.get_data('PBR', start, end, interval='1d')
price = dataf[['close']]
x_future = scaling(timer, price)
x_future = torch.tensor(x_future).type(torch.FloatTensor)
ypred_trainf = model(x_train).detach().numpy()
acc_train = r2_score(ypred_trainf, y_train)
print('Future year Accuracy', acc_train)


fig = plt.figure(figsize=(6, 4))
plt.plot(y_train, label='ground future')
plt.plot(ypred_train, label='predicted future')
plt.grid()
plt.xlabel('Future ')
plt.ylabel('Future scaled Price')
plt.legend()
plt.show()


# In[ ]:




