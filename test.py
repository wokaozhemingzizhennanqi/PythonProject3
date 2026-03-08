import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('dataset/income.csv')

plt.scatter(data['Education'], data['Income'])

plt.xlabel('Education')
plt.ylabel('Income')

w=torch.randn(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)

learning_rate=0.01

X = torch.from_numpy(data['Education'].values.reshape(-1,1)).type(torch.FloatTensor)
Y = torch.from_numpy(data['Income'].values.reshape(-1,1)).type(torch.FloatTensor)
# 训练过程
for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = torch.matmul(x,w ) + b
        #损失函数
        loss =  (y - y_pred).pow(2).sum()

        if w.grad is not None:
            w.grad.data.zero_()
        if b.grad is not None:
            b.grad.data.zero_()

        loss.backward()

        with torch.no_grad():
            w.data -= learning_rate * w.grad.data
            b.data -= learning_rate * b.grad.data

print(w)

plt.scatter(data['Education'], data['Income'])
plt.plot(X.numpy(),(torch.matmul(X,w) + b).detach().numpy(),c='r')
plt.show()