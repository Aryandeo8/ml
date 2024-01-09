import numpy as np
import pandas as pd
df=pd.read_csv('Lineardata_train.csv')
c=np.array(df)
x=c[0:,1:]
y=c[0:,0]
def calcost(x, y, w, b):
    m = len(y)
    predictions = np.dot(x, w) + b
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost
def gradient(x,y,w,b):
  m,n=x.shape
  djdw=np.zeros(n,).tolist()
  djdb=0.0
  for i in range(m):
    error=(np.dot(x[i],w)+b)-y[i]
    djdw=np.add(djdw,np.multiply(x[i],error))
    djdb=djdb + error
  djdw=np.divide(djdw,m)
  djdb=djdb/m
  return djdw,djdb
def grad(x, y, wt, bt, lr, iters):
    costh = []
    m, n = x.shape

    for i in range(iters):
        djdw, djdb = gradient(x, y, wt, bt)
        wt =np.subtract(wt,np.multiply(lr,djdw))
        bt -= lr * djdb
        costx = calcost(x, y, wt, bt)
        print(i,costx)
        costh.append(costx)
    return wt, bt, costh
win=np.zeros(20,)
bin=0.0
lr=0.000035555555555
iters=100
wf,bf,costh=grad(x,y,win,bin,lr,iters)
dft=pd.read_csv('Lineardata_test.csv')
c=np.array(df)
xt=c[:,1:]
yt=c[:,0]
pv=[]
for i in range(xt.shape[0]):
  v=np.dot(xt[i],wf )+bf
  pv.append(v)
print(costh[-1])
it=[]
for i in range (iters):
    it.append(i)
import matplotlib.pyplot as plt
costn=np.divide(costh,100000)
plt.scatter(it,costn,c='black')
