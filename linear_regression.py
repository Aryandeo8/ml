import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
s=str(input("what would you like to predict:"))
df = pd.read_csv('train_data_Linear.csv')
c=df.columns.values.tolist()
x=np.array(df['label'])
y=np.array(df[s])
slope,intercept=np.polyfit(x,y,deg=1)
line=x*slope+intercept
plt.plot(line)
dft=pd.read_csv('test_data_linear.csv')
pv=np.array(np.add(np.multiply(np.array(dft['ID']),slope),intercept))
print(pv)
l=len(pv)
na=np.subtract(dft[s],pv)
s1=np.square(na)
g=0
for i in s1:
  g=g+i 
cost_function=g/l
print(cost_function)
