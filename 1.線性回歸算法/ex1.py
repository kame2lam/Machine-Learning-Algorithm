#ex1.py
import matplotlib.pyplot as plt
import numpy as np
#生成數據集
x=np.linspace(-3,3,30)
y=2*x+1
#隨機生成0到1之間的擾亂
x=x+np.random.rand(30)
y=y+np.random.rand(30)
x=[[i] for i in x]
y=[[i] for i in y]

from sklearn import linear_model
#訓練線性回歸模型
model=linear_model.LinearRegression()
model.fit(x,y)
#進行預測
x_=[[-3],[3]]
y_=model.predict(x_)

for tx1, tx2 in x_,y_:
  print("預測值 f({0})={1}".format(tx1,tx2))
#法向量model.coef_, 截距model.intercept_
print("函數 y={0}*x+{1}".format(model.coef_,model.intercept_))
plt.scatter(x,y)
plt.plot(x_,y_)
plt.show()