# 線性回歸分析

預測值: \(\hat y=w^T x_i +b\)
 
損失函數 $L(x)=||\hat y -y||^2_2$

優化方法 $\min\limits_{w,b} ||\hat y-y||^2_2$

$w_新=w_舊-學習率*損失值$

L1范數 $||x||_1=\sum\limits_{i=1}^{n}|x_i|$

L2范數 $||x||_2=\sqrt{\sum\limits_{i=1}^{n}x_i^2}$

```python
#ch1-1.py
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
```

線性回歸算法的特點
| | |
|-|-|
|**優點** |線性模型形式簡單,可解釋性强,容易理解和實現 |
|**缺點**|線性模型不能表達複雜的模式, 對於非線性問題表現不佳 |
|**應用**|包括金融領域和氣象預報,適用於對能夠用線性關係進行的問題領域|