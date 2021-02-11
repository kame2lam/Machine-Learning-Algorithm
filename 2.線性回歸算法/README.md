# 線性回歸分析

預測值: <!-- $\hat y=w^T x_i +b$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Chat%20y%3Dw%5ET%20x_i%20%2Bb">
 
損失函數 <!-- $L(x)=||\hat y -y||^2_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L(x)%3D%7C%7C%5Chat%20y%20-y%7C%7C%5E2_2">

優化方法 <!-- $\min\limits_{w,b} ||\hat y-y||^2_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cmin%5Climits_%7Bw%2Cb%7D%20%7C%7C%5Chat%20y-y%7C%7C%5E2_2">

$w_新=w_舊-學習率*損失值$

L1范數: <!-- $||x||_1=\sum_{i=1}^{n}|x_i|$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%7C%7Cx%7C%7C_1%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Cx_i%7C">

L2范數: <!-- $ ||x||_2=\sqrt{\sum_{i=1}^{n}x_i^2}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%7C%7Cx%7C%7C_2%3D%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_i%5E2%7D">

ch2-1.py
```python
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