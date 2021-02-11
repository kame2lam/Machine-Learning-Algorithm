# Logistic回歸算法

Logistic函數
 <!-- $Logistic(t)={1\over{1+e^{-t}}} $ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Logistic(t)%3D%7B1%5Cover%7B1%2Be%5E%7B-t%7D%7D%7D">

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)

是可導的階躍函數。

線性方程表達式,
<!-- $H(x)={1\over{1+e^{-{w^T x_i+b}}}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H(x)%3D%7B1%5Cover%7B1%2Be%5E%7B-%7Bw%5ET%20x_i%2Bb%7D%7D%7D%7D">

損失函數,
<!-- $L(x)=-y \log H(x)-(1-y)\log(1-H(x))$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L(x)%3D-y%20%5Clog%20H(x)-(1-y)%5Clog(1-H(x))">

|||
|--|--|
|**算法名稱**|Logistic回歸|
|**問題域**|有監督學習的分類問題|
|**輸入**|向量X,樣本的多種特徵信息值. <br> 向量Y,對應的類別標籤|
|**輸出**|預測模型,為是否為正類的概率|
|**用法**|輸入向量X,輸出預測結果分類向量Y|

1. LinearRegression類

損失函數: <!-- $L(x)=\min\limits_{w}||Xw-y||_2^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L(x)%3D%5Cmin%5Climits_%7Bw%7D%7C%7CXw-y%7C%7C_2%5E2">

2. Ridge類

損失函數: <!-- $L(x)=\min\limits_{w}||Xw-y||_2^2+a ||w||_2^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L(x)%3D%5Cmin%5Climits_%7Bw%7D%7C%7CXw-y%7C%7C_2%5E2%2Ba%20%7C%7Cw%7C%7C_2%5E2">

3. Lasso類

損失函數: <!-- $L(x)=\min\limits_w {1\over 2n}||Xw-y||^2_2+a||w||_1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L(x)%3D%5Cmin%5Climits_w%20%7B1%5Cover%202n%7D%7C%7CXw-y%7C%7C%5E2_2%2Ba%7C%7Cw%7C%7C_1">

ch3-1.py
```python
from sklearn.linear_model import LogisticRegression
#Scikit-Learnh 的鳶尾花類數據集,是一個分類問題的數據集
from sklearn.datasets import load_iris
#載入鳶花數據集
X, y=load_iris(return_X_y=True)
#訓練模型
clf=LogisticRegression().fit(X, y)
#使用模型進行分數預測
x_=[[5.1,3.5,1.4,0.2],[5.9,3.,5.1,1.8]]
print(clf.predict(x_))
#性能評估器
print('性能:',clf.score(X,y))
```

| | |
|--|--|
|**優點**|線性模型形式簡單,可解釋性强,容易理解和實現,是計算代價較低的分類模型|
|**缺點**|分類的效果有時不好,容易欠擬合|
|**應用**|適用於二分類領域,或作為其他算法的"部件",如作為神經網絡算法的激活函數,如點擊率的變化規率|