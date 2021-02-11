# KNN分類算法:用多數表決進行分類

"同類相吸"可以說是KNN分類算法的指導思想。

| | |
|--|--|
|**算法**|KNN分類|
|**問題域**|有監督學習的分類問題|
|**輸入**|向量X:樣本的多種特徵信息值,<br>向量Y:對應的類別標簽|
|**輸出**|預測模型,表示是否為正類的概率|
|**用法**|輸入待預測的向量X,輸出預測結果分類向量|

KNN分類算法的思路:
1) 找K個最近鄰。
2) 統計最近鄰的類別占比。
3) 選取占比最多的類別作為待分類樣本的類別。

度量兩點之間的直線距離,歐幾里得矩離
<!-- $d_2(x,y)=\sqrt{\sum_{i=1}^n(x_i-y_i)^2}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=d_2(x%2Cy)%3D%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En(x_i-y_i)%5E2%7D">

ch4-1.py
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
#載入鳶花數據集
X, y=load_iris(return_X_y=True)
#訓練模型
clf=KNeighborsClassifier().fit(X,y)
#使用模型進行分數預測
x_=[[5.1,3.5,1.4,0.2],[5.9,3.,5.1,1.8]]
print(clf.predict(x_))
print('性能:',clf.score(X,y))
```

| | |
|--|--|
|**優點**|理論形式簡單,容易實現,新加入數據不必整個數據進行重新訓練,可以實現在線訓練|
|**缺點**|對樣本分佈比較敏感,正負樣本不平衡時會對預測有明顯影響,數據集規模大時計算量將加大|
|**應用**|模式識別,文本分類,多分類領域,OCR光學字符識別|