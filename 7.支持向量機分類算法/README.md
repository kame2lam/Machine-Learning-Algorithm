# 支持向量機分類算法

利用 "漏斗"使得二維的線性不可分數據變得可以分,
![](https://www.biaodianfu.com/wp-content/uploads/2020/09/svm-13.png)

(圖片引用: https://www.biaodianfu.com/svm.html)

| | |
|--|--|
|**算法**|支持向量機分類|
|**問題域**|有監督學習的分類問題|
|**輸入**|向量X:樣本的多種特徵信息值,<br>向量Y:對應的結果數值|
|**輸出**|預測模型,為線性函數|
|**用法**|輸入待預測的向量X,輸出預測結果向量Y|

使用支援向量機演算法，具體需要三步：
1）選擇核函數。
2）核函數完成高維映射並完成計算間隔所需的內積運算，求得間隔。
3）使用SMO等演算法使得間隔最大。

Kame Lam, [10.02.21 15:49]
在Scikit-Learn庫中，支援向量機演算法族都在sklearn.svm包中，當前版本一共有8個類。具體為：
* LinearSVC類：基於線性核函數的支援向量機分類演算法。
* LinearSVR類：基於線性核函數的支援向量機回歸演算法。
* SVC類：可選擇多種核函數的支援向量機分類演算法，通過“kernel”參數可以傳入“linear”選擇線性函數、傳入“polynomial”選擇多項式函數、傳入“rbf”選擇徑向基函數、傳入“sigmoid”選擇Logistics函數作為核函數，以及設置“precomputed”使用預設核值矩陣。預設以徑向基函數作為核函數。
* SVR類：可選擇多種核函數的支援向量機回歸演算法。
* NuSVC類：與SVC類非常相似，但可通過參數“nu”設置支持向量的數量。
* NuSVR類：與SVR類非常相似，但可通過參數“nu”設置支持向量的數量。
* OneClassSVM類：用支援向量機演算法解決無監督學習的異常點檢測問題。

ch7-1.py
```python
from sklearn.datasets import load_iris 
#從Scikit-Learn庫導入支援向量機演算法 
from sklearn.svm import SVC 
#載入鳶尾花資料集 
X, y = load_iris(return_X_y=True) 
#訓練模型 
clf = SVC().fit(X, y) 
#默認為徑向基rbf，可通過kernel查看 
print(clf.kernel)
print(clf.predict(X))
print('性能:',clf.score(X,y))
```

| | |
|--|--|
|**優點**|能解決非線性問題,訓練不依赖全部數據,能較好地解決小樣本分類問題,泛化能力强|
|**缺點**|對非線性問題缺乏通解,在部分情況下要找到合適的核函數並不容易,原始支持向量機只支持二分類|
|**應用**|原始支持向量機只支持二分類,但已有多種方法將支持向量拓展用於多分類問題,支持向量機作為一種熱算法，如360殺毒的WVM人工智能引擎|