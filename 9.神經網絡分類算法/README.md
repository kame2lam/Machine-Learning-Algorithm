# 神經網絡分類算法

| | |
|--|--|
|**算法**|神經網絡分類|
|**問題域**|有監督學習的分類問題|
|**輸入**|向量X:樣本的多種特徵信息值,<br>向量Y:對應的結果數值|
|**輸出**|預測模型,為線性函數|
|**用法**|輸入待預測的向量X,輸出預測結果向量Y|

使用神經網路分類演算法，具體需要五步：
1）初始化神經網路中所有神經元激勵函數的權值。
2）輸入層接收輸入，通過正向傳播產生輸出。
3）根據輸出的預測值，結合實際值計算偏差。
4）輸出層接收偏差，通過反向傳播機制讓所有神經元更新權值。
5）第2～4步是神經網路模型一次完整的訓練過程，重複進行訓練過程直到偏差最小。

在Scikit-Learn庫中，基於神經網路這一大類的演算法模型的相關類庫都在sklearn. neural_network包中，這個包只有三種演算法API。神經網路演算法在Scikit-Learn庫中被稱為多層感知機（Multi-layer Perceptron）演算法，這裡可以簡單地認為二者只有叫法上的區別，縮寫為MLP。神經網路演算法可以完成多種任務，前面所介紹的用於解決分類問題的神經網路分類演算法對應的API為MLPClassifier，除此之外，神經網路演算法也可以用來解決回歸問題，對應的API為MLPRegressor。該包還有一種演算法，為基於Bernoulli Restricted Boltzmann Machine模型的神經網路分類演算法類BernoulliRBM。

ch9-1.py
```python 
from sklearn.datasets import load_iris 
#從Scikit-Learn庫導入神經網路模型中的神經網路分類演算法 
from sklearn.neural_network import MLPClassifier 
#載入鳶尾花資料集 
X, y = load_iris(return_X_y=True) 
#訓練模型 
clf = MLPClassifier().fit(X, y) 
#使用模型進行分類預測 
print(clf.predict(X))
print('性能:',clf.score(X,y))
```

| | |
|--|--|
|**優點**|網絡結構拓展性好,對複雜的"神秘函數"如非線性函數, 只通過簡單地調節參數也往往能有令人意外的表現|
|**缺點**|可解釋性差,調參依賴經驗,可能陷入局部最優解|
|**應用**|神經網路算法擬合能力強,應用領域很廣,如圖像處理,語音識別,自然語言處理等|

ch9-2.py
```python
from sklearn.datasets import load_wine
wine=load_wine()
#print('紅酒分類各個特徵名:')
#print(wine.feature_names)
#print('紅酒分類:')
#print(wine.target_names)
X, y= wine.data, wine.target
#把X,y分割為兩個測試集和訓練集,30%為測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#數據預處理,提升準確率
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
# 構造多層感知分類模型,
# solver含義為"解題者",表示的是某種快速找解的優化算法
# hidden_layer_sizes=(10,10,)代表兩層隱含層,韽層有10個神經元,
# 這裡的"多層"通常不超過七層,即淺度神經網絡,模型由sklearn提供,
from sklearn.neural_network import MLPClassifier
model= MLPClassifier(solver="lbfgs",hidden_layer_sizes=(10,10,))
#訓練模型
model.fit(X_train, y_train)
#在訓練集和測試集上進行預測
y_predict_on_train=model.predict(X_train)
y_predict_on_test=model.predict(X_test)
#模型評估
from sklearn.metrics import accuracy_score
print('訓練集的準確率為:{:.2f}%'.format(100*accuracy_score(y_train,y_predict_on_train)))
print('測試集的準確率為:{:.2f}%'.format(100*accuracy_score(y_test,y_predict_on_test)))
```