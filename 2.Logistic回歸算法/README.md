# Logistic回歸算法

Logistic函數
 $Logistic(t)={1\over{1+e^{-t}}} $

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)
是可導的階躍函數。

線性方程表達式,
$H(x)={1\over{1+e^{-{w^T x_i+b}}}}$

損失函數,
$L(x)=-y \log H(x)-(1-y)\log(1-H(x))$

|||
|--|--|
|**算法名稱**|Logistic回歸|
|**問題域**|有監督學習的分類問題|
|**輸入**|向量X,樣本的多種特徵信息值. <br> 向量Y,對應的類別標籤|
|**輸出**|預測模型,為是否為正類的概率|
|**用法**|輸入向量X,輸出預測結果分類向量Y|

1. LinearRegression類

損失函數 $L(x)=\min\limits_{w}||Xw-y||_2^2$