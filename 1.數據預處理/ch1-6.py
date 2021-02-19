from sklearn import preprocessing
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#載入鳶花數據集
iris=load_iris()
X, y=iris.data, iris.target
#將鳶花數據集分隔為訓練集和測試集
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=0)
#初始化分位數轉化器
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans=quantile_transformer.fit_transform(X_train)
X_test_trans=quantile_transformer.fit_transform(X_test)
print('被轉化訓練集的五分位數')
print(np.percentile(X_train[:,0],[0,5,50,75,100]))
print(X_train_trans)