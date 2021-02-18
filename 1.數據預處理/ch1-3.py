from sklearn import preprocessing
import numpy as np
#原始數據X
X = np.array([[3, -2., 2.], [2., 0., 0.], [-1., 1., 3.]])
#初始化數據預處器,最大絶對值縮放
max_abs_scaler = preprocessing.MaxAbsScaler()
#數據轉換並打印
X_maxabs = max_abs_scaler.fit_transform(X)
print(X_maxabs)