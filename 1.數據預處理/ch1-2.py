from sklearn import preprocessing
import numpy as np
X = np.array([[3, -2., 2.],[2., 0., 0.],[-1, 1., 3.]])
min_max_scaler=preprocessing.MinMaxScaler()
X_minmax=min_max_scaler.fit_transform(X)
print('縮放規範化結果如下:')
print(X_minmax)
print('輸出其縮放倍數:')
print(min_max_scaler.scale_)
print('輸出每一列的最小調整:')
print(min_max_scaler.min_)
print('輸出每一列的最小值:')
print(min_max_scaler.data_min_)