from sklearn import preprocessing
import numpy as np
#原始數據X
X=np.array([[3, -2., 2.],[2., 0., 0.],[-1, 1., 3.]])
#二值化變換器,其中閾值為1
binarizer=preprocessing.Binarizer(threshold=1)
X_binarizer=binarizer.fit_transform(X)
print(X_binarizer)