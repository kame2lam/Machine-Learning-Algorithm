# 數據預處理

### 數據源
```python
#載入鳶花數據集
from sklearn.datasets import load_iris
X, y=load_iris(return_X_y=True)
```

```python
#引用團點生成器
from sklearn.datasets import make_blobs
#生成10個樣本,3個團點,數據特徵為2個,隨機狀態為0
X, y=make_blobs(n_samples=10, n_features=2, random_state=0)
```

ch1-1.py
```python
import numpy as np
from sklearn.impute import SimpleImputer
#缺失值補全策略,均值(mean),中位數(median),常數(constant),最高頻數(most_frequent)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant')
imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X = [[13,22],[5,3],[7, np.nan],[np.nan,5],[3,7]]
imp_mean.fit(X)
imp_median.fit(X)
imp_constant.fit(X)
imp_most_frequent.fit(X)
print('均值處理,結果如下:')
print(imp_mean.transform(X))
print('中位數處理,結果如下:')
print(imp_median.transform(X))
print('常數處理,結果如下:')
print(imp_constant.transform(X))
print('最高頻數處理,結果如下:')
print(imp_most_frequent.transform(X))
```

### 縮放規範化
是將數據按照比例縮放,使之落入一個較小的特定區間,如[0,1]。

<!-- $x_{i(k)}=\frac{x_{i(k)}-\min\limits_{1\leq j\leq n}\{x_{j(k)}\}}{\max\limits_{1\leq j\leq n}\{x_{j(k)}\}-\min\limits_{1\leq j\leq n}\{x_{j(k)}\}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bi(k)%7D%3D%5Cfrac%7Bx_%7Bi(k)%7D-%5Cmin%5Climits_%7B1%5Cleq%20j%5Cleq%20n%7D%5C%7Bx_%7Bj(k)%7D%5C%7D%7D%7B%5Cmax%5Climits_%7B1%5Cleq%20j%5Cleq%20n%7D%5C%7Bx_%7Bj(k)%7D%5C%7D-%5Cmin%5Climits_%7B1%5Cleq%20j%5Cleq%20n%7D%5C%7Bx_%7Bj(k)%7D%5C%7D%7D">

ch1-2.py
```python
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
```

最大絶對值縮放,其取值範圍為-1~1, 且數據分佈是以0為中心的,分布更稀疏、合理。

<!-- $x_{i(k)}=\frac{x_{i(k)}}{\big|\max\limits_{1\leq j\leq n}\{x_{j(k)}\}\big|}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bi(k)%7D%3D%5Cfrac%7Bx_%7Bi(k)%7D%7D%7B%5Cbig%7C%5Cmax%5Climits_%7B1%5Cleq%20j%5Cleq%20n%7D%5C%7Bx_%7Bj(k)%7D%5C%7D%5Cbig%7C%7D">

ch1-3.py
```python
from sklearn import preprocessing
import numpy as np
#原始數據X
X = np.array([[3, -2., 2.], [2., 0., 0.], [-1., 1., 3.]])
#初始化數據預處器,最大絶對值縮放
max_abs_scaler = preprocessing.MaxAbsScaler()
#數據轉換並打印
X_maxabs = max_abs_scaler.fit_transform(X)
print(X_maxabs)
```

數據分佈到特定的取值區間,數據變化滿足如下公式:
<!-- $x_{i(k)}=\frac{x_{i(k)} - \min\limits_{1\leq j\leq n}\{x_{j(k)}\}}{\max\limits_{1\leq j\leq n}\{x_{j(k)}\}-\min\limits_{1\leq j\leq n}\{x_{j(k)}\}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bi(k)%7D%3D%5Cfrac%7Bx_%7Bi(k)%7D%20-%20%5Cmin%5Climits_%7B1%5Cleq%20j%5Cleq%20n%7D%5C%7Bx_%7Bj(k)%7D%5C%7D%7D%7B%5Cmax%5Climits_%7B1%5Cleq%20j%5Cleq%20n%7D%5C%7Bx_%7Bj(k)%7D%5C%7D-%5Cmin%5Climits_%7B1%5Cleq%20j%5Cleq%20n%7D%5C%7Bx_%7Bj(k)%7D%5C%7D%7D">

<!-- $x\_scale_{i(k)}=(max-min)\times x_{i(k)}+min$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%5C_scale_%7Bi(k)%7D%3D(max-min)%5Ctimes%20x_%7Bi(k)%7D%2Bmin">

ch1-4.py
```python
from sklearn import preprocessing
import numpy as np
#原始數據X
X = np.array([[3, -2., 2.], [2., 0., 0.], [-1, 1., 3.]])
#初始化數據預處器,最大絶對值縮放
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-2,6))
#數據轉換並打印
X_minmax = min_max_scaler.fit_transform(X)
print(X_minmax)
```

### 非線性變換
通常需要以一些非線性的方式進行呈現,具體如下。
1.考試分數的及格線。
2.高等院校的錄取線。
3.家庭年收入劃分。

* 二值化變換

設定一個閾值,數據大於閾值時化為1;相反,數據小於閾值時化為0。公式為
<!-- $x'=\begin{cases}
1, & threshold>x,\\
0, & threshold\leq x
\end{cases}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x'%3D%5Cbegin%7Bcases%7D%0D%0A1%2C%20%26%20threshold%3Ex%2C%5C%5C%0D%0A0%2C%20%26%20threshold%5Cleq%20x%0D%0A%5Cend%7Bcases%7D">

ch1-5.py
```python
from sklearn import preprocessing
import numpy as np
#原始數據X
X=np.array([[3, -2., 2.],[2., 0., 0.],[-1, 1., 3.]])
#二值化變換器,其中閾值為1
binarizer=preprocessing.Binarizer(threshold=1)
X_binarizer=binarizer.fit_transform(X)
print(X_binarizer)
```