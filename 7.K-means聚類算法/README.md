# K-means聚類算法

| | |
|--|--|
|**算法**|K-means聚類|
|**問題域**|無監督的聚類算法|
|**輸入**|向量X:樣本的多種特徵信息值|
|**輸出**|預測模型,為是否為該類|
|**用法**|輸入數據X,輸出對應的簇編號|

K-means演算法的具體分五步：
1）隨機選取K個物件，以它們為質心。
2）計算資料集到質心的距離。
3）將物件劃歸（根據距離哪個質心最近）。
4）以本類內所有物件的均值重新計算質心，完成後進行第二步。
5）類不再變化後停止。

在Scikit-Learn機器學習庫中，最近鄰模型演算法族都在cluster類庫下，當前版本一共有10個類。具有代表性的幾個類如下。
● KMeans類：這個類就是本文介紹的K-means聚類演算法。
● MiniBatchKMeans類：這是K-means演算法的變體，使用mini-batch（小批量）來減少一次聚類所需的計算時間。mini-batch也是深度學習常使用的方法。
● DBSCAN類：使用DBSCAN聚類演算法，DBSCAN演算法的主要思想是將聚類的類視為被低密度區域分隔的高密度區域。
● MeanShift類：使用MeanShift聚類演算法，MeanShift演算法的主要方法是以任意點作為質心的起點，根據距離均值將質心不斷往高密度的地方移動，也即所謂均值漂移，當不滿足漂移條件後說明密度已經達到最高，就可以劃分成簇。
● AffinityPropagation類：使用Affinity Propagation聚類演算法，簡稱AP演算法，聚類過程是一個“不斷合併同類項”的過程，用類似於歸納法的思想方法完成聚類，這種方法被稱為“層次聚類”。
本文所介紹的K-means聚類演算法可以通過KMeans類調用，K-means演算法中的“K”，也即聚類得到的簇的個數可以通過參數“n_clusters”設置，默認為8。使用方法具體如下：
 
```python
#ch7-1.py
#導入繪圖庫 
import matplotlib.pyplot as plt 
#從Scikit-Learn庫導入聚類模型中的K-means聚類演算法 
from sklearn.cluster import KMeans 
#導入聚類資料生成工具 
from sklearn.datasets import make_blobs 

#用sklearn自帶的make_blobs方法生成聚類測試資料 
n_samples = 1500 
#該聚類資料集共1500個樣本 
X, y = make_blobs(n_samples=n_samples) 
#進行聚類，這裡n_clusters設定為3，也即聚成3個簇 
y_pred=KMeans(n_clusters=3).fit_predict(X) 

#用點狀圖顯示聚類效果 
plt.scatter(X[:, 0], X[:, 1], c=y_pred) 
plt.show() 
```
| | |
|--|--|
|**優點**|原理簡單,實現容易,運算效率高|
|**缺點**|需要人為地設置聚類的簇的個數,只適用於特徵維度為數值型的類據,隨機初始化可能影響聚類的最終效果,對孤立點非常敏感|
|**應用**|適用於特徵維度為數值型的聚類問題,如Google News的相同話題的新聞聚類|