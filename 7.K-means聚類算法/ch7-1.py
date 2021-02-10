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