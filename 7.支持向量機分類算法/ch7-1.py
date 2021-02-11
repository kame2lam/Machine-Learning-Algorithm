from sklearn.datasets import load_iris 
#從Scikit-Learn庫導入支援向量機演算法 
from sklearn.svm import SVC 
#載入鳶尾花資料集 
X, y = load_iris(return_X_y=True) 
#訓練模型 
clf = SVC().fit(X, y) 
#默認為徑向基rbf，可通過kernel查看 
clf.kernel
print(clf.predict(X))
print('性能:',clf.score(X,y))