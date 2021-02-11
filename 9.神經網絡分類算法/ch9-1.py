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