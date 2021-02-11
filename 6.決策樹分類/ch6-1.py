from sklearn.datasets import load_iris 
#從Scikit-Learn庫導入決策樹模型中的決策樹分類演算法 
from sklearn.tree import DecisionTreeClassifier 
#載入鳶尾花資料集 
X, y = load_iris(return_X_y=True) 
#訓練模型 
clf = DecisionTreeClassifier().fit(X, y) 
#使用模型進行分類預測 
print(clf.predict(X))
print('性能:',clf.score(X,y))