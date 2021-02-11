from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
#載入鳶花數據集
X, y=load_iris(return_X_y=True)
#訓練模型
clf=KNeighborsClassifier().fit(X,y)
#使用模型進行分數預測
x_=[[5.1,3.5,1.4,0.2],[5.9,3.,5.1,1.8]]
print(clf.predict(x_))
print('性能:',clf.score(X,y))