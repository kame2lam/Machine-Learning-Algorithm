from sklearn.datasets import load_wine
wine=load_wine()
#print('紅酒分類各個特徵名:')
#print(wine.feature_names)
#print('紅酒分類:')
#print(wine.target_names)
X, y= wine.data, wine.target
#把X,y分割為兩個測試集和訓練集,30%為測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#數據預處理,提升準確率
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
# 構造多層感知分類模型,
# solver含義為"解題者",表示的是某種快速找解的優化算法
# hidden_layer_sizes=(10,10,)代表兩層隱含層,韽層有10個神經元,
# 這裡的"多層"通常不超過七層,即淺度神經網絡,模型由sklearn提供,
from sklearn.neural_network import MLPClassifier
model= MLPClassifier(solver="lbfgs",hidden_layer_sizes=(10,10,))
#訓練模型
model.fit(X_train, y_train)
#在訓練集和測試集上進行預測
y_predict_on_train=model.predict(X_train)
y_predict_on_test=model.predict(X_test)
#模型評估
from sklearn.metrics import accuracy_score
print('訓練集的準確率為:{:.2f}%'.format(100*accuracy_score(y_train,y_predict_on_train)))
print('測試集的準確率為:{:.2f}%'.format(100*accuracy_score(y_test,y_predict_on_test)))
