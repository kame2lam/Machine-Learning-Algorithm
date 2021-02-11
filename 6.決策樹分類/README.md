# 5.決策樹分類:if-else進行選擇

信息熵:
<!-- $H(X)=-\sum_{k=1}^N p_k log_2(p_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H(X)%3D-%5Csum_%7Bk%3D1%7D%5EN%20p_k%20log_2(p_k)">

p就是概率,X是信息熵計算的集合。

CART算法是最常用的決策樹之一,采用了 基尼指數,簡法了計算。

基尼指數:
<!-- $Gini(D)=1-\sum_{k=1}^N p_k^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Gini(D)%3D1-%5Csum_%7Bk%3D1%7D%5EN%20p_k%5E2">

D是基尼指數計算的集合。

| | |
|--|--|
|**算法**|決策樹分類|
|**問題域**|有監督學習的分類問題|
|**輸入**|向量X:樣本的多種特徵信息值,<br>向量Y:對應的結果數值|
|**輸出**|預測模型,為線性函數|
|**用法**|輸入待預測的向量X,輸出預測結果向量Y|

決策樹分類思路:
1）選定純度度量指標。
2）利用純度度量指標，依次計算依據資料集中現有的各個特徵得到的純度，選取純度能達到最大的那個特徵作為該次的“條件判斷”。
3）利用該特徵作為“條件判斷”切分資料集，同時將該特徵從切分後的子集中剔除（也即不能再用該特徵切分子集了）。
4）重複第二、第三步，直到再沒有特徵，或切分後的資料集均為同一類。

在Scikit-Learn庫中，基於決策樹這一大類的演算法模型的相關類庫都在sklearn.tree包中。tree包中提供了7個類，但有3個類是用於匯出和繪製決策樹，實際的決策樹演算法只有4種，這4種又分為兩類，分別用於解決分類問題和回歸問題。
* DecisionTreeClassifier類：經典的決策樹分類演算法，其中有一個名為“criterion”的參數，給這個參數傳入字串“gini”，將使用基尼指數；傳入字串“entropy”，則使用資訊增益。默認使用的是基尼指數。餘下3個決策樹演算法都有這個參數。
* DecisionTreeRegressor類：用決策樹演算法解決反回歸問題。
* ExtraTreeClassifier類：這也是一款決策樹分類演算法，但與前面經典的決策樹分類演算法不同，該演算法在決策條件選擇環節加入了隨機性，不是從全部的特徵維度集中選取，而是首先隨機抽取n個特徵維度來構成新的集合，然後再在新集合中選取決策條件。n的值通過參數“max_features”設置，當max_features設置為1時，相當於決策條件完全通過隨機抽取得到。
* ExtraTreeRegressor類：與ExtraTreeClassifier類似，同樣在決策條件選擇環境加入隨機性，用於解決回歸問題。

ch6-1.py
```python
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
```

| | |
|--|--|
|**優點**|算法邏輯清晰,對程序尤其友好,樹形結構容易可視化,能夠比較直觀地觀察分類過程|
|**缺點**|最大也是最突出的缺點是容易過擬合,特徵維度在關聯關係時會對預測結果產生明顯影響|
|**應用**|適用於需要"決策"的領域,如商業決策,管理決策等,不過作為一種熱門算法,決策樹應用領域非常廣,如人體行為識別功能|