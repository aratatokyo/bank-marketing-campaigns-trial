import matplotlib
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn import preprocessing


df = pd.read_csv('./bank-additional-full.csv', sep=';')
mapvalue = {"y": {"yes": 1, "no": 0}}
df = df.replace(mapvalue)
df = pd.get_dummies(df)
X = df[df.columns[df.columns != 'y']]
y = df['y']

#%%
## Boosting tree
### 学習データとテストデータを4:1で分離
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


# 学習データからXGBoost用のデータを生成
dm_train = xgb.DMatrix(X_train, label=y_train)

# パラメータ
param = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'binary:logistic'
}

# XGBoostで学習
model = xgb.train(param, dm_train)

# 特徴量の重要度を表示
xgb.plot_importance(model)

# 木の表示
xgb.to_graphviz(model)

# テスト用のデータを生成
dm_test = xgb.DMatrix(X_test)

# 予測
y_pred = model.predict(dm_test)

# 精度
accuracy = sum(((y_pred > 0.5) & (y_test == 1)) | (
    (y_pred <= 0.5) & (y_test == 0))) / len(y_pred)

print(accuracy)
#%%
%matplotlib inline
## data preprocessing
sc = preprocessing.StandardScaler()
sc.fit(X)
X_norm = sc.transform(X)
## PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print('pca done.')
plt.subplot(2, 1, 1)
plt.scatter(X_pca[:,0],X_pca[:,1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.show()
#%%
## Kernel PCA
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=20.0)
X_kpca = kpca.fit_transform(X)
print('kpca done.')
plt.subplot(2, 1, 2)
plt.scatter(X_kpca[:,0],X_kpca[:,1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# %%
