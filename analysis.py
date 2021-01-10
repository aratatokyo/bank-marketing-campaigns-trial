#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./bank-additional-full.csv', sep=';')
mapvalue = {"y": {"yes": 1, "no": 0}}
df = df.replace(mapvalue)
df = pd.get_dummies(df)

corr = df.corr()
sns.heatmap(corr, square=True, annot=True)
plt.show()

# Know what component affect the result
# Find proper segmentation and measure sensibity

## Lasso regression

## PCR

## Boosting tree
