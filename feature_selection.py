import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

# Removed columns having missing values 
df = pd.read_csv("communities.data",header=None)
df = df._get_numeric_data()
df = df.drop([0,4],axis = 1)

dfs = np.split(df,[-1],axis=1)

x = dfs[0]
y = dfs[1]
y.columns= ["Crime Rate"]

# x.boxplot()
# plt.show()
# PCA
pca = PCA(0.9)
pca.fit(x)
x = pd.DataFrame(pca.transform(x))


### TSNE ###

tsne = TSNE(perplexity=50, learning_rate=1000)
tsne_res = tsne.fit_transform((x))
print(tsne_res.shape)
sns.scatterplot(
    x=tsne_res[:,0], y=tsne_res[:,1],
    hue=y["Crime Rate"],
    # palette=sns.color_palette("hls"),
    # legend="full",
    alpha=0.8
)
plt.show()

### Scatterplot ###
sns.scatterplot(x=x[0], y=y["Crime Rate"])
plt.show()

## Boxplot ###
for col in x:
    sns.boxplot(x=x[col])
x.boxplot()
plt.show()

## Visualising Correlation ###

corr = x.corr()


mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title("Correlation Plot before PCA")

plt.show()
            