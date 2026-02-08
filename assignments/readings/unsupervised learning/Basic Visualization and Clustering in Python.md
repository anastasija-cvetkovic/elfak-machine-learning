[Dhanya Jothimani's profile](https://www.kaggle.com/dhanyajothimani) Dhanya Jothimani  · 8y ago · 229,328 views

arrow\_drop\_up463

[Copy & Edit](https://www.kaggle.com/kernels/fork-version/2148091)
669

![silver medal](https://www.kaggle.com/static/images/medals/notebooks/silverl@1x.png)

more\_vert

# Basic Visualization and Clustering in Python

## Basic Visualization and Clustering in Python

[Notebook](https://www.kaggle.com/code/dhanyajothimani/basic-visualization-and-clustering-in-python/notebook) [Input](https://www.kaggle.com/code/dhanyajothimani/basic-visualization-and-clustering-in-python/input) [Output](https://www.kaggle.com/code/dhanyajothimani/basic-visualization-and-clustering-in-python/output) [Logs](https://www.kaggle.com/code/dhanyajothimani/basic-visualization-and-clustering-in-python/log) [Comments (19)](https://www.kaggle.com/code/dhanyajothimani/basic-visualization-and-clustering-in-python/comments)

historyVersion 9 of 9chevron\_right

## Runtime

play\_arrow

25m 22s

## Input

DATASETS

![](https://storage.googleapis.com/kaggle-datasets-images/894/1634/927f32f2cc2cc40d208ae384562ad743/dataset-thumbnail.jpg)

World Happiness Report

## Tags

[Global](https://www.kaggle.com/code?tagIds=3007-Global) [Social Science](https://www.kaggle.com/code?tagIds=11200-Social+Science) [Data Visualization](https://www.kaggle.com/code?tagIds=13208-Data+Visualization) [Clustering](https://www.kaggle.com/code?tagIds=13304-Clustering)

## Language

Python

\_\_notebook\_\_

linkcode

**Basic Visualization and Clustering in Python: World Happiness Report**

This kernel shows basic visualization of data using Choropleth maps. Further, it tries to cluster the data using few clustering algorithms including K-means and Guassian Mixture Model based on several factors such as GDP per capita, life expectancy, corruption etc. We have considered 2017 data only.

In \[1\]:

linkcode

```
#Call required libraries
import time                   # To time processes
import warnings               # To suppress warnings

import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio
import matplotlib.pyplot as plt                   # For graphics
import seaborn as sns
import plotly.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering

import os                     # For os related operations
import sys                    # For data size
```

In \[2\]:

linkcode

```
wh = pd.read_csv("../input/2017.csv") #Read the dataset
wh.describe()
```

Out\[2\]:

|  | Happiness.Rank | Happiness.Score | Whisker.high | Whisker.low | Economy..GDP.per.Capita. | Family | Health..Life.Expectancy. | Freedom | Generosity | Trust..Government.Corruption. | Dystopia.Residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 155.000000 | 155.000000 | 155.000000 | 155.000000 | 155.000000 | 155.000000 | 155.000000 | 155.000000 | 155.000000 | 155.000000 | 155.000000 |
| mean | 78.000000 | 5.354019 | 5.452326 | 5.255713 | 0.984718 | 1.188898 | 0.551341 | 0.408786 | 0.246883 | 0.123120 | 1.850238 |
| std | 44.888751 | 1.131230 | 1.118542 | 1.145030 | 0.420793 | 0.287263 | 0.237073 | 0.149997 | 0.134780 | 0.101661 | 0.500028 |
| min | 1.000000 | 2.693000 | 2.864884 | 2.521116 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.377914 |
| 25% | 39.500000 | 4.505500 | 4.608172 | 4.374955 | 0.663371 | 1.042635 | 0.369866 | 0.303677 | 0.154106 | 0.057271 | 1.591291 |
| 50% | 78.000000 | 5.279000 | 5.370032 | 5.193152 | 1.064578 | 1.253918 | 0.606042 | 0.437454 | 0.231538 | 0.089848 | 1.832910 |
| 75% | 116.500000 | 6.101500 | 6.194600 | 6.006527 | 1.318027 | 1.414316 | 0.723008 | 0.516561 | 0.323762 | 0.153296 | 2.144654 |
| max | 155.000000 | 7.537000 | 7.622030 | 7.479556 | 1.870766 | 1.610574 | 0.949492 | 0.658249 | 0.838075 | 0.464308 | 3.117485 |

In \[3\]:

linkcode

```
print("Dimension of dataset: wh.shape")
wh.dtypes
```

```
Dimension of dataset: wh.shape
```

Out\[3\]:

```
Country                           object
Happiness.Rank                     int64
Happiness.Score                  float64
Whisker.high                     float64
Whisker.low                      float64
Economy..GDP.per.Capita.         float64
Family                           float64
Health..Life.Expectancy.         float64
Freedom                          float64
Generosity                       float64
Trust..Government.Corruption.    float64
Dystopia.Residual                float64
dtype: object
```

linkcode

**Basic Visualization**

_Correlation among variables_

First, we will try to understand the correlation between few variables. For this, first compute the correlation matrix among the variables and plotted as heat map.

In \[4\]:

linkcode

```
wh1 = wh[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.', 'Freedom',\
          'Generosity','Trust..Government.Corruption.','Dystopia.Residual']] #Subsetting the data
cor = wh1.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = True) #Plot the correlation as heat map
```

Out\[4\]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7f294a80bc50>
```

![](https://www.kaggleusercontent.com/kf/2148091/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..GgwZNhiemSAT0_rDvBQLwg.PThcVVi65C8TFp0j2N9aIEau18UJL1GSWKEUTfsWIQVLTIDV6oUNLzhl3WdmGfi3wToANfIHDP-5S5YTCwYb8oJVoO1nkbEiYOsHTxoq_QpWvi0kksJUhOIDtN1sBniBJ5FGLkWp-jqYvffI8F-bPxHNonhIus9OMb0MqnehzlUVcgTtRZH1YUcGx1aeZTdNuR3m-L7e_JEXIECapGYiwTcb8sHuUK2aYCd5aX30AT13O9m-20FziTl8z1tvO0o1W3WVa2NAKSFOP50NBUDk1StlqpTPSZagRDItNbHcU9SvDf03R_3otyVCCrNzwT5CfO8j8Tch_sfqfRqZi8IeS4LHmhzLJVE3F5XnkqPNxcnmbaGuHaEsDFDKJ1382zpSFZ5QxYFfj0EvoHyWqf6ZIaDi0dsNUVAE21F96gc0pILQEoNIxfxBruwuinB2_UcfUeiisZifgP85ox1EYA5J2OigUIXjbdWqkJtU8gZIicprVZih9KmSXG23ODJVb2_-AQeoObs8N7YZzlD50SpdtXJe50anhQOHkOXC-koSBL3rgjbfX-rBOJLL4KkpILFBJutNPtaR4jctjH1vIY1Log8_UPG1xfrfsgzF-Kd6JJdwQPjhOlslA-ABt4axy-rZ8rZwTGB5qGQS7A5c2K9xWjEuqoB9iJhj_91mCRPETMM.2qkZ_cXfcirICCsE9r56lw/__results___files/__results___5_1.png)

linkcode

We have obtained the heatmap of correlation among the variables. The color palette in the side represents the amount of correlation among the variables. The lighter shade represents high correlation. We can see that happiness score is highly correlated with GDP per capita, family and life expectancy. It is least correlated with generosity.

linkcode

_Visualization of Happiness Score: Using Choropleth feature_

linkcode

We will try to plot the happiness score of countries in the world map. Hovering the mouse over the country shows the name of the country as well as its happiness score.

In \[5\]:

linkcode

```
#Ref: https://plot.ly/python/choropleth-maps/
data = dict(type = 'choropleth',
           locations = wh['Country'],
           locationmode = 'country names',
           z = wh['Happiness.Score'],
           text = wh['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Happiness Index 2017',
             geo = dict(showframe = False,
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
```

linkcode

**User-defined function in Python**

Before we proceed further, we will see the basics of defining a function (or user-defined function) in Python:

1. The user-defined function starts with a keyword def to declare the function and this is followed by adding the function name.

> def plus

1. Pass the arguments in the function and is provided within parantheses () and close the statement with colon: .

> def plus(x,y):

1. Add the statements that need to be executed.

> z = sum(x,y)

1. End the function with return statement to see the output. If return statement is not provided, there will be no output.

> return (z)

So, the complete function is

> def plus(x,y):
>
> ```
>  z = sum (x,y)
>
>   return(z)
> ```

a = plus(2,5) #Calling the function to add two numbers

So, whenever we execute a = plus(2,5), it would return a = 7.

For more details, refer [https://www.datacamp.com/community/tutorials/functions-python-tutorial](https://www.datacamp.com/community/tutorials/functions-python-tutorial)

linkcode

**Clustering Of Countries**

We are considering eight parameters, namely, happiness score, GDP per capita, family, life expectancy, freedom, generosity, corruption and Dystopia residual for clustering the countries. Since the clustering is sensitive to range of data. It is advisable to scale the data before proceeding.

In \[6\]:

linkcode

```
#Scaling of data
ss = StandardScaler()
ss.fit_transform(wh1)
```

Out\[6\]:

```
array([[ 1.93599602,  1.50618765,  1.20357658, ...,  0.8569643 ,\
         1.90308437,  0.85629599],\
       [ 1.92269283,  1.18651768,  1.26503623, ...,  0.80685634,\
         2.73999784,  0.92989102],\
       [ 1.90672969,  1.1823454 ,  1.47266877, ...,  1.70201314,\
         0.30006609,  0.94796425],\
       ...,\
       [-1.77816933, -1.12910094, -0.51306362, ...,  0.79923322,\
        -0.56334657, -2.4660431 ],\
       [-2.17193469, -2.12929212, -1.95262416, ..., -0.31596505,\
        -0.38459935, -0.33549229],\
       [-2.35994869, -2.34773594, -4.15212515, ...,  0.253028  ,\
        -0.65680192,  0.43290816]])
```

linkcode

**(1) k-means clustering**

In general, k-means is the first choice for clustering because of its simplicity. Here, the user has to define the number of clusters (Post on how to decide the number of clusters would be dealt later). The clusters are formed based on the closeness to the center value of the clusters. The initial center value is chosen randomly. K-means clustering is top-down approach, in the sense, we decide the number of clusters (k) and then group the data points into k clusters.

In \[7\]:

linkcode

```
#K means Clustering
def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(wh1, 2)
kmeans = pd.DataFrame(clust_labels)
wh1.insert((wh1.shape[1]),'kmeans',kmeans)
```

In \[8\]:

linkcode

```
#Plot the clusters obtained using k means
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)
```

Out\[8\]:

```
<matplotlib.colorbar.Colorbar at 0x7f29431d2588>
```

![](https://www.kaggleusercontent.com/kf/2148091/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..GgwZNhiemSAT0_rDvBQLwg.PThcVVi65C8TFp0j2N9aIEau18UJL1GSWKEUTfsWIQVLTIDV6oUNLzhl3WdmGfi3wToANfIHDP-5S5YTCwYb8oJVoO1nkbEiYOsHTxoq_QpWvi0kksJUhOIDtN1sBniBJ5FGLkWp-jqYvffI8F-bPxHNonhIus9OMb0MqnehzlUVcgTtRZH1YUcGx1aeZTdNuR3m-L7e_JEXIECapGYiwTcb8sHuUK2aYCd5aX30AT13O9m-20FziTl8z1tvO0o1W3WVa2NAKSFOP50NBUDk1StlqpTPSZagRDItNbHcU9SvDf03R_3otyVCCrNzwT5CfO8j8Tch_sfqfRqZi8IeS4LHmhzLJVE3F5XnkqPNxcnmbaGuHaEsDFDKJ1382zpSFZ5QxYFfj0EvoHyWqf6ZIaDi0dsNUVAE21F96gc0pILQEoNIxfxBruwuinB2_UcfUeiisZifgP85ox1EYA5J2OigUIXjbdWqkJtU8gZIicprVZih9KmSXG23ODJVb2_-AQeoObs8N7YZzlD50SpdtXJe50anhQOHkOXC-koSBL3rgjbfX-rBOJLL4KkpILFBJutNPtaR4jctjH1vIY1Log8_UPG1xfrfsgzF-Kd6JJdwQPjhOlslA-ABt4axy-rZ8rZwTGB5qGQS7A5c2K9xWjEuqoB9iJhj_91mCRPETMM.2qkZ_cXfcirICCsE9r56lw/__results___files/__results___15_1.png)

linkcode

**(2) Agglomerative Clustering**

Also known as Hierarchical clustering, does not require the user to specify the number of clusters. Initially, each point is considered as a separate cluster, then it recursively clusters the points together depending upon the distance between them. The points are clustered in such a way that the distance between points within a cluster is minimum and distance between the cluster is maximum. Commonly used distance measures are Euclidean distance, Manhattan distance or Mahalanobis distance. Unlike k-means clustering, it is "bottom-up" approach.

Python Tip: Though providing the number of clusters is not necessary but Python provides an option of providing the same for easy and simple use.

In \[9\]:

linkcode

```
def doAgglomerative(X, nclust=2):
    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
    clust_labels1 = model.fit_predict(X)
    return (clust_labels1)

clust_labels1 = doAgglomerative(wh1, 2)
agglomerative = pd.DataFrame(clust_labels1)
wh1.insert((wh1.shape[1]),'agglomerative',agglomerative)
```

In \[10\]:

linkcode

```
#Plot the clusters obtained using Agglomerative clustering or Hierarchical clustering
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=agglomerative[0],s=50)
ax.set_title('Agglomerative Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)
```

Out\[10\]:

```
<matplotlib.colorbar.Colorbar at 0x7f29430f84e0>
```

![](https://www.kaggleusercontent.com/kf/2148091/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..GgwZNhiemSAT0_rDvBQLwg.PThcVVi65C8TFp0j2N9aIEau18UJL1GSWKEUTfsWIQVLTIDV6oUNLzhl3WdmGfi3wToANfIHDP-5S5YTCwYb8oJVoO1nkbEiYOsHTxoq_QpWvi0kksJUhOIDtN1sBniBJ5FGLkWp-jqYvffI8F-bPxHNonhIus9OMb0MqnehzlUVcgTtRZH1YUcGx1aeZTdNuR3m-L7e_JEXIECapGYiwTcb8sHuUK2aYCd5aX30AT13O9m-20FziTl8z1tvO0o1W3WVa2NAKSFOP50NBUDk1StlqpTPSZagRDItNbHcU9SvDf03R_3otyVCCrNzwT5CfO8j8Tch_sfqfRqZi8IeS4LHmhzLJVE3F5XnkqPNxcnmbaGuHaEsDFDKJ1382zpSFZ5QxYFfj0EvoHyWqf6ZIaDi0dsNUVAE21F96gc0pILQEoNIxfxBruwuinB2_UcfUeiisZifgP85ox1EYA5J2OigUIXjbdWqkJtU8gZIicprVZih9KmSXG23ODJVb2_-AQeoObs8N7YZzlD50SpdtXJe50anhQOHkOXC-koSBL3rgjbfX-rBOJLL4KkpILFBJutNPtaR4jctjH1vIY1Log8_UPG1xfrfsgzF-Kd6JJdwQPjhOlslA-ABt4axy-rZ8rZwTGB5qGQS7A5c2K9xWjEuqoB9iJhj_91mCRPETMM.2qkZ_cXfcirICCsE9r56lw/__results___files/__results___18_1.png)

linkcode

**(3) Affinity Propagation**

It does not require the number of cluster to be estimated and provided before starting the algorithm. It makes no assumption regarding the internal structure of the data points. For further details on clustering, refer [http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/](http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/)

In \[11\]:

linkcode

```
def doAffinity(X):
    model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')
    model.fit(X)
    clust_labels2 = model.predict(X)
    cent2 = model.cluster_centers_
    return (clust_labels2, cent2)

clust_labels2, cent2 = doAffinity(wh1)
affinity = pd.DataFrame(clust_labels2)
wh1.insert((wh1.shape[1]),'affinity',affinity)
```

In \[12\]:

linkcode

```
#Plotting the cluster obtained using Affinity algorithm
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=affinity[0],s=50)
ax.set_title('Affinity Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)
```

Out\[12\]:

```
<matplotlib.colorbar.Colorbar at 0x7f29382c0e10>
```

![](https://www.kaggleusercontent.com/kf/2148091/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..GgwZNhiemSAT0_rDvBQLwg.PThcVVi65C8TFp0j2N9aIEau18UJL1GSWKEUTfsWIQVLTIDV6oUNLzhl3WdmGfi3wToANfIHDP-5S5YTCwYb8oJVoO1nkbEiYOsHTxoq_QpWvi0kksJUhOIDtN1sBniBJ5FGLkWp-jqYvffI8F-bPxHNonhIus9OMb0MqnehzlUVcgTtRZH1YUcGx1aeZTdNuR3m-L7e_JEXIECapGYiwTcb8sHuUK2aYCd5aX30AT13O9m-20FziTl8z1tvO0o1W3WVa2NAKSFOP50NBUDk1StlqpTPSZagRDItNbHcU9SvDf03R_3otyVCCrNzwT5CfO8j8Tch_sfqfRqZi8IeS4LHmhzLJVE3F5XnkqPNxcnmbaGuHaEsDFDKJ1382zpSFZ5QxYFfj0EvoHyWqf6ZIaDi0dsNUVAE21F96gc0pILQEoNIxfxBruwuinB2_UcfUeiisZifgP85ox1EYA5J2OigUIXjbdWqkJtU8gZIicprVZih9KmSXG23ODJVb2_-AQeoObs8N7YZzlD50SpdtXJe50anhQOHkOXC-koSBL3rgjbfX-rBOJLL4KkpILFBJutNPtaR4jctjH1vIY1Log8_UPG1xfrfsgzF-Kd6JJdwQPjhOlslA-ABt4axy-rZ8rZwTGB5qGQS7A5c2K9xWjEuqoB9iJhj_91mCRPETMM.2qkZ_cXfcirICCsE9r56lw/__results___files/__results___21_1.png)

linkcode

**(4) Guassian Mixture Modelling**

It is probabilistic based clustering or kernel density estimation based clusterig. The clusters are formed based on the Gaussian distribution of the centers. For further details and pictorial description, refer [https://home.deib.polimi.it/matteucc/Clustering/tutorial\_html/mixture.html](https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/mixture.html)

In \[13\]:

linkcode

```
def doGMM(X, nclust=2):
    model = GaussianMixture(n_components=nclust,init_params='kmeans')
    model.fit(X)
    clust_labels3 = model.predict(X)
    return (clust_labels3)

clust_labels3 = doGMM(wh1,2)
gmm = pd.DataFrame(clust_labels3)
wh1.insert((wh1.shape[1]),'gmm',gmm)
```

In \[14\]:

linkcode

```
#Plotting the cluster obtained using GMM
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=gmm[0],s=50)
ax.set_title('Affinity Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)
```

Out\[14\]:

```
<matplotlib.colorbar.Colorbar at 0x7f29381f0b70>
```

![](https://www.kaggleusercontent.com/kf/2148091/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..GgwZNhiemSAT0_rDvBQLwg.PThcVVi65C8TFp0j2N9aIEau18UJL1GSWKEUTfsWIQVLTIDV6oUNLzhl3WdmGfi3wToANfIHDP-5S5YTCwYb8oJVoO1nkbEiYOsHTxoq_QpWvi0kksJUhOIDtN1sBniBJ5FGLkWp-jqYvffI8F-bPxHNonhIus9OMb0MqnehzlUVcgTtRZH1YUcGx1aeZTdNuR3m-L7e_JEXIECapGYiwTcb8sHuUK2aYCd5aX30AT13O9m-20FziTl8z1tvO0o1W3WVa2NAKSFOP50NBUDk1StlqpTPSZagRDItNbHcU9SvDf03R_3otyVCCrNzwT5CfO8j8Tch_sfqfRqZi8IeS4LHmhzLJVE3F5XnkqPNxcnmbaGuHaEsDFDKJ1382zpSFZ5QxYFfj0EvoHyWqf6ZIaDi0dsNUVAE21F96gc0pILQEoNIxfxBruwuinB2_UcfUeiisZifgP85ox1EYA5J2OigUIXjbdWqkJtU8gZIicprVZih9KmSXG23ODJVb2_-AQeoObs8N7YZzlD50SpdtXJe50anhQOHkOXC-koSBL3rgjbfX-rBOJLL4KkpILFBJutNPtaR4jctjH1vIY1Log8_UPG1xfrfsgzF-Kd6JJdwQPjhOlslA-ABt4axy-rZ8rZwTGB5qGQS7A5c2K9xWjEuqoB9iJhj_91mCRPETMM.2qkZ_cXfcirICCsE9r56lw/__results___files/__results___24_1.png)

linkcode

**Visualization of countries based on the clustering results**

_(1) k-Means algorithm_

In \[15\]:

linkcode

```
wh1.insert(0,'Country',wh.iloc[:,0])
wh1.iloc[:,[0,9,10,11,12]]
data = [dict(type='choropleth',\
             locations = wh1['Country'],\
             locationmode = 'country names',\
             z = wh1['kmeans'],\
             text = wh1['Country'],\
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Clustering of Countries based on K-Means',
              geo=dict(showframe = False,
                       projection = {'type':'Mercator'}))
map1 = go.Figure(data = data, layout=layout)
iplot(map1)
```

linkcode

_(2) Agglomerative Clustering_

In \[16\]:

linkcode

```
data = [dict(type='choropleth',\
             locations = wh1['Country'],\
             locationmode = 'country names',\
             z = wh1['agglomerative'],\
             text = wh1['Country'],\
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Grouping of Countries based on Agglomerative Clustering',
              geo=dict(showframe = False,
                       projection = {'type':'Mercator'}))
map2 = dict(data=data, layout=layout)
iplot(map2)
```

linkcode

_(3) Affinity Propagation_

In \[17\]:

linkcode

```
data = [dict(type='choropleth',\
             locations = wh1['Country'],\
             locationmode = 'country names',\
             z = wh1['affinity'],\
             text = wh1['Country'],\
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Grouping of Countries based on Affinity Clustering',
              geo=dict(showframe = False, projection = {'type':'Mercator'}))
map3 = dict(data=data, layout=layout)
iplot(map3)
```

linkcode

_(4) GMM_

In \[18\]:

linkcode

```
data = [dict(type='choropleth',\
             locations = wh1['Country'],\
             locationmode = 'country names',\
             z = wh1['gmm'],\
             text = wh1['Country'],\
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Grouping of Countries based on GMM clustering',
              geo=dict(showframe = False, projection = {'type':'Mercator'}))
map4 = dict(data=data, layout=layout)
iplot(map4)
```

linkcode

Quick visual analysis of heat map of clustering of countries shows that k-means, Agglomerative and GMM gives similar results. Affinity propagation clustering has grouped the countries into 10 clusters. Since clustering is unsupervised learning algorithm and since there is no clustering/target provided in the dataset, we are not able to analyse which algorithm performs better.

Otherwise, the best model for our data can be determined using metrics such as Normalized Mutual Information and Adjusted Rand Score.

## License

This Notebook has been released under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) open source license.

## Continue exploring

- ![](https://www.kaggle.com/static/images/kernel/viewer/input_light.svg)







Input

1 file




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/output_light.svg)







Output

0 files




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/logs_light.svg)







Logs

1522.2 second run - successful




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/comments_light.svg)
