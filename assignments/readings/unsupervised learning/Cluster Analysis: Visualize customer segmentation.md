[Rishi Patel's profile](https://www.kaggle.com/patelris) Rishi Patel  · 4y ago · 1,500 views

arrow\_drop\_up33

[Copy & Edit](https://www.kaggle.com/kernels/fork-version/74680975)
43

![bronze medal](https://www.kaggle.com/static/images/medals/notebooks/bronzel@1x.png)

more\_vert

# ✨Cluster Analysis: Visualize customer segmentation

## ✨Cluster Analysis: Visualize customer segmentation

[Notebook](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation/notebook) [Input](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation/input) [Output](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation/output) [Logs](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation/log) [Comments (9)](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation/comments)

historyVersion 3 of 3chevron\_right

## Runtime

play\_arrow

10m 22s · GPU P100

## Input

DATASETS

![](https://storage.googleapis.com/kaggle-datasets-images/new-version-temp-images/default-backgrounds-23.png-982687/dataset-thumbnail.png)

Telcom Churn's Dataset

## Tags

[Data Visualization](https://www.kaggle.com/code?tagIds=13208-Data+Visualization) [Clustering](https://www.kaggle.com/code?tagIds=13304-Clustering) [K-Means](https://www.kaggle.com/code?tagIds=13408-KMeans)

## Language

Python

## Table of Contents

[Customer Segmentation](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#Customer-Segmentation) [Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#Table-of-Contents) [1\. Functions](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#1.-Functions) [2\. Preprocess Data](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#2.-Preprocess-Data) [3\. EDA](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#3.-EDA) [4\. Clustering](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#4.-Clustering) [5\. Visualization](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#5.-Visualization) [6\. Evaluation](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#6.-Evaluation) [7\. What makes a cluster unique?](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#7.-What-makes-a-cluster-unique?)

\_\_notebook\_\_

linkcode

# Customer Segmentation [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#Customer-Segmentation)

![Customer segmentation](https://miro.medium.com/max/1400/0*qxHSR7XeQAYrCn6n.gif)

linkcode

## Table of Contents [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#Table-of-Contents)

1. [Functions](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#functions)
2. [Preprocess Data](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#preprocess)

2.1 [Load Data](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#load)

2.2 [NaN Values](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#nan)

2.3 [Preprocessing Steps](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#preprocessing)


1. [EDA](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#eda)
2. [Clustering](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#clustering)

4.1 [k-Means](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#kmeans)

4.2 [Normalized k-Means](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#normalizedkmeans)

4.3 [DBSCAN](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#dbscan)


1. [Visualization](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#visualization)

5.1 [PCA](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#pca)

5.2 [t-SNE](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#tsne)

5.3 [3D Animation](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#animation)


1. [Evaluation](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#evaluation)

1. [What makes a cluster unique?](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#unique)

7.1 [Variance](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#variance)

7.2 [Feature Importance](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#feature)


linkcode

## 1\. Functions [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#1.-Functions)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[1\]:

linkcode

```
# Data handling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Visualization
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

%matplotlib inline

def load_preprocess_data():
    """ Load and preprocess data
    """

    # Load data
    df = pd.read_csv("../input/telcom-churns-dataset/TelcoChurn.csv")

    # remove empty values
    df = df.loc[df.TotalCharges!=" ", :]
    df.TotalCharges = df.TotalCharges.astype(float)

    # Label data correctly
    replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',\
                    'TechSupport','StreamingTV', 'StreamingMovies', 'Partner', 'Dependents',\
                   'PhoneService', 'MultipleLines', 'PaperlessBilling', 'Churn']
    for i in replace_cols :
        df.loc[:, i]  = df.loc[:, i].replace({'No internet service' : 'No', 'No phone service':'No'})
        df.loc[:, i]  = df.loc[:, i].map({'No':0, 'Yes':1})
    df.gender = df.gender.map({"Female":0, "Male":1})

    # One-hot encoding of variables
    others_categorical = ['Contract', 'PaymentMethod', 'InternetService']
    for i in others_categorical:
        df = df.join(pd.get_dummies(df[i], prefix=i))
    df.drop(others_categorical, axis=1, inplace=True)

    # Calculate number of services
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity',\
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',\
            'StreamingMovies', 'InternetService_DSL', 'InternetService_Fiber optic',\
            'InternetService_No']
    df['nr_services'] = df.apply(lambda row: sum([row[x] for x in services[:-1]]), 1)

    return df.drop('customerID', 1)

def plot_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

def plot_tsne(tnse_data, kmeans_labels):
    df_tsne = pd.DataFrame(tsne_data).rename({0: 'x', 1: 'y'}, axis=1)
    df_tsne['z'] = kmeans_labels
    sns.scatterplot(x=df_tsne.x, y=df_tsne.y, hue=df_tsne.z, palette="Set2")
    plt.show()

def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels

    return df_matrix

def prepare_tsne(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = TSNE(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels

    return df_matrix

def plot_3d(df, name='labels'):
    iris = px.data.iris()
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                  color=name, opacity=0.5)


    fig.update_traces(marker=dict(size=3))
    fig.show()

def plot_animation(df, label_column, name):
    def update(num):
        ax.view_init(200, num)

    N=360
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_3d_df['x'], tsne_3d_df['y'], tsne_3d_df['z'], c=tsne_3d_df[label_column],
               s=6, depthshade=True, cmap='Paired')
    ax.set_zlim(-15, 25)
    ax.set_xlim(-20, 20)
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, N, blit=False, interval=50)
    ani.save('{}.gif'.format(name), writer='imagemagick')
    plt.show()
```

linkcode

## 2\. Preprocess Data [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#2.-Preprocess-Data)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

Demographic

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure

Services

- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

Customer account information

- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

Target

- Churn

linkcode

No = 0
Yes = 1

Female = 0
Male = 1

In \[2\]:

linkcode

```
df = load_preprocess_data()
```

linkcode

## 3\. EDA [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#3.-EDA)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[3\]:

linkcode

```
sns.scatterplot(df.TotalCharges, df.tenure, df.nr_services)
```

```
/opt/conda/lib/python3.7/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y, hue. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

Out\[3\]:

```
<AxesSubplot:xlabel='TotalCharges', ylabel='tenure'>
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___8_2.png)

In \[4\]:

linkcode

```
plot_corr(df)
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___9_0.png)

linkcode

## 4\. Clustering [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#4.-Clustering)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[5\]:

linkcode

```
df = df.drop(["Churn"], 1)
```

linkcode

### 4.1. k-Means [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#4.1.-k-Means)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[6\]:

linkcode

```
scores = [KMeans(n_clusters=i+2).fit(df).inertia_ for i in range(10)]
sns.lineplot(np.arange(2, 12), scores)
plt.xlabel('Number of clusters')
plt.ylabel("Inertia")
plt.title("Inertia of k-Means versus number of clusters")
```

```
/opt/conda/lib/python3.7/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

Out\[6\]:

```
Text(0.5, 1.0, 'Inertia of k-Means versus number of clusters')
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___13_2.png)

In \[7\]:

linkcode

```
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
```

Out\[7\]:

```
KMeans(n_clusters=4)
```

linkcode

### 4.2. Normalized k-Means [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#4.2.-Normalized-k-Means)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[8\]:

linkcode

```
normalized_vectors = preprocessing.normalize(df)
scores = [KMeans(n_clusters=i+2).fit(normalized_vectors).inertia_ for i in range(10)]
sns.lineplot(np.arange(2, 12), scores)
plt.xlabel('Number of clusters')
plt.ylabel("Inertia")
plt.title("Inertia of Cosine k-Means versus number of clusters")
```

```
/opt/conda/lib/python3.7/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

Out\[8\]:

```
Text(0.5, 1.0, 'Inertia of Cosine k-Means versus number of clusters')
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___16_2.png)

In \[9\]:

linkcode

```
normalized_kmeans = KMeans(n_clusters=4)
normalized_kmeans.fit(normalized_vectors)
```

Out\[9\]:

```
KMeans(n_clusters=4)
```

linkcode

### 4.3. DBSCAN [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#4.3.-DBSCAN)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[10\]:

linkcode

```
min_samples = df.shape[1]+1 #  Rule of thumb; number of dimensions D in the data set, as minPts ≥ D + 1
dbscan = DBSCAN(eps=3.5, min_samples=min_samples).fit(df)
```

linkcode

## 5\. Visualization [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#5.-Visualization)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

linkcode

### 5.1. PCA [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#5.1.-PCA)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[11\]:

linkcode

```
pca_df = prepare_pca(3, df, normalized_kmeans.labels_)
sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set2")
```

Out\[11\]:

```
<AxesSubplot:xlabel='x', ylabel='y'>
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___22_1.png)

In \[12\]:

linkcode

```
pca_df = prepare_pca(3, df, normalized_kmeans.labels_)
plot_3d(pca_df)
```

linkcode

### 5.2. t-SNE [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#5.2.-t-SNE)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[13\]:

linkcode

```
tsne_3d_df = prepare_tsne(3, df, kmeans.labels_)
```

In \[14\]:

linkcode

```
plot_3d(tsne_3d_df)
```

In \[15\]:

linkcode

```
tsne_3d_df['normalized_kmeans'] = normalized_kmeans.labels_
plot_3d(tsne_3d_df, name='normalized_kmeans')
```

In \[16\]:

linkcode

```
tsne_3d_df['dbscan'] = dbscan.labels_
plot_3d(tsne_3d_df, name='normalized_kmeans')
```

In \[17\]:

linkcode

```
plot_animation(tsne_3d_df, 'normalized_kmeans', 'normalized_kmeans')
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___29_0.png)

In \[18\]:

linkcode

```
plot_animation(tsne_3d_df, 'dbscan', 'dbscan')
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___30_0.png)

linkcode

### 5.3. 3D Animation [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#5.3.-3D-Animation)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[19\]:

linkcode

```
tsne_3d_df.dbscan = tsne_3d_df.dbscan.astype(int)
plot_animation(tsne_3d_df, 'normalized_kmeans', 'normalized_kmeans_new')
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___32_0.png)

linkcode

## 6\. Evaluation [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#6.-Evaluation)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[20\]:

linkcode

```
kmeans = KMeans(n_clusters=4).fit(df)

normalized_vectors = preprocessing.normalize(df)
normalized_kmeans = KMeans(n_clusters=4).fit(normalized_vectors)

min_samples = df.shape[1]+1 #  Rule of thumb; number of dimensions D in the data set, as minPts ≥ D + 1
dbscan = DBSCAN(eps=3.5, min_samples=min_samples).fit(df)
```

In \[21\]:

linkcode

```
print('kmeans: {}'.format(silhouette_score(df, kmeans.labels_, metric='euclidean')))
print('Cosine kmeans: {}'.format(silhouette_score(normalized_vectors, normalized_kmeans.labels_, metric='cosine')))
print('DBSCAN: {}'.format(silhouette_score(df, dbscan.labels_, metric='cosine')))
```

```
kmeans: 0.6018792629176068
Cosine kmeans: 0.8635411390498181
DBSCAN: 0.8302013261718773
```

linkcode

## 7\. What makes a cluster unique? [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#7.-What-makes-a-cluster-unique?)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

linkcode

One way to see the differences between clusters is to take the average value of each cluster and visualize it.

In \[22\]:

linkcode

```
# Setting all variables between 0 and 1 in order to better visualize the results
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df))
df_scaled.columns = df.columns
df_scaled['dbscan'] = dbscan.labels_
```

In \[23\]:

linkcode

```
# df = load_preprocess_data()
df['dbscan'] = dbscan.labels_
tidy = df_scaled.melt(id_vars='dbscan')
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='dbscan', y='value', hue='variable', data=tidy, palette='Set3')
plt.legend([''])
# plt.savefig("mess.jpg", dpi=300)
plt.savefig("dbscan_mess.jpg", dpi=300)
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___39_0.png)

linkcode

The problem with this approach is that we simply have too many variables. Not all of them are likely to be important when creating the clusters. Instead, I will select the most important columns based on the following approach:

linkcode

### 7.1. Variance [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#7.1.-Variance)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

linkcode

What I essentially do is group datapoints by cluster and take the average. Then, I calculate the standard deviation between
those values for each variable. Variables with a high standard deviation indicate that there are large differences between clusters and that the variable might be important.

In \[24\]:

linkcode

```
df_mean = df_scaled.loc[df_scaled.dbscan!=-1, :].groupby('dbscan').mean().reset_index()
```

In \[25\]:

linkcode

```
df_mean
```

Out\[25\]:

|  | dbscan | gender | SeniorCitizen | Partner | Dependents | tenure | PhoneService | MultipleLines | OnlineSecurity | OnlineBackup | ... | Contract\_One year | Contract\_Two year | PaymentMethod\_Bank transfer (automatic) | PaymentMethod\_Credit card (automatic) | PaymentMethod\_Electronic check | PaymentMethod\_Mailed check | InternetService\_DSL | InternetService\_Fiber optic | InternetService\_No | nr\_services |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.593750 | 0.018750 | 0.168750 | 0.218750 | 0.0 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | ... | 0.031250 | 0.0125 | 0.081250 | 0.068750 | 0.112500 | 0.737500 | 0.000000 | 0.0 | 1.000000 | 0.000000 |
| 1 | 1 | 0.609756 | 0.105691 | 0.097561 | 0.154472 | 0.0 | 0.967480 | 0.073171 | 0.130081 | 0.089431 | ... | 0.000000 | 0.0000 | 0.081301 | 0.073171 | 0.398374 | 0.447154 | 1.000000 | 0.0 | 0.000000 | 0.183943 |
| 2 | 2 | 0.480447 | 0.251397 | 0.195531 | 0.145251 | 0.0 | 1.000000 | 0.184358 | 0.022346 | 0.083799 | ... | 0.000000 | 0.0000 | 0.039106 | 0.027933 | 0.731844 | 0.201117 | 0.000000 | 1.0 | 0.000000 | 0.194134 |
| 3 | 3 | 0.432432 | 0.216216 | 0.108108 | 0.189189 | 0.0 | 0.162162 | 0.162162 | 0.000000 | 0.000000 | ... | 0.027027 | 0.0000 | 0.054054 | 0.000000 | 0.432432 | 0.513514 | 0.837838 | 0.0 | 0.162162 | 0.020270 |

4 rows × 28 columns

In \[26\]:

linkcode

```
# Setting all variables between 0 and 1 in order to better visualize the results
# df = load_preprocess_data().drop("Churn", 1)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df))
df_scaled.columns = df.columns
df_scaled['dbscan'] = dbscan.labels_

# Calculate variables with largest differences (by standard deviation)
# The higher the standard deviation in a variable based on average values for each cluster
# The more likely that the variable is important when creating the cluster
df_mean = df_scaled.loc[df_scaled.dbscan!=-1, :].groupby('dbscan').mean().reset_index()
results = pd.DataFrame(columns=['Variable', 'Std'])
for column in df_mean.columns[1:]:
    results.loc[len(results), :] = [column, np.std(df_mean[column])]
selected_columns = list(results.sort_values('Std', ascending=False).head(7).Variable.values) + ['dbscan']

# Plot data
tidy = df_scaled[selected_columns].melt(id_vars='dbscan')
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='dbscan', y='value', hue='variable', data=tidy, palette='Set3')
plt.legend(loc='upper right')
plt.savefig("dbscan_results.jpg", dpi=300)
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___45_0.png)

linkcode

### 7.2. Feature Importance [¶](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation\#7.2.-Feature-Importance)

[Back to Table of Contents](https://www.kaggle.com/code/patelris/cluster-analysis-visualize-customer-segmentation#table)

In \[27\]:

linkcode

```
from sklearn.ensemble import RandomForestClassifier
y = df.iloc[:,-1]
X = df.iloc[:,:-1]
clf = RandomForestClassifier(n_estimators=100).fit(X, y)
selected_columns = list(pd.DataFrame(np.array([clf.feature_importances_, X.columns]).T, columns=['Importance', 'Feature'])
           .sort_values("Importance", ascending=False)
           .head(7)
           .Feature
           .values)
```

In \[28\]:

linkcode

```
# Plot data
tidy = df_scaled[selected_columns+['dbscan']].melt(id_vars='dbscan')
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='dbscan', y='value', hue='variable', data=tidy, palette='Set3')
plt.legend(loc='upper right')
plt.savefig('randomforest.jpg', dpi=300)
```

![](https://www.kaggleusercontent.com/kf/74680975/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mQwicWjUoh6s7V0BtW4dxg._toVeMAE1101f4VleRx18vN7t4FHJ9yx6T2lBrc8LqxD3Id7PsbrFx86N7M26G7UScslgX2qldxCgTDmdioGtPwyknl5hff8LzvrERWhA3-frcz4tCvz9HkQfEzl_Kd9ag5JxhguRaB62ENQE7kBTbJAI-Lt_PBWMSatGFTIzWuZoQgWxD5p3rveaIXTdPxq1E2m5tEWkdMo8LCh4COp3csz67pdCCTZtDMTgIrow2FSWPZOhYjvrJvO_RdYfvC7hd4Jf-SIWymKjISg4YQBRhTPjSyn_4tHOuCcYm4S6-Wi7OVFm1zn22ciN4s7pHoWYg9wepFpXs8cXMqmfZo5Pc8erZuUR3N7CILfT7BCCXGMgsfkFhTYg3C54nm4Kb1QT98Xtug3fbpZ7mhHrv98zSphH5BxLxOd9rgzj5CIdTZt2ofx26A_Cu36AcwfcT3VwhCVEUL08cjF7gx8KD5ltu_AMfcEjbcKlleImFhZCzrKsfVlnyzoWVXeritXWODokg1wzUh2_9QU_E136-o7OBeWD6lgv637Elu8cOsvp8D8Dti8Sw6r9BnA0t2wqmBLOlThIg9XsCAqBAK2Zj6ol17s-vQJGtNtzqzV7AsI03wHeymrtI6kv5u4gUetnhFHOY9ZBqtq8LG6M6-8vR0g5w.NVdbA0Vt_OJq4brXTPJOMA/__results___files/__results___48_0.png)

In \[ \]:

linkcode

```

```

## License

This Notebook has been released under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) open source license.

## Continue exploring

- ![](https://www.kaggle.com/static/images/kernel/viewer/input_light.svg)







Input

1 file




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/output_light.svg)







Output

6 files




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/logs_light.svg)







Logs

621.8 second run - successful




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/comments_light.svg)







Comments

9 comments




arrow\_right\_alt