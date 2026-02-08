[Skip to main content](https://www.datacamp.com/tutorial/k-means-clustering-python#main)

![](https://images.datacamp.com/image/upload/v1683911413/Clustering_k_means_b4a2d825ee.png)

## Introduction

In this tutorial, you will learn about k-means clustering. We'll cover:

- How the k-means clustering algorithm works
- How to visualize data to determine if it is a good candidate for clustering
- A case study of training and tuning a k-means clustering model using a real-world California housing dataset.

Note that this should not be confused with k-nearest neighbors, and readers wanting that should go to k-Nearest Neighbors (KNN) Classification with scikit-learn in Python instead.

This is useful to know as k-means clustering is a popular clustering algorithm that does a good job of grouping spherical data together into distinct groups. This is very valuable as both an analysis tool when the groupings of rows of data are unclear or as a feature-engineering step for improving supervised learning models.

We expect a basic understanding of Python and the ability to work with pandas Dataframes for this tutorial.

## Develop AI Applications

Learn to build AI applications using the OpenAI API.

[Start Upskilling For Free](https://www.datacamp.com/tracks/developing-ai-applications)

## An Overview of K-Means Clustering

Clustering models aim to group data into distinct “clusters” or groups. This can both serve as an interesting view in an analysis, or can serve as a feature in a supervised learning algorithm.

Consider a social setting where there are groups of people having discussions in different circles around a room. When you first look at the room, you just see a group of people. You could mentally start placing points in the center of each group of people and name that point as a unique identifier. You would then be able to refer to each group by a unique name to describe them. This is essentially what k-means clustering does with data.

![image7.png](https://media.datacamp.com/legacy/v1725630538/image_9c867e067e.png)

In the left-hand side of the diagram above, we can see 2 distinct sets of points that are unlabeled and colored as similar data points. Fitting a k-means model to this data (right-hand side) can reveal 2 distinct groups (shown in both distinct circles and colors).

In two dimensions, it is easy for humans to split these clusters, but with more dimensions, you need to use a model.

## The Dataset

In this tutorial, we will be using California housing data from Kaggle ( [here](https://www.kaggle.com/datasets/camnugent/california-housing-prices?resource=download)). We will use location data (latitude and longitude) as well as the median house value. We will cluster the houses by location and observe how house prices fluctuate across California. We save the dataset as a csv file called `‘housing.csv’` in our working directory and read it using `pandas`.

```python
import pandas as pd

home_data = pd.read_csv('housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
home_data.head()

Powered By

Was this AI assistant helpful? Yes No
```

![image4.png](https://media.datacamp.com/legacy/v1725630538/image_e3420cb5c3.png)

The data include 3 variables that we have selected using the `usecols` parameter:

- **longitude:** A value representing how far west a house is. Higher values represent houses that are further West.
- **latitude:** A value representing how far north a house is. Higher values represent houses that are further north.
- **median\_house\_value:** The median house price within a block measured in USD.

## k-Means Clustering Workflow

Like other Machine Learning algorithms, k-Means Clustering has a workflow (see [A Beginner's Guide to The Machine Learning Workflow](https://www.datacamp.com/blog/a-beginner-s-guide-to-the-machine-learning-workflow) for a more in depth breakdown of the Machine learning workflow).

In this tutorial, we will focus on collecting and splitting the data (in data preparation) and hyperparameter tuning, training your model, and assessing model performance (in modeling). Much of the work involved in unsupervised learning algorithms lies in the hyperparameter tuning and assessing performance to get the best results from your model.

## Visualize the Data

We start by visualizing our housing data. We look at the location data with a heatmap based on the median price in a block. We will use Seaborn to quickly create plots in this tutorial (see our [Introduction to Data Visualization with Seaborn course](https://www.datacamp.com/courses/introduction-to-data-visualization-with-seaborn) to better understand how these graphs are being created).

```python
import seaborn as sns

sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')

Powered By

Was this AI assistant helpful? Yes No
```

![image10.png](https://media.datacamp.com/legacy/v1725630538/image_4ea0d21c49.png)

We see that most of the expensive houses are on the west coast of California with different areas that have clusters of moderately priced houses. This is expected as typically waterfront properties are worth more than houses that are not on the coast.

Clusters are often easy to spot when you are only using 2 or 3 features. It becomes increasingly difficult or impossible when the

## Normalizing the Data

When working with distance-based algorithms, like k-Means Clustering, we must normalize the data. If we do not normalize the data, variables with different scaling will be weighted differently in the distance formula that is being optimized during training. For example, if we were to include price in the cluster, in addition to latitude and longitude, price would have an outsized impact on the optimizations because its scale is significantly larger and wider than the bounded location variables.

We first set up training and test splits using `train_test_split` from `sklearn`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

Powered By

Was this AI assistant helpful? Yes No
```

Next, we normalize the training and test data using the `preprocessing.normalize()` method from `sklearn`.

```python
from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

Powered By

Was this AI assistant helpful? Yes No
```

## Fitting and Evaluating the Model

For the first iteration, we will arbitrarily choose a number of clusters (referred to as k) of 3. Building and fitting models in `sklearn` is very simple. We will create an instance of `KMeans`, define the number of clusters using the `n_clusters` attribute, set `n_init`, which defines the number of iterations the algorithm will run with different centroid seeds, to “auto,” and we will set the `random_state` to 0 so we get the same result each time we run the code.  We can then fit the model to the normalized training data using the `fit()` method.

```python
from sklearn import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

Powered By

Was this AI assistant helpful? Yes No
```

Once the data are fit, we can access labels from the labels\_ attribute. Below, we visualize the data we just fit.

```python
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)

Powered By

Was this AI assistant helpful? Yes No
```

![image1.png](https://media.datacamp.com/legacy/v1725630538/image_3beb399da6.png)

We see that the data are now clearly split into 3 distinct groups (Northern California, Central California, and Southern California). We can also look at the distribution of median house prices in these 3 groups using a boxplot.

```python
sns.boxplot(x = kmeans.labels_, y = y_train['median_house_value'])

Powered By

Was this AI assistant helpful? Yes No
```

![image3.png](https://media.datacamp.com/legacy/v1725630538/image_053426eeea.png)

We clearly see that the Northern and Southern clusters have similar distributions of median house values (clusters 0 and 2) that are higher than the prices in the central cluster (cluster 1).

We can evaluate performance of the clustering algorithm using a Silhouette score which is a part of `sklearn.metrics` where a lower score represents a better fit.

```python
from sklearn.metrics import silhouette_score

silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

Powered By

Was this AI assistant helpful? Yes No
```

Since we have not looked at the strength of different numbers of clusters, we do not know how good of a fit the k = 3 model is. In the next section, we will explore different clusters and compare performance to make a decision on the best hyperparameter values for our model.

## Choosing the best number of clusters

The weakness of k-means clustering is that we don’t know how many clusters we need by just running the model. We need to test ranges of values and make a decision on the best value of k. We typically make a decision using the Elbow method to determine the optimal number of clusters where we are both not overfitting the data with too many clusters, and also not underfitting with too few.

We create the below loop to test and store different model results so that we can make a decision on the best number of clusters.

```python
K = range(2, 8)
fits = []
score = []

for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)

    # append the model to fits
    fits.append(model)

    # Append the silhouette score to scores
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

Powered By

Was this AI assistant helpful? Yes No
```

We can then first visually look at a few different values of k.

First we look at k = 2.

```python
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[0].labels_)

Powered By

Was this AI assistant helpful? Yes No
```

![image11.png](https://media.datacamp.com/legacy/v1725630538/image_79d773248c.png)

The model does an ok job of splitting the state into two halves, but probably doesn’t capture enough nuance in the California housing market.

Next, we look at k = 4.

```python
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[2].labels_)

Powered By

Was this AI assistant helpful? Yes No
```

![image13.png](https://media.datacamp.com/legacy/v1725630538/image_196829ac57.png)

We see this plot groups California into more logical clusters across the state based on how far North or South the houses are in the state. This model most likely captures more nuance in the housing market as we move across the state.

Finally, we look at k = 7.

```python
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[2].labels_)

Powered By

Was this AI assistant helpful? Yes No
```

![image2.png](https://media.datacamp.com/legacy/v1725630538/image_bf29b911c9.png)

The above graph appears to have too many clusters. We have sacrifice easy interpretation of the clusters for a “more accurate” geo-clustering result.

Typically, as we increase the value of K, we see improvements in clusters and what they represent until a certain point. We then start to see diminishing returns or even worse performance. We can visually see this to help make a decision on the value of k by using an elbow plot where the y-axis is a measure of goodness of fit and the x-axis is the value of k.

```python
sns.lineplot(x = K, y = score)

Powered By

Was this AI assistant helpful? Yes No
```

![image12.png](https://media.datacamp.com/legacy/v1725630538/image_b4d98a1e6b.png)

We typically choose the point where the improvements in performance start to flatten or get worse. We see k = 5 is probably the best we can do without overfitting.

We can also see that the clusters do a relatively good job of breaking California into distinct clusters and these clusters map relatively well to different price ranges as seen below.

```python
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[3].labels_)

Powered By

Was this AI assistant helpful? Yes No
```

![image6.png](https://media.datacamp.com/legacy/v1725630538/image_642fe596af.png)

```python
sns.boxplot(x = fits[3].labels_, y = y_train['median_house_value'])

Powered By

Was this AI assistant helpful? Yes No
```

![image8.png](https://media.datacamp.com/legacy/v1725630538/image_9d43e2c7f3.png)

## When will k-means cluster analysis fail?

K-means clustering performs best on data that are spherical. Spherical data are data that group in space in close proximity to each other either. This can be visualized in 2 or 3 dimensional space more easily. Data that aren’t spherical or should not be spherical do not work well with k-means clustering. For example, k-means clustering would not do well on the below data as we would not be able to find distinct centroids to cluster the two circles or arcs differently, despite them clearly visually being two distinct circles and arcs that should be labeled as such.

![image5.png](https://media.datacamp.com/legacy/v1725630538/image_1e9e228932.png)

[Image Source](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)

There are many other clustering algorithms that do a good job of clustering non-spherical data, covered in [Clustering in Machine Learning: 5 Essential Clustering Algorithms](https://www.datacamp.com/blog/clustering-in-machine-learning-5-essential-clustering-algorithms).

## Should you split your data into training and testing sets?

The decision to split your data depends on what your goals are for clustering. If the goal is to cluster your data as the end of your analysis, then it is not necessary. If you are using the clusters as a feature in a supervised learning model or for prediction (like we do in the [Scikit-Learn Tutorial: Baseball Analytics Pt 1](https://www.datacamp.com/tutorial/scikit-learn-tutorial-baseball-1) tutorial), then you will need to split your data before clustering to ensure you are following best practices for the supervised learning workflow.

## Take it to the Next Level

Now that we have covered the basics of k-means clustering in Python, you can check out this [Unsupervised Learning in Python](https://www.datacamp.com/courses/unsupervised-learning-in-python) course for a good introduction to k-means and other unsupervised learning algorithms. Our more advanced course, [Cluster Analysis in Python](https://www.datacamp.com/courses/cluster-analysis-in-python), gives a more in-depth look at clustering algorithms and how to build and tune them in Python. Finally, you can also check out the [An Introduction to Hierarchical Clustering in Python](https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python) tutorial as an approach which uses an alternative algorithm to create hierarchies from data.

## Earn a Top AI Certification

Demonstrate you can effectively and responsibly use AI.

[Get Certified, Get Hired](https://www.datacamp.com/certification/ai-fundamentals)

## Top Courses on Machine Learning

[See More\\
\\
Right Arrow](https://www.datacamp.com/category/machine-learning)

### [Cluster Analysis in Python](https://www.datacamp.com/courses/cluster-analysis-in-python)

BeginnerSkill Level

4 hr

57.6K learners

In this course, you will be introduced to unsupervised learning through techniques such as hierarchical and k-means clustering using the SciPy library.

[See DetailsRight Arrow](https://www.datacamp.com/courses/cluster-analysis-in-python)

### [Unsupervised Learning in Python](https://www.datacamp.com/courses/unsupervised-learning-in-python)

BeginnerSkill Level

4 hr

149.5K learners

Learn how to cluster, transform, visualize, and extract insights from unlabeled datasets using scikit-learn and scipy.

[See DetailsRight Arrow](https://www.datacamp.com/courses/unsupervised-learning-in-python)

[See More\\
\\
Right Arrow](https://www.datacamp.com/category/machine-learning)

Topics

[Machine Learning](https://www.datacamp.com/tutorial/category/machine-learning) [Data Science](https://www.datacamp.com/tutorial/category/data-science) [Python](https://www.datacamp.com/tutorial/category/python)

Learn more about Machine Learning

Course

### [Image Modeling with Keras](https://www.datacamp.com/courses/image-modeling-with-keras)

4 hr

38.9K

Learn to conduct image analysis using Keras with Python by constructing, training, and evaluating convolutional neural networks.

[See DetailsRight Arrow](https://www.datacamp.com/courses/image-modeling-with-keras) [Start Course](https://www.datacamp.com/users/sign_up?redirect=%2Fcourses%2Fimage-modeling-with-keras%2Fcontinue)

Course

### [Ensemble Methods in Python](https://www.datacamp.com/courses/ensemble-methods-in-python)

4 hr

12.2K

Learn how to build advanced and effective machine learning models in Python using ensemble techniques such as bagging, boosting, and stacking.

[See DetailsRight Arrow](https://www.datacamp.com/courses/ensemble-methods-in-python) [Start Course](https://www.datacamp.com/users/sign_up?redirect=%2Fcourses%2Fensemble-methods-in-python%2Fcontinue)

Course

### [Building Recommendation Engines in Python](https://www.datacamp.com/courses/building-recommendation-engines-in-python)

4 hr

12.5K

Learn to build recommendation engines in Python using machine learning techniques.

[See DetailsRight Arrow](https://www.datacamp.com/courses/building-recommendation-engines-in-python) [Start Course](https://www.datacamp.com/users/sign_up?redirect=%2Fcourses%2Fbuilding-recommendation-engines-in-python%2Fcontinue)

[See MoreRight Arrow](https://www.datacamp.com/category/machine-learning)

Related

![Hierarchical Clustering Python Tutorial](https://media.datacamp.com/legacy/v1674149821/Hierarchical_Clustering_Python_Tutorial_c96d4e4bb1.png?w=750)

[Tutorial\\
\\
**An Introduction to Hierarchical Clustering in Python**](https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python)

Understand the ins and outs of hierarchical clustering and its implementation in Python

[![Zoumana Keita 's photo](https://media.datacamp.com/legacy/v1658156655/zoumana_2042541b93.jpg?w=48)](https://www.datacamp.com/portfolio/keitazoumana)

Zoumana Keita

[Tutorial\\
\\
**Introduction to Machine Learning in Python**](https://www.datacamp.com/tutorial/introduction-machine-learning-python)

In this tutorial, you will be introduced to the world of Machine Learning (ML) with Python. To understand ML practically, you will be using a well-known machine learning algorithm called K-Nearest Neighbor (KNN) with Python.

[![Aditya Sharma's photo](https://media.datacamp.com/legacy/v1662144769/Aditya_Sharma_24bd7540bf.jpg?w=48)](https://www.datacamp.com/portfolio/adityasharma101993)

Aditya Sharma

[Tutorial\\
\\
**K-Means Clustering in R Tutorial**](https://www.datacamp.com/tutorial/k-means-clustering-r)

Learn what k-means is and discover why it’s one of the most used clustering algorithms in data science

![Eugenia Anello's photo](https://media.datacamp.com/legacy/v1658314239/1634300134363_d4c0d376a3.jpg?w=48)

Eugenia Anello

[Tutorial\\
\\
**Mean Shift Clustering: A Comprehensive Guide**](https://www.datacamp.com/tutorial/mean-shift-clustering)

Discover the mean shift clustering algorithm, its advantages, real-world applications, and step-by-step Python implementation. Compare it with K-means to understand key differences.

[![Vidhi Chugh's photo](https://media.datacamp.com/legacy/v1668604916/1661260473361_9c31f7294c.jpg?w=48)](https://www.datacamp.com/portfolio/vidhichugh3001)

Vidhi Chugh

[Tutorial\\
\\
**Python Machine Learning: Scikit-Learn Tutorial**](https://www.datacamp.com/tutorial/machine-learning-python)

An easy-to-follow scikit-learn tutorial that will help you get started with Python machine learning.

[![Kurtis Pykes 's photo](https://media.datacamp.com/legacy/v1658156357/Kurtis_e60df9583d.jpg?w=48)](https://www.datacamp.com/portfolio/kurtispykes)

Kurtis Pykes

![](https://media.datacamp.com/legacy/v1694029339/image_52f35c0c4f.jpg?w=750)

[code-along\\
\\
**Getting Started with Machine Learning in Python**](https://www.datacamp.com/code-along/getting-started-with-machine-learning-python)

Learn the fundamentals of supervised learning by using scikit-learn.

![George Boorman's photo](https://media.datacamp.com/legacy/v1680689573/bl52vasrrwy6y05ba5mi.jpg?w=48)

George Boorman

[See More](https://www.datacamp.com/tutorial/category/machine-learning) [See More](https://www.datacamp.com/tutorial/category/machine-learning)