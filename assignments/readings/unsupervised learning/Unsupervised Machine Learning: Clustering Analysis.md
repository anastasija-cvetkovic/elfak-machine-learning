# Unsupervised Machine Learning: Clustering Analysis

Learn the intuition and applications of the most popular clustering algorithms.

[Victor Roman](https://towardsdatascience.com/author/rromanss23/)

Mar 6, 2019

13 min read

Share

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1P3ixlaofL31Oycf4uoUL9Q.jpeg)

## Introduction to Unsupervised Learning

Up to know, we have only explored [supervised Machine Learning algorithms](https://towardsdatascience.com/supervised-learning-basics-of-linear-regression-1cbab48d0eba) and techniques to develop models where the data had labels previously known. In other words, our data had some target variables with specific values that we used to train our models.

However, when dealing with real-world problems, most of the time, data will not come with predefined labels, so we will want to develop machine learning models that can classify correctly this data, by finding by themselves some commonality in the features, that will be used to predict the classes on new data.

### **Unsupervised Learning Analysis Process**

The overall process that we will follow when developing an unsupervised learning model can be summarized in the following chart:

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1MQjzeJweEb2Sf1o3xcpqoA.png)

Unsupervised learning main applications are:

- Segmenting datasets by some shared atributes.
- Detecting anomalies that do not fit to any group.
- Simplify datasets by aggregating variables with similar atributes.

In summary, the main goal is to study the intrinsic (and commonly hidden) structure of the data.

This techniques can be condensed in two main types of problems that unsupervised learning tries to solve. This problems are:

- Clustering
- Dimensionality Reduction

Throughout this article we will focus on clustering problems and we will cover dimensionality reduction in future articles.

## Clustering Analysis

In basic terms, the objective of clustering is to find different groups within the elements in the data. To do so, clustering algorithms find the structure in the data so that elements of the same cluster (or group) are more similar to each other than to those from different clusters.

In a visual way: Imagine that we have a dataset of movies and want to classify them. We have the following reviews of films:

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1UrTFgcUrxq5C-wOUFvxCkQ.png)

The machine learning model will be able to infere that there are two different classes without knowing anything else from the data.

These unsupervised learning algorithms have an incredible wide range of applications and are quite useful to solve real world problems such as anomaly detection, recommending systems, documents grouping, or finding customers with common interests based on their purchases.

Some of the most common clustering algorithms, and the ones that will be explored thourghout the article, are:

- K-Means
- Hierarchichal Clustering
- Density Based Scan Clustering (DBSCAN)
- Gaussian Clustering Model

## K-Means Clustering

K-Means algorithms are extremely easy to implement and very efficient computationally speaking. Those are the main reasons that explain why they are so popular. But they are not very good to identify classes when dealing with in groups that do not have a spherical distribution shape.

The K-Means algorithms aims to find and group in classes the data points that have high similarity between them. In the terms of the algorithm, this similiarity is understood as the opposite of the distance between datapoints. The closer the data points are, the more similar and more likely to belong to the same cluster they will be.

**Key Concepts**

- Squared Euclidean Distance

The most commonly used distance in K-Means is the squared Euclidean distance. An example of this distance between two points _x and y_ in _m-dimensional_ space is:

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1svzWIVVO4k0tSu14pzSuFA.png)

Here, _j_ is the _jth_ dimension (or feature column) of the sample points _x and y._

- Cluster Inertia

Cluster inertia is the name given to the Sum of Squared Errors within the clustering context, and is represented as follows:

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1jO8AEM1Ttkc46ea7bIEl0Q.png)

Where _μ(j)_ is the centroid for cluster _j,_ and _w(i,j)_ is 1 if the sample _x(i)_ is in cluster _j_ and 0 otherwise.

K-Means can be understood as an algorithm that will try to minimize the cluster inertia factor.

**Algorithm Steps**

1. First, we need to choose k, the number of clusters that we want to be finded.
2. Then, the algorithm will select randomly the the centroids of each cluster.
3. It will be assigned each datapoint to the closest centroid (using euclidean distance).
4. It will be computed the cluster inertia.
5. The new centroids will be calculated as the mean of the points that belong to the centroid of the previous step. In other words, by calculating the minimum quadratic error of the datapoints to the center of each cluster, moving the center towards that point
6. Back to step 3.

**K-Means Hyperparameters**

- Number of clusters: The number of clusters and centroids to generate.
- Maximum iterations: Of the algorithm for a single run.
- Number initial: The numbe rof times the algorithm will be run with different centroid seeds. The final result will be the best output of the number defined of consecutives runs, in terms of inertia.

**Challeges of K-Means**

- The output for any fixed training set won’t be always the same, because the initial centroids are set randomly and that will influence the whole algorithm process.
- As stated beforee, due to the nature of Euclidean distance, it is not a suitable algorithm when dealing with clusters that adopt non-spherical shapes.

**Points to be Considered When Applying K-Means**

- Features must be measured on the same scale, so it may be necessay to perform z-score standardization or max-min scaling.
- When dealing with categorical data, we will use the get dummies function.
- Exploratory Data Analysis (EDA) is very helpful to have an overview of the data and determine if K-Means is the most appropiate algorithm.
- The minibatch method is very useful when there is a large number of columns, however, it is less accurate.

**How to Choose the Right K Number**

Choosing the right number of clusters is one of the key points of the K-Means algorithm. To find this number there are some methods:

- Field knowledge
- Bussiness decision
- Elbow Method

As being aligned with the motivation and nature of Data Science, the elbow mehtod is the prefered option as it relies on an analytical method backed with data, to make a decision.

**Elbow Method**

The elbow method is used for determining the correct number of clusters in a dataset. It works by plotting the ascending values of K versus the total error obtained when using that K.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1yD_1yRUKD_0ed-eYw_I9wA.png)

The goal is to find the k that for each cluster will not rise significantly the variance

![](https://towardsdatascience.com/wp-content/uploads/2019/03/186R1OByRi6JoLq1JPAUnpQ.png)

In this case, we will choose the k=3, where the elbow is located.

**K-Means Limitations**

Although K-Means is a great clustering algorithm, it is most useful when we know beforehand the exact number of clusters and when we are dealing with spherical-shaped distributions.

The following picture show what we would obtain if we use K-means clustering in each dataset even if we knew the exact number of clusters beforehand:

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1ykyaNxEi1QhICv8gbdI8aw.png)

It is quite common to take the K-Means algorithm as a benchmark to evaluate the performance of other clustering methods.

## Hierarchichal Clustering

Hierarchichal clustering is an alternative to prototyope-based clustering algorithms. The main advantage of Hierarchichal clustering is that we do not need to specify the number of clusters, it will find it by itself. In addition, it enables the plotting of dendograms. Dendograms are visualizations of a binary hierarchichal clustering.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1GDuQNu0Ioz0cuUgwnvCPGg.png)

Observations that fuse at the bottom are similarm while those that are at the top are quite different. With dendograms, conclutions are made based on the location of the vertical axis rather than on the horizontal one.

**Kinds of Hierarchichal Clustering**

There are two approaches to this type of clustering: Aglomerative and divisive.

- Divisive: this method starts by englobing all datapoints in one single cluster. Then, it will split the cluster iteratively into smaller ones until each one of them contains only one sample.
- Agglomerative: this method starts with each sample being a different cluster and then merging them by the ones that are closer from each other until there is only one cluster.

**Single Linkage & Complete Linkage**

These are the most common algorithms used for agglomerative hierarchichal clustering.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1KJ3R_iyQmlNUxPfM5QBIow.png)

- Single Linkage

As being an agglomerative algorithm, single linkage starts by assuming that each sample point is a cluster. Then, it computes the distances between the most similar members for each pair of clusters and merge the two clusters for which the distance between the most similar members is the smallest.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1HUOYokgnLlokcYvT2C4stg.png)

- Complete Linkage

Although being similar to its brother (single linkage) its philosophy is esactly the opposite, it compares the most dissimilar datapoints of a pair of clusters to perform the merge.

**Advantages of Hierarchichal Clustering**

- The resulting hierarchichal representations can be very informative.
- Dendograms provide an interesting and informative way of visualization.
- They are specially powerful when the dataset comtains real hierarchichal relationships.

**Disadvantages of Hierarchichal Clustering**

- They are very sensitive to outliers and, in their presence, the model performance decreases significantly.
- They are very expensive, computationally speaking.

## Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

Density-Based Spatial Clustering of Applications with Noise, or DBSCAN, is another clustering algorithm specially useful to correctly identify noise in data.

**DBSCAN Assigning Criteria**

It is based on a number of points with a specified radius ε and there is a special label assigned to each datapoint. The process of assigning this label is the following:

- It is a specified number (MinPts) of neighbour points. A core point will be assigned if there is this MinPts number of points that fall in the ε radius.
- A border point will fall in the ε radius of a core point, but will have less neighbors than the MinPts number.
- Every other point will be noise points.

**DBSCAN Algorithm**

The algorithm follows the logic:

1. Identify a core point and make a group for each one, or for each connected gorup of core points (if they staisfy the criteria to be core point).
2. Identify and assign border points to their respective core points.

The following figure summarize very well this process and the commented notation.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1USv6WLj3A-9De9D7am2iZQ.png)

**DBSCAN vs K-Means Clustering**

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1x48iVUvrWtYY31WEsVLdeQ.png)

**DBDSCAN Advantages**

- We do not need to specify the number of clusters.
- There is high flexibility in the shapes and sizes that the clusters may adopt.
- It is very useful to identify and deal with noise data and outliers.

**DBSCAN Disadvantages**

- It faces difficulties when dealing with boirder points that are reachable by two clusters.
- It doesn’t find well clusters of varying densities.

## **Gaussian Mixture Models (GMM)**

Gaussian Mixture Models are probabilistic models that assume that all samples are generated from a mix of a finitite number of Gaussian distribution with unkown parameters.

It belongs to the group of soft clustering algorithms in which every data point will belong to every cluster existing in the dataset, but with different levels of membership to each cluster. This membership is assigned as the probability of belonging to a certain cluster, ranging from 0 to 1.

For example, the highlighted point will belong to clusters A and B simultaneoulsy, but with higher membership to the group A, due to its closeness to it.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1jCGgXVlHGE3cXVncW3xdtw.png)

GMM is one of the most advanced clustering methods that we will study in this series, it assumes that each cluster follows a probabilistic distribution that can be Gaussian or Normal. It is a generalization of K-Means clustering that includes information about the covariance structure of the data as well as the centers of the latent Gaussians.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/11QDCWZS0AUZ-51VKG8INiQ.png)

**GMM Distribution in One Dimension**

The GMM will search for gaussian distributions in the dataset and mixture them.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1uc63ZNYZaVcW75QOcJyAtQ.png)

**GMM in Two Dimensions**

When having multivariate distributions as the following one, the mean centre would be µ + σ, for each axis of the de dataset distribution.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1wkTgfCOdSS06ia6KJDAENw.png)

**GMM Algorithm**

It is an expectation-maximization algorithm which process could be summarize as follows:

1. Initialize K Gaussian distributions. It does this with the µ (mean) and σ (standard deviation) values. They can be taken from the dataset (naive method) or by applying K-Means.
2. Soft cluster the data: this is the ‘Expectation’ phase in which all datapoints will be assigned to every cluster with their respective level of membership.
3. Re-estimate the gaussians: this is the ‘Maximization’ phase in which the expectations are checked and they are used to calculate new parameters for the gaussians: new µ and σ.
4. Evaluate the log-likelihood of the data to check for convergence. The higher the log-likehood is, the more probable is that the mixture of the model we created is likely to fit our dataset. So, this is the function to maximize.
5. Repeat from step 2 until convergence.

**GMM Advantages**

- It is a soft-clustering method, which assign sample membersips to multiple clusters. This characteristic makes it the fastest algorithm to learn mixture models
- There is high flexibility in the number and shape of the clusters.

**GMM Disadvantages**

- It is very sensitive to the initial values which will condition greatly its performance.
- GMM may converge to a local minimum, which would be a sub-optimal solution.
- When having insufficient points per mixture, the algorithm diverges and finds solutions with infinite likelihood unless we regularize the covariances between the data points artificially.

## Clustering Validation

Clustering validation is the process of evaluating the result of a cluster objectively and quantitatively. We will do this validation by applying cluster validation indices. There are three main categories:

**External Indices**

These are scoring methods that we use if the original data was labelled, which is not the most frequent case in this kind of problems. We will match a clusering structure to information known beforehand.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1MMtZnLHmzEYmF5K7zfMbHA.png)

The most used index is the Adjusted Rand index.

- Adjusted Rand Index (ARI) € \[-1,1\]

To understand it we should first define its components:

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1n5GidEL8cG-zzhyXQNWqCQ.png)

- a: is the number of points that are in the same cluster both in C and in K
- b: is the number of points that are in the different cluster both in C and in K.
- n = is the total number of samples

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1tSJ0NX7ZHwqpsRVhQrSebA.png)

The ARI can get values ranging from -1 to 1. The higher the value, the better it matches the original data.

**Internal Validation Indices**

In unsupervised learning, we will work with unlabeled data and this is when internal indices are more useful.

One of the most common indices is the Silhouette Coefficient.

- Silhouette Coefficient:

There is a Silhouette Coefficient for each data point.

![](https://towardsdatascience.com/wp-content/uploads/2019/03/1qydd2In0-dD7jo8yXfGVuQ.png)![](https://towardsdatascience.com/wp-content/uploads/2019/03/1tN6h-Ief1VDgj0cNsDdAKg.png)

- a = average distance to other sample i in the same cluster
- b = average distance to other sample i in closest neighbouring cluster

![](https://towardsdatascience.com/wp-content/uploads/2019/03/16E-zA1MfJJNBilIftRCX5w.png)

The Silhouette Coefficient (SC) can get values from -1 to 1. The higher the value, the better the K selected is. It penalized more if we surpass the ideal K than if we fall short.

It is only suitable for certain algorithms such as K-Means and hierarchical clustering. It is not suitable to work with DBSCAN, we will use DBCV instead.

## Conclusion

We have made a first introduction to unsupervised learning and the main clustering algorithms.

In the next article we will walk through an implementation that will serve as an example to build a K-means model and will review and put in practice the concepts explained.

Stay tuned!

* * *

Written By

Victor Roman

[See all from Victor Roman](https://towardsdatascience.com/author/rromanss23/)

[Clustering](https://towardsdatascience.com/tag/clustering/), [Data Science](https://towardsdatascience.com/tag/data-science/), [K Means](https://towardsdatascience.com/tag/k-means/), [Machine Learning](https://towardsdatascience.com/tag/machine-learning/), [Unsupervised Learning](https://towardsdatascience.com/tag/unsupervised-learning/)

Share This Article

- [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Ftowardsdatascience.com%2Funsupervised-machine-learning-clustering-analysis-d40f2b34ae7e%2F&title=Unsupervised%20Machine%20Learning%3A%20Clustering%20Analysis)
- [Share on LinkedIn](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Ftowardsdatascience.com%2Funsupervised-machine-learning-clustering-analysis-d40f2b34ae7e%2F&title=Unsupervised%20Machine%20Learning%3A%20Clustering%20Analysis)
- [Share on X](https://x.com/share?url=https%3A%2F%2Ftowardsdatascience.com%2Funsupervised-machine-learning-clustering-analysis-d40f2b34ae7e%2F&text=Unsupervised%20Machine%20Learning%3A%20Clustering%20Analysis)

Towards Data Science is a community publication. Submit your insights to reach our global audience and earn through the TDS Author Payment Program.

[Write for TDS](https://towardsdatascience.com/questions-96667b06af5/)

## Related Articles

- ![](https://towardsdatascience.com/wp-content/uploads/2024/08/0c09RmbCCpfjAbSMq.png)





## [Implementing Convolutional Neural Networks in TensorFlow](https://towardsdatascience.com/implementing-convolutional-neural-networks-in-tensorflow-bc1c4f00bd34/)

[Artificial Intelligence](https://towardsdatascience.com/category/artificial-intelligence/)





Step-by-step code guide to building a Convolutional Neural Network









[Shreya Rao](https://towardsdatascience.com/author/shreya-rao/)





August 20, 2024




6 min read

- ![Photo by Krista Mangulsone on Unsplash](https://towardsdatascience.com/wp-content/uploads/2024/08/0GyVVTbgotH-DhGPH-scaled.jpg)





## [How to Forecast Hierarchical Time Series](https://towardsdatascience.com/how-to-forecast-hierarchical-time-series-75f223f79793/)

[Artificial Intelligence](https://towardsdatascience.com/category/artificial-intelligence/)





A beginner’s guide to forecast reconciliation









[Dr. Robert Kübler](https://towardsdatascience.com/author/dr-robert-kuebler/)





August 20, 2024




13 min read

- ![Photo by davisuko on Unsplash](https://towardsdatascience.com/wp-content/uploads/2024/08/1bAABgtZtAIG5YW1oEjW3pA-scaled.jpeg)





## [Hands-on Time Series Anomaly Detection using Autoencoders, with Python](https://towardsdatascience.com/hands-on-time-series-anomaly-detection-using-autoencoders-with-python-7cd893bbc122/)

[Data Science](https://towardsdatascience.com/category/data-science/)





Here’s how to use Autoencoders to detect signals with anomalies in a few lines of…









[Piero Paialunga](https://towardsdatascience.com/author/piero-paialunga/)





August 21, 2024




12 min read

- ![Image from Canva.](https://towardsdatascience.com/wp-content/uploads/2024/08/1UAA9jQVdqMXnwzYiz8Q53Q.png)





## [3 AI Use Cases (That Are Not a Chatbot)](https://towardsdatascience.com/3-ai-use-cases-that-are-not-a-chatbot-f4f328a2707a/)

[Machine Learning](https://towardsdatascience.com/category/artificial-intelligence/machine-learning/)





Feature engineering, structuring unstructured data, and lead scoring









[Shaw Talebi](https://towardsdatascience.com/author/shawhin/)





August 21, 2024




7 min read

- ## [Solving a Constrained Project Scheduling Problem with Quantum Annealing](https://towardsdatascience.com/solving-a-constrained-project-scheduling-problem-with-quantum-annealing-d0640e657a3b/)

[Data Science](https://towardsdatascience.com/category/data-science/)





Solving the resource constrained project scheduling problem (RCPSP) with D-Wave’s hybrid constrained quadratic model (CQM)









[Luis Fernando PÉREZ ARMAS, Ph.D.](https://towardsdatascience.com/author/luisfernandopa1212/)





August 20, 2024




29 min read

- ![](https://towardsdatascience.com/wp-content/uploads/2023/02/1VEUgT5T4absnTqBMOEuNig.png)





## [Back To Basics, Part Uno: Linear Regression and Cost Function](https://towardsdatascience.com/back-to-basics-part-uno-linear-regression-cost-function-and-gradient-descent-590dcb3eee46/)

[Data Science](https://towardsdatascience.com/category/data-science/)





An illustrated guide on essential machine learning concepts









[Shreya Rao](https://towardsdatascience.com/author/shreya-rao/)





February 3, 2023




6 min read

- ![](https://towardsdatascience.com/wp-content/uploads/2024/08/1kM8tfYcdaoccB1HX71YDig.png)





## [Must-Know in Statistics: The Bivariate Normal Projection Explained](https://towardsdatascience.com/must-know-in-statistics-the-bivariate-normal-projection-explained-ace7b2f70b5b/)

[Data Science](https://towardsdatascience.com/category/data-science/)





Derivation and practical examples of this powerful concept









[Luigi Battistoni](https://towardsdatascience.com/author/lu-battistoni/)
