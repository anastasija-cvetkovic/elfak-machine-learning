[Sitemap](https://medium.com/sitemap/sitemap.xml)

[Open in app](https://play.google.com/store/apps/details?id=com.medium.reader&referrer=utm_source%3DmobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Funsupervised-learning-and-data-clustering-eeecb78b422a&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

[Medium Logo](https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------)

[Write](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------)

[Search](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Funsupervised-learning-and-data-clustering-eeecb78b422a&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

[**TDS Archive**](https://medium.com/data-science?source=post_page---publication_nav-7f60cf5620c9-eeecb78b422a---------------------------------------)

·

Follow publication

[![TDS Archive](https://miro.medium.com/v2/resize:fill:38:38/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_sidebar-7f60cf5620c9-eeecb78b422a---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow publication

# Unsupervised Learning and Data Clustering

[![Sanatan Mishra](https://miro.medium.com/v2/resize:fill:32:32/0*yatFB3GjZ4Zkqc6k.jpg)](https://medium.com/@sanatanmishra?source=post_page---byline--eeecb78b422a---------------------------------------)

[Sanatan Mishra](https://medium.com/@sanatanmishra?source=post_page---byline--eeecb78b422a---------------------------------------)

Follow

15 min read

·

May 19, 2017

1.4K

1

[Listen](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3Deeecb78b422a&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Funsupervised-learning-and-data-clustering-eeecb78b422a&source=---header_actions--eeecb78b422a---------------------post_audio_button------------------)

Share

A task involving machine learning may not be linear, but it has a number of well known steps:

- Problem definition.

- Preparation of Data.
- Learn an underlying model.
- Improve the underlying model by quantitative and qualitative evaluations.
- Present the model.

One good way to come to terms with a new problem is to work through identifying and defining the problem in the best possible way and learn a model that captures meaningful information from the data. While problems in Pattern Recognition and Machine Learning can be of various types, they can be broadly classified into three categories:

- Supervised Learning:

The system is presented with example inputs and their desired outputs, given by a “teacher”, and the goal is to learn a general rule that maps inputs to outputs.
- Unsupervised Learning:

No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning).
- Reinforcement Learning:

A system interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent). The system is provided feedback in terms of rewards and punishments as it navigates its problem space.

Between supervised and unsupervised learning is semi-supervised learning, where the teacher gives an incomplete training signal: a training set with some (often many) of the target outputs missing. We will focus on unsupervised learning and data clustering in this blog post.

**Unsupervised Learning**

In some pattern recognition problems, the training data consists of a set of input vectors x without any corresponding target values. The goal in such unsupervised learning problems may be to discover groups of similar examples within the data, where it is called _clustering_, or to determine how the data is distributed in the space, known as _density estimation_. To put forward in simpler terms, for a n-sampled space x1 to xn, true class labels are not provided for each sample, hence known as _learning without teacher_.

**_Issues with Unsupervised Learning:_**

- Unsupervised Learning is harder as compared to Supervised Learning tasks..
- How do we know if results are meaningful since no answer labels are available?
- Let the expert look at the results (external evaluation)
- Define an objective function on clustering (internal evaluation)

**_Why Unsupervised Learning is needed despite of these issues?_**

- Annotating large datasets is very costly and hence we can label only a few examples manually. Example: Speech Recognition
- There may be cases where we don’t know how many/what classes is the data divided into. Example: Data Mining
- We may want to use clustering to gain some insight into the structure of the data before designing a classifier.

Unsupervised Learning can be further classified into two categories:

- _Parametric Unsupervised Learning_ In this case, we assume a parametric distribution of data. It assumes that sample data comes from a population that follows a probability distribution based on a fixed set of parameters. Theoretically, in a normal family of distributions, all members have the same shape and are _parameterized_ by mean and standard deviation. That means if you know the mean and standard deviation, and that the distribution is normal, you know the probability of any future observation. Parametric Unsupervised Learning involves construction of Gaussian Mixture Models and using Expectation-Maximization algorithm to predict the class of the sample in question. This case is much harder than the standard supervised learning because there are no answer labels available and hence there is no correct measure of accuracy available to check the result.
- _Non-parametric Unsupervised Learning_ In non-parameterized version of unsupervised learning, the data is grouped into clusters, where each cluster(hopefully) says something about categories and classes present in the data. This method is commonly used to model and analyze data with small sample sizes. Unlike parametric models, nonparametric models do not require the modeler to make any assumptions about the distribution of the population, and so are sometimes referred to as a distribution-free method.

**_What is Clustering?_**

Clustering can be considered the most important _unsupervised learning_ problem; so, as every other problem of this kind, it deals with finding a _structure_ in a collection of unlabeled data. A loose definition of clustering could be “the process of organizing objects into groups whose members are similar in some way”. A _cluster_ is therefore a collection of objects which are “similar” between them and are “dissimilar” to the objects belonging to other clusters.

![](https://miro.medium.com/v2/resize:fit:684/0*9ksfYh14C-ARETav.)

**_Distance-based clustering_.**

Given a set of points, with a notion of distance between points, grouping the points into some number of _clusters_, such that

- internal (within the cluster) distances should be small i.e members of clusters are close/similar to each other.
- external (intra-cluster) distances should be large i.e. members of different clusters are dissimilar.

**_The Goals of Clustering_**

The goal of clustering is to determine the internal grouping in a set of unlabeled data. But how to decide what constitutes a good clustering? It can be shown that there is no absolute “best” criterion which would be independent of the final aim of the clustering. Consequently, it is the user who should supply this criterion, in such a way that the result of the clustering will suit their needs.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/0*9O8GsapVqofVvQE1.)

In the above image, how do we know what is the best clustering solution?

To find a particular clustering solution , we need to define the similarity measures for the clusters.

**_Proximity Measures_**

For clustering, we need to define a proximity measure for two data points. Proximity here means how similar/dissimilar the samples are with respect to each other.

- Similarity measure S(xi,xk): large if xi,xk are similar
- Dissimilarity(or distance) measure D(xi,xk): small if xi,xk are similar

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/0*znkBub0hCWTrm7Wi.)

There are various similarity measures which can be used.

- Vectors: Cosine Distance

![](https://miro.medium.com/v2/resize:fit:228/0*deKvV-Rp5kfuyaLo.)

- Sets: Jaccard Distance

![](https://miro.medium.com/v2/resize:fit:427/0*riBDEDp_cfTgEKPS.)

- Points: Euclidean Distance

q=2

![](https://miro.medium.com/v2/resize:fit:371/0*k3o1FnfwXKD0dk2J.)

A “good” proximity measure is VERY application dependent. The clusters should be invariant under the transformations “natural” to the problem. Also, while clustering it is not advised to normalize data that are drawn from multiple distributions.

![](https://miro.medium.com/v2/resize:fit:549/0*wuxAuZHdjY9IY4jT.)

**Clustering Algorithms**

Clustering algorithms may be classified as listed below:

- Exclusive Clustering
- Overlapping Clustering
- Hierarchical Clustering
- Probabilistic Clustering

In the first case data are grouped in an exclusive way, so that if a certain data point belongs to a definite cluster then it could not be included in another cluster. A simple example of that is shown in the figure below, where the separation of points is achieved by a straight line on a bi-dimensional plane.

![](https://miro.medium.com/v2/resize:fit:257/0*Cy6ulWGnXRiNS-Sr.)

On the contrary, the second type, the overlapping clustering, uses fuzzy sets to cluster data, so that each point may belong to two or more clusters with different degrees of membership. In this case, data will be associated to an appropriate membership value.

A hierarchical clustering algorithm is based on the union between the two nearest clusters. The beginning condition is realized by setting every data point as a cluster. After a few iterations it reaches the final clusters wanted.

Finally, the last kind of clustering uses a completely probabilistic approach.

In this blog we will talk about four of the most used clustering algorithms:

- K-means
- Fuzzy K-means
- Hierarchical clustering
- Mixture of Gaussians

Each of these algorithms belongs to one of the clustering types listed above. While, K-means is an _exclusive clustering_ algorithm, Fuzzy K-means is an _overlapping clustering_ algorithm, Hierarchical clustering is obvious and lastly Mixture of Gaussians is a _probabilistic clustering_ algorithm. We will discuss about each clustering method in the following paragraphs.

**K-Means Clustering**

K-means is one of the simplest unsupervised learning algorithms that solves the well known clustering problem. The procedure follows a simple and easy way to classify a given data set through a certain number of clusters (assume k clusters) fixed a priori. The main idea is to define k centres, one for each cluster. These centroids should be placed in a smart way because of different location causes different result. So, the better choice is to place them as much as possible far away from each other. The next step is to take each point belonging to a given data set and associate it to the nearest centroid. When no point is pending, the first step is completed and an early groupage is done. At this point we need to re-calculate k new centroids as barycenters of the clusters resulting from the previous step. After we have these k new centroids, a new binding has to be done between the same data set points and the nearest new centroid. A loop has been generated. As a result of this loop we may notice that the k centroids change their location step by step until no more changes are done. In other words centroids do not move any more.

Finally, this algorithm aims at minimizing an _objective function_, in this case a squared error function. The objective function

![](https://miro.medium.com/v2/resize:fit:132/0*xbJHK6NM1yJQtCfI.)

where

![](https://miro.medium.com/v2/resize:fit:66/0*BkFA5O7_KF_ZKjPH.)

is a chosen distance measure between a data point xi and the cluster centre cj, is an indicator of the distance of the _n_ data points from their respective cluster centres.

## Get Sanatan Mishra’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

Subscribe

The algorithm is composed of the following steps:

- _Let X = {x1,x2,x3,……..,xn} be the set of data points and V = {v1,v2,…….,vc} be the set of centers._
- _Randomly select ‘c’ cluster centers._
- _Calculate the distance between each data point and cluster centers._
- _Assign the data point to the cluster center whose distance from the cluster center is minimum of all the cluster centers._
- _Recalculate the new cluster center using:_

![](https://miro.medium.com/v2/resize:fit:230/0*4TTvJHDjwl2WfoxQ.)

_where, ‘ci’ represents the number of data points in ith cluster._

- _Recalculate the distance between each data point and new obtained cluster centers._
- _If no data point was reassigned then stop, otherwise repeat from step 3)._

Although it can be proved that the procedure will always terminate, the k-means algorithm does not necessarily find the most optimal configuration, corresponding to the global objective function minimum. The algorithm is also significantly sensitive to the initial randomly selected cluster centres. The k-means algorithm can be run multiple times to reduce this effect.

K-means is a simple algorithm that has been adapted to many problem domains. As we are going to see, it is a good candidate for extension to work with fuzzy feature vectors.

![](https://miro.medium.com/v2/resize:fit:207/0*wDgdEnedDMPMb9Jl.)

The k-means procedure can be viewed as a greedy algorithm for partitioning the n samples into k clusters so as to minimize the sum of the squared distances to the cluster centers. It does have some weaknesses:

- The way to initialize the means was not specified. One popular way to start is to randomly choose k of the samples.
- It can happen that the set of samples closest to **m** i is empty, so that **m** i cannot be updated. This is a problem which needs to be handled during the implementation, but is generally ignored.
- The results depend on the value of k and there is no optimal way to describe a best “k”.

This last problem is particularly troublesome, since we often have no way of knowing how many clusters exist. In the example shown above, the same algorithm applied to the same data produces the following 3-means clustering. Is it better or worse than the 2-means clustering?

![](https://miro.medium.com/v2/resize:fit:217/0*qC0BiWCdXt2uu1hN.)

Unfortunately there is no general theoretical solution to find the optimal number of clusters for any given data set. A simple approach is to compare the results of multiple runs with different k classes and choose the best one according to a given criterion, but we need to be careful because increasing k results in smaller error function values by definition, but also increases the risk of overfitting.

**Fuzzy K-Means Clustering**

In fuzzy clustering, each point has a probability of belonging to each cluster, rather than completely belonging to just one cluster as it is the case in the traditional k-means. Fuzzy k-means specifically tries to deal with the problem where points are somewhat in between centers or otherwise ambiguous by replacing distance with probability, which of course could be some function of distance, such as having probability relative to the inverse of the distance. Fuzzy k-means uses a weighted centroid based on those probabilities. Processes of initialization, iteration, and termination are the same as the ones used in k-means. The resulting clusters are best analyzed as probabilistic distributions rather than a hard assignment of labels. One should realize that k-means is a special case of fuzzy k-means when the probability function used is simply 1 if the data point is closest to a centroid and 0 otherwise.

The fuzzy k-means algorithm is the following:

- **Assume** a fixed number of clusters _K._
- **Initialization:** Randomly initialize the k-means _μk_ associated with the clusters and compute the probability that each data point _Xi_ is a member of a given cluster _K_, _P(PointXiHasLabelK\|Xi,K)._
- **Iteration:** Recalculate the centroid of the cluster as the weighted centroid given the probabilities of membership of all data points _Xi_ :

![](https://miro.medium.com/v2/resize:fit:271/0*3FBkZCzjzT3_Ifte.)

- **Termination**: Iterate until convergence or until a user-specified number of iterations has been reached (the iteration may be trapped at some local maxima or minima)

For a better understanding, we may consider this simple mono-dimensional example. Given a certain data set, suppose to represent it as distributed on an axis. The figure below shows this:

![](https://miro.medium.com/v2/resize:fit:322/0*ZCzMAHi3RMrcMv4j.)

Looking at the picture, we may identify two clusters in proximity of the two data concentrations. We will refer to them using ‘A’ and ‘B’. In the first approach shown in this tutorial — the k-means algorithm — we associated each data point to a specific centroid; therefore, this membership function looked like this:

![](https://miro.medium.com/v2/resize:fit:347/0*RFrVn2efM9ETrgQ2.)

In the Fuzzy k-means approach, instead, the same given data point does not belong exclusively to a well defined cluster, but it can be placed in a middle way. In this case, the membership function follows a smoother line to indicate that every data point may belong to several clusters with different extent of membership.

![](https://miro.medium.com/v2/resize:fit:359/0*e6wMnmuKeaz3Tg7c.)

In the figure above, the data point shown as a red marked spot belongs more to the B cluster rather than the A cluster. The value 0.2 of ‘m’ indicates the degree of membership to A for such data point.

**Hierarchical Clustering Algorithms**

Given a set of N items to be clustered, and an N\*N distance (or similarity) matrix, the basic process of hierarchical clustering is this:

- Start by assigning each item to a cluster, so that if you have N items, you now have N clusters, each containing just one item. Let the distances (similarities) between the clusters the same as the distances (similarities) between the items they contain.
- Find the closest (most similar) pair of clusters and merge them into a single cluster, so that now you have one cluster less.
- Compute distances (similarities) between the new cluster and each of the old clusters.
- Repeat steps 2 and 3 until all items are clustered into a single cluster of size N.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/0*CBb_vzMzxXj0OTpg.)

**Clustering as a Mixture of Gaussians**

There’s another way to deal with clustering problems: a _model-based_ approach, which consists in using certain models for clusters and attempting to optimize the fit between the data and the model.

In practice, each cluster can be mathematically represented by a parametric distribution, like a Gaussian. The entire data set is therefore modelled by a _mixture_ of these distributions.

A mixture model with high likelihood tends to have the following traits:

- component distributions have high “peaks” (data in one cluster are tight);
- the mixture model “covers” the data well (dominant patterns in the data are captured by component distributions).

Main advantages of model-based clustering:

- well-studied statistical inference techniques available;
- flexibility in choosing the component distribution;
- obtain a density estimation for each cluster;
- a “soft” classification is available.

_Mixture of Gaussians_ The most widely used clustering method of this kind is based on learning a _mixture of Gaussians_:

![](https://miro.medium.com/v2/resize:fit:600/0*vJEEvhFCewVcN_5C.)

A mixture model is a mixture of k component distributions that collectively make a mixture distribution _f_( _x_):

![](https://miro.medium.com/v2/resize:fit:178/0*pBEyIFNoFV8v6SP8.)

The _αk_ represents the contribution of the _kth_ component in constructing _f(x)_. In practice, parametric distribution (e.g. gaussians), are often used since a lot work has been done to understand their behaviour. If you substitute each _fk_( _x_) for a gaussian you get what is known as a gaussian mixture models (GMM).

_The EM Algorithm_

Expectation-Maximization assumes that your data is composed of multiple multivariate normal distributions (note that this is a _very_ strong assumption, in particular when you fix the number of clusters!). Alternatively put, EM is an algorithm for maximizing a likelihood function when some of the variables in your model are unobserved (i.e. when you have latent variables).

You might fairly ask, if we’re just trying to maximize a function, why don’t we just use the existing machinery for maximizing a function. Well, if you try to maximize this by taking derivatives and setting them to zero, you find that in many cases the first-order conditions don’t have a solution. There’s a chicken-and-egg problem in that to solve for your model parameters you need to know the distribution of your unobserved data; but the distribution of your unobserved data is a function of your model parameters.

Expectation-Maximization tries to get around this by iteratively guessing a distribution for the unobserved data, then estimating the model parameters by maximizing something that is a lower bound on the actual likelihood function, and repeating until convergence:

The Expectation-Maximization algorithm

- Start with guess for values of your model parameters
- **E-step**: For each datapoint that has missing values, use your model equation to solve for the distribution of the missing data given your current guess of the model parameters and given the observed data (note that you are solving for a distribution for each missing value, not for the expected value). Now that we have a distribution for each missing value, we can calculate the _expectation_ of the likelihood function with respect to the unobserved variables. If our guess for the model parameter was correct, this expected likelihood will be the actual likelihood of our observed data; if the parameters were not correct, it will just be a lower bound.
- **M-step**: Now that we’ve got an expected likelihood function with no unobserved variables in it, maximize the function as you would in the fully observed case, to get a new estimate of your model parameters.
- Repeat until convergence.

**_Problems associated with clustering_**

There are a number of problems with clustering. Among them:

- dealing with large number of dimensions and large number of data items can be problematic because of time complexity;
- the effectiveness of the method depends on the definition of “distance” (for distance-based clustering). If an _obvious_ distance measure doesn’t exist we must “define” it, which is not always easy, especially in multidimensional spaces;
- the result of the clustering algorithm (that in many cases can be arbitrary itself) can be interpreted in different ways.

**_Possible Applications_**

Clustering algorithms can be applied in many fields, for instance:

- _Marketing_: finding groups of customers with similar behavior given a large database of customer data containing their properties and past buying records;
- _Biology_: classification of plants and animals given their features;
- _Insurance_: identifying groups of motor insurance policy holders with a high average claim cost; identifying frauds;
- _Earthquake studies_: clustering observed earthquake epicenters to identify dangerous zones;
- _World Wide Web_: document classification; clustering weblog data to discover groups of similar access patterns.

[Machine Learning](https://medium.com/tag/machine-learning?source=post_page-----eeecb78b422a---------------------------------------)

[Clustering](https://medium.com/tag/clustering?source=post_page-----eeecb78b422a---------------------------------------)

[Data Science](https://medium.com/tag/data-science?source=post_page-----eeecb78b422a---------------------------------------)

[Unsupervised Learning](https://medium.com/tag/unsupervised-learning?source=post_page-----eeecb78b422a---------------------------------------)

[Towards Data Science](https://medium.com/tag/towards-data-science?source=post_page-----eeecb78b422a---------------------------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:48:48/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--eeecb78b422a---------------------------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:64:64/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--eeecb78b422a---------------------------------------)

Follow

[**Published in TDS Archive**](https://medium.com/data-science?source=post_page---post_publication_info--eeecb78b422a---------------------------------------)

[828K followers](https://medium.com/data-science/followers?source=post_page---post_publication_info--eeecb78b422a---------------------------------------)

· [Last published Feb 3, 2025](https://medium.com/data-science/diy-ai-how-to-build-a-linear-regression-model-from-scratch-7b4cc0efd235?source=post_page---post_publication_info--eeecb78b422a---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow

[![Sanatan Mishra](https://miro.medium.com/v2/resize:fill:48:48/0*yatFB3GjZ4Zkqc6k.jpg)](https://medium.com/@sanatanmishra?source=post_page---post_author_info--eeecb78b422a---------------------------------------)

[![Sanatan Mishra](https://miro.medium.com/v2/resize:fill:64:64/0*yatFB3GjZ4Zkqc6k.jpg)](https://medium.com/@sanatanmishra?source=post_page---post_author_info--eeecb78b422a---------------------------------------)

Follow

[**Written by Sanatan Mishra**](https://medium.com/@sanatanmishra?source=post_page---post_author_info--eeecb78b422a---------------------------------------)

[282 followers](https://medium.com/@sanatanmishra/followers?source=post_page---post_author_info--eeecb78b422a---------------------------------------)

· [225 following](https://medium.com/@sanatanmishra/following?source=post_page---post_author_info--eeecb78b422a---------------------------------------)

Follow

## Responses (1)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

Write a response

[What are your thoughts?](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Funsupervised-learning-and-data-clustering-eeecb78b422a&source=---post_responses--eeecb78b422a---------------------respond_sidebar------------------)

Cancel

Respond

[![Hossein Moosavi](https://miro.medium.com/v2/resize:fill:32:32/0*pAjQzyeLUjgHawf8.)](https://medium.com/@hosseinmoosavi_13578?source=post_page---post_responses--eeecb78b422a----0-----------------------------------)

[Hossein Moosavi](https://medium.com/@hosseinmoosavi_13578?source=post_page---post_responses--eeecb78b422a----0-----------------------------------)

[May 9, 2018](https://medium.com/@hosseinmoosavi_13578/nice-introduction-sanatan-4325e1590fc2?source=post_page---post_responses--eeecb78b422a----0-----------------------------------)

```
Nice Introduction Sanatan! Can you give opinion about the usage of the EM Algorithm compared to NMF based clustering?

Thanks a lot
```

Reply

## More from Sanatan Mishra and TDS Archive

![Is it really an end or a new beginning?](https://miro.medium.com/v2/resize:fit:679/format:webp/319daec016b12da748373d51bd9d27c727580bba9947096fe89b8e2efb8e03a6)

[![Sanatan Mishra](https://miro.medium.com/v2/resize:fill:20:20/0*yatFB3GjZ4Zkqc6k.jpg)](https://medium.com/@sanatanmishra?source=post_page---author_recirc--eeecb78b422a----0---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

[Sanatan Mishra](https://medium.com/@sanatanmishra?source=post_page---author_recirc--eeecb78b422a----0---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

Aug 6, 2017

[A clap icon5](https://medium.com/@sanatanmishra/is-it-really-an-end-or-a-new-beginning-c7376e13a548?source=post_page---author_recirc--eeecb78b422a----0---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

![How to Implement Graph RAG Using Knowledge Graphs and Vector Databases](https://miro.medium.com/v2/resize:fit:679/format:webp/1*hrwv6zmmgogVNpQQlOIwIA.png)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:20:20/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---author_recirc--eeecb78b422a----1---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--eeecb78b422a----1---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

by

[Steve Hedden](https://medium.com/@stevehedden?source=post_page---author_recirc--eeecb78b422a----1---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

Sep 6, 2024

[A clap icon2K\\
\\
A response icon20](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759?source=post_page---author_recirc--eeecb78b422a----1---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

![Understanding LLMs from Scratch Using Middle School Math](https://miro.medium.com/v2/resize:fit:679/format:webp/1*9D2HQj6EBw0NC4c7YU0bWg.png)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:20:20/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---author_recirc--eeecb78b422a----2---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--eeecb78b422a----2---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

by

[Rohit Patel](https://medium.com/@rohit-patel?source=post_page---author_recirc--eeecb78b422a----2---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

Oct 19, 2024

[A clap icon8.2K\\
\\
A response icon103](https://medium.com/data-science/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876?source=post_page---author_recirc--eeecb78b422a----2---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

![जब कोई हद से ज्यादा चुप रहता है !](https://miro.medium.com/v2/resize:fit:679/format:webp/91b4cb73a944e1d27d5b9c057d310030c6cc236b371df86464ea194b87159845)

[![Sanatan Mishra](https://miro.medium.com/v2/resize:fill:20:20/0*yatFB3GjZ4Zkqc6k.jpg)](https://medium.com/@sanatanmishra?source=post_page---author_recirc--eeecb78b422a----3---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

[Sanatan Mishra](https://medium.com/@sanatanmishra?source=post_page---author_recirc--eeecb78b422a----3---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

Mar 5, 2017

[A clap icon6](https://medium.com/@sanatanmishra/%E0%A4%9C%E0%A4%AC-%E0%A4%95%E0%A5%8B%E0%A4%88-%E0%A4%B9%E0%A4%A6-%E0%A4%B8%E0%A5%87-%E0%A4%9C%E0%A5%8D%E0%A4%AF%E0%A4%BE%E0%A4%A6%E0%A4%BE-%E0%A4%9A%E0%A5%81%E0%A4%AA-%E0%A4%B0%E0%A4%B9%E0%A4%A4%E0%A4%BE-%E0%A4%B9%E0%A5%88-ed1dabf822d0?source=post_page---author_recirc--eeecb78b422a----3---------------------60adcf5d_f260_455b_8e70_3cf6929b037f--------------)

[See all from Sanatan Mishra](https://medium.com/@sanatanmishra?source=post_page---author_recirc--eeecb78b422a---------------------------------------)

[See all from TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--eeecb78b422a---------------------------------------)
