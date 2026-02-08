[Sitemap](https://medium.com/sitemap/sitemap.xml)

[Open in app](https://play.google.com/store/apps/details?id=com.medium.reader&referrer=utm_source%3DmobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fan-introduction-to-clustering-algorithms-in-python-123438574097&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

[Medium Logo](https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------)

[Write](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------)

[Search](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fan-introduction-to-clustering-algorithms-in-python-123438574097&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

[**TDS Archive**](https://medium.com/data-science?source=post_page---publication_nav-7f60cf5620c9-123438574097---------------------------------------)

·

Follow publication

[![TDS Archive](https://miro.medium.com/v2/resize:fill:38:38/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_sidebar-7f60cf5620c9-123438574097---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow publication

# An Introduction to Clustering Algorithms in Python

[![Jake Huneycutt](https://miro.medium.com/v2/resize:fill:32:32/1*I8aZXsNi_sfrT_uOeAeRdg.jpeg)](https://medium.com/@hjhuney?source=post_page---byline--123438574097---------------------------------------)

[Jake Huneycutt](https://medium.com/@hjhuney?source=post_page---byline--123438574097---------------------------------------)

Follow

6 min read

·

May 29, 2018

1.8K

11

[Listen](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3D123438574097&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fan-introduction-to-clustering-algorithms-in-python-123438574097&source=---header_actions--123438574097---------------------post_audio_button------------------)

Share

In data science, we often think about how to use data to make predictions on new data points. This is called “supervised learning.” Sometimes, however, rather than ‘making predictions’, we instead want to categorize data into buckets. This is termed “unsupervised learning.”

To illustrate the difference, let’s say we’re at a major pizza chain and we’ve been tasked with creating a feature in the order management software that will predict delivery times for customers. In order to achieve this, we are given a dataset that has delivery times, distances traveled, day of week, time of day, staff on hand, and volume of sales for several deliveries in the past. From this data, we can make predictions on future delivery times. This is a good example of supervised learning.

Now, let’s say the pizza chain wants to send out targeted coupons to customers. It wants to segment its customers into 4 groups: large families, small families, singles, and college students. We are given prior ordering data (e.g. size of order, price, frequency, etc) and we’re tasked with putting each customer into one of the four buckets. This would be an example of “unsupervised learning” since we’re not making predictions; we’re merely categorizing the customers into groups.

Clustering is one of the most frequently utilized forms of unsupervised learning. In this article, we’ll explore two of the most common forms of clustering: k-means and hierarchical.

**Understanding the K-Means Clustering Algorithm**

Let’s look at how k-means clustering works. First, let me introduce you to my good friend, blobby; i.e. the [make\_blobs](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) function in Python’s [sci-kit learn library](http://scikit-learn.org/stable/). We’ll create four random clusters using make\_blobs to aid in our task.

```
# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt# create blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)# create np array for data points
points = data[0]# create scatter plot
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='viridis')
plt.xlim(-15,15)
plt.ylim(-15,15)
```

You can see our “blobs” below:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*vwBYTWsoS622bChMfWJOZg.jpeg)

We have four colored clusters, but there is some overlap with the two clusters on top, as well as the two clusters on the bottom. The first step in k-means clustering is to select random centroids. Since our k=4 in this instance, we’ll need 4 random centroids. Here is how it looked in my implementation from scratch.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*8JwQLh2MbojLnfigXLm1wQ.jpeg)

Next, we take each point and find the nearest centroid. There are different ways to measure distance, but I used [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), which can be measured using [np.linalg.norm](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html) in Python.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*gNvJrDX1I39_TBeanTn5tQ.jpeg)

Now that we have 4 clusters, we find the new centroids of the clusters.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*bOT_8MU8imj9eGvtC9hD9A.jpeg)

Then we match each point to the closest centroid again, repeating the process, until we can improve the clusters no more. In this case, when the process finished, I ended up with the result below.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*cx_zK9sDqYi7amU9-hdnpw.jpeg)

Note that these clusters are a bit different than my original clusters. This is the result of the random initialization trap. Essentially, our starting centroids can dictate the location of our clusters in k-mean clustering.

This isn’t the result we wanted, but one way to combat this is with the [k-means ++ algorithm,](http://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) which provides better initial seeding in order to find the best clusters. Fortunately, this is automatically done in k-means implementation we’ll be using in Python.

**Implementing K-Means Clustering in Python**

## Get Jake Huneycutt’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

Subscribe

To run k-means in Python, we’ll need to import [KMeans from sci-kit learn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

```
# import KMeans
from sklearn.cluster import KMeans
```

Note that in the documentation, k-means ++ is the default, so we don’t need to make any changes in order to run this improved methodology. Now, let’s run k-means on our blobs (which were put into a numpy array called ‘points’).

```
# create kmeans object
kmeans = KMeans(n_clusters=4)# fit kmeans object to data
kmeans.fit(points)# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)# save new clusters for chart
y_km = kmeans.fit_predict(points)
```

Now, we can see the results by running the following code in matplotlib.

```
plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=100, c='red')
plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=100, c='black')
plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=100, c='blue')
plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=100, c='cyan')
```

And voila! We have our 4 clusters. Note that the k-means++ algorithm did a better job than the plain ole’ k-means I ran in the example, as it nearly perfectly captured the boundaries of the initial clusters we created.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*Vhvxj6d09cumLM3rk933yg.jpeg)

K-means is the most frequently used form of clustering due to its speed and simplicity. Another very common clustering method is hierarchical clustering.

**Implementing Agglomerative Hierarchical Clustering**

Agglomerative hierarchical clustering differs from k-means in a key way. Rather than choosing a number of clusters and starting out with random centroids, we instead begin with every point in our dataset as a “cluster.” Then we find the two closest points and combine them into a cluster. Then, we find the next closest points, and those become a cluster. We repeat the process until we only have one big giant cluster.

Along the way, we create what’s called a dendrogram. This is our “history.” You can see the dendrogram for our data points below to get a sense of what’s happening.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*p3Z5TOp0egX3ItAXw0JZlQ.jpeg)

The dendrogram plots out each cluster and the distance. We can use the dendrogram to find the clusters for any number we chose. In the dendrogram above, it’s easy to see the starting points for the first cluster (blue), the second cluster (red), and the third cluster (green). Only the first 3 are color-coded here, but if you look over at the red side of the dendrogram, you can spot the starting point for the 4th cluster as well. The dendrogram runs all the way until every point is its own individual cluster.

Let’s see how agglomerative hierarchical clustering works in Python. First, let’s import the necessary libraries from [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) and [sklearn.clustering](http://scikit-learn.org/stable/modules/clustering.html).

```
# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
```

Now, let’s create our dendrogram (which I’ve already shown you above), determine how many clusters we want, and save the data points from those clusters to chart them out.

```
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')# save clusters for chart
y_hc = hc.fit_predict(points)
```

Now, we’ll do as we did with the k-means algorithm and see our clusters using matplotlib.

```
plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=100, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=100, c='black')
plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='blue')
plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='cyan')
```

Here are the results:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*I7Ze7FaLE_XkrHw9hnDTzQ.jpeg)

In this instance, the results between k-means and hierarchical clustering were pretty similar. This is not always the case, however. In general, the advantage of agglomerative hierarchical clustering is that it tends to produce more accurate results. The downside is that hierarchical clustering is more difficult to implement and more time/resource consuming than k-means.

**Further Reading**

If you want to know more about clustering, I highly recommend George Seif’s article, “ [The 5 Clustering Algorithms Data Scientists Need to Know](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68).”

**Additional Resources**

1. G. James, D. Witten, et. al. _Introduction to Statistical Learning_, Chapter 10: Unsupervised Learning, [Link](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf) (PDF)
2. Andrea Trevino, _Introduction to K-Means Clustering_, [Link](https://www.datascience.com/blog/k-means-clustering)
3. Kirill Eremenko, _Machine Learning A-Z_ (Udemy course), [Link](https://www.udemy.com/machinelearning/learn/v4/overview)

[Machine Learning](https://medium.com/tag/machine-learning?source=post_page-----123438574097---------------------------------------)

[Data Science](https://medium.com/tag/data-science?source=post_page-----123438574097---------------------------------------)

[Python](https://medium.com/tag/python?source=post_page-----123438574097---------------------------------------)

[Clustering](https://medium.com/tag/clustering?source=post_page-----123438574097---------------------------------------)

[Unsupervised Learning](https://medium.com/tag/unsupervised-learning?source=post_page-----123438574097---------------------------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:48:48/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--123438574097---------------------------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:64:64/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--123438574097---------------------------------------)

Follow

[**Published in TDS Archive**](https://medium.com/data-science?source=post_page---post_publication_info--123438574097---------------------------------------)

[828K followers](https://medium.com/data-science/followers?source=post_page---post_publication_info--123438574097---------------------------------------)

· [Last published Feb 3, 2025](https://medium.com/data-science/diy-ai-how-to-build-a-linear-regression-model-from-scratch-7b4cc0efd235?source=post_page---post_publication_info--123438574097---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow

[![Jake Huneycutt](https://miro.medium.com/v2/resize:fill:48:48/1*I8aZXsNi_sfrT_uOeAeRdg.jpeg)](https://medium.com/@hjhuney?source=post_page---post_author_info--123438574097---------------------------------------)

[![Jake Huneycutt](https://miro.medium.com/v2/resize:fill:64:64/1*I8aZXsNi_sfrT_uOeAeRdg.jpeg)](https://medium.com/@hjhuney?source=post_page---post_author_info--123438574097---------------------------------------)

Follow

[**Written by Jake Huneycutt**](https://medium.com/@hjhuney?source=post_page---post_author_info--123438574097---------------------------------------)

[437 followers](https://medium.com/@hjhuney/followers?source=post_page---post_author_info--123438574097---------------------------------------)

· [41 following](https://medium.com/@hjhuney/following?source=post_page---post_author_info--123438574097---------------------------------------)

Machine Learning @ Lambda School, Former Portfolio Manager

Follow

## Responses (11)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

Write a response

[What are your thoughts?](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fan-introduction-to-clustering-algorithms-in-python-123438574097&source=---post_responses--123438574097---------------------respond_sidebar------------------)

Cancel

Respond

[![Yeh Young](https://miro.medium.com/v2/resize:fill:32:32/0*YkYQcPiy39qGLq5U)](https://medium.com/@andyang94?source=post_page---post_responses--123438574097----0-----------------------------------)

[Yeh Young](https://medium.com/@andyang94?source=post_page---post_responses--123438574097----0-----------------------------------)

[Oct 1, 2018](https://medium.com/@andyang94/sounds-like-a-classification-problem-here-a-problem-of-classifying-people-into-different-groups-fcb28702e168?source=post_page---post_responses--123438574097----0-----------------------------------)

It wants to segment its customers into 4 groups: large families, small families, singles, and college students. We are given prior ordering data (e.g. size of order, price, frequency, e...

```
Sounds like a classification problem here, a problem of classifying people into different groups. And classification is a supervised learning method right? A better way of explaining should state without specific groups, then after clustering, we figure out those groups looks like large families, small families…
```

5

Reply

[![Joseph Kay](https://miro.medium.com/v2/resize:fill:32:32/1*BLEXQKoL_cEYQUpbMUBIKg.jpeg)](https://medium.com/@joseph.kay?source=post_page---post_responses--123438574097----1-----------------------------------)

[Joseph Kay](https://medium.com/@joseph.kay?source=post_page---post_responses--123438574097----1-----------------------------------)

[Sep 24, 2018](https://medium.com/@joseph.kay/do-you-have-any-thoughts-on-using-manhattan-average-affinity-instead-of-euclidean-ward-affinity-in-cfee12e30772?source=post_page---post_responses--123438574097----1-----------------------------------)

```
Do you have any thoughts on using manhattan/average affinity instead of euclidean/ward affinity in agglomerative hierarchical clustering?

In the context of segmenting customers, I always thought that the euclidean option didn’t fit the metaphor. If…more
```

2

Reply

[![Carlos H Brandt](https://miro.medium.com/v2/resize:fill:32:32/1*gDvOTozm3EJio1F1S8HMgA.png)](https://medium.com/@chbrandt?source=post_page---post_responses--123438574097----2-----------------------------------)

[Carlos H Brandt](https://medium.com/@chbrandt?source=post_page---post_responses--123438574097----2-----------------------------------)

[Dec 5, 2018](https://medium.com/@chbrandt/hi-jake-72bcff6cbd3a?source=post_page---post_responses--123438574097----2-----------------------------------)

In data science, we often think about how to use data to make predictions on new data points. This is called “supervised learning.” Sometimes, however, rather than ‘making predictions’,...

```
Hi Jake,

Allow me to suggest a correction to this first paragraph.

Supervised and unsupervised learning are concepts completely different to what you suggest here. Although they may be to "making predictions" or "categorize data into buckets"…more
```

12

Reply

See all responses

## More from Jake Huneycutt and TDS Archive

![Implementing a Random Forest Classification Model in Python](https://miro.medium.com/v2/resize:fit:679/format:webp/1*xz0jfZazpnD2ncPqU54zYg.jpeg)

[**Implementing a Random Forest Classification Model in Python**\\
\\
**Random forests algorithms are used for classification and regression. The random forest is an ensemble learning method, composed of…**](https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652?source=post_page---author_recirc--123438574097----0---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

May 18, 2018

[A clap icon289\\
\\
A response icon6](https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652?source=post_page---author_recirc--123438574097----0---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

![How to Implement Graph RAG Using Knowledge Graphs and Vector Databases](https://miro.medium.com/v2/resize:fit:679/format:webp/1*hrwv6zmmgogVNpQQlOIwIA.png)

[**How to Implement Graph RAG Using Knowledge Graphs and Vector Databases**\\
\\
**A Step-by-Step Tutorial on Implementing Retrieval-Augmented Generation (RAG), Semantic Search, and Recommendations**](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759?source=post_page---author_recirc--123438574097----1---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

Sep 6, 2024

[A clap icon2K\\
\\
A response icon20](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759?source=post_page---author_recirc--123438574097----1---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

![Understanding LLMs from Scratch Using Middle School Math](https://miro.medium.com/v2/resize:fit:679/format:webp/1*9D2HQj6EBw0NC4c7YU0bWg.png)

[**Understanding LLMs from Scratch Using Middle School Math**\\
\\
**In this article, we talk about how LLMs work, from scratch — assuming only that you know how to add and multiply two numbers. The article…**](https://medium.com/data-science/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876?source=post_page---author_recirc--123438574097----2---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

Oct 19, 2024

[A clap icon8.2K\\
\\
A response icon103](https://medium.com/data-science/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876?source=post_page---author_recirc--123438574097----2---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

![Convert LaTeX into HTML with MathJax](https://miro.medium.com/v2/resize:fit:679/format:webp/1*oGKSLAyukkOr-1bZbMY0Cg.png)

[**Convert LaTeX into HTML with MathJax**\\
\\
**LaTeX (pronounced LAY-tek) is a high-quality typesetting and document preparation system, primarily used in academia. You can create LaTeX…**](https://medium.com/@hjhuney/how-to-convert-latex-into-html-a4334ffda3f4?source=post_page---author_recirc--123438574097----3---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

Apr 26, 2018

[A clap icon118\\
\\
A response icon3](https://medium.com/@hjhuney/how-to-convert-latex-into-html-a4334ffda3f4?source=post_page---author_recirc--123438574097----3---------------------67e2d8b0_797b_4af0_9d16_795796fc7441--------------)

[See all from Jake Huneycutt](https://medium.com/@hjhuney?source=post_page---author_recirc--123438574097---------------------------------------)

[See all from TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--123438574097---------------------------------------)

## Recommended from Medium

![Stanford Just Killed Prompt Engineering With 8 Words (And I Can’t Believe It Worked)](https://miro.medium.com/v2/resize:fit:679/format:webp/1*va3sFwIm26snbj5ly9ZsgA.jpeg)

[**Stanford Just Killed Prompt Engineering With 8 Words (And I Can’t Believe It Worked)**\\
\\
**ChatGPT keeps giving you the same boring response? This new technique unlocks 2× more creativity from ANY AI model — no training required…**](https://medium.com/generative-ai/stanford-just-killed-prompt-engineering-with-8-words-and-i-cant-believe-it-worked-8349d6524d2b?source=post_page---read_next_recirc--123438574097----0---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

Oct 19, 2025

[A clap icon23K\\
\\
A response icon608](https://medium.com/generative-ai/stanford-just-killed-prompt-engineering-with-8-words-and-i-cant-believe-it-worked-8349d6524d2b?source=post_page---read_next_recirc--123438574097----0---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

![6 brain images](https://miro.medium.com/v2/resize:fit:679/format:webp/1*Q-mzQNzJSVYkVGgsmHVjfw.png)

[**As a Neuroscientist, I Quit These 5 Morning Habits That Destroy Your Brain**\\
\\
**Most people do \#1 within 10 minutes of waking (and it sabotages your entire day)**](https://medium.com/write-a-catalyst/as-a-neuroscientist-i-quit-these-5-morning-habits-that-destroy-your-brain-3efe1f410226?source=post_page---read_next_recirc--123438574097----1---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

Jan 14

[A clap icon26K\\
\\
A response icon441](https://medium.com/write-a-catalyst/as-a-neuroscientist-i-quit-these-5-morning-habits-that-destroy-your-brain-3efe1f410226?source=post_page---read_next_recirc--123438574097----1---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

![Data Engineering Design Patterns You Must Learn in 2026](https://miro.medium.com/v2/resize:fit:679/format:webp/1*0cuVBpD9ZUDcnV3U1mV8cg.png)

[**Data Engineering Design Patterns You Must Learn in 2026**\\
\\
**These are the 8 data engineering design patterns every modern data stack is built on. Learn them once, and every data engineering tool…**](https://medium.com/aws-in-plain-english/data-engineering-design-patterns-you-must-learn-in-2026-c25b7bd0b9a7?source=post_page---read_next_recirc--123438574097----0---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

Jan 5

[A clap icon857\\
\\
A response icon17](https://medium.com/aws-in-plain-english/data-engineering-design-patterns-you-must-learn-in-2026-c25b7bd0b9a7?source=post_page---read_next_recirc--123438574097----0---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

![The AI Bubble Is About To Burst, But The Next Bubble Is Already Growing](https://miro.medium.com/v2/resize:fit:679/format:webp/0*jQ7Z0Y2Rw8kblsEX)

[**The AI Bubble Is About To Burst, But The Next Bubble Is Already Growing**\\
\\
**Techbros are preparing their latest bandwagon.**](https://medium.com/@wlockett/the-ai-bubble-is-about-to-burst-but-the-next-bubble-is-already-growing-383c0c0c7ede?source=post_page---read_next_recirc--123438574097----1---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

Sep 14, 2025

[A clap icon22K\\
\\
A response icon936](https://medium.com/@wlockett/the-ai-bubble-is-about-to-burst-but-the-next-bubble-is-already-growing-383c0c0c7ede?source=post_page---read_next_recirc--123438574097----1---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

![Building a Scalable, Production-Grade Agentic RAG Pipeline](https://miro.medium.com/v2/resize:fit:679/format:webp/1*vPuXR0bvfvmSAGECGOub0A.png)

[**Building a Scalable, Production-Grade Agentic RAG Pipeline**\\
\\
**Autoscaling, Evaluation, AI Compute Workflows and more**](https://medium.com/gitconnected/building-a-scalable-production-grade-agentic-rag-pipeline-1168dcd36260?source=post_page---read_next_recirc--123438574097----2---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

Dec 31, 2025

[A clap icon1.4K\\
\\
A response icon16](https://medium.com/gitconnected/building-a-scalable-production-grade-agentic-rag-pipeline-1168dcd36260?source=post_page---read_next_recirc--123438574097----2---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

![My ML Interview Notebook: A Revision Guide for Data Science Interviews](https://miro.medium.com/v2/resize:fit:679/format:webp/1*Bh2UIPk3nxj6H5tFpTfvJw.webp)

[**My ML Interview Notebook: A Revision Guide for Data Science Interviews**\\
\\
**ML Revision Notes**](https://medium.com/@mdmoseena22/my-ml-interview-notebook-a-revision-guide-for-data-science-interviews-d7b89cccfe33?source=post_page---read_next_recirc--123438574097----3---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

Nov 18, 2025

[A clap icon312\\
\\
A response icon11](https://medium.com/@mdmoseena22/my-ml-interview-notebook-a-revision-guide-for-data-science-interviews-d7b89cccfe33?source=post_page---read_next_recirc--123438574097----3---------------------18b2bb33_bfc7_4d89_b343_dc67e47f30e7--------------)

[See more recommendations](https://medium.com/?source=post_page---read_next_recirc--123438574097---------------------------------------)

[Help](https://help.medium.com/hc/en-us?source=post_page-----123438574097---------------------------------------)

[Status](https://status.medium.com/?source=post_page-----123438574097---------------------------------------)

[About](https://medium.com/about?autoplay=1&source=post_page-----123438574097---------------------------------------)

[Careers](https://medium.com/jobs-at-medium/work-at-medium-959d1a85284e?source=post_page-----123438574097---------------------------------------)

[Press](mailto:pressinquiries@medium.com)

[Blog](https://blog.medium.com/?source=post_page-----123438574097---------------------------------------)

[Privacy](https://policy.medium.com/medium-privacy-policy-f03bf92035c9?source=post_page-----123438574097---------------------------------------)

[Rules](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page-----123438574097---------------------------------------)

[Terms](https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----123438574097---------------------------------------)

[Text to speech](https://speechify.com/medium?source=post_page-----123438574097---------------------------------------)

reCAPTCHA

Recaptcha requires verification.

[Privacy](https://www.google.com/intl/en/policies/privacy/) \- [Terms](https://www.google.com/intl/en/policies/terms/)

protected by **reCAPTCHA**

[Privacy](https://www.google.com/intl/en/policies/privacy/) \- [Terms](https://www.google.com/intl/en/policies/terms/)