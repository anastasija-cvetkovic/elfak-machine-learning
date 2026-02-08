[Sitemap](https://medium.com/sitemap/sitemap.xml)

[Open in app](https://play.google.com/store/apps/details?id=com.medium.reader&referrer=utm_source%3DmobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fclustering-based-unsupervised-learning-8d705298ae51&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

[Medium Logo](https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------)

[Write](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------)

[Search](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fclustering-based-unsupervised-learning-8d705298ae51&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

[**TDS Archive**](https://medium.com/data-science?source=post_page---publication_nav-7f60cf5620c9-8d705298ae51---------------------------------------)

·

Follow publication

[![TDS Archive](https://miro.medium.com/v2/resize:fill:38:38/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_sidebar-7f60cf5620c9-8d705298ae51---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow publication

# Clustering Based Unsupervised Learning

[![Syed Sadat Nazrul](https://miro.medium.com/v2/resize:fill:32:32/1*8I0_wgEjC0CPRDAovzrfJQ.png)](https://medium.com/@sadatnazrul?source=post_page---byline--8d705298ae51---------------------------------------)

[Syed Sadat Nazrul](https://medium.com/@sadatnazrul?source=post_page---byline--8d705298ae51---------------------------------------)

Follow

6 min read

·

Apr 2, 2018

896

2

[Listen](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3D8d705298ae51&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fclustering-based-unsupervised-learning-8d705298ae51&source=---header_actions--8d705298ae51---------------------post_audio_button------------------)

Share

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*6hfFWPITJxbtw4ztoC1YeA.png)

Unsupervised machine learning is the machine learning task of inferring a function to describe hidden structure from “unlabeled” data (a classification or categorization is not included in the observations). Common scenarios for using unsupervised learning algorithms include:

\- Data Exploration

\- Outlier Detection

\- Pattern Recognition

## Get Syed Sadat Nazrul’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

Subscribe

While there is an exhaustive list of clustering algorithms available (whether you use R or Python’s Scikit-Learn), I will attempt to cover the basic concepts.

## K-Means

The most common and simplest clustering algorithm out there is the K-Means clustering. This algorithms involve you telling the algorithms how many possible cluster (or K) there are in the dataset. The algorithm then iteratively moves the k-centers and selects the datapoints that are closest to that centroid in the cluster.

![](https://miro.medium.com/v2/resize:fit:430/1*F9CpDKLeeowBIbpjPGrptQ.png)

Taking K=3 as an example, the iterative process is given below:

![](https://miro.medium.com/v2/resize:fit:637/1*Nx6IyGfRAV1ly6uDGnVCxQ.gif)

One obvious question that may come to mind is the methodology for picking the K value. This is done using an elbow curve, where the x-axis is the K-value and the y axis is some objective function. A common objective function is the average distance between the datapoints and the nearest centroid.

![](https://miro.medium.com/v2/resize:fit:526/1*IpOrjgpUmNOFddP31xQmGw.png)

The best number for K is the “elbow” or kinked region. After this point, it is generally established that adding more clusters will not add significant value to your analysis. Below is an example script for K-Means using Scikit-Learn on the iris dataset:

```
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
%matplotlib inline
from sklearn import datasets#Iris Dataset
iris = datasets.load_iris()
X = iris.data#KMeans
km = KMeans(n_clusters=3)
km.fit(X)
km.predict(X)
labels = km.labels_#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)
```

![](https://miro.medium.com/v2/resize:fit:500/1*XPClT1UesyqHRLxoS20V9Q.png)

One issue with K-means, as see in the 3D diagram above, is that it does hard labels. However, you can see that datapoints at the boundary of the purple and yellow clusters can be either one. For such circumstances, a different approach may be necessary.

## Mixture Models

In K-Means, we do what is called “hard labeling”, where we simply add the label of the maximum probability. However, certain data points that exist at the boundary of clusters may simply have similar probabilities of being on either clusters. In such circumstances, we look at all the probabilities instead of the max probability. This is known as “soft labeling”.

```
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
%matplotlib inline
from sklearn import datasets#Iris Dataset
iris = datasets.load_iris()
X = iris.data#Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
proba_lists = gmm.predict_proba(X)#Plotting
colored_arrays = np.matrix(proba_lists)
colored_tuples = [tuple(i.tolist()[0]) for i in colored_arrays]
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=colored_tuples, edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("Gaussian Mixture Model", fontsize=14)
```

![](https://miro.medium.com/v2/resize:fit:500/1*RWOzGs9LJQ5E-1SkPR_ZxA.png)

For the above Gaussian Mixure Model, the colors of the datapoints are based on the Gaussian probability of being near the cluster. The RGB values are based on the nearness to each of the red, blue and green clusters. If you look at the datapoints near the boundary of the blue and red cluster, you shall see purple, indicating the datapoints are close to either clusters.

## Topic Modelling

Since we have talked about numerical values, let’s take a turn towards categorical values. One such application is text analytics. Common approach for such problems is topic modelling, where documents or words in a document are categorized into topics. The simplest of these is the TF-IDF model. The TF-IDF model classifies words based on their importance. This is determined by how frequent are they in specific documents (e.g. specific science topics in scientific journals) and words that are common among all documents (e.g. stop words).

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:319/1*Uucq42G4ntPGJKzI84b3aA.png)

One of my favorite algorithms is the Latent Dirichlet Allocation or LDA model. In this model, each word in the document is given a topic based on the entire document corpus. Below, I have attached a slide from the University of Washington’s Machine Learning specialization course:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*1njiK8zRvWLO4DWaaeOxfQ.png)

The mechanics behind the LDA model itself is hard to explain in this blog. However, a common question people have is deciding on the number of topics. While there is no established answer for this, personally I prefer to implement a elbow curve of K-Means of the word vector of each document. The closeness of each word vector can be determined by the cosine distance.

![](https://miro.medium.com/v2/resize:fit:697/1*Q4xQoV8k_7S7xB-NfvFdrw.png)

## Hidden Markov Model

Finally, let’s cover some timeseries analysis. For clustering, my favourite is using Hidden Markov Models or HMM. In a Markov Model, we look for states and the probability of the next state given the current state. An example below is of a dog’s life in Markov Model.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*XSHrKIysFYhaOJCLBgZyKA.png)

Let’s assume the dog is sick. Given the current state, there is a 0.6 chance it will continue being sick the next hour, 0.4 that it is sleeping, 05 pooping, 0.1 eating and 0.4 that it will be healthy again. In an HMM, you provide how many states there may be inside the timeseries data for the model to compute. An example of the Boston house prices dataset is given below with 3 states.

```
from hmmlearn import hmm
import numpy as np
%matplotlib inline
from sklearn import datasets#Data
boston = datasets.load_boston()
ts_data = boston.data[1,:]#HMM Model
gm = hmm.GaussianHMM(n_components=3)
gm.fit(ts_data.reshape(-1, 1))
states = gm.predict(ts_data.reshape(-1, 1))#Plot
color_dict = {0:"r",1:"g",2:"b"}
color_array = [color_dict[i] for i in states]
plt.scatter(range(len(ts_data)), ts_data, c=color_array)
plt.title("HMM Model")
```

![](https://miro.medium.com/v2/resize:fit:524/1*DxjolQX59HHv_KRhvGb1Aw.png)

As with every clustering problem, deciding the number of states is also a common issue. This may either be domain based. e.g. in voice recognition, it is common practice to use 3 states. Another possibility is using an elbow curve.

## Final Thoughts

As I have mentioned at the beginning of this blog, it is not possible for me to cover every single unsupervised models out there. At the same time, based on your use case, you may need a combination of algorithms to get a different perspective of the same data. With that I would like to leave you off with Scikit-Learn’s famous clustering demonstrations on the toy dataset:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*oNt9G9UpVhtyFLDBwEMf8Q.png)

[Unsupervised Learning](https://medium.com/tag/unsupervised-learning?source=post_page-----8d705298ae51---------------------------------------)

[Clustering](https://medium.com/tag/clustering?source=post_page-----8d705298ae51---------------------------------------)

[K Means](https://medium.com/tag/k-means?source=post_page-----8d705298ae51---------------------------------------)

[Lda](https://medium.com/tag/lda?source=post_page-----8d705298ae51---------------------------------------)

[Topic Modeling](https://medium.com/tag/topic-modeling?source=post_page-----8d705298ae51---------------------------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:48:48/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--8d705298ae51---------------------------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:64:64/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--8d705298ae51---------------------------------------)

Follow

[**Published in TDS Archive**](https://medium.com/data-science?source=post_page---post_publication_info--8d705298ae51---------------------------------------)

[828K followers](https://medium.com/data-science/followers?source=post_page---post_publication_info--8d705298ae51---------------------------------------)

· [Last published Feb 3, 2025](https://medium.com/data-science/diy-ai-how-to-build-a-linear-regression-model-from-scratch-7b4cc0efd235?source=post_page---post_publication_info--8d705298ae51---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow

[![Syed Sadat Nazrul](https://miro.medium.com/v2/resize:fill:48:48/1*8I0_wgEjC0CPRDAovzrfJQ.png)](https://medium.com/@sadatnazrul?source=post_page---post_author_info--8d705298ae51---------------------------------------)

[![Syed Sadat Nazrul](https://miro.medium.com/v2/resize:fill:64:64/1*8I0_wgEjC0CPRDAovzrfJQ.png)](https://medium.com/@sadatnazrul?source=post_page---post_author_info--8d705298ae51---------------------------------------)

Follow

[**Written by Syed Sadat Nazrul**](https://medium.com/@sadatnazrul?source=post_page---post_author_info--8d705298ae51---------------------------------------)

[2.9K followers](https://medium.com/@sadatnazrul/followers?source=post_page---post_author_info--8d705298ae51---------------------------------------)

· [86 following](https://medium.com/@sadatnazrul/following?source=post_page---post_author_info--8d705298ae51---------------------------------------)

Using Machine Learning to catch cyber and financial criminals by day … and writing cool blogs by night. Views expressed are of my own.

Follow

## Responses (2)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

Write a response

[What are your thoughts?](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fclustering-based-unsupervised-learning-8d705298ae51&source=---post_responses--8d705298ae51---------------------respond_sidebar------------------)

Cancel

Respond

[![Michael Galarnyk](https://miro.medium.com/v2/resize:fill:32:32/2*FsG9-gMI-jiPBDcH0LLl7g.jpeg)](https://medium.com/@GalarnykMichael?source=post_page---post_responses--8d705298ae51----0-----------------------------------)

[Michael Galarnyk](https://medium.com/@GalarnykMichael?source=post_page---post_responses--8d705298ae51----0-----------------------------------)

[Apr 3, 2018](https://medium.com/@GalarnykMichael/excellent-work-i-will-advise-some-people-studying-for-interviews-to-view-this-df6154175bbb?source=post_page---post_responses--8d705298ae51----0-----------------------------------)

```
Excellent work! I will advise some people studying for interviews to view this.
```

63

Reply

[![Rajaphysharma](https://miro.medium.com/v2/resize:fill:32:32/0*KvnCvSiigpedq1Pl)](https://medium.com/@rajaphysharma?source=post_page---post_responses--8d705298ae51----1-----------------------------------)

[Rajaphysharma](https://medium.com/@rajaphysharma?source=post_page---post_responses--8d705298ae51----1-----------------------------------)

[Dec 30, 2020](https://medium.com/@rajaphysharma/i-simply-liked-it-specially-the-conluding-remarks-with-scikit-learns-demonstration-b631cf88e3d4?source=post_page---post_responses--8d705298ae51----1-----------------------------------)

```
I simply liked it. Specially the conluding remarks with Scikit-Learn's demonstration.
```

Reply

## More from Syed Sadat Nazrul and TDS Archive

![Basics of IP Addresses in Computer Networking](https://miro.medium.com/v2/resize:fit:679/format:webp/1*K05qR30OtmmWviQHt8sUwQ.png)

[![Syed Sadat Nazrul](https://miro.medium.com/v2/resize:fill:20:20/1*8I0_wgEjC0CPRDAovzrfJQ.png)](https://medium.com/@sadatnazrul?source=post_page---author_recirc--8d705298ae51----0---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

[Syed Sadat Nazrul](https://medium.com/@sadatnazrul?source=post_page---author_recirc--8d705298ae51----0---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

Oct 24, 2018

[A clap icon400\\
\\
A response icon3](https://medium.com/@sadatnazrul/basics-of-ip-addresses-in-computer-networking-f1a4661ea85c?source=post_page---author_recirc--8d705298ae51----0---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

![How to Implement Graph RAG Using Knowledge Graphs and Vector Databases](https://miro.medium.com/v2/resize:fit:679/format:webp/1*hrwv6zmmgogVNpQQlOIwIA.png)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:20:20/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---author_recirc--8d705298ae51----1---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--8d705298ae51----1---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

by

[Steve Hedden](https://medium.com/@stevehedden?source=post_page---author_recirc--8d705298ae51----1---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

Sep 6, 2024

[A clap icon2K\\
\\
A response icon20](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759?source=post_page---author_recirc--8d705298ae51----1---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

![Understanding LLMs from Scratch Using Middle School Math](https://miro.medium.com/v2/resize:fit:679/format:webp/1*9D2HQj6EBw0NC4c7YU0bWg.png)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:20:20/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---author_recirc--8d705298ae51----2---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--8d705298ae51----2---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

by

[Rohit Patel](https://medium.com/@rohit-patel?source=post_page---author_recirc--8d705298ae51----2---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

Oct 19, 2024

[A clap icon8.2K\\
\\
A response icon103](https://medium.com/data-science/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876?source=post_page---author_recirc--8d705298ae51----2---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

![Intro To Computer Networking And Internet Protocols](https://miro.medium.com/v2/resize:fit:679/format:webp/1*3WccK2T1uREd-6VEC5xk0w.png)

[![Syed Sadat Nazrul](https://miro.medium.com/v2/resize:fill:20:20/1*8I0_wgEjC0CPRDAovzrfJQ.png)](https://medium.com/@sadatnazrul?source=post_page---author_recirc--8d705298ae51----3---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

[Syed Sadat Nazrul](https://medium.com/@sadatnazrul?source=post_page---author_recirc--8d705298ae51----3---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

Oct 24, 2018

[A clap icon400\\
\\
A response icon3](https://medium.com/@sadatnazrul/intro-to-computer-networking-and-internet-protocols-8f03710ca409?source=post_page---author_recirc--8d705298ae51----3---------------------21babaae_1e5e_46a7_ad30_45d669c06f02--------------)

[See all from Syed Sadat Nazrul](https://medium.com/@sadatnazrul?source=post_page---author_recirc--8d705298ae51---------------------------------------)

[See all from TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--8d705298ae51---------------------------------------)