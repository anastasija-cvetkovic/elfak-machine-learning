[Sitemap](https://medium.com/sitemap/sitemap.xml)

[Open in app](https://play.google.com/store/apps/details?id=com.medium.reader&referrer=utm_source%3DmobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fcluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

[Medium Logo](https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------)

[Write](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------)

[Search](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fcluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

[**TDS Archive**](https://medium.com/data-science?source=post_page---publication_nav-7f60cf5620c9-474e55d00ebb---------------------------------------)

·

Follow publication

[![TDS Archive](https://miro.medium.com/v2/resize:fill:38:38/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_sidebar-7f60cf5620c9-474e55d00ebb---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow publication

Member-only story

## customer segmentation

# Cluster Analysis: Create, Visualize and Interpret Customer Segments

## Exploring methods for cluster analysis, visualizing clusters through dimensionality reduction and interpreting clusters through exploring impactful features.

[![Maarten Grootendorst](https://miro.medium.com/v2/resize:fill:32:32/1*k0XjRqlY6vxKmBSr-uySMQ.jpeg)](https://medium.com/@maartengrootendorst?source=post_page---byline--474e55d00ebb---------------------------------------)

[Maarten Grootendorst](https://medium.com/@maartengrootendorst?source=post_page---byline--474e55d00ebb---------------------------------------)

Follow

9 min read

·

Jul 30, 2019

916

11

[Listen](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3D474e55d00ebb&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fcluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb&source=---header_actions--474e55d00ebb---------------------post_audio_button------------------)

Share

Although we have seen a large influx of supervised machine learning techniques being used in organizations these methods suffer from, typically, one large issue; a need for labeled data. Fortunately, many unsupervised methods exist for clustering data into previously unseen groups, thereby extracting new insights from your clientele.

This article will guide you through the ins and outs of clustering customers. Note that I will not only show you which sklearn package you can use but more importantly, **how** they can be used and what to look out for.

As always, the [data](https://www.kaggle.com/blastchar/telco-customer-churn/) is relatively straightforward and you can follow along with the notebook [here](https://github.com/MaartenGr/CustomerSegmentation). It contains customer information from a Telecom company and is typically used to predict churn:

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1000/1*sBIBzy7vW2Xy9DHm1XGTug.png)

## Clustering Algorithms

There are many unsupervised clustering algorithms out there and although each of them has significant strengths in certain situations, I will…

### k-Means Clustering

In my experience, this is by far the most frequently used algorithm for clustering data. k-Means starts by choosing _k_ random centers which you can set yourself. Then, all data points are assigned to the closest center based on their Euclidean distance. Next, new centers are calculated and the data points are updated (see gif below). This process continuous until clusters do not change between iterations.

![](https://miro.medium.com/v2/resize:fit:480/1*umzqxI8Oeje8nU5EItF5dw.gif)

_An animation demonstrating the inner workings of k-means — Courtesy: Mubaris NK_

Now in the example above the three cluster centers start very close to each other. This typically does not work well as it will have a harder time finding clusters. Instead, you can use _k-means++_ to improve the initialization of the centers. It starts with an initial center…

## Create an account to read the full story.

The author made this story available to Medium members only.

If you’re new to Medium, create a new account to read this story on us.

[Continue in app](https://play.google.com/store/apps/details?id=com.medium.reader&referrer=utm_source%3Dregwall&source=-----474e55d00ebb---------------------post_regwall------------------)

Or, continue in mobile web

[Sign up with Google](https://medium.com/m/connect/google?state=google-%7Chttps%3A%2F%2Fmedium.com%2Fdata-science%2Fcluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb%3Fsource%3D-----474e55d00ebb---------------------post_regwall------------------%26skipOnboarding%3D1%7Cregister&source=-----474e55d00ebb---------------------post_regwall------------------)

[Sign up with Facebook](https://medium.com/m/connect/facebook?state=facebook-%7Chttps%3A%2F%2Fmedium.com%2Fdata-science%2Fcluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb%3Fsource%3D-----474e55d00ebb---------------------post_regwall------------------%26skipOnboarding%3D1%7Cregister&source=-----474e55d00ebb---------------------post_regwall------------------)

Sign up with email

Already have an account? [Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fcluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb&source=-----474e55d00ebb---------------------post_regwall------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:48:48/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--474e55d00ebb---------------------------------------)

[![TDS Archive](https://miro.medium.com/v2/resize:fill:64:64/1*JEuS4KBdakUcjg9sC7Wo4A.png)](https://medium.com/data-science?source=post_page---post_publication_info--474e55d00ebb---------------------------------------)

Follow

[**Published in TDS Archive**](https://medium.com/data-science?source=post_page---post_publication_info--474e55d00ebb---------------------------------------)

[828K followers](https://medium.com/data-science/followers?source=post_page---post_publication_info--474e55d00ebb---------------------------------------)

· [Last published Feb 3, 2025](https://medium.com/data-science/diy-ai-how-to-build-a-linear-regression-model-from-scratch-7b4cc0efd235?source=post_page---post_publication_info--474e55d00ebb---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Follow

[![Maarten Grootendorst](https://miro.medium.com/v2/resize:fill:48:48/1*k0XjRqlY6vxKmBSr-uySMQ.jpeg)](https://medium.com/@maartengrootendorst?source=post_page---post_author_info--474e55d00ebb---------------------------------------)

[![Maarten Grootendorst](https://miro.medium.com/v2/resize:fill:64:64/1*k0XjRqlY6vxKmBSr-uySMQ.jpeg)](https://medium.com/@maartengrootendorst?source=post_page---post_author_info--474e55d00ebb---------------------------------------)

Follow

[**Written by Maarten Grootendorst**](https://medium.com/@maartengrootendorst?source=post_page---post_author_info--474e55d00ebb---------------------------------------)

[7.2K followers](https://medium.com/@maartengrootendorst/followers?source=post_page---post_author_info--474e55d00ebb---------------------------------------)

· [7 following](https://medium.com/@maartengrootendorst/following?source=post_page---post_author_info--474e55d00ebb---------------------------------------)

Data Scientist \| Psychologist. Passionate about anything AI-related! Get in touch: [www.linkedin.com/in/mgrootendorst/](http://www.linkedin.com/in/mgrootendorst/)

Follow

## Responses (11)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

Write a response

[What are your thoughts?](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fdata-science%2Fcluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb&source=---post_responses--474e55d00ebb---------------------respond_sidebar------------------)

Cancel

Respond

[![Jorge Del Río](https://miro.medium.com/v2/resize:fill:32:32/0*DLMvLTvl9ngYs0dg)](https://medium.com/@fri0?source=post_page---post_responses--474e55d00ebb----0-----------------------------------)

[Jorge Del Río](https://medium.com/@fri0?source=post_page---post_responses--474e55d00ebb----0-----------------------------------)

[Jun 23, 2020](https://medium.com/@fri0/hi-great-article-fb37fe411c7f?source=post_page---post_responses--474e55d00ebb----0-----------------------------------)

```
Hi, great article!
I would like to ask you how many features are necesary to be considered as high dimensional data? more than 10? 20? Thanks!
```

2

Reply

[![Sandra Silva](https://miro.medium.com/v2/resize:fill:32:32/1*NGP9TpK2cGFbGsdYUNEWuQ.jpeg)](https://medium.com/@sandrasilva_2912?source=post_page---post_responses--474e55d00ebb----1-----------------------------------)

[Sandra Silva](https://medium.com/@sandrasilva_2912?source=post_page---post_responses--474e55d00ebb----1-----------------------------------)

[Jun 2, 2020](https://medium.com/@sandrasilva_2912/thank-you-so-much-for-this-explanation-great-work-1f3414c7d8c6?source=post_page---post_responses--474e55d00ebb----1-----------------------------------)

```
Thank you so much for this explanation. Great work!
```

2

Reply

[![Manisha Dhingra](https://miro.medium.com/v2/resize:fill:32:32/1*TZ0UctUPSRfauG8VdfDjvQ.jpeg)](https://medium.com/@manishadhingra?source=post_page---post_responses--474e55d00ebb----2-----------------------------------)

[Manisha Dhingra](https://medium.com/@manishadhingra?source=post_page---post_responses--474e55d00ebb----2-----------------------------------)

[Jul 30, 2019](https://manishadhingra.medium.com/very-well-explained-a4a5376ca40a?source=post_page---post_responses--474e55d00ebb----2-----------------------------------)

```
Very well explained!
```

2

1 reply

Reply

See all responses

## More from Maarten Grootendorst and TDS Archive

![Topic Modeling with BERT](https://miro.medium.com/v2/resize:fit:679/format:webp/1*W94GjvT6vBzDGY50qPHZDA.png)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--474e55d00ebb----0---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

by

[Maarten Grootendorst](https://medium.com/@maartengrootendorst?source=post_page---author_recirc--474e55d00ebb----0---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

[**Topic Modeling with BERT**\\
\\
**Leveraging BERT and TF-IDF to create easily interpretable topics.**](https://medium.com/data-science/topic-modeling-with-bert-779f7db187e6?source=post_page---author_recirc--474e55d00ebb----0---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

Oct 5, 2020

[A clap icon3.1K\\
\\
A response icon26](https://medium.com/data-science/topic-modeling-with-bert-779f7db187e6?source=post_page---author_recirc--474e55d00ebb----0---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

![How to Implement Graph RAG Using Knowledge Graphs and Vector Databases](https://miro.medium.com/v2/resize:fit:679/format:webp/1*hrwv6zmmgogVNpQQlOIwIA.png)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--474e55d00ebb----1---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

by

[Steve Hedden](https://medium.com/@stevehedden?source=post_page---author_recirc--474e55d00ebb----1---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

[**How to Implement Graph RAG Using Knowledge Graphs and Vector Databases**\\
\\
**A Step-by-Step Tutorial on Implementing Retrieval-Augmented Generation (RAG), Semantic Search, and Recommendations**](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759?source=post_page---author_recirc--474e55d00ebb----1---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

Sep 6, 2024

[A clap icon2K\\
\\
A response icon20](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759?source=post_page---author_recirc--474e55d00ebb----1---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

![Understanding LLMs from Scratch Using Middle School Math](https://miro.medium.com/v2/resize:fit:679/format:webp/1*9D2HQj6EBw0NC4c7YU0bWg.png)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--474e55d00ebb----2---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

by

[Rohit Patel](https://medium.com/@rohit-patel?source=post_page---author_recirc--474e55d00ebb----2---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

[**Understanding LLMs from Scratch Using Middle School Math**\\
\\
**In this article, we talk about how LLMs work, from scratch — assuming only that you know how to add and multiply two numbers. The article…**](https://medium.com/data-science/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876?source=post_page---author_recirc--474e55d00ebb----2---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

Oct 19, 2024

[A clap icon8.2K\\
\\
A response icon103](https://medium.com/data-science/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876?source=post_page---author_recirc--474e55d00ebb----2---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

![9 Distance Measures in Data Science](https://miro.medium.com/v2/resize:fit:679/format:webp/1*FTVRr_Wqz-3_k6Mk6G4kew.png)

In

[TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--474e55d00ebb----3---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

by

[Maarten Grootendorst](https://medium.com/@maartengrootendorst?source=post_page---author_recirc--474e55d00ebb----3---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

[**9 Distance Measures in Data Science**\\
\\
**The advantages and pitfalls of common distance measures**](https://medium.com/data-science/9-distance-measures-in-data-science-918109d069fa?source=post_page---author_recirc--474e55d00ebb----3---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

Feb 1, 2021

[A clap icon4.2K\\
\\
A response icon26](https://medium.com/data-science/9-distance-measures-in-data-science-918109d069fa?source=post_page---author_recirc--474e55d00ebb----3---------------------e9f55995_ecc6_4a90_818c_bcf8832e448b--------------)

[See all from Maarten Grootendorst](https://medium.com/@maartengrootendorst?source=post_page---author_recirc--474e55d00ebb---------------------------------------)

[See all from TDS Archive](https://medium.com/data-science?source=post_page---author_recirc--474e55d00ebb---------------------------------------)

## Recommended from Medium

![Chapter 4: Propensity Score Matching](https://miro.medium.com/v2/resize:fit:679/format:webp/1*HcnTHGtrZ4EHgvh7DSdbnQ.png)

In

[Modern Causal Inference: Methods and Applications](https://medium.com/causal-inference-methods-models-and-applications?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

by

[Chris Kuo/Dr. Dataman](https://medium.com/@dataman-ai?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

[**Chapter 4: Propensity Score Matching**\\
\\
**Alex, a data scientist at a government agency, was tasked with evaluating a new job training program. Eager to deliver results quickly, he…**](https://medium.com/causal-inference-methods-models-and-applications/propensity-score-matching-c589abd4c291?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

Nov 5, 2025

[A clap icon33\\
\\
A response icon1](https://medium.com/causal-inference-methods-models-and-applications/propensity-score-matching-c589abd4c291?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

![Stanford Just Killed Prompt Engineering With 8 Words (And I Can’t Believe It Worked)](https://miro.medium.com/v2/resize:fit:679/format:webp/1*va3sFwIm26snbj5ly9ZsgA.jpeg)

In

[Generative AI](https://medium.com/generative-ai?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

by

[Adham Khaled](https://medium.com/@adham__khaled__?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

[**Stanford Just Killed Prompt Engineering With 8 Words (And I Can’t Believe It Worked)**\\
\\
**ChatGPT keeps giving you the same boring response? This new technique unlocks 2× more creativity from ANY AI model — no training required…**](https://medium.com/generative-ai/stanford-just-killed-prompt-engineering-with-8-words-and-i-cant-believe-it-worked-8349d6524d2b?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

Oct 19, 2025

[A clap icon23K\\
\\
A response icon608](https://medium.com/generative-ai/stanford-just-killed-prompt-engineering-with-8-words-and-i-cant-believe-it-worked-8349d6524d2b?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

![Data Engineering Design Patterns You Must Learn in 2026](https://miro.medium.com/v2/resize:fit:679/format:webp/1*0cuVBpD9ZUDcnV3U1mV8cg.png)

In

[AWS in Plain English](https://medium.com/aws-in-plain-english?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

by

[Khushbu Shah](https://medium.com/@khushbu.shah_661?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

[**Data Engineering Design Patterns You Must Learn in 2026**\\
\\
**These are the 8 data engineering design patterns every modern data stack is built on. Learn them once, and every data engineering tool…**](https://medium.com/aws-in-plain-english/data-engineering-design-patterns-you-must-learn-in-2026-c25b7bd0b9a7?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

Jan 5

[A clap icon857\\
\\
A response icon17](https://medium.com/aws-in-plain-english/data-engineering-design-patterns-you-must-learn-in-2026-c25b7bd0b9a7?source=post_page---read_next_recirc--474e55d00ebb----0---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

![6 brain images](https://miro.medium.com/v2/resize:fit:679/format:webp/1*Q-mzQNzJSVYkVGgsmHVjfw.png)

In

[Write A Catalyst](https://medium.com/write-a-catalyst?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

by

[Dr. Patricia Schmidt](https://medium.com/@creatorschmidt?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

[**As a Neuroscientist, I Quit These 5 Morning Habits That Destroy Your Brain**\\
\\
**Most people do \#1 within 10 minutes of waking (and it sabotages your entire day)**](https://medium.com/write-a-catalyst/as-a-neuroscientist-i-quit-these-5-morning-habits-that-destroy-your-brain-3efe1f410226?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

Jan 14

[A clap icon26K\\
\\
A response icon441](https://medium.com/write-a-catalyst/as-a-neuroscientist-i-quit-these-5-morning-habits-that-destroy-your-brain-3efe1f410226?source=post_page---read_next_recirc--474e55d00ebb----1---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

![A high-tech robotic claw extracting a glowing data cube from a complex digital interface, symbolizing web scraping automation and data extraction.](https://miro.medium.com/v2/resize:fit:679/format:webp/1*HCULJH-MauL4rDajdbgKWg.png)

[Emre Yunusoglu](https://medium.com/@yunsoftofficial?source=post_page---read_next_recirc--474e55d00ebb----2---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

[**What is Clawdbot?**\\
\\
**Moving beyond the hype to understand what Clawbot actually does in real-world workflows.**](https://medium.com/@yunsoftofficial/what-is-clawbot-88c64a8e537c?source=post_page---read_next_recirc--474e55d00ebb----2---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

Jan 30

[A clap icon56](https://medium.com/@yunsoftofficial/what-is-clawbot-88c64a8e537c?source=post_page---read_next_recirc--474e55d00ebb----2---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

![Building a Synthetic Healthcare Insurance Claims Dataset for Fraud Detection](https://miro.medium.com/v2/resize:fit:679/format:webp/1*2hGcP7-WhFMKZfE5FBuGoQ.png)

[Dr.Tiya Vaj](https://medium.com/@vtiya?source=post_page---read_next_recirc--474e55d00ebb----3---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

[**Building a Synthetic Healthcare Insurance Claims Dataset for Fraud Detection**\\
\\
**Fraud detection in healthcare insurance claims is a crucial problem for payers, providers, and regulators. Real-world claims data, however…**](https://medium.com/@vtiya/building-a-synthetic-healthcare-insurance-claims-dataset-for-fraud-detection-95350c29fe78?source=post_page---read_next_recirc--474e55d00ebb----3---------------------e5344cb1_1dae_4c84_90b5_30cae53f414e--------------)

Sep 13, 2025

[See more recommendations](https://medium.com/?source=post_page---read_next_recirc--474e55d00ebb---------------------------------------)

[Help](https://help.medium.com/hc/en-us?source=post_page-----474e55d00ebb---------------------------------------)

[Status](https://status.medium.com/?source=post_page-----474e55d00ebb---------------------------------------)

[About](https://medium.com/about?autoplay=1&source=post_page-----474e55d00ebb---------------------------------------)

[Careers](https://medium.com/jobs-at-medium/work-at-medium-959d1a85284e?source=post_page-----474e55d00ebb---------------------------------------)

[Press](mailto:pressinquiries@medium.com)

[Blog](https://blog.medium.com/?source=post_page-----474e55d00ebb---------------------------------------)

[Privacy](https://policy.medium.com/medium-privacy-policy-f03bf92035c9?source=post_page-----474e55d00ebb---------------------------------------)

[Rules](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page-----474e55d00ebb---------------------------------------)

[Terms](https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----474e55d00ebb---------------------------------------)

[Text to speech](https://speechify.com/medium?source=post_page-----474e55d00ebb---------------------------------------)

reCAPTCHA

Recaptcha requires verification.

[Privacy](https://www.google.com/intl/en/policies/privacy/) \- [Terms](https://www.google.com/intl/en/policies/terms/)

protected by **reCAPTCHA**

[Privacy](https://www.google.com/intl/en/policies/privacy/) \- [Terms](https://www.google.com/intl/en/policies/terms/)