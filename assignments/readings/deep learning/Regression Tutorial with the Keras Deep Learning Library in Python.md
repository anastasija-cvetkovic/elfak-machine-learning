[Sitemap](https://sanchittanwar75.medium.com/sitemap/sitemap.xml)

[Open in app](https://play.google.com/store/apps/details?id=com.medium.reader&referrer=utm_source%3DmobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fsanchittanwar75.medium.com%2Fintroduction-to-machine-learning-and-deep-learning-bd25b792e488&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

[Medium Logo](https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------)

Get app

[Write](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------)

[Search](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------)

Sign up

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fsanchittanwar75.medium.com%2Fintroduction-to-machine-learning-and-deep-learning-bd25b792e488&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

# Introduction to machine learning and deep learning.

[![Sanchit Tanwar](https://miro.medium.com/v2/da:true/resize:fill:32:32/0*AeFRlTZDXD2zdudL)](https://sanchittanwar75.medium.com/?source=post_page---byline--bd25b792e488---------------------------------------)

[Sanchit Tanwar](https://sanchittanwar75.medium.com/?source=post_page---byline--bd25b792e488---------------------------------------)

Follow

4 min read

·

May 31, 2019

39

[Listen](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3Dbd25b792e488&operation=register&redirect=https%3A%2F%2Fsanchittanwar75.medium.com%2Fintroduction-to-machine-learning-and-deep-learning-bd25b792e488&source=---header_actions--bd25b792e488---------------------post_audio_button------------------)

Share

Signup for my live computer vision course: [https://bit.ly/cv\_coursem](https://bit.ly/cv_coursem)

This is the first part of deep learning workshop. The link to lessons will be given below as soon as I update them. Github link of this repo is [here](https://github.com/sanchit2843/dlworkshop). First two parts will cover the basics of machine learning and the background. Later We will jump to code and maths.

1. Introduction to machine learning and deep learning. <— You are here
2. [Introduction to neural networks.](https://medium.com/@sanchittanwar75/introduction-to-neural-networks-660f6909fba9?postPublishedType=repub)
3. [Introduction to python.](https://github.com/sanchit2843/dlworkshop/blob/master/Lesson%203%20-%20Introduction%20to%20python.ipynb)
4. [Building our first neural network in keras.](https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5)
5. [A comprehensive guide to CNN.](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
6. [Image classification with CNN.](https://medium.com/@sanchittanwar75/making-our-first-cnn-based-project-using-keras-c3a7790b1e02)

## What is machine learning?

Machine learning is a field of study which allows machines(computers) to learn from data or experience and make a prediction based on the experience.

## Get Sanchit Tanwar’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

Subscribe

Remember me for faster sign in

It enables the computers or the machines to make data-driven decisions rather than being explicitly programmed for carrying out a certain task. These programs or algorithms are designed in a way that they learn and improve over time when are exposed to new data.

## Types of machine learning

### Machine learning can be broadly divided into 3 subcategories

## 1\. Supervised learning -

In supervised learning, we have a labeled data containing input X and a label Y. In supervised learning, our task is to find the mapping between the input variable(X) called the independent variable and output variable(Y) called the dependent variable. Supervised learning can further be divided into two types of tasks:-

1. Regression — Regression problem is when the output variable is continuous and real value. For example price, weight, etc.
2. Classification — Classification is a problem when the output variable is a category, such as “red” or “blue” or “disease” or “no disease”.

## 2\. Unsupervised learning -

Unsupervised learning is where you only have input data (X) and no corresponding output variables. The goal of unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. Unsupervised learning can further be divided into two types of tasks:-

1. Clustering: A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.
2. Association: An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y.

## 3\. Reinforcement learning -

Reinforcement learning, in the context of artificial intelligence, is a type of dynamic programming that trains algorithms using a system of reward and punishment. A reinforcement learning algorithm, or agent, learns by interacting with its environment. The agent receives rewards by performing correctly and penalties for performing incorrectly. The agent learns without intervention from a human by maximizing its reward and minimizing its penalty.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*AOh7p7f0MYj-fkHSrrNdGg.png)

Machine learning categories with few applications

Now we will directly jump to deep learning without discussing machine learning algorithms, to learn about various machine learning algorithms refer [here](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/).

## What is Deep Learning?

Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*AT6RRGh5FXZFWZZHay48Xg.jpeg)

## Why there is hype for deep learning?

Neural networks which are the basic building block of deep learning are not new, and even older than most machine learning algorithms. But they were not mainstream in earlier years of its development because of lack of computation power and lack of data. In the year 2012 when a team of researcher won imagenet yearly computer vision competition using convolution neural network for the first time in imagenet competition lead made deep learning mainstream ai. They created alexnet CNN(convolution neural network) architecture, the network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. The idea of CNN was not new and it was proposed in the year 1998 by Yan Le Cun one of the pioneers of modern ai. In 2012, neural networks due to available computation power and enough data showed their true potential.

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:700/1*2XbrnZPBZxWTdPUc-9Y7OQ.png)

## Applications of deep learning: -

Applications of deep learning are vast and many of great technologies now use deep learning to improve the task. Some of the examples are:-

1. Self-driving cars
2. Voice search and virtual assistants
3. Machine translation
4. Image caption generation
5. Colorization of Black and White Images
6. Game playing ai(Open Ai dota bot, google brain alpha go).
7. Real-time object recognition in the image (Google lens).

![](https://miro.medium.com/v2/resize:fit:638/1*cTFOLJmBPhv61vtEbfCkFw.jpeg)

This list can become huge rather endless. See these links to learn about more applications of deep learning.

[https://machinelearningmastery.com/inspirational-applications-deep-learning/](https://machinelearningmastery.com/inspirational-applications-deep-learning/)

[https://medium.com/@vratulmittal/top-15-deep-learning-applications-that-will-rule-the-world-in-2018-and-beyond-7c6130c43b01](https://medium.com/@vratulmittal/top-15-deep-learning-applications-that-will-rule-the-world-in-2018-and-beyond-7c6130c43b01)

[http://www.yaronhadad.com/deep-learning-most-amazing-applications/](http://www.yaronhadad.com/deep-learning-most-amazing-applications/)

We will discuss neural networks in the next chapter and will start with mathematics. Link to other chapters is at the beginning.

[Machine Learning](https://medium.com/tag/machine-learning?source=post_page-----bd25b792e488---------------------------------------)

[Tutorials](https://medium.com/tag/tutorials?source=post_page-----bd25b792e488---------------------------------------)

[Artificial Intelligence](https://medium.com/tag/artificial-intelligence?source=post_page-----bd25b792e488---------------------------------------)

[Deep Learning](https://medium.com/tag/deep-learning?source=post_page-----bd25b792e488---------------------------------------)

[Beginners Guide](https://medium.com/tag/beginners-guide?source=post_page-----bd25b792e488---------------------------------------)

[![Sanchit Tanwar](https://miro.medium.com/v2/resize:fill:48:48/0*AeFRlTZDXD2zdudL)](https://sanchittanwar75.medium.com/?source=post_page---post_author_info--bd25b792e488---------------------------------------)

[![Sanchit Tanwar](https://miro.medium.com/v2/resize:fill:64:64/0*AeFRlTZDXD2zdudL)](https://sanchittanwar75.medium.com/?source=post_page---post_author_info--bd25b792e488---------------------------------------)

Follow

[**Written by Sanchit Tanwar**](https://sanchittanwar75.medium.com/?source=post_page---post_author_info--bd25b792e488---------------------------------------)

[281 followers](https://sanchittanwar75.medium.com/followers?source=post_page---post_author_info--bd25b792e488---------------------------------------)

· [64 following](https://sanchittanwar75.medium.com/following?source=post_page---post_author_info--bd25b792e488---------------------------------------)

Follow

## No responses yet

![](https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png)

Write a response

[What are your thoughts?](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fsanchittanwar75.medium.com%2Fintroduction-to-machine-learning-and-deep-learning-bd25b792e488&source=---post_responses--bd25b792e488---------------------respond_sidebar------------------)

Cancel

Respond

## More from Sanchit Tanwar

![Markov chains and Markov Decision process](https://miro.medium.com/v2/resize:fit:679/format:webp/1*Uh11rrUKKsHLLRmmv0ss2w.jpeg)

[![Sanchit Tanwar](https://miro.medium.com/v2/resize:fill:20:20/0*AeFRlTZDXD2zdudL)](https://sanchittanwar75.medium.com/?source=post_page---author_recirc--bd25b792e488----0---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

[Sanchit Tanwar](https://sanchittanwar75.medium.com/?source=post_page---author_recirc--bd25b792e488----0---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

[**This is the second part of the reinforcement learning tutorial series for beginners if you have not read part 1 please follow this link to…**](https://sanchittanwar75.medium.com/markov-chains-and-markov-decision-process-e91cda7fa8f2?source=post_page---author_recirc--bd25b792e488----0---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

Jan 24, 2019

[A clap icon60\\
\\
A response icon1](https://sanchittanwar75.medium.com/markov-chains-and-markov-decision-process-e91cda7fa8f2?source=post_page---author_recirc--bd25b792e488----0---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

![Bellman Equation and dynamic programming](https://miro.medium.com/v2/resize:fit:679/format:webp/1*CiDCpUjj_3mGm3vdGrxu4g.png)

[![Analytics Vidhya](https://miro.medium.com/v2/resize:fill:20:20/1*Qw8AOQSnnlz7SLiwAda2jw.png)](https://medium.com/analytics-vidhya?source=post_page---author_recirc--bd25b792e488----1---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

In

[Analytics Vidhya](https://medium.com/analytics-vidhya?source=post_page---author_recirc--bd25b792e488----1---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

by

[Sanchit Tanwar](https://sanchittanwar75.medium.com/?source=post_page---author_recirc--bd25b792e488----1---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

[**This is a series of articles on reinforcement learning and if you are new and have not studied earlier one please do read(links at the…**](https://sanchittanwar75.medium.com/bellman-equation-and-dynamic-programming-773ce67fc6a7?source=post_page---author_recirc--bd25b792e488----1---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

Jan 28, 2019

[A clap icon90](https://sanchittanwar75.medium.com/bellman-equation-and-dynamic-programming-773ce67fc6a7?source=post_page---author_recirc--bd25b792e488----1---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

![Review: Spatial Pyramid Pooling[1406.4729]](https://miro.medium.com/v2/resize:fit:679/format:webp/1*u2yrYj7SrUffyOAXpGaoUw.jpeg)

[![Analytics Vidhya](https://miro.medium.com/v2/resize:fill:20:20/1*Qw8AOQSnnlz7SLiwAda2jw.png)](https://medium.com/analytics-vidhya?source=post_page---author_recirc--bd25b792e488----2---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

In

[Analytics Vidhya](https://medium.com/analytics-vidhya?source=post_page---author_recirc--bd25b792e488----2---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

by

[Sanchit Tanwar](https://sanchittanwar75.medium.com/?source=post_page---author_recirc--bd25b792e488----2---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

[**Passing variable size input to CNN**](https://sanchittanwar75.medium.com/review-spatial-pyramid-pooling-1406-4729-bfc142988dd2?source=post_page---author_recirc--bd25b792e488----2---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

Apr 18, 2020

[A clap icon32\\
\\
A response icon1](https://sanchittanwar75.medium.com/review-spatial-pyramid-pooling-1406-4729-bfc142988dd2?source=post_page---author_recirc--bd25b792e488----2---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

![Image Augmentation](https://miro.medium.com/v2/resize:fit:679/format:webp/1*1M6C3JDHvPz4oK-q2Q5W4g.png)

[![Analytics Vidhya](https://miro.medium.com/v2/resize:fill:20:20/1*Qw8AOQSnnlz7SLiwAda2jw.png)](https://medium.com/analytics-vidhya?source=post_page---author_recirc--bd25b792e488----3---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

In

[Analytics Vidhya](https://medium.com/analytics-vidhya?source=post_page---author_recirc--bd25b792e488----3---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

by

[Sanchit Tanwar](https://sanchittanwar75.medium.com/?source=post_page---author_recirc--bd25b792e488----3---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

[**Improving Deep learning models**](https://sanchittanwar75.medium.com/image-augmentation-9b7be3972e27?source=post_page---author_recirc--bd25b792e488----3---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

Aug 4, 2021

[A clap icon40\\
\\
A response icon1](https://sanchittanwar75.medium.com/image-augmentation-9b7be3972e27?source=post_page---author_recirc--bd25b792e488----3---------------------ccf1911c_2985_49c0_aa8a_1fc41778ecfe--------------)

[See all from Sanchit Tanwar](https://sanchittanwar75.medium.com/?source=post_page---author_recirc--bd25b792e488---------------------------------------)

## Recommended from Medium

![6 brain images](https://miro.medium.com/v2/resize:fit:679/format:webp/1*Q-mzQNzJSVYkVGgsmHVjfw.png)

[![Write A Catalyst](https://miro.medium.com/v2/resize:fill:20:20/1*KCHN5TM3Ga2PqZHA4hNbaw.png)](https://medium.com/write-a-catalyst?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

In

[Write A Catalyst](https://medium.com/write-a-catalyst?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

by

[Dr. Patricia Schmidt](https://medium.com/@creatorschmidt?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[**Most people do \#1 within 10 minutes of waking (and it sabotages your entire day)**](https://medium.com/@creatorschmidt/as-a-neuroscientist-i-quit-these-5-morning-habits-that-destroy-your-brain-3efe1f410226?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

Jan 14

[A clap icon36K\\
\\
A response icon649](https://medium.com/@creatorschmidt/as-a-neuroscientist-i-quit-these-5-morning-habits-that-destroy-your-brain-3efe1f410226?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

![Stanford Just Killed Prompt Engineering With 8 Words (And I Can’t Believe It Worked)](https://miro.medium.com/v2/resize:fit:679/format:webp/1*va3sFwIm26snbj5ly9ZsgA.jpeg)

[![Generative AI](https://miro.medium.com/v2/resize:fill:20:20/1*M4RBhIRaSSZB7lXfrGlatA.png)](https://generativeai.pub/?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

In

[Generative AI](https://generativeai.pub/?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

by

[Adham Khaled](https://medium.com/@adham__khaled__?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[**ChatGPT keeps giving you the same boring response? This new technique unlocks 2× more creativity from ANY AI model — no training required…**](https://medium.com/@adham__khaled__/stanford-just-killed-prompt-engineering-with-8-words-and-i-cant-believe-it-worked-8349d6524d2b?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

Oct 19, 2025

[A clap icon24K\\
\\
A response icon656](https://medium.com/@adham__khaled__/stanford-just-killed-prompt-engineering-with-8-words-and-i-cant-believe-it-worked-8349d6524d2b?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

![I Stopped Using ChatGPT for 30 Days. What Happened to My Brain Was Terrifying.](https://miro.medium.com/v2/resize:fit:679/format:webp/1*z4UOJs0b33M4UJXq5MXkww.png)

[![Level Up Coding](https://miro.medium.com/v2/resize:fill:20:20/1*5D9oYBd58pyjMkV_5-zXXQ.jpeg)](https://levelup.gitconnected.com/?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

In

[Level Up Coding](https://levelup.gitconnected.com/?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

by

[Teja Kusireddy](https://medium.com/@kusireddy?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[**91% of you will abandon 2026 resolutions by January 10th. Here’s how to be in the 9% who actually win.**](https://medium.com/@kusireddy/i-stopped-using-chatgpt-for-30-days-what-happened-to-my-brain-was-terrifying-70d2a62246c0?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

Dec 28, 2025

[A clap icon9.8K\\
\\
A response icon358](https://medium.com/@kusireddy/i-stopped-using-chatgpt-for-30-days-what-happened-to-my-brain-was-terrifying-70d2a62246c0?source=post_page---read_next_recirc--bd25b792e488----0---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

![The AI Bubble Is About To Burst, But The Next Bubble Is Already Growing](https://miro.medium.com/v2/resize:fit:679/format:webp/0*jQ7Z0Y2Rw8kblsEX)

[![Will Lockett](https://miro.medium.com/v2/resize:fill:20:20/1*V0qWMQ8V5_NaF9yUoHAdyg.jpeg)](https://wlockett.medium.com/?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[Will Lockett](https://wlockett.medium.com/?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[**Techbros are preparing their latest bandwagon.**](https://wlockett.medium.com/the-ai-bubble-is-about-to-burst-but-the-next-bubble-is-already-growing-383c0c0c7ede?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

Sep 14, 2025

[A clap icon23K\\
\\
A response icon996](https://wlockett.medium.com/the-ai-bubble-is-about-to-burst-but-the-next-bubble-is-already-growing-383c0c0c7ede?source=post_page---read_next_recirc--bd25b792e488----1---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

![An example of a perfect, human designed dashboard interface for desktop and mobile phone](https://miro.medium.com/v2/resize:fit:679/format:webp/1*C8RVDKs_uZrVUdgpsF6Fmw.png)

[![Michal Malewicz](https://miro.medium.com/v2/resize:fill:20:20/1*149zXrb2FXvS_mctL4NKSg.png)](https://michalmalewicz.medium.com/?source=post_page---read_next_recirc--bd25b792e488----2---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[Michal Malewicz](https://michalmalewicz.medium.com/?source=post_page---read_next_recirc--bd25b792e488----2---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[**Design is becoming quietly human again.**](https://michalmalewicz.medium.com/the-end-of-dashboards-and-design-systems-5d98ec9de627?source=post_page---read_next_recirc--bd25b792e488----2---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

Nov 26, 2025

[A clap icon6.1K\\
\\
A response icon242](https://michalmalewicz.medium.com/the-end-of-dashboards-and-design-systems-5d98ec9de627?source=post_page---read_next_recirc--bd25b792e488----2---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

![AI Agents: Complete Course](https://miro.medium.com/v2/resize:fit:679/format:webp/1*PvPPSGJ9779FTWmtK_Yeyw.png)

[![Data Science Collective](https://miro.medium.com/v2/resize:fill:20:20/1*0nV0Q-FBHj94Kggq00pG2Q.jpeg)](https://medium.com/data-science-collective?source=post_page---read_next_recirc--bd25b792e488----3---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

In

[Data Science Collective](https://medium.com/data-science-collective?source=post_page---read_next_recirc--bd25b792e488----3---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

by

[Marina Wyss](https://medium.com/@gratitudedriven?source=post_page---read_next_recirc--bd25b792e488----3---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[**From beginner to intermediate to production.**](https://medium.com/@gratitudedriven/ai-agents-complete-course-f226aa4550a1?source=post_page---read_next_recirc--bd25b792e488----3---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

Dec 6, 2025

[A clap icon4.1K\\
\\
A response icon159](https://medium.com/@gratitudedriven/ai-agents-complete-course-f226aa4550a1?source=post_page---read_next_recirc--bd25b792e488----3---------------------9e5280b4_83d9_4700_9ceb_2bf2aa637fb6--------------)

[See more recommendations](https://medium.com/?source=post_page---read_next_recirc--bd25b792e488---------------------------------------)

[Help](https://help.medium.com/hc/en-us?source=post_page-----bd25b792e488---------------------------------------)

[Status](https://status.medium.com/?source=post_page-----bd25b792e488---------------------------------------)

[About](https://medium.com/about?autoplay=1&source=post_page-----bd25b792e488---------------------------------------)

[Careers](https://medium.com/jobs-at-medium/work-at-medium-959d1a85284e?source=post_page-----bd25b792e488---------------------------------------)

[Press](mailto:pressinquiries@medium.com)

[Blog](https://blog.medium.com/?source=post_page-----bd25b792e488---------------------------------------)

[Privacy](https://policy.medium.com/medium-privacy-policy-f03bf92035c9?source=post_page-----bd25b792e488---------------------------------------)

[Rules](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page-----bd25b792e488---------------------------------------)

[Terms](https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----bd25b792e488---------------------------------------)

[Text to speech](https://speechify.com/medium?source=post_page-----bd25b792e488---------------------------------------)

reCAPTCHA

Recaptcha requires verification.

[Privacy](https://www.google.com/intl/en/policies/privacy/) \- [Terms](https://www.google.com/intl/en/policies/terms/)

protected by **reCAPTCHA**

[Privacy](https://www.google.com/intl/en/policies/privacy/) \- [Terms](https://www.google.com/intl/en/policies/terms/)