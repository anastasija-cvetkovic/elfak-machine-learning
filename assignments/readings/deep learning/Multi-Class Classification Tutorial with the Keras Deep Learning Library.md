### [Navigation](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/\#navigation)

[![Go from Data to Strategy: Tepper School of Business](https://machinelearningmastery.com/wp-content/uploads/2023/02/mlm1-cmu-banner-728-1.jpg)](https://www.cmu.edu/tepper/)

[Go from Data to Strategy: Tepper School of Business](https://www.cmu.edu/tepper/)

By[Jason Brownlee](https://machinelearningmastery.com/author/jasonb/ "Posts by Jason Brownlee")onAugust 7, 2022in[Deep Learning](https://machinelearningmastery.com/category/deep-learning/ "View all items in Deep Learning")[614](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comments)

Share _Post_Share

Keras is a Python library for deep learning that wraps the efficient numerical libraries Theano and TensorFlow.

In this tutorial, you will discover how to use Keras to develop and evaluate neural network models for multi-class classification problems.

After completing this step-by-step tutorial, you will know:

- How to load data from CSV and make it available to Keras
- How to prepare multi-class classification data for modeling with neural networks
- How to evaluate Keras neural network models with scikit-learn

**Kick-start your project** with my new book [Deep Learning With Python](https://machinelearningmastery.com/deep-learning-with-python/), including _step-by-step tutorials_ and the _Python source code_ files for all examples.

Let’s get started.

- **Update Oct/2016**: Updated for Keras 1.1.0 and scikit-learn v0.18
- **Update Mar/2017**: Updated for Keras 2.0.2, TensorFlow 1.0.1 and Theano 0.9.0
- **Update Jun/2017**: Updated to use softmax activation in output layer, larger hidden layer, default weight initialization
- **Update Aug/2019**: Added complete working example for convenience, removed random seed
- **Update Sep/2019**: Updated for Keras 2.2.5 API

![Multi-Class Classification Tutorial with the Keras Deep Learning Library](https://machinelearningmastery.com/wp-content/uploads/2016/06/Multi-Class-Classification-Tutorial-with-the-Keras-Deep-Learning-Library.jpg)

Multi-class classification tutorial with the Keras deep learning library

Photo by [houroumono](https://www.flickr.com/photos/hourou/8922014724/), some rights reserved.

## 1\. Problem Description

In this tutorial, you will use the standard machine learning problem called the [iris flowers dataset](http://archive.ics.uci.edu/ml/datasets/Iris).

This dataset is well studied and makes a good problem for practicing on neural networks because all four input variables are numeric and have the same scale in centimeters. Each instance describes the properties of an observed flower’s measurements, and the output variable is a specific iris species.

This is a multi-class classification problem, meaning that there are more than two classes to be predicted. In fact, there are three flower species. This is an important problem for practicing with neural networks because the three class values require specialized handling.

The iris flower dataset is a well-studied problem, and as such, you can [expect to achieve a model accuracy](http://www.is.umk.pl/projects/rules.html#Iris) in the range of 95% to 97%. This provides a good target to aim for when developing your models.

You can [download the iris flowers dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) from the UCI Machine Learning repository and place it in your current working directory with the filename “ _iris.csv_“.

- [Iris Flowers Dataset (iris.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv)

### Need help with Deep Learning in Python?

Take my free 2-week email course and discover MLPs, CNNs and LSTMs (with code).

Click to sign-up now and also get a free PDF Ebook version of the course.

Start Your FREE Mini-Course Now

## 2\. Import Classes and Functions

You can begin by importing all the classes and functions you will need in this tutorial.

This includes both the functionality you require from Keras and the data loading from [pandas](http://pandas.pydata.org/), as well as data preparation and model evaluation from [scikit-learn](http://scikit-learn.org/).

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10 | import pandas<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from keras.wrappers.scikit\_learn import KerasClassifier<br>from keras.utils import np\_utils<br>from sklearn.model\_selection import cross\_val\_score<br>from sklearn.model\_selection import KFold<br>from sklearn.preprocessing import LabelEncoder<br>from sklearn.pipeline import Pipeline<br>... |

## 3\. Load the Dataset

The dataset can be loaded directly. Because the output variable contains strings, it is easiest to load the data using pandas. You can then split the attributes (columns) into input variables (X) and output variables (Y).

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6 | ...<br>\# load dataset<br>dataframe=pandas.read\_csv("iris.csv",header=None)<br>dataset=dataframe.values<br>X=dataset\[:,0:4\].astype(float)<br>Y=dataset\[:,4\] |

## 4\. Encode the Output Variable

The output variable contains three different string values.

When modeling multi-class classification problems using neural networks, it is good practice to reshape the output attribute from a vector that contains values for each class value to a matrix with a Boolean for each class value and whether a given instance has that class value or not.

This is called [one-hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) or creating dummy variables from a categorical variable.

For example, in this problem, three class values are Iris-setosa, Iris-versicolor, and Iris-virginica. If you had the observations:

|     |     |
| --- | --- |
| 1<br>2<br>3 | Iris-setosa<br>Iris-versicolor<br>Iris-virginica |

You can turn this into a one-hot encoded binary matrix for each data instance that would look like this:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4 | Iris-setosa, Iris-versicolor, Iris-virginica<br>1, 0, 0<br>0, 1, 0<br>0, 0, 1 |

You can first encode the strings consistently to integers using the scikit-learn class LabelEncoder. Then convert the vector of integers to a one-hot encoding using the Keras function to\_categorical().

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | ...<br>\# encode class values as integers<br>encoder=LabelEncoder()<br>encoder.fit(Y)<br>encoded\_Y=encoder.transform(Y)<br>\# convert integers to dummy variables (i.e. one hot encoded)<br>dummy\_y=np\_utils.to\_categorical(encoded\_Y) |

## 5\. Define the Neural Network Model

If you are new to Keras or deep learning, see this [helpful Keras tutorial](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/).

The Keras library provides wrapper classes to allow you to use neural network models developed with Keras in scikit-learn.

There is a KerasClassifier class in Keras that can be used as an Estimator in scikit-learn, the base type of model in the library. The KerasClassifier takes the name of a function as an argument. This function must return the constructed neural network model, ready for training.

Below is a function that will create a baseline neural network for the iris classification problem. It creates a simple, fully connected network with one hidden layer that contains eight neurons.

The hidden layer uses a rectifier activation function which is a good practice. Because you used a one-hot encoding for your iris dataset, the output layer must create three output values, one for each class. The output value with the largest value will be taken as the class predicted by the model.

The network topology of this simple one-layer neural network can be summarized as follows:

|     |     |
| --- | --- |
| 1 | 4 inputs -> \[8 hidden nodes\] -> 3 outputs |

Note that a “ _softmax_” activation function was used in the output layer. This ensures the output values are in the range of 0 and 1 and may be used as predicted probabilities.

Finally, the network uses the efficient Adam gradient descent optimization algorithm with a logarithmic loss function, which is called “ _categorical\_crossentropy_” in Keras.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10 | ...<br>\# define baseline model<br>def baseline\_model():<br>\# create model<br>model=Sequential()<br>model.add(Dense(8,input\_dim=4,activation='relu'))<br>model.add(Dense(3,activation='softmax'))<br>\# Compile model<br>model.compile(loss='categorical\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>returnmodel |

You can now create your KerasClassifier for use in scikit-learn.

You can also pass arguments in the construction of the KerasClassifier class that will be passed on to the fit() function internally used to train the neural network. Here, you pass the number of epochs as 200 and batch size as 5 to use when training the model. Debugging is also turned off when training by setting verbose to 0.

|     |     |
| --- | --- |
| 1<br>2 | ...<br>estimator=KerasClassifier(build\_fn=baseline\_model,epochs=200,batch\_size=5,verbose=0) |

## 6\. Evaluate the Model with k-Fold Cross Validation

You can now evaluate the neural network model on our training data.

The scikit-learn has excellent capability to evaluate models using a suite of techniques. The gold standard for evaluating machine learning models is k-fold cross validation.

First, define the model evaluation procedure. Here, you set the number of folds to 10 (an excellent default) and shuffle the data before partitioning it.

|     |     |
| --- | --- |
| 1<br>2 | ...<br>kfold=KFold(n\_splits=10,shuffle=True) |

Now, you can evaluate your model (estimator) on your dataset (X and dummy\_y) using a 10-fold cross-validation procedure (k-fold).

Evaluating the model only takes approximately 10 seconds and returns an object that describes the evaluation of the ten constructed models for each of the splits of the dataset.

|     |     |
| --- | --- |
| 1<br>2<br>3 | ...<br>results=cross\_val\_score(estimator,X,dummy\_y,cv=kfold)<br>print("Baseline: %.2f%% (%.2f%%)"%(results.mean()\*100,results.std()\*100)) |

## 7\. Complete Example

You can tie all of this together into a single program that you can save and run as a script:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36 | \# multi-class classification with Keras<br>import pandas<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from keras.wrappers.scikit\_learn import KerasClassifier<br>from keras.utils import np\_utils<br>from sklearn.model\_selection import cross\_val\_score<br>from sklearn.model\_selection import KFold<br>from sklearn.preprocessing import LabelEncoder<br>from sklearn.pipeline import Pipeline<br>\# load dataset<br>dataframe=pandas.read\_csv("iris.data",header=None)<br>dataset=dataframe.values<br>X=dataset\[:,0:4\].astype(float)<br>Y=dataset\[:,4\]<br>\# encode class values as integers<br>encoder=LabelEncoder()<br>encoder.fit(Y)<br>encoded\_Y=encoder.transform(Y)<br>\# convert integers to dummy variables (i.e. one hot encoded)<br>dummy\_y=np\_utils.to\_categorical(encoded\_Y)<br>\# define baseline model<br>def baseline\_model():<br>\# create model<br>model=Sequential()<br>model.add(Dense(8,input\_dim=4,activation='relu'))<br>model.add(Dense(3,activation='softmax'))<br>\# Compile model<br>model.compile(loss='categorical\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>returnmodel<br>estimator=KerasClassifier(build\_fn=baseline\_model,epochs=200,batch\_size=5,verbose=0)<br>kfold=KFold(n\_splits=10,shuffle=True)<br>results=cross\_val\_score(estimator,X,dummy\_y,cv=kfold)<br>print("Baseline: %.2f%% (%.2f%%)"%(results.mean()\*100,results.std()\*100)) |

The results are summarized as both the mean and standard deviation of the model accuracy on the dataset.

**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

This is a reasonable estimation of the performance of the model on unseen data. It is also within the realm of known top results for this problem.

|     |     |
| --- | --- |
| 1 | Accuracy: 97.33% (4.42%) |

## Summary

In this post, you discovered how to develop and evaluate a neural network using the Keras Python library for deep learning.

By completing this tutorial, you learned:

- How to load data and make it available to Keras
- How to prepare multi-class classification data for modeling using one-hot encoding
- How to use Keras neural network models with scikit-learn
- How to define a neural network using Keras for multi-class classification
- How to evaluate a Keras neural network model using scikit-learn with k-fold cross validation

Do you have any questions about deep learning with Keras or this post?

Ask your questions in the comments below, and I will do my best to answer them.

Share _Post_Share

### More On This Topic

- [![Binary Classification Worked Example with the Keras Deep Learning Library](https://machinelearningmastery.com/wp-content/uploads/2016/06/Binary-Classification-Worked-Example-with-the-Keras-Deep-Learning-Library.jpg)Binary Classification Tutorial with the Keras Deep…](https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/)
- [![Regression Tutorial with Keras Deep Learning Library in Python](https://machinelearningmastery.com/wp-content/uploads/2016/06/Regression-Tutorial-with-Keras-Deep-Learning-Library-in-Python.jpg)Regression Tutorial with the Keras Deep Learning…](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)
- [![Small Sample of CIFAR-10 Images](https://machinelearningmastery.com/wp-content/uploads/2016/05/cifar10-sample-images.png)Object Classification with CNNs Using the Keras Deep…](https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/)
- [![Learning Curves of Cross-Entropy Loss for a Deep Learning Model](https://machinelearningmastery.com/wp-content/uploads/2019/12/Learning-Curves-of-Cross-Entropy-Loss-for-a-Deep-Learning-Model.png)TensorFlow 2 Tutorial: Get Started in Deep Learning…](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)
- [![Box and Whisker Plot of Machine Learning Models on the Imbalanced Glass Identification Dataset](https://machinelearningmastery.com/wp-content/uploads/2019/12/Box-and-Whisker-Plot-of-Machine-Learning-Models-on-the-Imbalanced-Glass-Identification-Dataset.png)Imbalanced Multiclass Classification with the Glass…](https://machinelearningmastery.com/imbalanced-multiclass-classification-with-the-glass-identification-dataset/)
- [![Histogram of Variables in the E.coli Dataset](https://machinelearningmastery.com/wp-content/uploads/2019/12/Histogram-of-Variables-in-the-E.coli-Dataset.png)Imbalanced Multiclass Classification with the E.coli Dataset](https://machinelearningmastery.com/imbalanced-multiclass-classification-with-the-e-coli-dataset/)

[How To Compare Machine Learning Algorithms in Python with scikit-learn](https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/)

[Ensemble Machine Learning Algorithms in Python with scikit-learn](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)

### 614 Responses to _Multi-Class Classification Tutorial with the Keras Deep Learning Library_

001. ![](https://secure.gravatar.com/avatar/461d9069ad8f4cb1d4eff299dd870ee6f6ac8d3ab6f724076483755510d6fae4?s=40&d=mm&r=g)



     JackJune 19, 2016 at 3:12 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-354404 "Direct link to this comment")





     Thanks for this cool tutorial! I have a question about the input data. If the datatypes of input variables are different (i.e. string and numeric). How to preprocess the train data to fit keras?



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-354404)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 20, 2016 at 5:41 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-354449 "Direct link to this comment")





       Great question. Eventually, all of the data need to be turned into real values.



       With categorical variables, you can create dummy variables and use one-hot encoding. For string data, you can use word embeddings.



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-354449)




       - ![](https://secure.gravatar.com/avatar/6c438da07d986024fbe08efb3aeba638faa0e15bf4185c208265318001b6574d?s=40&d=mm&r=g)



         ShraddhaFebruary 10, 2017 at 8:32 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-387003 "Direct link to this comment")





         Could you please let me know how to convert string data into word embeddings in large datasets?


         Would really appreciate it


         Thanks so much



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-387003)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)February 11, 2017 at 5:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-387107 "Direct link to this comment")





           Hi Shraddha,



           First, convert the chars to vectors of integers. You can then pad all vectors to the same length. Then away you go.



           I hope that helps.



           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-387107)




           - ![](https://secure.gravatar.com/avatar/6c438da07d986024fbe08efb3aeba638faa0e15bf4185c208265318001b6574d?s=40&d=mm&r=g)



             Shraddha SunilFebruary 13, 2017 at 4:52 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-387717 "Direct link to this comment")





             Thanks so much Jason!

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)February 14, 2017 at 10:04 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-387882 "Direct link to this comment")





             You’re welcome.

           - ![](https://secure.gravatar.com/avatar/75b1d156dd1fb698de382cc3cafc4f9ac035e7fe7cc548ae13caab39709912ff?s=40&d=mm&r=g)



             sasiAugust 5, 2017 at 7:51 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-408640 "Direct link to this comment")





             can you give an example for that..

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)August 6, 2017 at 7:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-408692 "Direct link to this comment")





             I have many tutorials for encoding and padding sequences on the blog. Please use the search.
       - ![](https://secure.gravatar.com/avatar/d08d78601b3105cb21c02afc54b0ae89ef676e3a15ed7d554fd04d57f6fd59b3?s=40&d=mm&r=g)



         ChandanFebruary 14, 2019 at 3:17 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-468315 "Direct link to this comment")





         query:



         which type of properties of an observed flower measurements is taken


         Told me what is the 4 attributes, you taken



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-468315)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)February 15, 2019 at 7:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-468507 "Direct link to this comment")





           For more on the dataset, see this post:

           [https://en.wikipedia.org/wiki/Iris\_flower\_data\_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)



           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-468507)

         - ![](https://secure.gravatar.com/avatar/2dc0bc1b94d9bbe25f2b48844ab483176d3d1cc9b54b74dff9a42700a725f6aa?s=40&d=mm&r=g)



           Manohar NookalaMarch 22, 2020 at 12:06 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-526358 "Direct link to this comment")





           Class indices are 7. Then how manu output variables i need to mentions



           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-526358)
       - ![](https://secure.gravatar.com/avatar/0a2cdd99d95a91ba16f2791ca2ef7fe38394d33f8d79322f2c1901b903302c99?s=40&d=mm&r=g)



         SJJuly 16, 2020 at 3:44 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-544490 "Direct link to this comment")





         IF i choose to use Entity Embeddings for categorical data, can you please suggest how to feed them to a MLP. I am able to do that in pytorch by using your article on pytorch.


         Can you please suggest how to convert the below architecture into an MLP.



         (all\_embeddings): ModuleList(


         (0): Embedding(24, 12)


         (1): Embedding(2, 1)


         (2): Embedding(7, 4)


         )


         (embedding\_dropout): Dropout(p=0.4, inplace=False)


         (batch\_norm\_num): BatchNorm1d(7, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)


         (layers): Sequential(


         (0): Linear(in\_features=24, out\_features=200, bias=True)


         (1): ReLU(inplace=True)


         (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)


         (3): Dropout(p=0.4, inplace=False)


         (4): Linear(in\_features=200, out\_features=100, bias=True)


         (5): ReLU(inplace=True)


         (6): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)


         (7): Dropout(p=0.4, inplace=False)


         (8): Linear(in\_features=100, out\_features=1, bias=True)


         )


         )



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-544490)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)July 17, 2020 at 6:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-544578 "Direct link to this comment")





           sorry, I don’t have an example for pytorch, but I have an example for keras that might help:

           [https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/](https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/)



           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-544578)
       - ![](https://secure.gravatar.com/avatar/bdff9a830705a5ec12dc545e75ce29b2f3162c4f3d25915117e21f50a12b949e?s=40&d=mm&r=g)



         LamOctober 27, 2021 at 1:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-632506 "Direct link to this comment")





         A great tutorial. What should I do to preprocess mixed input data (data includes both numeric and categorical variables) for the fitting model? Thanks in advance.



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-632506)




         - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



           Adrian TamOctober 27, 2021 at 3:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-632574 "Direct link to this comment")





           Numeric usually just presented as-is, but sometimes we apply scaling to it too. Categorical is usually one-hot encoded.



           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-632574)
     - ![](https://secure.gravatar.com/avatar/b1e74986d5e6d159ef8e1cb761a376b5f0fd814d7fae8b395b397283df4374dd?s=40&d=mm&r=g)



       Harale Vandana RangraoApril 21, 2018 at 5:55 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435584 "Direct link to this comment")





       Thank you very much, sir, for sharing so much information, but sir I want to a dataset of greenhouse for tomato crop with climate variable like Temperature, Humidity, Soil Moisture, pH Scale, CO2, Light Intensity. Can you provide me this type dataset?



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435584)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)April 22, 2018 at 5:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435620 "Direct link to this comment")





         I answer this question here:

         [https://machinelearningmastery.com/faq/single-faq/where-can-i-get-a-dataset-on-\_\_\_](https://machinelearningmastery.com/faq/single-faq/where-can-i-get-a-dataset-on-___)



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435620)
     - ![](https://secure.gravatar.com/avatar/efb30200f44f89f6fb6eed750f87d3dd0728750d0e180c937bbe68961be117f9?s=40&d=mm&r=g)



       RameshApril 17, 2019 at 7:00 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-480915 "Direct link to this comment")





       Hey,


       Can we use this module for array of string



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-480915)




       - ![](https://secure.gravatar.com/avatar/efb30200f44f89f6fb6eed750f87d3dd0728750d0e180c937bbe68961be117f9?s=40&d=mm&r=g)



         RameshApril 17, 2019 at 7:02 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-480917 "Direct link to this comment")





         Hey can we use this method for arrays



         array(\[\[”, u’multios’, u’dsdsds’, u’DW\_SAL\_CANNOT\_INITIALIZE’, u’av\_sw’\],\
\
\
         \[”, u’android-l’, u’dsssd’, u’SYS\_SW’, u’syssw’\],\
\
\
         \[”, u’gnu\_linux-k4.9′, u’dssss’, u’USB\_IO\_Error’, u’syssw’\],\
\
\
         …,\
\
\
         \[”, u’android-p’, u’fddfdfdf’, u’mm\_nvmedia\_video\_decoder\_create’,\
\
\
         u’multimedia’\],\
\
\
         \[”, u’android-o’, u’sasa’, u’mm\_log\_tag’,\
\
\
         u’multimedia’\],\
\
\
         \[u’rel-32′, u’android-p’, u’dsdsd’,\
\
\
         u’mm\_parsevp9\_incorrect\_sync\_code\_for\_vp9′, u’multimedia’\]\],


         dtype=object)



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-480917)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)April 18, 2019 at 8:25 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-481016 "Direct link to this comment")





           I would recommend using a bag of words model when starting with text:

           [https://machinelearningmastery.com/gentle-introduction-bag-words-model/](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)



           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-481016)
     - ![](https://secure.gravatar.com/avatar/1b2498f8fd6debbf48762fb6269123c592004b64d61956f401ec85c64b2f7ea2?s=40&d=mm&r=g)



       hieund198September 7, 2019 at 12:57 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500186 "Direct link to this comment")





       Hi Mr Jason,


       What’s name the model you use to train?


       Sorry, I am newbie.


       Thanks



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500186)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)September 7, 2019 at 5:35 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500237 "Direct link to this comment")





         The model in this tutorial a neural network or a multilayer neural network, often called an MLP or a fully connected network.



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500237)




         - ![](https://secure.gravatar.com/avatar/41dabe568699b92b664f2a3fe5fb89eb8ca45e91dee91cc3fadfeea5e5eeaac7?s=40&d=mm&r=g)



           hieund198September 11, 2019 at 3:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500916 "Direct link to this comment")





           Dear Mr Jason,



           I run your example code I noticed that softmax in your tutorial has different result with softmax used in CNN model.


           I would like to confirm with you this is a behavior of CNN



           Exemple my code:



           model = Sequential()


           model.add(Conv1D(64, 3, activation=’relu’, input\_shape=(8,1)))


           model.add(Conv1D(64, 3, activation=’relu’))


           model.add(Dropout(0.5))


           model.add(MaxPooling1D())


           model.add(Flatten())


           model.add(Dense(100, activation=’relu’))


           model.add(Dense(4, activation=’softmax’))


           model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])



           X\_train, X\_test, Y\_train, Y\_test = train\_test\_split(X, dummy\_y, test\_size=0.33, random\_state=seed)


           model.fit(X\_train, Y\_train)


           predictions = model.predict(X\_test)


           print(predictions)



           >\> Out put



           \[\[0.5863281 0.11777738 0.16206734 0.13382716\]\
\
\
           \[0.5863281 0.11777738 0.16206734 0.13382716\]\
\
\
           \[0.39733416 0.19241211 0.2283105 0.1819432 \]\
\
\
           \[0.54646176 0.12707633 0.20596607 0.12049587\]\
\
\
\
           I think that softmax in CNN model will return % for each result need to classification.\
\
\
\
           And your model will return value dummy\_y prediction.\
\
\
\
           Thank you\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500916)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)September 11, 2019 at 5:43 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500936 "Direct link to this comment")\
\
\
\
\
\
             The softmax is a standard implementation.\
\
\
\
             Perhaps I don’t follow, what is the problem you have exactly?\
\
           - ![](https://secure.gravatar.com/avatar/1b2498f8fd6debbf48762fb6269123c592004b64d61956f401ec85c64b2f7ea2?s=40&d=mm&r=g)\
\
\
\
             hieund1994September 11, 2019 at 12:41 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500979 "Direct link to this comment")\
\
\
\
\
\
             I got found solution from another article of you.\
\
\
             Thanks\
\
\
\
             “First, the raw 17-element prediction vector is printed. If we wish, we could pretty-print this vector and summarize the predicted confidence that the photo would be assigned each label.\
\
\
\
             Next, the prediction is rounded and the vector indexes that contain a 1 value are reverse-mapped to their tag string values. The predicted tags are then printed. we can see that the model has correctly predicted the known tags for the provided photo.\
\
\
\
             It might be interesting to repeat this test with an entirely new photo, such as a photo from the test dataset, after you have already manually suggested tags.\
\
\
\
             \[9.0940112e-01 3.6541668e-03 1.5959743e-02 6.8241461e-05 8.5694155e-05\
\
\
             9.9828100e-01 7.4096164e-08 5.5998818e-05 3.6668104e-01 1.2538023e-01\
\
\
             4.6371704e-04 3.7660234e-04 9.9999273e-01 1.9014676e-01 5.6060363e-04\
\
\
             1.4613305e-03 9.5227945e-01\]\
\
\
\
             \[‘agriculture’, ‘clear’, ‘primary’, ‘water’\] ”\
\
\
\
             [https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)September 11, 2019 at 2:29 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500994 "Direct link to this comment")\
\
\
\
\
\
             Happy to hear that.\
     - ![](https://secure.gravatar.com/avatar/989f6bb13724e3268a2c75e6310c4ab8d29507ea4a6954ddf3229e550f2f690f?s=40&d=mm&r=g)\
\
\
\
       PreetkaranJanuary 17, 2020 at 6:42 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-518268 "Direct link to this comment")\
\
\
\
\
\
       Hi Jason\
\
\
\
       I’m doing an image localization and classification task on Keras-FRCNN, on Theano Backend. I’m getting the following error:\
\
\
\
       Traceback (most recent call last):\
\
\
       File “train\_frcnn.py”, line 208, in\
\
\
       model\_classifier.compile(optimizer=optimizer\_classifier, loss=\[losses.class\_loss\_cls, losses.class\_loss\_regr(len(classes\_count)-1)\], metrics={‘dense\_class\_{}’.format(len(classes\_count)): ‘accuracy’})\
\
\
       File “C:\\Users\\singh\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py”, line 229, in compile\
\
\
       self.total\_loss = self.\_prepare\_total\_loss(masks)\
\
\
       File “C:\\Users\\singh\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py”, line 692, in \_prepare\_total\_loss\
\
\
       y\_true, y\_pred, sample\_weight=sample\_weight)\
\
\
       File “C:\\Users\\singh\\Anaconda3\\lib\\site-packages\\keras\\losses.py”, line 71, in \_\_call\_\_\
\
\
       losses = self.call(y\_true, y\_pred)\
\
\
       File “C:\\Users\\singh\\Anaconda3\\lib\\site-packages\\keras\\losses.py”, line 132, in call\
\
\
       return self.fn(y\_true, y\_pred, \*\*self.\_fn\_kwargs)\
\
\
       File “F:\\ML\\keras-frcnn-moded\\keras\_frcnn\\losses.py”, line 55, in class\_loss\_cls\
\
\
       return lambda\_cls\_class\*K.mean(categorical\_crossentropy(y\_true\[0, :, :\], y\_pred\[0, :, :\]))\
\
\
       File “C:\\Users\\singh\\Anaconda3\\lib\\site-packages\\keras\\losses.py”, line 691, in categorical\_crossentropy\
\
\
       return K.categorical\_crossentropy(y\_true, y\_pred, from\_logits=from\_logits)\
\
\
       File “C:\\Users\\singh\\Anaconda3\\lib\\site-packages\\keras\\backend\\theano\_backend.py”, line 1831, in categorical\_crossentropy\
\
\
       output\_dimensions = list(range(len(int\_shape(output))))\
\
\
       TypeError: object of type ‘NoneType’ has no len()\
\
\
\
       When I use Tensorflow backend, then I don’t face this error. So, I think it’s something related to Keras and Theano. Using tensorflow as keras backend serves useful but it’s quite slow for the model (takes days for training).\
\
\
\
       Any clue/fix for the issue, will be very helpful…..\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-518268)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)January 17, 2020 at 1:48 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-518286 "Direct link to this comment")\
\
\
\
\
\
         Perhaps post your code and error to stackoverflow?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-518286)\
002. ![](https://secure.gravatar.com/avatar/7a27627e116f4cba0488887ce6fa2a5466acbef50eb67c1d390e26b7b97a7061?s=40&d=mm&r=g)\
\
\
\
     Aakash NainJuly 4, 2016 at 2:25 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355669 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
     It’s a very nice tutorial to learn. I implemented the same model but on my work station I achieved a score of 88.67% only. After modifying the number of hidden layers, I achieved an accuracy of 93.04%. But I am not able to achieve the score of 95% or above. Any particular reason behind it ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355669)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2016 at 6:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355770 "Direct link to this comment")\
\
\
\
\
\
       Interesting Aakash.\
\
\
\
       I used the Theano backend. Are you using the same?\
\
\
\
       Are all your libraries up to date? (Keras, Theano, NumPy, etc…)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355770)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/7a27627e116f4cba0488887ce6fa2a5466acbef50eb67c1d390e26b7b97a7061?s=40&d=mm&r=g)\
\
\
\
         Aakash NainJuly 7, 2016 at 12:03 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355856 "Direct link to this comment")\
\
\
\
\
\
         Yes Jason . Backend is theano and all libraries are up to date.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355856)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 7, 2016 at 9:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355908 "Direct link to this comment")\
\
\
\
\
\
           Interesting. Perhaps seeding the random number generator is not having the desired effect for reproducibility. It perhaps it has different effects on different platforms.\
\
\
\
           Perhaps re-run the above code example a few times and see the spread of accuracy scores you achieve?\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355908)\
     - ![](https://secure.gravatar.com/avatar/1b2498f8fd6debbf48762fb6269123c592004b64d61956f401ec85c64b2f7ea2?s=40&d=mm&r=g)\
\
\
\
       hieund1994September 11, 2019 at 10:57 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500971 "Direct link to this comment")\
\
\
\
\
\
       Because Label I use LabelEncoder() to endcoe label.\
\
\
\
       I could not encoder.inverse\_transform(predictions)\
\
\
\
       Expected :output must be follow format:\
\
\
\
       —\
\
\
       \[\[1 0 0 0\]\
\
\
       \[1 0 0 0\]\
\
\
       \[1 0 0 0\]\
\
\
       —\
\
\
       But current output is:\
\
\
\
       \[\[0.5863281 0.11777738 0.16206734 0.13382716\]\
\
\
       \[0.5863281 0.11777738 0.16206734 0.13382716\]\
\
\
       \[0.39733416 0.19241211 0.2283105 0.1819432 \]\
\
\
\
       So I can not encoder.inverse\_transform(predictions)\
\
\
\
       Are you have any suggest?\
\
\
\
       Thank you\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500971)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)September 11, 2019 at 2:29 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500993 "Direct link to this comment")\
\
\
\
\
\
         First, you must reverse the prediction to an integer via argmax, then integer to category via the inverse\_transform.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500993)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/fa7c3c2818ac92cc9097e3fad0a70a1a76cde2e1a246a1d51ce8cb65161e6aae?s=40&d=mm&r=g)\
\
\
\
           Mbonu ChineduApril 24, 2020 at 2:34 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-531503 "Direct link to this comment")\
\
\
\
\
\
           lols, exactly !!!!!\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-531503)\
003. ![](https://secure.gravatar.com/avatar/908c82db1568b6f40b6d41afe565ebe88b9dea6480ce4c7313bc448e989be606?s=40&d=mm&r=g)\
\
\
\
     La Tuan NghiaJuly 6, 2016 at 1:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355738 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     In chapter 10 of the book “Deep Learning With Python”, there is a fraction of code:\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200, batch\_size=5, verbose=0)\
\
\
     kfold = KFold(n=len(X), n\_folds=10, shuffle=True, random\_state=seed)\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
     print(“Accuracy: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     How to save this model and weights to file, then how to load these file to predict a new input data?\
\
\
\
     Many thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355738)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2016 at 6:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355769 "Direct link to this comment")\
\
\
\
\
\
       Really good question.\
\
\
\
       Keras does provide functions to save network weights to HDF5 and network structure to JSON or YAML. The problem is, once you wrap the network in a scikit-learn classifier, how do you access the model and save it. Or can you save the whole wrapped model.\
\
\
\
       Perhaps a simple but inefficient place to start would be to try and simply pickle the whole classifier?\
\
       [https://docs.python.org/2/library/pickle.html](https://docs.python.org/2/library/pickle.html)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-355769)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/08e3a1ef4bf69be0be41d7a2d30a6371cc80c0012635a284c89d89a69feaef8c?s=40&d=mm&r=g)\
\
\
\
         Constantin WeisserJuly 30, 2016 at 4:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358321 "Direct link to this comment")\
\
\
\
\
\
         I tried doing that. It works for a normal sklearn classifier, but apparently not for a Keras Classifier:\
\
\
\
         import pickle\
\
\
         with open(“name.p”,”wb”) as fw:\
\
\
         pickle.dump(clf,fw)\
\
\
\
         with open(name+”.p”,”rb”) as fr:\
\
\
         clf\_saved = pickle.load(fr)\
\
\
         print(clf\_saved)\
\
\
\
         prob\_pred=clf\_saved.predict\_proba(X\_test)\[:,1\]\
\
\
\
         This gives:\
\
\
\
         theano.gof.fg.MissingInputError: An input of the graph, used to compute DimShuffle{x,x}(keras\_learning\_phase), was not provided and not given a value.Use the Theano flag exception\_verbosity=’high’,for more information on this error.\
\
\
\
         Backtrace when the variable is created:\
\
\
         File “nn\_systematics\_I\_evaluation\_of\_optimised\_classifiers.py”, line 6, in\
\
\
         import classifier\_eval\_simplified\
\
\
         File “../../../../classifier\_eval\_simplified.py”, line 26, in\
\
\
         from keras.utils import np\_utils\
\
\
         File “/usr/local/lib/python2.7/site-packages/keras/\_\_init\_\_.py”, line 2, in\
\
\
         from . import backend\
\
\
         File “/usr/local/lib/python2.7/site-packages/keras/backend/\_\_init\_\_.py”, line 56, in\
\
\
         from .theano\_backend import \*\
\
\
         File “/usr/local/lib/python2.7/site-packages/keras/backend/theano\_backend.py”, line 17, in\
\
\
         \_LEARNING\_PHASE = T.scalar(dtype=’uint8′, name=’keras\_learning\_phase’) # 0 = test, 1 = train\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358321)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 30, 2016 at 7:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358330 "Direct link to this comment")\
\
\
\
\
\
           I provide examples of saving and loading Keras models here:\
\
           [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
           Sorry, I don’t have any examples of saving/loading the wrapped Keras classifier. Perhaps the internal model can be seralized and later deserialized and put back inside the wrapper.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358330)\
004. ![](https://secure.gravatar.com/avatar/61b63c96b485b88adcb84a23aeadc5d5a295bdedc28bad7c6ad049ca0c3097cb?s=40&d=mm&r=g)\
\
\
\
     SallyJuly 15, 2016 at 4:10 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-356849 "Direct link to this comment")\
\
\
\
\
\
     Dear Dr. Jason,\
\
\
\
     Thanks very much for this great tutorial . I got extra benefit from it, but I need to calculate precision, recall and confusion matrix for such multi-class classification. I tried to did it but each time I got a different problem. could you please explain me how to do this\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-356849)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 15, 2016 at 9:04 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-356873 "Direct link to this comment")\
\
\
\
\
\
       Hi Sally, you could perhaps use the tools in scikit-learn to summarize the performance of your model.\
\
\
\
       For example, you could use sklearn.metrics.confusion\_matrix() to calculate the confusion matrix for predictions, etc.\
\
\
\
       See the metrics package:\
\
       [http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-356873)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/616bf1c70d7fac0f1ae02c428c1acfc89edf2595abb67d54cbfad0928826e4a9?s=40&d=mm&r=g)\
\
\
\
         PrabhatApril 13, 2018 at 5:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434874 "Direct link to this comment")\
\
\
\
\
\
         Could you tell how to use that in this code you have provided above? I am very new Keras.\
\
\
\
         Thanks in Advance\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434874)\
     - ![](https://secure.gravatar.com/avatar/4a72e33d9c282a4b10217cba4f311d0bf8e070ee693e5d212f817855278f949d?s=40&d=mm&r=g)\
\
\
\
       olfaAugust 3, 2018 at 11:23 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445175 "Direct link to this comment")\
\
\
\
\
\
       please how we can implemente python code using recall and precision to evaluate prediction model\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445175)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)August 4, 2018 at 6:11 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445204 "Direct link to this comment")\
\
\
\
\
\
         You can use the sklearn library to calculate these scores:\
\
         [http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445204)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/2086756c4b944b1f492b9ae6e250cd23049541dcb54ac8b5623f3deba6d5db17?s=40&d=mm&r=g)\
\
\
\
           Kaddy S.January 28, 2020 at 9:29 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-519603 "Direct link to this comment")\
\
\
\
\
\
           Hi jason.. your tutorials are a great help.. i am a student working on deep learning for detection of diabetic retinopathy and its stages.. using the code u gave for multi class, for my dataset.. i am getting a very low baseline.. 23%..can help me on improving the accuracy.. also how to classify images using deep learning?\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-519603)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)January 29, 2020 at 6:35 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-519655 "Direct link to this comment")\
\
\
\
\
\
             Thanks!\
\
\
\
             Yes, this will give you ideas:\
\
             [https://machinelearningmastery.com/start-here/#better](https://machinelearningmastery.com/start-here/#better)\
005. ![](https://secure.gravatar.com/avatar/461dfa11ae766f19f98b8913bf3672f500e6c4ada992563249451f2ffacd0833?s=40&d=mm&r=g)\
\
\
\
     [Fabian Leon](http://www.sundevs.com/)July 31, 2016 at 4:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358407 "Direct link to this comment")\
\
\
\
\
\
     Hi jason, Reading the tutorial and the same example in your book, you still don’t tell us how can use the model to make predictions, you have only show us how to train and evaluate it but I would like to see you using this model to make predictions on at least one example of iris flowers data no matters if is dummy data.\
\
\
\
     I would like to see how can I load my own instance of an iris-flower and use the above model to predict what kind is the flower?\
\
\
\
     could you do that for us?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358407)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 31, 2016 at 7:31 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358418 "Direct link to this comment")\
\
\
\
\
\
       Hi Fabian, no problem.\
\
\
\
       In the tutorial above, we are using the scikit-learn wrapper. That means we can use the standard model.predict() function to make predictions from scikit-learn.\
\
\
\
       For example, below is an an example adapted from the above where we split the dataset, train on 67% and make predictions on 33%. Remember that we have encoded the output class value as integers, so the predictions are integers. We can then use encoder.inverse\_transform() to turn the predicted integers back into strings.\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       |     |     |\
       | --- | --- |\
       | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38 | \# Train model and make predictions<br>import numpy<br>import pandas<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from keras.wrappers.scikit\_learn import KerasClassifier<br>from keras.utils import np\_utils<br>from sklearn.cross\_validation import train\_test\_split<br>from sklearn.preprocessing import LabelEncoder<br>\# fix random seed for reproducibility<br>seed=7<br>numpy.random.seed(seed)<br>\# load dataset<br>dataframe=pandas.read\_csv("iris.csv",header=None)<br>dataset=dataframe.values<br>X=dataset\[:,0:4\].astype(float)<br>Y=dataset\[:,4\]<br>\# encode class values as integers<br>encoder=LabelEncoder()<br>encoder.fit(Y)<br>encoded\_Y=encoder.transform(Y)<br>\# convert integers to dummy variables (i.e. one hot encoded)<br>dummy\_y=np\_utils.to\_categorical(encoded\_Y)<br>\# define baseline model<br>def baseline\_model():<br>\# create model<br>model=Sequential()<br>model.add(Dense(4,input\_dim=4,init='normal',activation='relu'))<br>model.add(Dense(3,init='normal',activation='sigmoid'))<br>\# Compile model<br>model.compile(loss='categorical\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>returnmodel<br>estimator=KerasClassifier(build\_fn=baseline\_model,nb\_epoch=200,batch\_size=5,verbose=0)<br>X\_train,X\_test,Y\_train,Y\_test=train\_test\_split(X,dummy\_y,test\_size=0.33,random\_state=seed)<br>estimator.fit(X\_train,Y\_train)<br>predictions=estimator.predict(X\_test)<br>print(predictions)<br>print(encoder.inverse\_transform(predictions)) |\
\
\
\
\
\
\
\
\
\
\
\
       Running this example prints:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       |     |     |\
       | --- | --- |\
       | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15 | \[2101201101210202220012122211222102100<br>0022122101120\]<br>\['Iris-virginica''Iris-versicolor''Iris-setosa''Iris-versicolor'<br>'Iris-virginica''Iris-setosa''Iris-versicolor''Iris-versicolor'<br>'Iris-setosa''Iris-versicolor''Iris-virginica''Iris-versicolor'<br>'Iris-setosa''Iris-virginica''Iris-setosa''Iris-virginica'<br>'Iris-virginica''Iris-virginica''Iris-setosa''Iris-setosa'<br>'Iris-versicolor''Iris-virginica''Iris-versicolor''Iris-virginica'<br>'Iris-virginica''Iris-virginica''Iris-versicolor''Iris-versicolor'<br>'Iris-virginica''Iris-virginica''Iris-virginica''Iris-versicolor'<br>'Iris-setosa''Iris-virginica''Iris-versicolor''Iris-setosa'<br>'Iris-setosa''Iris-setosa''Iris-setosa''Iris-virginica'<br>'Iris-virginica''Iris-versicolor''Iris-virginica''Iris-virginica'<br>'Iris-versicolor''Iris-setosa''Iris-versicolor''Iris-versicolor'<br>'Iris-virginica''Iris-setosa'\] |\
\
\
\
\
\
\
\
\
\
\
\
       I hope that is clear and useful. Let me know if you have any more questions.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-358418)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/f4e6edda8349a0f1d81a4b436034d96331810d5c1ace001d1d5a37494eb63e8f?s=40&d=mm&r=g)\
\
\
\
         DevendraNovember 27, 2016 at 9:40 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-372607 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
\
         I was facing error while converting string to float and so I had to make a minor correction to my code\
\
\
         X = dataset\[1:,0:4\].astype(float)\
\
\
         Y = dataset\[1:,4\]\
\
\
\
         However, I am still unable to run since I am getting the following error for line\
\
\
\
         “—-\> 1 results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)”\
\
\
         ……………….\
\
\
         “Exception: Error when checking model target: expected dense\_4 to have shape (None, 3) but got array with shape (135L, 22L)”\
\
\
\
         I would appreciate your help. Thanks.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-372607)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/f4e6edda8349a0f1d81a4b436034d96331810d5c1ace001d1d5a37494eb63e8f?s=40&d=mm&r=g)\
\
\
\
           DevendraNovember 28, 2016 at 5:41 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-372647 "Direct link to this comment")\
\
\
\
\
\
           I found the issue. It was with with the indexes.\
\
\
           I had to take \[1:,1:5\] for X and \[1:,5\] for Y.\
\
\
\
           I am using Jupyter notebook to run my code.\
\
\
           The index range seems to be different in my case.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-372647)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)November 28, 2016 at 8:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-372669 "Direct link to this comment")\
\
\
\
\
\
             I’m glad you worked it out Devendra.\
       - ![](https://secure.gravatar.com/avatar/c158948fb90006a8bc3f8f0a024dd55a15a8c63f358c0019e2c9ee736db2983e?s=40&d=mm&r=g)\
\
\
\
         CristinaMarch 24, 2017 at 2:23 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393938 "Direct link to this comment")\
\
\
\
\
\
         For some reason, when I run this example I get 0 as prediction value for all the samples. What could be happening?\
\
\
\
         I’ve the same problem on prediction with other code I’m executing, and decided to run yours to check if i could be doing something wrong?\
\
\
\
         I’m lost now, this is very strange.\
\
\
\
         Thanks a in advance!\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393938)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/c158948fb90006a8bc3f8f0a024dd55a15a8c63f358c0019e2c9ee736db2983e?s=40&d=mm&r=g)\
\
\
\
           CristinaMarch 24, 2017 at 2:42 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393941 "Direct link to this comment")\
\
\
\
\
\
           Hello again,\
\
\
\
           This is happening with Keras 2.0, with Keras 1 works fine.\
\
\
\
           Thanks,\
\
\
\
           Cristina\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393941)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)March 24, 2017 at 8:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393979 "Direct link to this comment")\
\
\
\
\
\
             Thanks for the note.\
\
           - ![](https://secure.gravatar.com/avatar/25ab70c6620090ee8f1c10cf51bb5873d5bd04e9dec77ab40d09e807e535517b?s=40&d=mm&r=g)\
\
\
\
             FawziApril 5, 2018 at 5:55 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434224 "Direct link to this comment")\
\
\
\
\
\
             Hi all,\
\
\
             I faced the same problem it works well with keras 1 but gives all 0 with keras 2 !\
\
\
\
             Thanks for this great tuto !\
\
\
\
             Fawzi\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)April 6, 2018 at 6:21 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434263 "Direct link to this comment")\
\
\
\
\
\
             Does this happen every time you train the model?\
\
           - ![](https://secure.gravatar.com/avatar/a668af798be65e1d0d2bb232ec2efec04f0fa6b1e1bdcf680db0498e3716c0c7?s=40&d=mm&r=g)\
\
\
\
             Tharindu RanganaDecember 27, 2018 at 4:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460201 "Direct link to this comment")\
\
\
\
\
\
             Hello Cristina,\
\
\
             I have faced the same problem with keras 2. And then I change keras to 1.2 and worked well. Thank you for the information\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 24, 2017 at 7:57 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393975 "Direct link to this comment")\
\
\
\
\
\
           Very strange.\
\
\
\
           Maybe check that your data file is correct, that you have all of the code and that your environment is installed and is working correctly.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393975)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/a17db2c69f644358d79794a362db0bc55245469b5ad7dce12517f724e6013f75?s=40&d=mm&r=g)\
\
\
\
             [Andrea](https://deland77@gmail.com/)December 12, 2017 at 7:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-422837 "Direct link to this comment")\
\
\
\
\
\
             Jason, I’m getting the same prediction (all zeroes) with Keras 2. If we could be able to nail the cause, it would be great. After all, as of now it’s more than likely that people will try to run your great examples with keras 2.\
\
\
\
             Plus, a couple of questions:\
\
\
\
             1\. why did you use a sigmoid for the output layer instead of a softmax?\
\
\
\
             2\. why did you provide initialization even for the last layer?\
\
\
\
             Thanks a lot.\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)December 12, 2017 at 4:02 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-422870 "Direct link to this comment")\
\
\
\
\
\
             The example does use softmax, perhaps check that you have copied all of the code from the post?\
         - ![](https://secure.gravatar.com/avatar/f65477121a1c68bd5bde814be27146cc3d1b4abba08724e6152ebf84389fc527?s=40&d=mm&r=g)\
\
\
\
           kristiJanuary 18, 2018 at 3:17 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426827 "Direct link to this comment")\
\
\
\
\
\
           I’m having same issue. How did u resolve it? could you please help me\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426827)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/7d05c35c543a7e3ace8a5bc6f77163c8ba6a62c234f8abef87c666d4c4cc59bb?s=40&d=mm&r=g)\
\
\
\
             YousufMarch 21, 2018 at 1:41 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-432791 "Direct link to this comment")\
\
\
\
\
\
             Has anyone resolved the issue with the output being all zeros?\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)March 21, 2018 at 3:07 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-432797 "Direct link to this comment")\
\
\
\
\
\
             Perhaps try re-train the model to see if the issue occurs again?\
\
           - ![](https://secure.gravatar.com/avatar/a2586a76eed7c4640eb2a3fcbf23e7d047dec32d79d25789791d8585231af893?s=40&d=mm&r=g)\
\
\
\
             JacksonMay 6, 2019 at 5:35 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-484233 "Direct link to this comment")\
\
\
\
\
\
             I changed the seed=7 to seed= 0, which should make each random number different, and the result will no longer be all 0.\
\
           - ![](https://secure.gravatar.com/avatar/0dde9c4d216508545376628f06107c3d20d9765c8a02bf8d4dc95e0e7dd1b7e4?s=40&d=mm&r=g)\
\
\
\
             [Yme](https://stackoverflow.com/users/7253901/psychotechnopath)August 15, 2019 at 12:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496770 "Direct link to this comment")\
\
\
\
\
\
             Issue is still present! If I use keras >2.0, the model simply predicts the same class for every training example in the dataset.\
\
\
\
             – Have tried varying loss functions\
\
\
             – changing activation function from sigmoid to softmax in the output layer\
\
\
             – using Theano/tensorflow backends\
\
\
             – Changing the number of hidden neurons in the hidden layer\
\
\
\
             And for all these fixes the error persists. Only thing that solves the issue, and makes me get similar results to the ones you’re getting in your tutorial, is downgrading to Keras <2.0 (In my case I downgraded to Keras 1.2.2.)\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2019 at 8:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496815 "Direct link to this comment")\
\
\
\
\
\
             I can confirm the example works as stated with Keras 2.2.4, TensorFlow 1.14 and Python 3.6.\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
             |     |     |\
             | --- | --- |\
             | 1 | Baseline:98.00%(3.06%) |\
\
\
\
\
\
\
\
\
\
\
\
             I believe there is an issue with your development environment. This may help:\
\
             [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)\
\
           - ![](https://secure.gravatar.com/avatar/0dde9c4d216508545376628f06107c3d20d9765c8a02bf8d4dc95e0e7dd1b7e4?s=40&d=mm&r=g)\
\
\
\
             YmeAugust 16, 2019 at 4:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496911 "Direct link to this comment")\
\
\
\
\
\
             Could you share with me the entire code you use? I don’t think its environment related, have tried with a fresh conda environment, and am able to reproduce the issue on 2 seperate machines.\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)August 16, 2019 at 8:04 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496951 "Direct link to this comment")\
\
\
\
\
\
             The entire code listing is provided in the post, I updated it to provide it all together.\
\
           - ![](https://secure.gravatar.com/avatar/0dde9c4d216508545376628f06107c3d20d9765c8a02bf8d4dc95e0e7dd1b7e4?s=40&d=mm&r=g)\
\
\
\
             YmeAugust 16, 2019 at 7:34 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-497019 "Direct link to this comment")\
\
\
\
\
\
             Managed to find the problem!!!\
\
\
\
             In the code above, as well as in your book (Which I am following) we are using code that I think is written for keras1. The code carries over to keras2, apart from some warnings, but predicts poor. The reason for this is the nb\_epoch parameter in the KerasClassifier class. When you leave that as is, the model predicts the same class for every training example. When you change it to “epochs” in keras2, everything is fine. I don’t know if this is Intented behavior or a bug.\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)August 17, 2019 at 5:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-497084 "Direct link to this comment")\
\
\
\
\
\
             No.\
\
\
\
             The example in the post uses “epochs” for Keras 2.\
\
\
\
             So does the most recent version of the book.\
\
\
\
             I think you are not referring to the above tutorial and are in fact referring to a very old version of the book. You can contact me here to get the most recent version:\
\
             [https://machinelearningmastery.com/contact/](https://machinelearningmastery.com/contact/)\
       - ![](https://secure.gravatar.com/avatar/76dbf69ef013fae979c9a50643c28e8bb71b45c6626aaf0dd9d5270139cac700?s=40&d=mm&r=g)\
\
\
\
         [Tanvir.](http://tanviranik.datascientistsbd.com/)March 27, 2017 at 7:43 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394284 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
         Thanks for your awesome tutorials. I had a curious question:\
\
\
         As we are using KerasClassifier or KerasRegressor of Scikit-Learn wrapper, then how to save them as a file after fitting ?\
\
\
\
         For example, I am predicting regression or multiclass classification. I have to use KerasRegressor or KerasClassifier then. After fitting a large volume of data, I want to save the trained neural network model to use it for prediction purpose only. How to save them and how to restore them from saved files ? Your answer will help me a lot.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394284)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 27, 2017 at 8:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394298 "Direct link to this comment")\
\
\
\
\
\
           Great question, I’m not sure you can easily do this. You might be better served fitting the Keras model directly then using the Keras API to save the model:\
\
           [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394298)\
       - ![](https://secure.gravatar.com/avatar/ac78d9c7d543d468ac77c25ae5bd23ec97429597646d27b470fb7b0fab2d8470?s=40&d=mm&r=g)\
\
\
\
         ReinierMay 4, 2017 at 2:04 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398455 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason, Thank your very much for those nice explainations.\
\
\
         I’m having some problems and I trying very hard to get it solved but it wont work..\
\
\
         If I simply copy-past your code from your comment on 31-july 2016 I keep getting the following Error:\
\
\
\
         Traceback (most recent call last): File “/Users/reinier/PycharmProjects/Test-IRIS/TESTIRIS.py”, line 43, in estimator.fit(X\_train, Y\_train) File “/Users/reinier/Library/Python/3.6/lib/python/site-packages/keras/wrappers/scikit\_learn.py”, line 206, in fit return super(KerasClassifier, self).fit(x, y, \*\*kwargs) File “/Users/reinier/Library/Python/3.6/lib/python/site-packages/keras/wrappers/scikit\_learn.py”, line 149, in fit history = self.model.fit(x, y, \*\*fit\_args) File “/Users/reinier/Library/Python/3.6/lib/python/site-packages/keras/models.py”, line 856, in fit initial\_epoch=initial\_epoch) File “/Users/reinier/Library/Python/3.6/lib/python/site-packages/keras/engine/training.py”, line 1429, in fit batch\_size=batch\_size) File “/Users/reinier/Library/Python/3.6/lib/python/site-packages/keras/engine/training.py”, line 1309, in \_standardize\_user\_data exception\_prefix=’target’) File “/Users/reinier/Library/Python/3.6/lib/python/site-packages/keras/engine/training.py”, line 139, in \_standardize\_input\_data str(array.shape)) ValueError: Error when checking target: expected dense\_2 to have shape (None, 3) but got array with shape (67, 40)\
\
\
\
         It seems like something is wrong with the fit function. Is this the cause of a new Keras version? Thanks you very much in advance,\
\
\
\
         Reinier\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398455)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)May 4, 2017 at 8:09 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398494 "Direct link to this comment")\
\
\
\
\
\
           Sorry, it is not clear what is going on.\
\
\
\
           Does the example in the blog post work as expected?\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398494)\
       - ![](https://secure.gravatar.com/avatar/dcf52d2a2e8d7fb1f7fcd56212eb7eb3fef3895c788f65b6d71acb50d259d555?s=40&d=mm&r=g)\
\
\
\
         PriyeshJuly 12, 2017 at 3:02 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405561 "Direct link to this comment")\
\
\
\
\
\
         Hello Jason,\
\
\
\
         Thank you for such a wonderful and detailed explanation. Please can guide me on how to plot the graphs for clustering for this data set and code (both for training and predictions).\
\
\
\
         Thanks.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405561)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 12, 2017 at 9:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405607 "Direct link to this comment")\
\
\
\
\
\
           Sorry, I do not have examples of clustering.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405607)\
       - ![](https://secure.gravatar.com/avatar/dcf52d2a2e8d7fb1f7fcd56212eb7eb3fef3895c788f65b6d71acb50d259d555?s=40&d=mm&r=g)\
\
\
\
         PriyeshJuly 12, 2017 at 5:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405572 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
\
         Thank you so much for such an elegant and detailed explanation. I wanted to learn on how to plot graphs for the same. I went through the comments and you said we can’t plot accuracy but I wish to plot the graphs for input data sets and predictions to show like a cluster (as we show K-means like a scattered plot). Please can you guide me with the same.\
\
\
\
         Thank you.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405572)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 12, 2017 at 9:53 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405610 "Direct link to this comment")\
\
\
\
\
\
           Sorry I do not have any examples for clustering.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405610)\
       - ![](https://secure.gravatar.com/avatar/18586999cf3d1073098cea7939efd7ef42633a2e330cf4fa470509246a1f05b4?s=40&d=mm&r=g)\
\
\
\
         BudiJanuary 19, 2018 at 2:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426863 "Direct link to this comment")\
\
\
\
\
\
         Woahh,, it’s work’s again…\
\
\
         it’s nice result,\
\
\
\
         btw, how, it we want make just own sentences, not use test data?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426863)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)January 19, 2018 at 6:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426893 "Direct link to this comment")\
\
\
\
\
\
           This is called NLP, learn more here:\
\
           [https://machinelearningmastery.com/start-here/#nlp](https://machinelearningmastery.com/start-here/#nlp)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426893)\
       - ![](https://secure.gravatar.com/avatar/862a7b574928eacbc5a379e0e79e85ce3be11a8cdec27f02e0a19f5f45ebae0d?s=40&d=mm&r=g)\
\
\
\
         BonoboJune 24, 2018 at 12:06 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-441798 "Direct link to this comment")\
\
\
\
\
\
         I think the line\
\
\
\
         model = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200, batch\_size=5, verbose=0)\
\
\
\
         must be\
\
\
\
         model = KerasClassifier(build\_fn=baseline\_model, epochs=200, batch\_size=5, verbose=0)\
\
\
\
         for newer Keras versions.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-441798)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 24, 2018 at 7:34 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-441841 "Direct link to this comment")\
\
\
\
\
\
           Correct.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-441841)\
       - ![](https://secure.gravatar.com/avatar/3d18398d5a68114ff9a1fd2d3c5a47122df05c28e4ad70ab7111ee160535fdf4?s=40&d=mm&r=g)\
\
\
\
         PrakharJuly 10, 2018 at 5:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443062 "Direct link to this comment")\
\
\
\
\
\
         hello Sir,\
\
\
         i used the following code in keras backend, but when using categorical\_crossentropy\
\
\
         all the rows of a columns have same predictions,but when i use binary\_crossentropy the predictions are correct.Can u plz explain why?\
\
\
         And my predictions are also in the form of HotEncoding an and not like 2,1,0,2. Kindly help me out in this.\
\
\
         Thank you\
\
\
\
         import numpy as np\
\
\
         import matplotlib.pyplot as plt\
\
\
         import pandas as pd\
\
\
\
         train=pd.read\_csv(‘iris\_train.csv’)\
\
\
         test=pd.read\_csv(‘iris\_test.csv’)\
\
\
\
         xtrain=train.iloc\[:,0:4\].values\
\
\
         ytrain=train.iloc\[:,4\].values\
\
\
         xtest=test.iloc\[:,0:4\].values\
\
\
         ytest=test.iloc\[:,4\].values\
\
\
\
         import keras\
\
\
         from keras.models import Sequential\
\
\
         from keras.layers import Dense\
\
\
         from keras.utils import to\_categorical\
\
\
\
         from sklearn.preprocessing import LabelEncoder,OneHotEncoder\
\
\
         ytrain2=ytrain.reshape(len(ytrain),1)\
\
\
         encoder1=LabelEncoder()\
\
\
         ytrain2\[:,0\]=encoder1.fit\_transform(ytrain2\[:,0\])\
\
\
         encoder=OneHotEncoder(categorical\_features=\[0\])\
\
\
         ytrain2=encoder.fit\_transform(ytrain2).toarray()\
\
\
\
         classifier=Sequential()\
\
\
         classifier.add(Dense(output\_dim=4,init=’uniform’,activation=’relu’,input\_dim=4))\
\
\
         classifier.add(Dense(output\_dim=4,init=’uniform’,activation=’relu’))\
\
\
         classifier.add(Dense(output\_dim=3,init=’uniform’,activation=’sigmoid’))\
\
\
\
         classifier.compile(optimizer=’adam’,loss=’categorical\_crossentropy’,metrics=\[‘accuracy’\])\
\
\
         classifier.fit(xtrain,ytrain2,batch\_size=5,epochs=300)\
\
\
\
         y\_pred=classifier.predict(xtest)\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443062)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 10, 2018 at 6:54 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443093 "Direct link to this comment")\
\
\
\
\
\
           Sorry, I do not have the capacity to debug your code. Perhaps post to stackoverflow.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443093)\
       - ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
         ShooterAugust 10, 2018 at 7:15 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445773 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason, this code gives the accuracy of 98%. But when i add k-fold cross validation code, accuracy decreases to 75%.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445773)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 11, 2018 at 6:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445811 "Direct link to this comment")\
\
\
\
\
\
           Perhaps try tuning the model further?\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445811)\
       - ![](https://secure.gravatar.com/avatar/05a177ecd3d34f5984130b424665be75a528f69b9cf9d896eca0f6fe7397c59b?s=40&d=mm&r=g)\
\
\
\
         TitusNovember 9, 2020 at 6:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-573873 "Direct link to this comment")\
\
\
\
\
\
         Hello Jason,\
\
\
\
         This code does not work form me. I am using the exact same code but I get error with estimator.fit(). The error looks like that:\
\
\
\
         —————————————————————————\
\
\
         TypeError Traceback (most recent call last)\
\
\
         in\
\
\
         34 estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200, batch\_size=5, verbose=0)\
\
\
         35 X\_train, X\_test, Y\_train, Y\_test = train\_test\_split(X, dummy\_y, test\_size=0.33, random\_state=seed)\
\
\
         —\> 36 estimator.fit(X\_train, Y\_train)\
\
\
         37 predictions = estimator.predict(X\_test)\
\
\
         38 print(predictions)\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-573873)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)November 9, 2020 at 7:51 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-573882 "Direct link to this comment")\
\
\
\
\
\
           I can confirm that the code works with the latest version of scikit-learn, tensorflow and keras.\
\
\
\
           Perhaps some of these tips will help:\
\
           [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-573882)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/05a177ecd3d34f5984130b424665be75a528f69b9cf9d896eca0f6fe7397c59b?s=40&d=mm&r=g)\
\
\
\
             TitusNovember 9, 2020 at 6:33 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-573999 "Direct link to this comment")\
\
\
\
\
\
             Thanks Jason,\
\
\
\
             I have resolved the issue. I don’t know why but the problem is from the model.add() function.\
\
\
\
             model.add(Dense(3, init=’normal’, activation=’sigmoid’))\
\
\
\
             If I remove the argument init = ‘normal’ from model.add() I get the correct result but if I add it then I get error with the estimator.fit() function. I don’t know what the reason maybe but simply removing init = ‘normal’ from model.add() resolves the error.\
\
\
\
             Thanks.\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)November 10, 2020 at 6:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-574123 "Direct link to this comment")\
\
\
\
\
\
             Nice work!\
006. ![](https://secure.gravatar.com/avatar/ed4a5cd35f01649fc6fcbcf5deb00b6734d12048dc045092aba95f36718a33b4?s=40&d=mm&r=g)\
\
\
\
     PrashAugust 14, 2016 at 9:15 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-359957 "Direct link to this comment")\
\
\
\
\
\
     Jason, boss you are too good! You have really helped me out especially in implementation of Deep learning part. I was rattled and lost and was desperately looking for some technology and came across your blogs. thanks a lot.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-359957)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2016 at 12:38 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360052 "Direct link to this comment")\
\
\
\
\
\
       I’m glad I have helped in some small way Prash.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360052)\
007. ![](https://secure.gravatar.com/avatar/54ff6ba5981b93339c742f018e29f6b0a10f367a958759786027ff3a8e2b1684?s=40&d=mm&r=g)\
\
\
\
     HarshaAugust 18, 2016 at 7:03 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360394 "Direct link to this comment")\
\
\
\
\
\
     It is a great tutorial Dr. Jason. Very clear and crispy. I am a beginner in Keras. I have a small doubt.\
\
\
\
     Is it necessary to use scikit-learn. Can we solve the same problem using basic keras?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360394)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 19, 2016 at 5:25 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360449 "Direct link to this comment")\
\
\
\
\
\
       You can use basic Keras, but scikit-learn make Keras better. They work very well together.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360449)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/54ff6ba5981b93339c742f018e29f6b0a10f367a958759786027ff3a8e2b1684?s=40&d=mm&r=g)\
\
\
\
         HarshaAugust 19, 2016 at 11:06 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360513 "Direct link to this comment")\
\
\
\
\
\
         Thank You Jason for your prompt reply\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360513)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 20, 2016 at 6:05 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360579 "Direct link to this comment")\
\
\
\
\
\
           You’re welcome Harsha.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360579)\
       - ![](https://secure.gravatar.com/avatar/a0c141abec846cf8782e21e3f780772e5670724932e43d2599463d42db8d26d7?s=40&d=mm&r=g)\
\
\
\
         joklaJanuary 12, 2017 at 7:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-381577 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason, nice tutorial!\
\
\
\
         I have a question. You mentioned that scikit-learn make Keras better, why?\
\
\
\
         Thanks!\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-381577)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)January 12, 2017 at 9:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-381603 "Direct link to this comment")\
\
\
\
\
\
           Hi jokla, great question.\
\
\
\
           The reason is that we can access all of sklearn’s features using the Keras Wrapper classes. Tools like grid searching, cross validation, ensembles, and more.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-381603)\
008. ![](https://secure.gravatar.com/avatar/5deb8f550f7032c0647d85cd7f8b53dabefb3cce245c01165f2fbe350d7568e3?s=40&d=mm&r=g)\
\
\
\
     moeyzfAugust 21, 2016 at 10:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360721 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I’m a CS student currently studying sentiment analysis and was wondering how to use keras for multi classification of text, ideally I would like the functionality of the TFidvectoriser from sklearn so a one hot vector representation against a given vocabulary is used, within a neural net to determine the final classification.\
\
\
\
     I am having trouble understanding the initial steps in transforming and feeding word data into vector representations. Can you help me out with some basic code examples of this first step in the sense that say I have a text file with 5000 words for example, which also include emoji (to use as the vocabulary), how can I feed in a training file in csv format text,sentiment and convert each text into a one hot representation then feed it into the neural net, for a final output vector of size e.g 1×7 to denote the various class labels.\
\
\
\
     I have tried to find help online and most of the solutions use helper methods to load in text data such as imdb, while others use word2vec which isnt what i need.\
\
\
\
     Hope you can help, I would really appreciate it!\
\
\
\
     Cheers,\
\
\
\
     Mo\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-360721)\
\
009. ![](https://secure.gravatar.com/avatar/87edc44cc96bf7dedbc34231f14da3846a1cb2ce2d3a5aa81916d33c4e83a98b?s=40&d=mm&r=g)\
\
\
\
     QichangSeptember 12, 2016 at 3:01 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364285 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for the great tutorial!\
\
\
\
     Just one question regarding the output variable encoding. You mentioned that it is a good practice to convert the output variable to one hot encoding matrix. Is this a necessary step? If the output varible consists of discrete integters, say 1, 2, 3, do we still need to to\_categorical() to perform one hot encoding?\
\
\
\
     I check some example codes in keras github, it seems this is required. Can you please kindly shed some lights on it?\
\
\
\
     Thanks in advance.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364285)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 13, 2016 at 8:09 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364344 "Direct link to this comment")\
\
\
\
\
\
       Hi Qichang, great question.\
\
\
\
       A one hot encoding is not required, you can train the network to predict an integer, it is just a MUCH harder problem.\
\
\
\
       By using a one hot encoding, you greatly simplify the prediction problem making it easier to train for and achieve better performance.\
\
\
\
       Try it and compare the results.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364344)\
010. ![](https://secure.gravatar.com/avatar/2b77c834021cc3d13dd0f26dcb93ef12c2e606531bdeb9fa46e8fc50719b2536?s=40&d=mm&r=g)\
\
\
\
     Pedro A. CastilloSeptember 16, 2016 at 12:31 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364583 "Direct link to this comment")\
\
\
\
\
\
     Hello,\
\
\
     I have followed your tutorial and I get an error in the following line:\
\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
\
     Traceback (most recent call last):\
\
\
     File “k.py”, line 84, in\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
     File “/Library/Python/2.7/site-packages/scikit\_learn-0.17.1-py2.7-macosx-10.9-intel.egg/sklearn/cross\_validation.py”, line 1433, in cross\_val\_score\
\
\
     for train, test in cv)\
\
\
     File “/Library/Python/2.7/site-packages/scikit\_learn-0.17.1-py2.7-macosx-10.9-intel.egg/sklearn/externals/joblib/parallel.py”, line 800, in \_\_call\_\_\
\
\
     while self.dispatch\_one\_batch(iterator):\
\
\
     File “/Library/Python/2.7/site-packages/scikit\_learn-0.17.1-py2.7-macosx-10.9-intel.egg/sklearn/externals/joblib/parallel.py”, line 658, in dispatch\_one\_batch\
\
\
     self.\_dispatch(tasks)\
\
\
     File “/Library/Python/2.7/site-packages/scikit\_learn-0.17.1-py2.7-macosx-10.9-intel.egg/sklearn/externals/joblib/parallel.py”, line 566, in \_dispatch\
\
\
     job = ImmediateComputeBatch(batch)\
\
\
     File “/Library/Python/2.7/site-packages/scikit\_learn-0.17.1-py2.7-macosx-10.9-intel.egg/sklearn/externals/joblib/parallel.py”, line 180, in \_\_init\_\_\
\
\
     self.results = batch()\
\
\
     File “/Library/Python/2.7/site-packages/scikit\_learn-0.17.1-py2.7-macosx-10.9-intel.egg/sklearn/externals/joblib/parallel.py”, line 72, in \_\_call\_\_\
\
\
     return \[func(\*args, \*\*kwargs) for func, args, kwargs in self.items\]\
\
\
     File “/Library/Python/2.7/site-packages/scikit\_learn-0.17.1-py2.7-macosx-10.9-intel.egg/sklearn/cross\_validation.py”, line 1531, in \_fit\_and\_score\
\
\
     estimator.fit(X\_train, y\_train, \*\*fit\_params)\
\
\
     File “/Library/Python/2.7/site-packages/keras/wrappers/scikit\_learn.py”, line 135, in fit\
\
\
     \*\*self.filter\_sk\_params(self.build\_fn.\_\_call\_\_))\
\
\
     TypeError: \_\_call\_\_() takes at least 2 arguments (1 given)\
\
\
\
     Do you have received this error before? do you have an idea how to fix that?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364583)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 16, 2016 at 9:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364628 "Direct link to this comment")\
\
\
\
\
\
       I have not seen this before Pedro.\
\
\
\
       Perhaps it is something simple like a copy-paste error from the tutorial?\
\
\
\
       Are you able to double check the code matches the tutorial exactly?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-364628)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/0a0572f5ded1e863349f7986251066b5dad3dad45a4514f22e6c2fa73ed7233b?s=40&d=mm&r=g)\
\
\
\
         VictorOctober 8, 2016 at 10:15 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-366627 "Direct link to this comment")\
\
\
\
\
\
         I have exactly the same problem.\
\
\
         Double checked the code,\
\
\
         have all the versions of keras etc, updated.\
\
\
         🙁\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-366627)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)October 9, 2016 at 6:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-366652 "Direct link to this comment")\
\
\
\
\
\
           Hi Victor, are you able to share your version of Keras, scikit-learn, TensorFlow/Theano?\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-366652)\
011. ![](https://secure.gravatar.com/avatar/cd96165eef2d82ee2617703b6d04cb892f2740ff0546085983d1b037626680ba?s=40&d=mm&r=g)\
\
\
\
     YunitaSeptember 25, 2016 at 12:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365422 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for the great tutorial.\
\
\
     But I have a question, why did you use sigmoid activation function together with categorical\_crossentropy loss function?\
\
\
     Usually, for multiclass classification problem, I found implementations always using softmax activation function with categorical\_cross entropy.\
\
\
     In addition, does one-hot encoding in the output make it as binary classification instead of multiclass classification? Could you please give some explanations on it?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365422)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 25, 2016 at 8:04 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365453 "Direct link to this comment")\
\
\
\
\
\
       Yes, you could use a softmax instead of sigmoid. Try it and see.\
\
\
\
       The one hot encoding creates 3 binary output features. This too would be required with the softmax activation function.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365453)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/c2a02a253e951388898a7136d71f600b6a49538400ec2b02fcb85d933fa399c0?s=40&d=mm&r=g)\
\
\
\
         PrestonSeptember 12, 2017 at 11:14 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413591 "Direct link to this comment")\
\
\
\
\
\
         Jason,\
\
\
\
         Great site, great resource. Is it possible to see the old example with the one hot encoding output? I’m interested in creating a network with multiple binary outputs and have been searching around for an example.\
\
\
\
         Many thanks.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413591)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)September 13, 2017 at 12:31 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413649 "Direct link to this comment")\
\
\
\
\
\
           I have many examples on the blog of categorical outputs from LSTMs, try the search.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413649)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/c2a02a253e951388898a7136d71f600b6a49538400ec2b02fcb85d933fa399c0?s=40&d=mm&r=g)\
\
\
\
             PrestonSeptember 14, 2017 at 5:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413737 "Direct link to this comment")\
\
\
\
\
\
             Thank you.\
012. ![](https://secure.gravatar.com/avatar/4d3beb73a99dfa0328ebfbe08a2f53de671266c3ceec11dd596b71a83c12c322?s=40&d=mm&r=g)\
\
\
\
     MarcusSeptember 26, 2016 at 6:49 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365527 "Direct link to this comment")\
\
\
\
\
\
     For Text classification or to basically assign them a category based on the text. How would the baseline\_model change????\
\
\
\
     I’m trying to have an inner layer of 24 nodes and an output of 17 categories but the input\_dim=4 as specified in the tutorial wouldn’t be right cause the text length will change depending on the number of words.\
\
\
\
     I’m a little confused. Your help would be much appreciated.\
\
\
\
     model.add(Dense(24, init=’normal’, activation=’relu’))\
\
\
\
     def baseline\_model():\
\
\
     # create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(24, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(17, init=’normal’, activation=’sigmoid’))\
\
\
     # Compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365527)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 26, 2016 at 7:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365532 "Direct link to this comment")\
\
\
\
\
\
       You will need to use padding on the input vectors of encoded words.\
\
\
\
       See this post for an example of working with text:\
\
       [https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/](https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-365532)\
013. ![](https://secure.gravatar.com/avatar/1a0879e18008a79f62702c1e607d473ade41cffcfce37f87bc4e54fb4389ce9c?s=40&d=mm&r=g)\
\
\
\
     VishnuOctober 19, 2016 at 9:07 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367508 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thank you for your tutorial. I was really interested in Deep Learning and was looking for a place to start, this helped a lot.\
\
\
\
     But while I was running the code, I came across two errors. The first one was, that while loading the data through pandas, just like your code i set “header= None” but in the next line when we convert the value to float i got the following error message.\
\
\
\
     “ValueError: could not convert string to float: ‘Petal.Length'”.\
\
\
\
     This problem went away after I took the header=None condition off.\
\
\
\
     The second one came at the end, during the Kfold validation. during the one hot encoding it’s binning the values into 22 categories and not 3. which is causing this error:\
\
\
\
     “Exception: Error when checking model target: expected dense\_2 to have shape (None, 3) but got array with shape (135, 22)”\
\
\
\
     I haven’t been able to get around this. Any suggestion would be appreciated.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367508)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 20, 2016 at 8:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367555 "Direct link to this comment")\
\
\
\
\
\
       That is quite strange Vishnu, I think perhaps you have the wrong dataset.\
\
\
\
       You can download the CSV here:\
\
       [http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367555)\
014. ![](https://secure.gravatar.com/avatar/41058d1b837772d3d8ada9fd0e49402f3117560aa10fc6fad6465212093218f1?s=40&d=mm&r=g)\
\
\
\
     Homagni SahaOctober 20, 2016 at 10:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367571 "Direct link to this comment")\
\
\
\
\
\
     Hello, I tried to use the exact same code for another dataset , the only difference being the dataset had 78 columns and 100000 rows . I had to predict the last column taking the remaining 77 columns as features . I must also say that the last column has 23 different classes.(types basically) and the 23 different classes are all integers not strings like you have used.\
\
\
\
     model = Sequential()\
\
\
     model.add(Dense(77, input\_dim=77, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(10, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(23, init=’normal’, activation=’sigmoid’))\
\
\
\
     also I used nb\_epoch=20 and batch\_size=1000\
\
\
\
     also in estimator I changed the verbose to 1, and now the accuracy is a dismal of 0.52% at the end. Also while running I saw strange outputs in the verbose as :\
\
\
\
     93807/93807 \[==============================\] – 0s – loss: nan – acc: 0.0052\
\
\
\
     why is the loss always as loss: nan ??\
\
\
\
     Can you please tell me how to modify the code to make it run correctly for my dataset?(remaining everything in the code is unchanged)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367571)\
\
015. ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
     [Jason Brownlee](https://machinelearningmastery.com/)October 21, 2016 at 8:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367752 "Direct link to this comment")\
\
\
\
\
\
     Hi Homagni,\
\
\
\
     That is a lot of classes for 100K records. If you can reduce that by splitting up the problem, that might be good.\
\
\
\
     Your batch size is probably too big and your number of epochs is way too small. Dramatically increase the number of epochs bu 2-3 orders of magnitude.\
\
\
\
     Start there and let me know how you go.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-367752)\
\
016. ![](https://secure.gravatar.com/avatar/385f1e14385f6040a9cb1f43a6d036c1c0848ada7ba40b92bf6ca254556675c8?s=40&d=mm&r=g)\
\
\
\
     AbuZekryOctober 30, 2016 at 12:02 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-368626 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I’ve edited the first layer’s activation to ‘softplus’ instead of ‘relu’ and number of neurons to 8 instead of 4\
\
\
     Then I edited the second layer’s activation to ‘softmax’ instead of sigmoid and I got 97.33% (4.42%) performance. Do you have an explanation to this enhancement in performance ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-368626)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 30, 2016 at 8:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-368678 "Direct link to this comment")\
\
\
\
\
\
       Well done AbuZekry.\
\
\
\
       Neural nets are infinitely configurable.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-368678)\
017. ![](https://secure.gravatar.com/avatar/0bef2f378bc253526be59bf376d16ad0fe2941a69e3576196c0140d8629ce1b7?s=40&d=mm&r=g)\
\
\
\
     PanandNovember 7, 2016 at 3:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-369729 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     Is there a error in your code? You said the network has 4 input neurons , 4 hidden neurons and 3 output neurons.But in the code you haven’t added the hidden neurons.You just specified only the input and output neurons… Will it effect the output in anyway?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-369729)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 7, 2016 at 7:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-369746 "Direct link to this comment")\
\
\
\
\
\
       Hi Panand,\
\
\
\
       The network structure is as follows:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       |     |     |\
       | --- | --- |\
       | 1 | 4inputs->\[4hidden nodes\]->3outputs |\
\
\
\
\
\
\
\
\
\
\
\
       Line 5 of the code in section 6 adds both the input and hidden layer:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       |     |     |\
       | --- | --- |\
       | 1 | model.add(Dense(4,input\_dim=4,init='normal',activation='relu')) |\
\
\
\
\
\
\
\
\
\
\
\
       The input\_dim argument defines the shape of the input.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-369746)\
018. ![](https://secure.gravatar.com/avatar/0f8848a326fb58e1d5ce65d774b459b94d5c4ffc8bab1f969a8356947e5cb305?s=40&d=mm&r=g)\
\
\
\
     JDNovember 13, 2016 at 5:28 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-370445 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     I have a set of categorical features and continuous features, I have this model:\
\
\
     model = Sequential()\
\
\
     model.add(Dense(117, input\_dim=117, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(10, activation=’softmax’))\
\
\
\
     I am getting a dismal : (‘Test accuracy:’, 0.43541752685249119) :\
\
\
     Details:\
\
\
     Total records 45k, 10 classes to predict\
\
\
     batch\_size=1000, nb\_epoch=25\
\
\
\
     Any improvements also I would like to put LSTM how to go about doing that as I am getting errors if I add\
\
\
     model.add(Dense(117, input\_dim=117, init=’normal’, activation=’relu’))\
\
\
     model.add(LSTM(117,dropout\_W=0.2, dropout\_U=0.2, return\_sequences=True))\
\
\
     model.add(Dense(10, activation=’softmax’))\
\
\
     Error:\
\
\
     Exception: Input 0 is incompatible with layer lstm\_6: expected ndim=3, found ndim=2\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-370445)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 14, 2016 at 7:41 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-370500 "Direct link to this comment")\
\
\
\
\
\
       Hi JD,\
\
\
\
       Here is a long list of ideas to improve the skill of your deep learning model:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       Not sure about the exception, you may need to double check the input dimensions of your data and confirm that your model definition matches.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-370500)\
019. ![](https://secure.gravatar.com/avatar/9efe4bef34bed89cff0bec46d47479a8aed4ae79385e09c37b2856031c82fb51?s=40&d=mm&r=g)\
\
\
\
     YANovember 17, 2016 at 7:00 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-370974 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I have a set of categorical features(events) from a real system, and i am trying to build a deep learning model for event prediction.\
\
\
     The event’s are not appears equally in the training set and one of them is relatively rare compared to the others.\
\
\
     event count in training set\
\
\
     1 22000\
\
\
     2 6000\
\
\
     3 13000\
\
\
     4 12000\
\
\
     5 26000\
\
\
\
     Should i continue with this training set? or should i restructure the training set?\
\
\
     What is your recommendation?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-370974)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 18, 2016 at 8:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-371039 "Direct link to this comment")\
\
\
\
\
\
       Hi YA, I would try as many different “views” on your problem as you can think of and see which best exposes the problem to the learning algorithms (gets the best performance when everything else is held constant).\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-371039)\
020. ![](https://secure.gravatar.com/avatar/786cd44a5e6443d426e113ec9f39119613928232d35cb146db1574ea007f0aef?s=40&d=mm&r=g)\
\
\
\
     TomDecember 9, 2016 at 12:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-374387 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
     Great work on your website and tuturials! I was wondering if you could show a multi hot encoding, I think you can call it al multi label classification.\
\
\
     Now you have (only one option on and the rest off)\
\
\
     \[1,0,0\]\
\
\
     \[0,1,0\]\
\
\
     \[0,0,1\]\
\
\
\
     And do like (each classification has the option on or off)\
\
\
     \[0,0,0\]\
\
\
     \[0,1,1\]\
\
\
     \[1,0,1\]\
\
\
     \[1,1,0\]\
\
\
     \[1,1,1\]\
\
\
     etc..\
\
\
\
     This would really help for me\
\
\
     Thanks!!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-374387)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/786cd44a5e6443d426e113ec9f39119613928232d35cb146db1574ea007f0aef?s=40&d=mm&r=g)\
\
\
\
       TomDecember 9, 2016 at 1:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-374391 "Direct link to this comment")\
\
\
\
\
\
       Extra side note, with k-Fold Cross Validation. I got it working with binary\_crossentropy with quite bad results. Therefore I wanted to optimize the model and add cross validation which unfortunately didn’t work.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-374391)\
021. ![](https://secure.gravatar.com/avatar/0a9547e12a695688c7d4d9c9cfa17ee9b7b72644c5e5d55cc1a8a5824d73f3b1?s=40&d=mm&r=g)\
\
\
\
     MartinDecember 26, 2016 at 6:02 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-377751 "Direct link to this comment")\
\
\
\
\
\
     Hi, Jason: Regarding this, I have 2 questions:\
\
\
     1) You said this is a “simple one-layer neural network”. However, I feel it’s still 3-layer network: input layer, hidden layer and output layer.\
\
\
\
     4 inputs -> \[4 hidden nodes\] -> 3 outputs\
\
\
\
     2) However, in your model definition:\
\
\
     model.add(Dense(4, input\_dim=4, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(3, init=’normal’, activation=’sigmoid’))\
\
\
\
     Seems that only two layers, input and output, there is no hidden layer. So this is actually a 2-layer network. Is this right?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-377751)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 27, 2016 at 5:24 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-377853 "Direct link to this comment")\
\
\
\
\
\
       Hi Martin, yes. One hidden layer. I take the input and output layers as assumed, the work happens in the hidden layer.\
\
\
\
       The first line defines the number of inputs (input\_dim=4) AND the number of nodes in the hidden layer:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       |     |     |\
       | --- | --- |\
       | 1 | model.add(Dense(4,input\_dim=4,init=’normal’,activation=’relu’)) |\
\
\
\
\
\
\
\
\
\
\
\
       I hope that helps.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-377853)\
022. ![](https://secure.gravatar.com/avatar/cdcff46c91e22fed0ed93c1250316bc1cf82140ac80aa23b3d01544e3572ad3e?s=40&d=mm&r=g)\
\
\
\
     SeunJanuary 16, 2017 at 3:58 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-382423 "Direct link to this comment")\
\
\
\
\
\
     Hi, Jason: I ran this same code but got this error:\
\
\
\
     Traceback (most recent call last):\
\
\
\
     File “”, line 1, in\
\
\
     runfile(‘C:/Users/USER/Documents/keras-master/examples/iris\_val.py’, wdir=’C:/Users/USER/Documents/keras-master/examples’)\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\spyder\\utils\\site\\sitecustomize.py”, line 866, in runfile\
\
\
     execfile(filename, namespace)\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\spyder\\utils\\site\\sitecustomize.py”, line 87, in execfile\
\
\
     exec(compile(scripttext, filename, ‘exec’), glob, loc)\
\
\
\
     File “C:/Users/USER/Documents/keras-master/examples/iris\_val.py”, line 46, in\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\sklearn\\model\_selection\\\_validation.py”, line 140, in cross\_val\_score\
\
\
     for train, test in cv\_iter)\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 758, in \_\_call\_\_\
\
\
     while self.dispatch\_one\_batch(iterator):\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 603, in dispatch\_one\_batch\
\
\
     tasks = BatchedCalls(itertools.islice(iterator, batch\_size))\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 127, in \_\_init\_\_\
\
\
     self.items = list(iterator\_slice)\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\sklearn\\model\_selection\\\_validation.py”, line 140, in\
\
\
     for train, test in cv\_iter)\
\
\
\
     File “C:\\Users\\USER\\Anaconda2\\lib\\site-packages\\sklearn\\base.py”, line 67, in clone\
\
\
     new\_object\_params = estimator.get\_params(deep=False)\
\
\
\
     TypeError: get\_params() got an unexpected keyword argument ‘deep’\
\
\
\
     Please, I need your help on how to resolve this.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-382423)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 17, 2017 at 7:35 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-382528 "Direct link to this comment")\
\
\
\
\
\
       Hi Seun, it is not clear what is going on here.\
\
\
\
       You may have added an additional line or whitespace or perhaps your environment has a problem?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-382528)\
\
     - ![](https://secure.gravatar.com/avatar/9c4799da4c7e8243e4aeeb940dc6260c18a9af4a7bcd14f420007490846335a5?s=40&d=mm&r=g)\
\
\
\
       DavidJanuary 25, 2017 at 3:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-383938 "Direct link to this comment")\
\
\
\
\
\
       Hello Seun, perhaps this could help you: [http://stackoverflow.com/questions/41796618/python-keras-cross-val-score-error/41832675#41832675](http://stackoverflow.com/questions/41796618/python-keras-cross-val-score-error/41832675#41832675)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-383938)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)January 25, 2017 at 10:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384023 "Direct link to this comment")\
\
\
\
\
\
         I have reproduced the fault and understand the cause.\
\
\
\
         The error is caused by a bug in Keras 1.2.1 and I have two candidate fixes for the issue.\
\
\
\
         I have written up the problem and fixes here:\
\
         [http://stackoverflow.com/a/41841066/78453](http://stackoverflow.com/a/41841066/78453)\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384023)\
023. ![](https://secure.gravatar.com/avatar/295fe0aa0334a1a1bcacf86ecead3a1c5a18e21d4d6741bbcf0f5dd21150d38e?s=40&d=mm&r=g)\
\
\
\
     shazzJanuary 25, 2017 at 7:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-383979 "Direct link to this comment")\
\
\
\
\
\
     I have the same issue….\
\
\
     File “/usr/local/lib/python3.5/dist-packages/sklearn/base.py”, line 67, in clone\
\
\
     new\_object\_params = estimator.get\_params(deep=False)\
\
\
     TypeError: get\_params() got an unexpected keyword argument ‘deep’\
\
\
\
     Looks to be an old issue fixed last year so I don’t understand which lib is in the wrong version…\
\
     [https://github.com/fchollet/keras/issues/1385](https://github.com/fchollet/keras/issues/1385)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-383979)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 25, 2017 at 10:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384022 "Direct link to this comment")\
\
\
\
\
\
       Hi shazz,\
\
\
\
       I have reproduced the fault and understand the cause.\
\
\
\
       The error is caused by a bug in Keras 1.2.1 and I have two candidate fixes for the issue.\
\
\
\
       I have written up the problem and fixes here:\
\
       [http://stackoverflow.com/a/41841066/78453](http://stackoverflow.com/a/41841066/78453)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384022)\
024. ![](https://secure.gravatar.com/avatar/cdcff46c91e22fed0ed93c1250316bc1cf82140ac80aa23b3d01544e3572ad3e?s=40&d=mm&r=g)\
\
\
\
     SeunJanuary 25, 2017 at 10:13 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384140 "Direct link to this comment")\
\
\
\
\
\
     Hi Jasson,\
\
\
     Thanks so much. The second fix worked for me.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384140)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 26, 2017 at 4:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384215 "Direct link to this comment")\
\
\
\
\
\
       Glad to hear it Seun.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-384215)\
025. ![](https://secure.gravatar.com/avatar/f3096ad938d7055137da250c6f4791a9117aff8222286a4f63074192dd44af08?s=40&d=mm&r=g)\
\
\
\
     SulthanJanuary 31, 2017 at 3:08 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385100 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason,\
\
\
\
     With the help of your example i am trying to use the same for handwritten digits pixel data to classify the no input is 5000rows with example 20\*20 pixels so totally x matrix is (5000,400) and Y is (5000,1), i am not able to successfully run the model getting error as below in the end of the code.\
\
\
\
     #importing the needed libraries\
\
\
     import scipy.io\
\
\
     import numpy\
\
\
     from sklearn.preprocessing import LabelEncoder\
\
\
     from keras.models import Sequential\
\
\
     from keras.layers import Dense\
\
\
     from keras.wrappers.scikit\_learn import KerasClassifier\
\
\
     from keras.utils import np\_utils\
\
\
     from sklearn.model\_selection import cross\_val\_score\
\
\
     from sklearn.model\_selection import KFold\
\
\
     from sklearn.preprocessing import LabelEncoder\
\
\
     from sklearn.pipeline import Pipeline\
\
\
     ​\
\
\
\
     In \[158\]:\
\
\
\
     #Intializing random no for reproductiblity\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed)\
\
\
\
     In \[159\]:\
\
\
\
     #loading the dataset from mat file\
\
\
     mat = scipy.io.loadmat(‘C:\Users\Sulthan\Desktop\NeuralNet\ex3data1.mat’)\
\
\
     print(mat)\
\
\
\
     {‘X’: array(\[\[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     …,\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\]\]), ‘\_\_header\_\_’: b’MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Sun Oct 16 13:09:09 2011′, ‘\_\_version\_\_’: ‘1.0’, ‘y’: array(\[\[10\],\
\
\
     \[10\],\
\
\
     \[10\],\
\
\
     …,\
\
\
     \[ 9\],\
\
\
     \[ 9\],\
\
\
     \[ 9\]\], dtype=uint8), ‘\_\_globals\_\_’: \[\]}\
\
\
\
     Type Markdown and LaTeX:\
\
\
     α\
\
\
     2\
\
\
     α2\
\
\
     In \[ \]:\
\
\
\
     ​\
\
\
\
     In \[ \]:\
\
\
\
     ​\
\
\
\
     In \[160\]:\
\
\
\
     #Splitting of X and Y of DATA\
\
\
     X\_train = mat\[‘X’\]\
\
\
\
     In \[161\]:\
\
\
\
     X\_train\
\
\
\
     Out\[161\]:\
\
\
     array(\[\[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     …,\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\]\])\
\
\
     In \[162\]:\
\
\
\
     Y\_train = mat\[‘y’\]\
\
\
\
     In \[163\]:\
\
\
\
     Y\_train\
\
\
\
     Out\[163\]:\
\
\
     array(\[\[10\],\
\
\
     \[10\],\
\
\
     \[10\],\
\
\
     …,\
\
\
     \[ 9\],\
\
\
     \[ 9\],\
\
\
     \[ 9\]\], dtype=uint8)\
\
\
     In \[164\]:\
\
\
\
     X\_train.shape\
\
\
\
     Out\[164\]:\
\
\
     (5000, 400)\
\
\
     In \[165\]:\
\
\
\
     Y\_train.shape\
\
\
\
     Out\[165\]:\
\
\
     (5000, 1)\
\
\
     In \[166\]:\
\
\
\
     data\_trainX = X\_train\[2500:,0:400\]\
\
\
\
     In \[167\]:\
\
\
\
     data\_trainX\
\
\
\
     Out\[167\]:\
\
\
     array(\[\[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     …,\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\],\
\
\
     \[ 0., 0., 0., …, 0., 0., 0.\]\])\
\
\
     In \[168\]:\
\
\
\
     data\_trainX.shape\
\
\
\
     Out\[168\]:\
\
\
     (2500, 400)\
\
\
     In \[256\]:\
\
\
\
     data\_trainY = Y\_train\[:2500,:\].reshape(-1)\
\
\
\
     In \[257\]:\
\
\
\
     data\_trainY\
\
\
     data\_trainY.shape\
\
\
\
     Out\[257\]:\
\
\
     (2500,)\
\
\
     In \[284\]:\
\
\
\
     #enocode class values as integers\
\
\
     encoder = LabelEncoder()\
\
\
     encoder.fit(data\_trainY)\
\
\
     encoded\_Y = encoder.transform(data\_trainY)\
\
\
     \# convert integers to dummy variables\
\
\
     dummy\_Y= np\_utils.to\_categorical(encoded\_Y)\
\
\
\
     In \[285\]:\
\
\
\
     dummy\_Y\
\
\
     ​\
\
\
     ​\
\
\
\
     Out\[285\]:\
\
\
     array(\[\[ 0., 0., 0., 0., 1.\],\
\
\
     \[ 0., 0., 0., 0., 1.\],\
\
\
     \[ 0., 0., 0., 0., 1.\],\
\
\
     …,\
\
\
     \[ 0., 0., 0., 1., 0.\],\
\
\
     \[ 0., 0., 0., 1., 0.\],\
\
\
     \[ 0., 0., 0., 1., 0.\]\])\
\
\
     In \[298\]:\
\
\
\
     newy = dummy\_Y.reshape(-1,1)\
\
\
     ​\
\
\
     ​\
\
\
\
     In \[300\]:\
\
\
\
     newy\
\
\
\
     Out\[300\]:\
\
\
     array(\[\[ 0.\],\
\
\
     \[ 0.\],\
\
\
     \[ 0.\],\
\
\
     …,\
\
\
     \[ 0.\],\
\
\
     \[ 1.\],\
\
\
     \[ 0.\]\])\
\
\
     In \[293\]:\
\
\
\
     #define baseline model\
\
\
     def baseline\_model():\
\
\
     #create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(15,input\_dim=400,init=’normal’,activation=’relu’))\
\
\
     model.add(Dense(10,init=’normal’,activation=’sigmoid’))\
\
\
     #compilemodel\
\
\
     model.compile(loss=’categorical\_crossentropy’,optimizer=’adam’,metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200,batch\_size=5,verbose=0)\
\
\
     print(estimator)\
\
\
\
     In \[295\]:\
\
\
\
     kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
\
     results = cross\_val\_score(estimator, data\_trainX, newy, cv=kfold)\
\
\
     print(“Baseline: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     —————————————————————————\
\
\
     ValueError Traceback (most recent call last)\
\
\
     in ()\
\
\
     —-\> 1 results = cross\_val\_score(estimator, data\_trainX, newy, cv=kfold)\
\
\
     2 print(“Baseline: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     C:\\Users\\Sulthan\\Anaconda3\\lib\\site-packages\\sklearn\\model\_selection\\\_validation.py in cross\_val\_score(estimator, X, y, groups, scoring, cv, n\_jobs, verbose, fit\_params, pre\_dispatch)\
\
\
     126\
\
\
     127 “””\
\
\
     –\> 128 X, y, groups = indexable(X, y, groups)\
\
\
     129\
\
\
     130 cv = check\_cv(cv, y, classifier=is\_classifier(estimator))\
\
\
\
     C:\\Users\\Sulthan\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py in indexable(\*iterables)\
\
\
     204 else:\
\
\
     205 result.append(np.array(X))\
\
\
     –\> 206 check\_consistent\_length(\*result)\
\
\
     207 return result\
\
\
     208\
\
\
\
     C:\\Users\\Sulthan\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py in check\_consistent\_length(\*arrays)\
\
\
     179 if len(uniques) > 1:\
\
\
     180 raise ValueError(“Found input variables with inconsistent numbers of”\
\
\
     –\> 181 ” samples: %r” % \[int(l) for l in lengths\])\
\
\
     182\
\
\
     183\
\
\
\
     ValueError: Found input variables with inconsistent numbers of samples: \[2500, 12500\]\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385100)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 1, 2017 at 10:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385401 "Direct link to this comment")\
\
\
\
\
\
       Hi Sulthan, the trace is a little hard to read.\
\
\
\
       Sorry, I have no off the cuff ideas.\
\
\
\
       Perhaps try cutting your example back to the minimum to help isolate the fault?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385401)\
026. ![](https://secure.gravatar.com/avatar/600ad56462a1757bfae32aecfdbd3c3635118923844a53c6a534876a0139b21a?s=40&d=mm&r=g)\
\
\
\
     LinmuFebruary 3, 2017 at 2:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385704 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for your tutorial!\
\
\
\
     Just one question regarding the output. In this problem, we got three classes (setosa, versicolor and virginica), and since each data instance should be classified into only one category, the problem is more specifically “single-lable, multi-class classification”. What if each data instance belonged to multiple categories. Then we are facing “multi-lable, multi-class classification”. In our case, each flower belongs to at least two species (Let’s just forget the biology 🙂 ).\
\
\
\
     My solution is to modify the output variable (Y) with mutiple ‘1’ in it, i.e. \[1 1 0\], \[0 1 1\], \[1 1 1 \]……. This is definitely not one-hot encoding any more (maybe two or three-hot?)\
\
\
\
     Will my method work out? If not, how do you think the problem of “multi-lable, multi-class classification” should be solved?\
\
\
\
     Thanks in advance\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385704)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 3, 2017 at 10:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385764 "Direct link to this comment")\
\
\
\
\
\
       Your method sounds very reasonable.\
\
\
\
       You may also want to use sigmoid activation functions on the output layer to allow binary class membership to each available class.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-385764)\
027. ![](https://secure.gravatar.com/avatar/885bbaf18a8db779bd9bff85d28d419008b384b77fcc87dabef9e65cfa2ae7f9?s=40&d=mm&r=g)\
\
\
\
     [solarenqu](http://x.hu/)February 19, 2017 at 9:28 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-389064 "Direct link to this comment")\
\
\
\
\
\
     Hello, how can I use the model to create predictions?\
\
\
\
     if i try this: print(‘predict: ‘,estimator.predict(\[\[5.7,4.4,1.5,0.4\]\])) i got this exception:\
\
\
\
     AttributeError: ‘KerasClassifier’ object has no attribute ‘model’\
\
\
     Exception ignored in: <bound method BaseSession.\_\_del\_\_ of >\
\
\
     Traceback (most recent call last):\
\
\
     File “/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorflow/python/client/session.py”, line 581, in \_\_del\_\_\
\
\
     AttributeError: ‘NoneType’ object has no attribute ‘TF\_DeleteStatus’\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-389064)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 20, 2017 at 9:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-389151 "Direct link to this comment")\
\
\
\
\
\
       I have not seen this error before.\
\
\
\
       What versions of Keras/TF/sklearn/Python are you using?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-389151)\
028. ![](https://secure.gravatar.com/avatar/a9a2d0d2639a955c558f0694386ffad4eed55ce12460818e917adbf5636299ad?s=40&d=mm&r=g)\
\
\
\
     SuvamMarch 1, 2017 at 7:34 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390698 "Direct link to this comment")\
\
\
\
\
\
     Hi,\
\
\
     Thanks for the great tutorial.\
\
\
     It would be great if you could outline what changes would be necessary if I want to do a multi-class classification with text data: the training data assigns scores to different lines of text, and the problem is to infer the score for a new line of text. It seems that the estimator above cannot handle strings. What would be the fix for this?\
\
\
\
     Thanks in advance for the help.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390698)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 1, 2017 at 8:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390719 "Direct link to this comment")\
\
\
\
\
\
       Consider encoding your words as integers, using a word embedding and a fixed sequence length.\
\
\
\
       See this tutorial:\
\
       [https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/](https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390719)\
029. ![](https://secure.gravatar.com/avatar/42e745aa7c9961b97c353f04234ac494ad6d801b04fc53eaf6b49dfdd48043ad?s=40&d=mm&r=g)\
\
\
\
     SwetaMarch 1, 2017 at 9:10 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390803 "Direct link to this comment")\
\
\
\
\
\
     This was a great tutorial to enhance the skills in deep learning. My question: is it possible to use this same dataset for LSTM? Can you please help with this how to solve in LSTM?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390803)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2017 at 8:15 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390869 "Direct link to this comment")\
\
\
\
\
\
       Hi Sweta,\
\
\
\
       You could use an LSTM, but it would not be appropriate because LSTMs are intended for sequence prediction problems and this is not a sequence prediction problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-390869)\
030. ![](https://secure.gravatar.com/avatar/2234a6aa9722ef7782927fdfb7feb37d17e3d5f05f3b6f17e3cb7d1260681609?s=40&d=mm&r=g)\
\
\
\
     AkashMarch 22, 2017 at 5:47 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393776 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I have this problem where I have 1500 features as input to my DNN and 2 output classes, can you explain how do I decide the size of neurons in my hidden layer and how many hidden layers I need to process such high features with accuracy.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393776)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 23, 2017 at 8:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393846 "Direct link to this comment")\
\
\
\
\
\
       Lots of trial and error.\
\
\
\
       Start with a small network and keep adding neurons and layers and epochs until no more benefit is seen.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-393846)\
031. ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
     Ananya MohapatraMarch 24, 2017 at 9:39 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394049 "Direct link to this comment")\
\
\
\
\
\
     sir, the following code is showing an error message.. could you help me figure it out. i am trying to do a multi class classification with 5 datasets combined in one( 4 non epileptic patients and 1 epileptic) …500 x 25 dataset and the 26th column is the class.\
\
\
\
     \# Train model and make predictions\
\
\
     import numpy\
\
\
     import pandas\
\
\
     from keras.models import Sequential\
\
\
     from keras.layers import Dense\
\
\
     from keras.wrappers.scikit\_learn import KerasClassifier\
\
\
     from keras.utils import np\_utils\
\
\
     from sklearn.model\_selection import cross\_val\_score\
\
\
     from sklearn.cross\_validation import train\_test\_split\
\
\
     from sklearn.preprocessing import LabelEncoder\
\
\
     from sklearn.model\_selection import KFold\
\
\
\
     \# fix random seed for reproducibility\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed)\
\
\
     \# load dataset\
\
\
     dataframe = pandas.read\_csv(“DemoNSO.csv”, header=None)\
\
\
     dataset = dataframe.values\
\
\
     X = dataset\[:,0:25\].astype(float)\
\
\
     Y = dataset\[:,25\]\
\
\
     \# encode class values as integers\
\
\
     encoder = LabelEncoder()\
\
\
     encoder.fit(Y)\
\
\
     encoded\_Y = encoder.transform(Y)\
\
\
     \# convert integers to dummy variables (i.e. one hot encoded)\
\
\
     dummy\_y = np\_utils.to\_categorical(encoded\_Y)\
\
\
     \# define baseline model\
\
\
     def baseline\_model():\
\
\
     # create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(700, input\_dim=25, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(2, init=’normal’, activation=’sigmoid’))\
\
\
\
     # Compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=50, batch\_size=20)\
\
\
\
     kfold = KFold(n\_splits=5, shuffle=True, random\_state=seed)\
\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
     print(“Baseline: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     X\_train, X\_test, Y\_train, Y\_test = train\_test\_split(X, dummy\_y, test\_size=0.55, random\_state=seed)\
\
\
     estimator.fit(X\_train, Y\_train)\
\
\
     predictions = estimator.predict(X\_test)\
\
\
\
     print(predictions)\
\
\
     print(encoder.inverse\_transform(predictions))\
\
\
\
     error message:\
\
\
     str(array.shape))\
\
\
     ValueError: Error when checking model target: expected dense\_56 to have shape (None, 2) but got array with shape (240, 3)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394049)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 25, 2017 at 7:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394102 "Direct link to this comment")\
\
\
\
\
\
       Confirm the size of your output (y) matches the dimension of your output layer.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394102)\
032. ![](https://secure.gravatar.com/avatar/b2b3903077c9eb1634d8cb7fdc9a71d66b6cb4c8abc74faad9a2ebe9d59f2223?s=40&d=mm&r=g)\
\
\
\
     AlicanMarch 28, 2017 at 4:05 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394378 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     I got your model to work using Python 2.7.13, Keras 2.0.2, Theano 0.9.0.dev…, by copying the codes exactly, however the results that I get are not only very bad (59.33%, 48.67%, 38.00% on different trials), but they are also different.\
\
\
\
     I was under the impression that using a fixed seed would allow us to reproduce the same results.\
\
\
\
     Do you have any idea what could have caused such bad results?\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394378)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/b2b3903077c9eb1634d8cb7fdc9a71d66b6cb4c8abc74faad9a2ebe9d59f2223?s=40&d=mm&r=g)\
\
\
\
       AlicanMarch 28, 2017 at 4:28 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394381 "Direct link to this comment")\
\
\
\
\
\
       edit: I was re-executing only the results=cross\_val\_score(…) line to get different results I listed above.\
\
\
\
       Running the whole script over and over generates the same result: “Baseline: 59.33% (21.59%)”\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394381)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)March 28, 2017 at 8:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394410 "Direct link to this comment")\
\
\
\
\
\
         Glad to hear it.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394410)\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 28, 2017 at 8:25 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394409 "Direct link to this comment")\
\
\
\
\
\
       Not sure why the results are so bad. I’ll take a look.\
\
\
\
       The fixed seed does not seem to have an effect on the Theano or TensorFlow backends. Try running examples multiple times and take the average performance.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394409)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/b2b3903077c9eb1634d8cb7fdc9a71d66b6cb4c8abc74faad9a2ebe9d59f2223?s=40&d=mm&r=g)\
\
\
\
         AlicanApril 2, 2017 at 2:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395052 "Direct link to this comment")\
\
\
\
\
\
         Did you have time to look into this?\
\
\
\
         I had my colleague run this script on Theano 1.0.1, and it gave the expected performance of 95.33%. I then installed Theano 1.0.1, and got the same result again.\
\
\
\
         However, using Theano 2.0.2 I was getting 59.33% with seed=7, and similar performances with different seeds. Is it possible the developers made some crucial changes with the new version?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395052)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)April 2, 2017 at 6:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395077 "Direct link to this comment")\
\
\
\
\
\
           The most recent version of Theano is 0.9:\
\
           [https://github.com/Theano/Theano/releases](https://github.com/Theano/Theano/releases)\
\
\
\
           Do you mean Keras versions?\
\
\
\
           It may not be the Keras version causing the difference in the run. The fixed random seed may not be having an effect in general, or may not be having when a Theano backend is being used.\
\
\
\
           Neural networks are stochastic algorithms and will produce a different result each run:\
\
           [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395077)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/b2b3903077c9eb1634d8cb7fdc9a71d66b6cb4c8abc74faad9a2ebe9d59f2223?s=40&d=mm&r=g)\
\
\
\
             AlicanApril 2, 2017 at 6:59 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395088 "Direct link to this comment")\
\
\
\
\
\
             Yes I meant Keras, sorry.\
\
\
\
             There is no issue with the seed, I’m getting the same result with you on multiple computers using Keras 1.1.1. But with Keras 2.0.2, the results are absymally bad.\
\
           - ![](https://secure.gravatar.com/avatar/3df5eef2c10411931b8880240d5a5b19b87f7850e73f5915b1e53823bca2b86b?s=40&d=mm&r=g)\
\
\
\
             JonathanJuly 11, 2017 at 4:28 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405442 "Direct link to this comment")\
\
\
\
\
\
             not sure if this was every resolved, but I’m getting the same thing with most recent versions of Theano and Keras\
\
\
\
             59.33% with seed=7\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2017 at 10:33 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405485 "Direct link to this comment")\
\
\
\
\
\
             Try running the example a few times with different seeds.\
\
\
\
             Neural networks are stochastic:\
\
             [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
033. ![](https://secure.gravatar.com/avatar/71997c5aed97aa1902d34196e2d21e77fd33b75bd2ec7b5fc4c0b6ddb473ec74?s=40&d=mm&r=g)\
\
\
\
     NaliniMarch 29, 2017 at 3:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394524 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
\
     in this code for multiclass classification can u suggest me how to plot graph to display the accuracy and also what should be the axis represent\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394524)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 29, 2017 at 9:10 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394565 "Direct link to this comment")\
\
\
\
\
\
       No, we normally do not graph accuracy, unless you want to graph it over training epochs?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394565)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/f4025c3937c1f5666c6b00787f1cf5e367c63952f0c07dde44c8537688d6d77d?s=40&d=mm&r=g)\
\
\
\
         SebastianSeptember 4, 2019 at 3:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499567 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
\
         First of all, I’d like to thank you for your blog. I’m currently trying to build a multiclass classifier just as the one you have explained above. I was wondering: how could I plot the history of loss and accuracy for training and validation per epoch as it is done using the historry=model.fit()?.\
\
\
\
         Many thanks in advance for your help.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499567)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)September 4, 2019 at 6:03 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499608 "Direct link to this comment")\
\
\
\
\
\
           Yes, see this post:\
\
           [https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499608)\
034. ![](https://secure.gravatar.com/avatar/71997c5aed97aa1902d34196e2d21e77fd33b75bd2ec7b5fc4c0b6ddb473ec74?s=40&d=mm&r=g)\
\
\
\
     NaliniMarch 31, 2017 at 1:42 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394800 "Direct link to this comment")\
\
\
\
\
\
     thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394800)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 31, 2017 at 5:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394844 "Direct link to this comment")\
\
\
\
\
\
       You’re welcome.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-394844)\
035. ![](https://secure.gravatar.com/avatar/83f9139e28e53d51696c1338c9cdcae5e7eeef9ef77d31b6958d017f74567069?s=40&d=mm&r=g)\
\
\
\
     FrankApril 6, 2017 at 8:47 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395530 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason,\
\
\
     I have found this tutorial very interesting and helpful.\
\
\
     What I wanted to ask is, I am currently trying to classify poker hands as this kaggle competition: [https://www.kaggle.com/c/poker-rule-induction](https://www.kaggle.com/c/poker-rule-induction) (For a school project) I wish to create a neural network as you have created above. What do you suggest for me to start this?\
\
\
     Your help would be greatly appreciated!\
\
\
     Thanks.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395530)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 9, 2017 at 2:39 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395794 "Direct link to this comment")\
\
\
\
\
\
       This process will help you work through your modeling problem:\
\
       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395794)\
036. ![](https://secure.gravatar.com/avatar/9a152e7dfc889b4f329403c64e71a54535129300e9b6d55656c4038cc966a1e8?s=40&d=mm&r=g)\
\
\
\
     shivaApril 8, 2017 at 12:28 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395667 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Its an awesome tutorial. It would be great if you can come up with a blog post on multiclass medical image classification with Keras Deep Learning library. It would serve as a great asset for researchers like me, working with medical image classification. Looking forward.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395667)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 9, 2017 at 2:56 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395822 "Direct link to this comment")\
\
\
\
\
\
       Thanks for the suggestion.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395822)\
037. ![](https://secure.gravatar.com/avatar/65c5b8b8718f279c15ea925e0b7cae858c0fe897ecf50e58c904303d2b0c4d52?s=40&d=mm&r=g)\
\
\
\
     TobyApril 9, 2017 at 4:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395727 "Direct link to this comment")\
\
\
\
\
\
     Thanks for the great tutorial!\
\
\
     I duplicated the result using Theano as backend.\
\
\
     However, using Tensorflow yield a worse accuracy, 88.67%.\
\
\
     Any explanation?\
\
\
     Thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395727)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 9, 2017 at 3:00 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395827 "Direct link to this comment")\
\
\
\
\
\
       It may be related to the stochastic nature of neural nets and the difficulty of making results with the TF backend reproducible.\
\
\
\
       You can learn more about the stochastic nature of machine learning algorithms here:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-395827)\
038. ![](https://secure.gravatar.com/avatar/1643e788b8d1868869b171ba82eaddacf29e58ea679d1b50dbfab02f4a9dc87e?s=40&d=mm&r=g)\
\
\
\
     AnupamApril 11, 2017 at 6:11 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396094 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, How to find the Precision, Recall and f1 score of your example?\
\
\
\
     Case-1 I have used like :\
\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’Nadam’, metrics=\[‘acc’, ‘fmeasure’, ‘precision’, ‘recall’\])\
\
\
\
     Case-2 and also used :\
\
\
\
     def score(yh, pr):\
\
\
     coords = \[np.where(yhh > 0)\[0\]\[0\] for yhh in yh\]\
\
\
     yh = \[yhh\[co:\] for yhh, co in zip(yh, coords)\]\
\
\
     ypr = \[prr\[co:\] for prr, co in zip(pr, coords)\]\
\
\
     fyh = \[c for row in yh for c in row\]\
\
\
     fpr = \[c for row in ypr for c in row\]\
\
\
     return fyh, fpr\
\
\
\
     pr = model.predict\_classes(X\_train)\
\
\
     yh = y\_train.argmax(2)\
\
\
     fyh, fpr = score(yh, pr)\
\
\
     print ‘Training accuracy:’, accuracy\_score(fyh, fpr)\
\
\
     print ‘Training confusion matrix:’\
\
\
     print confusion\_matrix(fyh, fpr)\
\
\
     precision\_recall\_fscore\_support(fyh, fpr)\
\
\
\
     pr = model.predict\_classes(X\_test)\
\
\
     yh = y\_test.argmax(2)\
\
\
     fyh, fpr = score(yh, pr)\
\
\
     print ‘Testing accuracy:’, accuracy\_score(fyh, fpr)\
\
\
     print ‘Testing confusion matrix:’\
\
\
     print confusion\_matrix(fyh, fpr)\
\
\
     precision\_recall\_fscore\_support(fyh, fpr)\
\
\
\
     What I have observed is that, accuracy of case-1 and case-2 are different?\
\
\
\
     Any solution?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396094)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 12, 2017 at 7:52 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396168 "Direct link to this comment")\
\
\
\
\
\
       You can make predictions on your test data and use the tools from sklearn:\
\
       [http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396168)\
039. ![](https://secure.gravatar.com/avatar/3dfdc4c14260fce59a2557c37887eeab01d86feb61e3f285f61d12ead68e5852?s=40&d=mm&r=g)\
\
\
\
     [Raynier van Egmond](http://www.xfintell.com/)April 15, 2017 at 12:19 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396518 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Like a student earlier in the comments my accuracy results are exactly the same as his:\
\
\
\
     \\*\\*\\*\\*\\*\\*\\*\\*\\*\\* Baseline: 88.67% (21.09%)\
\
\
\
     and I think this is related to having Tensorflow as the backend rather than the Theano backend.\
\
\
\
     I am working this through in a Jupyter notebook\
\
\
\
     I went through your earlier tutorials on setting up the environment:\
\
\
\
     scipy: 0.18.1\
\
\
     numpy: 1.11.3\
\
\
     matplotlib: 2.0.0\
\
\
     pandas: 0.19.2\
\
\
     statsmodels: 0.6.1\
\
\
     sklearn: 0.18.1\
\
\
     theano: 0.9.0.dev-c697eeab84e5b8a74908da654b66ec9eca4f1291\
\
\
     tensorflow: 1.0.1\
\
\
     Using TensorFlow backend.\
\
\
     keras: 2.0.3\
\
\
\
     The Tensorflow is a Python3.6 recompile picked up from the web at:\
\
\
\
     [http://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow](http://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow)\
\
\
\
     Do you know have I can force the Keras library to take Theano as a backend rather than the Tensorflow library?\
\
\
\
     Thanks for the great work on your tutorials… for beginners it is such in invaluable thing to have tutorials that actually work !!!\
\
\
\
     Looking forward to get more of your books\
\
\
\
     Rene\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396518)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/3dfdc4c14260fce59a2557c37887eeab01d86feb61e3f285f61d12ead68e5852?s=40&d=mm&r=g)\
\
\
\
       [Raynier van Egmond](http://www.xfintell.com/)April 15, 2017 at 12:42 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396521 "Direct link to this comment")\
\
\
\
\
\
       Changing to the Theano backend doesn’t change the results:\
\
\
\
       Managed to change to a Theano backend by setting the Keras config file:\
\
\
       {\
\
\
       “image\_data\_format”: “channels\_last”,\
\
\
       “epsilon”: 1e-07,\
\
\
       “floatx”: “float32”,\
\
\
       “backend”: “theano”\
\
\
       }\
\
\
\
       as instructed at: [https://keras.io/backend/#keras-backends](https://keras.io/backend/#keras-backends)\
\
\
\
       The notebook no longer reports it is using Tensorflow so I guess the switch worked but the results are still:\
\
\
\
       \\*\\*\\*\\*\\*\\* Baseline: 88.67% (21.09%)\
\
\
\
       Will need to look a little deeper and play with the actual architecture a bit.\
\
\
\
       All the same great material to get started with\
\
\
\
       Thanks again\
\
\
\
       Rene\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396521)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/3dfdc4c14260fce59a2557c37887eeab01d86feb61e3f285f61d12ead68e5852?s=40&d=mm&r=g)\
\
\
\
         [Raynier van Egmond](http://www.xfintell.com/)April 15, 2017 at 1:26 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396524 "Direct link to this comment")\
\
\
\
\
\
         Confirmed that changes to the model as someone above mentioned\
\
\
\
         model.add(Dense(8, input\_dim=4, kernel\_initializer=’normal’, activation=’relu’))\
\
\
         model.add(Dense(3, kernel\_initializer=’normal’, activation=’softmax’))\
\
\
\
         nodes makes a substantial difference:\
\
\
\
         \\*\\*\\*\\* Baseline: 96.67% (4.47%)\
\
\
\
         but there is no difference between the Tensorflow and Theano backend results. I guess that’s as far as I can take this for now.\
\
\
\
         Take care,\
\
\
\
         Rene\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396524)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)April 16, 2017 at 9:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396584 "Direct link to this comment")\
\
\
\
\
\
           Nice.\
\
\
\
           Also, note that MLPs are stochastic. This means that if you don’t fix the random seed, you will get different results for each run of the algorithm.\
\
\
\
           Ideally, you should take the average performance of the algorithm across multiple runs to evaluate its performance.\
\
\
\
           See this post:\
\
           [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396584)\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 16, 2017 at 9:22 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396581 "Direct link to this comment")\
\
\
\
\
\
       You can change the back-end used by Keras in the Kersas config file. See this post:\
\
       [https://machinelearningmastery.com/introduction-python-deep-learning-library-keras/](https://machinelearningmastery.com/introduction-python-deep-learning-library-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396581)\
040. ![](https://secure.gravatar.com/avatar/d7aea65780b951404b99221b62503a03e04e88b26f1934ad6ce48fbaa8a89d3b?s=40&d=mm&r=g)\
\
\
\
     TursunApril 16, 2017 at 9:18 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396630 "Direct link to this comment")\
\
\
\
\
\
     Jason,\
\
\
     Thank you very much first. These tutorials are excellent. They are very practical. Your are an excellent educator.\
\
\
     I want classify my data into multiple classes of 25-30. Your IRIS example is nearest classification. They DL4J previously has IRIS classification with DBN; but disappeared in new community version.\
\
\
     I have following issues:\
\
\
     1.>\
\
\
     It takes so long. My laptop is TOSHIBA L745, 4GB RAM, i3 processor. it has CUDA.\
\
\
     My classification problem is solved with SVM in very short time. I’d say in split second.\
\
\
     Do you think speed would increase if we use DBN or CNN something ?\
\
\
     2.>\
\
\
     My result :\
\
\
     Baseline: 88.67% (21.09%),\
\
\
     Once I have installed Docker (tensorflow in it),then run IRIS classification. It shows 96%.\
\
\
     I wish similar or better accuracy. How to reach that level ?\
\
\
\
     Thank you\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396630)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 17, 2017 at 5:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396656 "Direct link to this comment")\
\
\
\
\
\
       MLP is the right algorithm for multi-class classification algorithms.\
\
\
\
       If it is slow, consider running it on AWS:\
\
       [https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)\
\
\
\
       There are many things you can do to lift performance, see this post:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396656)\
041. ![](https://secure.gravatar.com/avatar/d7bce2e99f92798a4403d1cda36dd50c3503320de89d6cf0dac87c51a7debf8f?s=40&d=mm&r=g)\
\
\
\
     ChrisApril 17, 2017 at 5:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396657 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
     first of all, your tutorials are really well done when you start working with keras.\
\
\
\
     I have a question about the epochs and batch\_size in this tutorial. I think I haven’t understood it correctly.\
\
\
\
     I loaded the record and it contains 150 entries.\
\
\
\
     You choose 200 epochs and batch\_size=5. So you use 5\*200=1000 examples for training. So does keras use the same entries multiple times or does it stop automatically?\
\
\
\
     Thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396657)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 18, 2017 at 8:23 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396767 "Direct link to this comment")\
\
\
\
\
\
       One epoch involves exposing each pattern in the training dataset to the model.\
\
\
\
       One epoch is comprised of one or more batches.\
\
\
\
       One batch involves showing a subset of the patterns in the training data to the model and updating weights.\
\
\
\
       The number of patterns in the dataset for one epoch must be a factor of the batch size (e.g. divide evenly).\
\
\
\
       Does that help?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396767)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/d7bce2e99f92798a4403d1cda36dd50c3503320de89d6cf0dac87c51a7debf8f?s=40&d=mm&r=g)\
\
\
\
         ChrisApril 22, 2017 at 3:43 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397168 "Direct link to this comment")\
\
\
\
\
\
         Hi,\
\
\
         thank you for the explanation.\
\
\
         The explanation helped me, and in the meantime I have read and tried several LSTM tutorials from you and it became much clearer to me.\
\
\
         greetings, Chris\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397168)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)April 22, 2017 at 9:28 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397198 "Direct link to this comment")\
\
\
\
\
\
           I’m glad to hear that Chris.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397198)\
042. ![](https://secure.gravatar.com/avatar/ab8149c9f06e67c0c6dc908ab4224dcd7b4035b96d893fe25b5b206a31dd69ca?s=40&d=mm&r=g)\
\
\
\
     Abhilash MenonApril 17, 2017 at 1:27 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396686 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason,\
\
\
\
     I have been following your tutorials and they have been very very helpful!. Especially, the most useful section being the comments where people like me get to ask you questions and some of them are the same ones I had in my mind.\
\
\
\
     Although, I have one that I think hasn’t been asked before, at least on this page!\
\
\
\
     What changes should I make to the regular program you illustrated with the “pima\_indians\_diabetes.csv” in order to take a dataset that has 5 categorical inputs and 1 binary output.\
\
\
\
     This would be a huge help! Thanks in advance!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396686)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 18, 2017 at 8:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396777 "Direct link to this comment")\
\
\
\
\
\
       Great question.\
\
\
\
       Consider using an integer encoding followed by a binary encoding of the categorical inputs.\
\
\
\
       This post will show you how:\
\
       [https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/](https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396777)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/85fd5b84eb0c649a4a171c300bd4fd46d24b2e89dfa0f774ee504090122ebe70?s=40&d=mm&r=g)\
\
\
\
         Abhilash MenonJuly 18, 2017 at 12:47 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406323 "Direct link to this comment")\
\
\
\
\
\
         Hello Dr. Brownlee,\
\
\
\
         The link that you shared was very helpful and I have been able to one hot encode and use the data set but at this point of time I am not able to find relevant information regarding what the perfect batch size and no. of epochs should be. My data has 5 categorical inputs and 1 binary output (2800 instances). Could you tell me what factors I should take into consideration before arriving at a perfect batch size and epoch number? The following are the configuration details of my neural net:\
\
\
\
         model.add(Dense(28, input\_dim=43, init=’uniform’, activation=’relu’))\
\
\
         model.add(Dense(28, init=’uniform’, activation=’relu’))\
\
\
         model.add(Dense(1, init=’uniform’, activation=’sigmoid’))\
\
\
         model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406323)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 18, 2017 at 5:01 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406342 "Direct link to this comment")\
\
\
\
\
\
           I recommend testing a suite of different batch sizes.\
\
\
\
           I have a post this friday with advice on tuning the batch size, watch out for it.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406342)\
043. ![](https://secure.gravatar.com/avatar/148079eb764992c39030e66247008057d1433c5df4ca429e1db651f3d4c90f4c?s=40&d=mm&r=g)\
\
\
\
     TubaApril 18, 2017 at 8:43 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396785 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     First of all, your tutorials are really very interesting.\
\
\
\
     I was facing error this when i run it . I’m work with python 3 and the same file input .\
\
\
\
     Error :\
\
\
     ImportError: Traceback (most recent call last):\
\
\
     File “/home/indatacore/anaconda3/lib/python3.5/site-packages/tensorflow/python/\_\_init\_\_.py”, line 61, in\
\
\
     from tensorflow.python import pywrap\_tensorflow\
\
\
     File “/home/indatacore/anaconda3/lib/python3.5/site-packages/tensorflow/python/pywrap\_tensorflow.py”, line 28, in\
\
\
     \_pywrap\_tensorflow = swig\_import\_helper()\
\
\
     File “/home/indatacore/anaconda3/lib/python3.5/site-packages/tensorflow/python/pywrap\_tensorflow.py”, line 24, in swig\_import\_helper\
\
\
     \_mod = imp.load\_module(‘\_pywrap\_tensorflow’, fp, pathname, description)\
\
\
     File “/home/indatacore/anaconda3/lib/python3.5/imp.py”, line 242, in load\_module\
\
\
     return load\_dynamic(name, filename, file)\
\
\
     File “/home/indatacore/anaconda3/lib/python3.5/imp.py”, line 342, in load\_dynamic\
\
\
     return \_load(spec)\
\
\
     ImportError: libcudart.so.8.0: cannot open shared object file: No such file or directory\
\
\
\
     Failed to load the native TensorFlow runtime.\
\
\
\
     See [https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get\_started/os\_setup.md#import\_error](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#import_error)\
\
\
\
     for some common reasons and solutions. Include the entire stack trace\
\
\
     above this error message when asking for help.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396785)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 19, 2017 at 7:44 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396870 "Direct link to this comment")\
\
\
\
\
\
       Ouch. I have not seen this error before.\
\
\
\
       Consider trying the Theano backend and see if that makes a difference.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-396870)\
044. ![](https://secure.gravatar.com/avatar/d7aea65780b951404b99221b62503a03e04e88b26f1934ad6ce48fbaa8a89d3b?s=40&d=mm&r=g)\
\
\
\
     TursunApril 21, 2017 at 2:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397062 "Direct link to this comment")\
\
\
\
\
\
     Jason,\
\
\
     Thank you. I got your notion: there is no key which opens all doors.\
\
\
\
     Here, I have multi class classification problem.\
\
\
     My data can be downloaded from here:\
\
     [https://www.dropbox.com/s/w2en6ewdsed69pc/tursun\_deep\_p6.csv?dl=0](https://www.dropbox.com/s/w2en6ewdsed69pc/tursun_deep_p6.csv?dl=0)\
\
\
\
     size of my data set : 512\*16, last column is 21 classes, they are digits 1-21\
\
\
     note: number of samples (rows in my data) for each class is different. mostly 20 rows, but sometimes 17 or 31 rows\
\
\
     my network has:\
\
\
     first layer (input) has 15 neurons\
\
\
     second layer (hidden) has 30 neurons\
\
\
     last layer (output) has 21 neurons\
\
\
     in last layer I used “softmax” based on this recommendation from\
\
     [https://github.com/fchollet/keras/issues/1013](https://github.com/fchollet/keras/issues/1013)\
\
\
     “The softmax function transforms your hidden units into probability scores of the class labels you have; and thus is more suited to classification problems ”\
\
\
     error message:\
\
\
     alueError: Error when checking model target: expected dense\_8 to have shape (None, 21) but got array with shape (512, 1)\
\
\
\
     I would be thankful if you can help me to run this code.\
\
\
\
     I modified this code from yours:\
\
\
     ———–keras code start ———–\
\
\
     from keras.models import Sequential\
\
\
     from keras.layers import Dense\
\
\
     import numpy\
\
\
     \# fix random seed for reproducibility\
\
\
     numpy.random.seed(7)\
\
\
     \# load pima indians dataset\
\
\
     dataset = numpy.loadtxt(“tursun\_deep\_p6.csv”, delimiter=”,”)\
\
\
     \# split into input (X) and output (Y) variables\
\
\
     X = dataset\[:,0:15\]\
\
\
     Y = dataset\[:,15\]\
\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(30, input\_dim=15, activation=’relu’)) # not sure if 30 too much. not sure #about lower and upper limits\
\
\
     #model.add(Dense(25, activation=’relu’)) # think about to add one more hidden layer\
\
\
     model.add(Dense(21, activation=’softmax’)) # they say softmax at last L does classification\
\
\
     \# Compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     \# Fit the model\
\
\
     model.fit(X, Y, epochs=150, batch\_size=5)\
\
\
     \# evaluate the model\
\
\
     scores = model.evaluate(X, Y)\
\
\
     print(“\\n%s: %.2f%%” % (model.metrics\_names\[1\], scores\[1\]\*100))\
\
\
\
     ———–keras code start ———–\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397062)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 21, 2017 at 8:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397100 "Direct link to this comment")\
\
\
\
\
\
       I see the problem, your output layer expects 8 columns and you only have 1.\
\
\
\
       You need to transform your output variable int 8 variables. You can do this using a one hot encoding.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397100)\
045. ![](https://secure.gravatar.com/avatar/9a152e7dfc889b4f329403c64e71a54535129300e9b6d55656c4038cc966a1e8?s=40&d=mm&r=g)\
\
\
\
     ShivaApril 23, 2017 at 5:54 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397293 "Direct link to this comment")\
\
\
\
\
\
     Hi jason, I am following your book deep learning with python and i have an issue with the script. I have succesfully read my .csv datafile through pandas and trying to adopt a decay based learning rate as discussed in the book. I define the initial lrate, drop, epochs\_drop and the formula for lrate update as said in the book. I then created the model like this (works best for my problem) and started creating a pipeline in contrary to the model fitting strategy used by you in the book:\
\
\
\
     def baseline\_model():\
\
\
     # create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(50, input\_dim=15, kernel\_initializer=’normal’, activation=’relu’))\
\
\
     model.add(Dense(3, kernel\_initializer=’normal’, activation=’sigmoid’))\
\
\
     sgd = SGD(lr=0.0, momentum=0.9, decay=0, nesterov=False)\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=sgd, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
     #learning schedule callback\
\
\
     lrate = LearningRateScheduler(step\_decay)\
\
\
     callbacks\_list = \[lrate\]\
\
\
\
     estimators = \[\]\
\
\
     estimators.append((‘standardize’, StandardScaler()))\
\
\
     estimators.append((‘mlp’, KerasClassifier(build\_fn=baseline\_model, epochs=100,\
\
\
     batch\_size=5, callbacks=\[lrate\], verbose=1)))\
\
\
     pipeline = Pipeline(estimators)\
\
\
     kfold = StratifiedKFold(n\_splits=2, shuffle=True, random\_state=seed)\
\
\
     results = cross\_val\_score(pipeline, X, encoded\_Y, cv=kfold)\
\
\
\
     I’m getting the error “Cannot clone object , as the constructor does not seem to set parameter callbacks”. According to keras documentation, I can see that i can pass callbacks to the kerasclassifier wrapper. kindly suggest what to do in this occasion. Looking forward.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397293)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 24, 2017 at 5:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397379 "Direct link to this comment")\
\
\
\
\
\
       I have not tried to use callbacks with the sklearn wrapper sorry.\
\
\
\
       Perhaps it is a limitation that you can’t? Though, I’d be surprised.\
\
\
\
       you may have to use the keras API directly.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397379)\
046. ![](https://secure.gravatar.com/avatar/9a152e7dfc889b4f329403c64e71a54535129300e9b6d55656c4038cc966a1e8?s=40&d=mm&r=g)\
\
\
\
     ShivaApril 25, 2017 at 6:23 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397514 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     I’m trying to apply the image augmentation techniques discussed in your book to the data I have stored in my system under C:\\images\\train and C:\\images\\test. Could you help me with the syntax on how to load my own data with a modification to the syntax available in the book:\
\
\
\
     \# load data\
\
\
     (X\_train, y\_train), (X\_test, y\_test) = mnist.load\_data()\
\
\
\
     Thanks in advance.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397514)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 25, 2017 at 7:52 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397531 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I don’t have an example of how to load image data from disk, I hope to cover it in the future.\
\
\
\
       This post may help as a start:\
\
       [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397531)\
047. ![](https://secure.gravatar.com/avatar/b659c54a72c638bd6757d5ad41d2dce2be3b121e593cfad0a4cc0987c1e9c0fd?s=40&d=mm&r=g)\
\
\
\
     Michael NgApril 28, 2017 at 12:49 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397802 "Direct link to this comment")\
\
\
\
\
\
     Hi,\
\
\
\
     By implementing neural network in Keras, how can we get the associated probabilities for each predicted class?’\
\
\
\
     Many Thanks!\
\
\
     Michael Ng\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397802)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 28, 2017 at 7:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397852 "Direct link to this comment")\
\
\
\
\
\
       Review the outputs from the softmax, although not strictly probabilities, they can be used as such.\
\
\
\
       Also see the keras function model.predict\_proba() for predicting probabilities directly.\
\
       [https://keras.io/models/sequential/](https://keras.io/models/sequential/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397852)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/b659c54a72c638bd6757d5ad41d2dce2be3b121e593cfad0a4cc0987c1e9c0fd?s=40&d=mm&r=g)\
\
\
\
         Michael NgApril 30, 2017 at 11:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398058 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
\
         ‘Note that we use a sigmoid activation function in the output layer. This is to ensure the output values are in the range of 0 and 1 and may be used as predicted probabilities.’\
\
\
\
         Instead of using softmax function, how do I review the sigmoidal outputs (as per the tutorial) for each of 3 output nodes? Mind to share the code to list the sigmoidal outputs?\
\
\
\
         Regards,\
\
\
         Michael Ng\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398058)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)May 1, 2017 at 5:52 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398128 "Direct link to this comment")\
\
\
\
\
\
           I would recommend softmax for multi-class classification.\
\
\
\
           You can learn more about sigmoid here:\
\
           [https://en.wikipedia.org/wiki/Logistic\_function](https://en.wikipedia.org/wiki/Logistic_function)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398128)\
       - ![](https://secure.gravatar.com/avatar/a17db2c69f644358d79794a362db0bc55245469b5ad7dce12517f724e6013f75?s=40&d=mm&r=g)\
\
\
\
         [Andrea](https://deland77@gmail.com/)December 12, 2017 at 7:59 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-422841 "Direct link to this comment")\
\
\
\
\
\
         Jason,\
\
\
\
         may you elaborate further (or provide a link) about “the outputs from the softmax, although not strictly probabilities”?\
\
\
\
         I thought they were probabilities even in the most formal sense.\
\
\
\
         Thanks!\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-422841)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)December 12, 2017 at 4:04 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-422872 "Direct link to this comment")\
\
\
\
\
\
           No, they are normalized to look like probabilities.\
\
\
\
           This might be a good place to start:\
\
           [https://en.wikipedia.org/wiki/Softmax\_function](https://en.wikipedia.org/wiki/Softmax_function)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-422872)\
048. ![](https://secure.gravatar.com/avatar/8ab32df8b377cfe1c0f06d49e88fc9aa40ece03570d96c9a2c6b4f022d5bc6e6?s=40&d=mm&r=g)\
\
\
\
     AnnApril 28, 2017 at 2:08 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397808 "Direct link to this comment")\
\
\
\
\
\
     Hi, Jason! I’m exactly newbie to Keras, and I want to figure out confusion matrix by using sklearn.confusion\_matrix(y\_test, predict). But I was facing error this when i run it .\
\
\
\
     —————————————————————————\
\
\
     ValueError Traceback (most recent call last)\
\
\
     in ()\
\
\
     —-\> 1 confusion\_matrix(y\_test, predict)\
\
\
\
     C:\\Users\\Ann\\Anaconda3\\envs\\py27\\lib\\site-packages\\sklearn\\metrics\\classification.pyc in confusion\_matrix(y\_true, y\_pred, labels, sample\_weight)\
\
\
     240 y\_type, y\_true, y\_pred = \_check\_targets(y\_true, y\_pred)\
\
\
     241 if y\_type not in (“binary”, “multiclass”):\
\
\
     –\> 242 raise ValueError(“%s is not supported” % y\_type)\
\
\
     243\
\
\
     244 if labels is None:\
\
\
\
     ValueError: multilabel-indicator is not supported\
\
\
\
     I’ve checked that y\_test and predict have same shape (231L, 2L).\
\
\
     Any solution?\
\
\
     Your help would be greatly appreciated!\
\
\
     Thanks.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397808)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 28, 2017 at 7:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397855 "Direct link to this comment")\
\
\
\
\
\
       Consider checking the dimensionality of both y and yhat to ensure they are the same (e.g. print the shape of them).\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-397855)\
049. ![](https://secure.gravatar.com/avatar/e6917781026f5b16c017680e0d924dce452087d7db2e198fa9fda6324d3443c2?s=40&d=mm&r=g)\
\
\
\
     Mohammed ZahranApril 30, 2017 at 4:49 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398031 "Direct link to this comment")\
\
\
\
\
\
     can we use the same approach to classify MNIST in (0,1…) and the same time classify the numbers to even and odd numbers ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398031)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 30, 2017 at 5:35 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398043 "Direct link to this comment")\
\
\
\
\
\
       Machine learning is not needed to check for odd and even numbers, just a little math.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398043)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/545b1dca478b288fa1e5e9a5ee37b66fa708391b4f0388230a9799887f8241f1?s=40&d=mm&r=g)\
\
\
\
         TAM.GApril 30, 2017 at 4:46 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398079 "Direct link to this comment")\
\
\
\
\
\
         but if we too it as a simple try to learn about multi-labeling ,, how could we do this\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398079)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/e6917781026f5b16c017680e0d924dce452087d7db2e198fa9fda6324d3443c2?s=40&d=mm&r=g)\
\
\
\
           MohMay 1, 2017 at 10:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398163 "Direct link to this comment")\
\
\
\
\
\
           @Jason Brownlee I totally agree with you. We are using this problem as proxy for more complex problems like classifying a scene with multiple cars and we want to classify the models of these cars. The same approach is needed in tackling neurological images\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398163)\
050. ![](https://secure.gravatar.com/avatar/545b1dca478b288fa1e5e9a5ee37b66fa708391b4f0388230a9799887f8241f1?s=40&d=mm&r=g)\
\
\
\
     TAM.GApril 30, 2017 at 3:22 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398070 "Direct link to this comment")\
\
\
\
\
\
     first this is a great tutorial , but , am confused a little ,, am i loading my training files and labeling files or what ??\
\
\
     as i tried to apply this tutorial to my case ,, I’ve about 10 folder each has its own images these images are related together for one class ,, but i need to make multi labeling for each folder of them for example folder number 1 has about 1500 .png imgs of owl bird , here i need to make a multi label for this to train it as a bird and owl , and here comes the problem ,, as i’m seraching for a tool to make labeling for all images in each folder and label them as \[ owl, bird\] together … any idea about how to build my own multi label classifier ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398070)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 1, 2017 at 5:53 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398129 "Direct link to this comment")\
\
\
\
\
\
       I would recommend using a CNN instead of an MLP for image classification, see this post:\
\
       [https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/](https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-398129)\
051. ![](https://secure.gravatar.com/avatar/697474550f7d0eb81ef7ade1955a78c4b8a48ebe506665566ce3cef78abb25c3?s=40&d=mm&r=g)\
\
\
\
     Ik.OMay 14, 2017 at 10:58 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-399666 "Direct link to this comment")\
\
\
\
\
\
     I implemented the same code on my system and achieved a score of 88.67% at seed = 7 and 96.00% at seed = 4. Any particular reason for this?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-399666)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 15, 2017 at 5:52 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-399685 "Direct link to this comment")\
\
\
\
\
\
       Nice work!\
\
\
\
       Yes, deep learning algorithms are stochastic:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-399685)\
052. ![](https://secure.gravatar.com/avatar/1643e788b8d1868869b171ba82eaddacf29e58ea679d1b50dbfab02f4a9dc87e?s=40&d=mm&r=g)\
\
\
\
     AnupamMay 18, 2017 at 4:58 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-399959 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, Just gone through your blog [https://machinelearningmastery.com/](https://machinelearningmastery.com/) .Just to know as a beginner in Deep learning, can you give any hint to do the task sequence learning for word language identification problem.\
\
\
     Here each word is a variable sequence of characters and the id of each word must be classified with a language tag.\
\
\
     Like, Suppose if we have a dataset like:\
\
\
\
     hello/L1 bahiya/L2 hain/L2 brother/L1 ,/L3 :)/L4\
\
\
\
     where L1,L2,L3 and L4 are the Language-tag\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-399959)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 19, 2017 at 8:14 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400008 "Direct link to this comment")\
\
\
\
\
\
       Hi Anupam, that sounds like a great problem.\
\
\
\
       I would suggest starting with a high-quality dataset, then consider modeling the problem using a seq2seq architecture.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400008)\
053. ![](https://secure.gravatar.com/avatar/a68b4ae2fcaea226ca13343b6da227ad273df8d9580a89abbc9ab1fed3ea7c08?s=40&d=mm&r=g)\
\
\
\
     A.MalathiMay 19, 2017 at 7:30 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400048 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Your tutorials are great and very helpful to me. Have you written any article on Autoencoder.\
\
\
     I have constructed an autoencoder network for a dataset with labels. The output is a vector of\
\
\
     errors(Euclidean Distance). From that errors, classification or prediction on the test set is possible since labels are given??\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400048)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 20, 2017 at 5:37 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400090 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I don’t currently have any material on autoencoders.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400090)\
054. ![](https://secure.gravatar.com/avatar/63485612610e3622aade95544a43005108321ccf22c185eea355deccfddf1941?s=40&d=mm&r=g)\
\
\
\
     J. A. GildeaMay 22, 2017 at 2:57 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400220 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, thank you so much for your helpful tutorials.\
\
\
     I have one question regarding one-hot encoding:\
\
\
     I am working on using a CNN for sentiment analysis and I have a total of six labels for my output variable, string values (P+, P, NONE, NEU, N, N+) representing sentiments.\
\
\
     I one-hot encoded my output variable the same way as you showed in this tutorial, but the shape after one-hot encoding appears to be (, 7). Shouln’t it be 6 instead of 7? Any idea what might be going on? I checked for issues in my dataset such as null values in a certain row, and got rid of all of them yet this persists.\
\
\
     Thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400220)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 22, 2017 at 7:54 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400241 "Direct link to this comment")\
\
\
\
\
\
       It should be 7.\
\
\
\
       Consider loading your data in Python and printing the set of values in the column to get an idea of what is in your data.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400241)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/63485612610e3622aade95544a43005108321ccf22c185eea355deccfddf1941?s=40&d=mm&r=g)\
\
\
\
         J. A. GildeaMay 22, 2017 at 6:39 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400264 "Direct link to this comment")\
\
\
\
\
\
         I checked my data a bit deeper and it seems it had a couple of null values that I removed.\
\
\
         I am however getting very poor results, could this be due to the fact that my data is a bit unbalanced? Some of the classes appear twice as others, so I imagine I would have to change the metrics in my compile function (using accuracy at the moment).\
\
\
         Can a slight imbalance in the dataset yield such poor results (under 40% validation accuracy)?\
\
\
\
         Thanks.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400264)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)May 23, 2017 at 7:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400300 "Direct link to this comment")\
\
\
\
\
\
           With multiple classes, it might be better to use another metric like log loss (cross entropy) or AUC.\
\
\
\
           Accuracy will not capture the true performance of the model.\
\
\
\
           Also, imbalanced classes can be a problem. You could look at removing some classes or rebalancing the data:\
\
           [https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400300)\
055. ![](https://secure.gravatar.com/avatar/71997c5aed97aa1902d34196e2d21e77fd33b75bd2ec7b5fc4c0b6ddb473ec74?s=40&d=mm&r=g)\
\
\
\
     NaliniMay 24, 2017 at 6:10 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400442 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason!\
\
\
     I can’t seem to add more layers in my code.\
\
\
     model.add(Dense(12, input\_dim=25, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(5, init=’normal’, activation=’sigmoid’))\
\
\
     This is a part of the existing code. if i try to add more layers along with them i get a warning for indentation fault.\
\
\
     can you please specify which one of the above layers is the input layer and which one is hidden….\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400442)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2017 at 11:32 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401111 "Direct link to this comment")\
\
\
\
\
\
       This is a Python issue. Ensure you understand the role of whitespace in Python:\
\
       [http://www.diveintopython.net/getting\_to\_know\_python/indenting\_code.html](http://www.diveintopython.net/getting_to_know_python/indenting_code.html)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401111)\
056. ![](https://secure.gravatar.com/avatar/b769cb00ad3823edf12ba4bed43c04fd1faae3b7b4a5cb02d2b678035fb3eaf7?s=40&d=mm&r=g)\
\
\
\
     MichaelMay 28, 2017 at 4:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400662 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     I have two questions:\
\
\
\
     1\. I didn’t see the code in this post calling the fit method. Is the fitting process executed in KerasClassifier?\
\
\
\
     2\. I have only one dataset as training set (No dedicated test set).\
\
\
     Is the KFold method using this single dataset for evaluation in the KerasClassifier class?\
\
\
     Or should I use the “validation\_split parameter in the fit method?\
\
\
\
     Thank’s\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400662)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2017 at 12:06 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401156 "Direct link to this comment")\
\
\
\
\
\
       Hi Michael,\
\
\
\
       Yes, we use the sklearn infrastructure to fit and evaluate the model.\
\
\
\
       You can try both methods. The best evaluation test harness is really problem dependent. k-fold cross validation generally gives a less biased estimate of performance and is often recommended.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401156)\
057. ![](https://secure.gravatar.com/avatar/21c7e8ce29985257625ba5ff4a66c90bd04ac322aa7a982fffea49276b2eaa63?s=40&d=mm&r=g)\
\
\
\
     [Nimesh](http://www.morafitweebly.com/)May 29, 2017 at 4:20 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400743 "Direct link to this comment")\
\
\
\
\
\
     I am classifying mp3s into 7 genre classes. I have 1200 mp3 files dataset with 7 features as input. I got basic Neural network as your example shows and it gives nearly 60% of accuracy. Any suggestions on how to improve accuracy? your suggestions will be very helpful for me.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-400743)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2017 at 12:22 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401180 "Direct link to this comment")\
\
\
\
\
\
       Yes, see this post:\
\
       [https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)\
\
\
\
       And this post:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401180)\
058. ![](https://secure.gravatar.com/avatar/63485612610e3622aade95544a43005108321ccf22c185eea355deccfddf1941?s=40&d=mm&r=g)\
\
\
\
     J. A. GildeaJune 9, 2017 at 3:35 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401918 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     I posted here a while back and I’m back for more wisdom!\
\
\
\
     I have my own model and dataset for text classification (6 labels representing sentiment of tweets). I am not sure on how to evaluate it, I have tried using k fold just as in your example and it yields 100% accuracy which I assume is not the reality.\
\
\
     Just using model.fit() I obtain a result of 99%, which also makes me think I am not evaluating my model correctly.\
\
\
     I have been looking for a way to do this and apparently a good approach is to use a confusion matrix. Is this necessary to evaluate a multiclass model for text classification, or will other methods suffice?\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401918)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 9, 2017 at 6:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401946 "Direct link to this comment")\
\
\
\
\
\
       Generally, I would recommend this process to work through your problem systematically:\
\
       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)\
\
\
\
       I would recommend this post to get a robust estimate of the skill of a deep learning model on unseen data:\
\
       [https://machinelearningmastery.com/evaluate-skill-deep-learning-models/](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)\
\
\
\
       For multi-class classification, I would recommend a confusion matrix, but also measures like logloss.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-401946)\
059. ![](https://secure.gravatar.com/avatar/3bdd35e916968f0fee18d77f5540e21f9c989fee85022cf48c180aaa47590d3a?s=40&d=mm&r=g)\
\
\
\
     zakariaJune 11, 2017 at 3:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402157 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, I need your help I use tensorflow and keras to classify cifar10 images. My question is how to make prediction (make prediction for only one image)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402157)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 11, 2017 at 8:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402181 "Direct link to this comment")\
\
\
\
\
\
       Like this:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       |     |     |\
       | --- | --- |\
       | 1 | yhat=model.predict(X) |\
\
\
\
\
\
\
\
\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402181)\
060. ![](https://secure.gravatar.com/avatar/3bdd35e916968f0fee18d77f5540e21f9c989fee85022cf48c180aaa47590d3a?s=40&d=mm&r=g)\
\
\
\
     zakariaJune 12, 2017 at 6:35 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402283 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     To make the prediction I used this function Y\_pred = model.predict (x\_test)\
\
\
     print (Y\_pred)\
\
\
     Y\_pred = np.argmax (Y\_pred, axis = 1)\
\
\
     print (y\_pred)\
\
\
\
     And I got these results\
\
\
     \[\[0, 0, …, 0, 0, 0\]\]\
\
\
     \[0, 1, 0, …, 0, 0, 0\]\
\
\
     \[1\. 0. 0. …, 0. 0. 0.\]\
\
\
     …\
\
\
     \[0, 0, 0, …, 0, 0, 0\]\]\
\
\
     \[1\. 0. 0. …, 0. 0. 0.\]\
\
\
     \[0\. 0. 0. …, 1. 0. 0.\]\]\
\
\
     \[0 1 0 …, 5 0 7\]\
\
\
     What these results mean\
\
\
     And how to display for example the first 10 images of the test database to see if the model works well\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402283)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 13, 2017 at 8:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402345 "Direct link to this comment")\
\
\
\
\
\
       The prediction result may be an outcome (probability-like value) for each class.\
\
\
\
       You can take an argmax() of each vector to find the selected class.\
\
\
\
       Alternately, you can call predict\_classes() to predict the class directly.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402345)\
061. ![](https://secure.gravatar.com/avatar/1674f0d85174bbeff112cf1184cc22fde91b1fdcd2894d74ff0283eecdca44e5?s=40&d=mm&r=g)\
\
\
\
     HuongJune 12, 2017 at 11:55 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402300 "Direct link to this comment")\
\
\
\
\
\
     Dear @Jason,\
\
\
     Thank you for your useful post. I have a issues.\
\
\
     My dataset have 3 columns (features) for output data. Each column has multi-classes. So how can I process in this case?\
\
\
     Thanks.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402300)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 13, 2017 at 8:22 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402349 "Direct link to this comment")\
\
\
\
\
\
       I don’t have a great answer for you off the cuff. I would suggest doing a little research to see how this type of problem has been handled in the literature.\
\
\
\
       Maybe you can model each class separately?\
\
\
\
       Maybe you can one-hot encode each output variable and use a neural network to output everyone directly.\
\
\
\
       Let me know how you go.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402349)\
062. ![](https://secure.gravatar.com/avatar/e040a30b80801ceb1de4e76d569874c185b9cb51b985c4e14437001c81be3d5b?s=40&d=mm&r=g)\
\
\
\
     [Anastasios](http://soulis.tech/)June 17, 2017 at 10:05 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402904 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     great post on multiclass classification. I am trying to do a gridsearch on a multiclass dataset i created, but I get an error when calling the fit function on the gridsearch. Can we apply gridsearch on a multiclass dataset ?\
\
\
\
     My code looks like: [https://pastebin.com/eB35aJmW](https://pastebin.com/eB35aJmW)\
\
\
\
     And the error I get is: [https://pastebin.com/C1ch7709](https://pastebin.com/C1ch7709)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402904)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 18, 2017 at 6:31 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402941 "Direct link to this comment")\
\
\
\
\
\
       Yes, I believe you can grid search a multi-class classification problem.\
\
\
\
       Sorry, it is not clear to me what the cause of the error might be. You will need to cut your example back to a minimum case that still produces the error.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-402941)\
063. ![](https://secure.gravatar.com/avatar/0321fa1b97e6ab4e8b8f575cd48a395a342d1df4ca03bce6fef1803404262733?s=40&d=mm&r=g)\
\
\
\
     Anupam SamantaJune 29, 2017 at 3:42 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404140 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Excellent tutorials! I have been able to learn a lot reading your articles.\
\
\
\
     I ran into some problem while implementing this program\
\
\
     My accuracy was around Accuracy: 70.67% (12.00%)\
\
\
     I dont know why the accuracy is so dismal!\
\
\
     I tried changing some parameters, mostly that are mentioned in the comments, such as removing kernel\_initializer, changing activation function, also the number of hidden nodes. But the best I was able to achieve was 70 %\
\
\
\
     Any reason something is going wrong here in my code?!\
\
\
\
     \# Modules\
\
\
     import numpy\
\
\
     import pandas\
\
\
     from keras.models import Sequential\
\
\
     from keras.layers import Dense\
\
\
     from keras.utils import np\_utils\
\
\
     from keras.wrappers.scikit\_learn import KerasClassifier\
\
\
     from sklearn.model\_selection import cross\_val\_score\
\
\
     from sklearn.model\_selection import KFold\
\
\
     from sklearn.preprocessing import LabelEncoder\
\
\
     from keras import backend as K\
\
\
     import os\
\
\
\
     def set\_keras\_backend(backend):\
\
\
     if K.backend() != backend:\
\
\
     os.environ\[‘KERAS\_BACKEND’\] = backend\
\
\
     reload(K)\
\
\
     assert K.backend() == backend\
\
\
\
     set\_keras\_backend(“theano”)\
\
\
     \# seed\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed)\
\
\
\
     \# load dataset\
\
\
     dataFrame = pandas.read\_csv(“iris.csv”, header=None)\
\
\
     dataset = dataFrame.values\
\
\
\
     X = dataset\[:, 0:4\].astype(float)\
\
\
     Y = dataset\[:, 4\]\
\
\
\
     \# encode class values\
\
\
     encoder = LabelEncoder()\
\
\
     encoder.fit(Y)\
\
\
     encoded\_Y = encoder.transform(Y)\
\
\
\
     dummy\_Y = np\_utils.to\_categorical(encoded\_Y)\
\
\
\
     \# baseline model\
\
\
     def baseline\_model():\
\
\
     # create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(8, input\_dim=4, kernel\_initializer=’normal’, activation=’softplus’))\
\
\
     model.add(Dense(3, kernel\_initializer=’normal’, activation=’softmax’))\
\
\
     # compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200, batch\_size=5, verbose=0)\
\
\
     kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
\
     results = cross\_val\_score(estimator, X, dummy\_Y, cv=kfold)\
\
\
\
     print(“Accuracy: %.2f%% (%.2f%%)” % (results.mean() \* 100, results.std() \* 100))\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404140)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/0321fa1b97e6ab4e8b8f575cd48a395a342d1df4ca03bce6fef1803404262733?s=40&d=mm&r=g)\
\
\
\
       Anupam SamantaJune 29, 2017 at 3:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404141 "Direct link to this comment")\
\
\
\
\
\
       I added my code here: [https://pastebin.com/3Kr7P6Kw](https://pastebin.com/3Kr7P6Kw)\
\
\
       Its better formatted here!\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404141)\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 29, 2017 at 6:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404180 "Direct link to this comment")\
\
\
\
\
\
       There are more ideas here:\
\
       [https://machinelearningmastery.com/deploy-machine-learning-model-to-production/](https://machinelearningmastery.com/deploy-machine-learning-model-to-production/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404180)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/0321fa1b97e6ab4e8b8f575cd48a395a342d1df4ca03bce6fef1803404262733?s=40&d=mm&r=g)\
\
\
\
         Anupam SamantaJune 30, 2017 at 3:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404284 "Direct link to this comment")\
\
\
\
\
\
         But isnt it strange, that when I use the same code as yours, my program in my machine returns such bad results!\
\
\
         Is there anything I am doing wrong in my code?!\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404284)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 30, 2017 at 8:14 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404310 "Direct link to this comment")\
\
\
\
\
\
           No. Try running the example a few times. Neural networks are stochastic and give different results each time they are run.\
\
\
\
           See this post on why:\
\
           [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
           See this post on how to address it and get a robust estimate of model performance:\
\
           [https://machinelearningmastery.com/evaluate-skill-deep-learning-models/](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404310)\
     - ![](https://secure.gravatar.com/avatar/4de52ecec91a1596dc0c8e092b44898a739cc99f81e2a38c9044e5490fc54e4b?s=40&d=mm&r=g)\
\
\
\
       Zefeng WuJune 30, 2017 at 11:05 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404368 "Direct link to this comment")\
\
\
\
\
\
       Hi, my codes is as followings, but keras gave a extremebad results,\
\
\
       import numpy\
\
\
       import pandas\
\
\
       from keras.models import Sequential\
\
\
       from keras.layers import Dense\
\
\
       from keras.wrappers.scikit\_learn import KerasClassifier\
\
\
       from keras.utils import np\_utils\
\
\
       from sklearn.model\_selection import cross\_val\_score\
\
\
       from sklearn.model\_selection import KFold\
\
\
       from sklearn.preprocessing import LabelEncoder\
\
\
       from sklearn.pipeline import Pipeline\
\
\
       \# fix random seed for reproducibility\
\
\
       seed = 7\
\
\
       numpy.random.seed(seed)\
\
\
       \# load dataset\
\
\
       dataframe = pandas.read\_csv(“iris.csv”, header=None)\
\
\
       dataset = dataframe.values\
\
\
       X = dataset\[:,0:4\].astype(float)\
\
\
       Y = dataset\[:,4\]\
\
\
       \# encode class values as integers\
\
\
       encoder = LabelEncoder()\
\
\
       encoder.fit(Y)\
\
\
       encoded\_Y = encoder.transform(Y)\
\
\
       \# convert integers to dummy variables (i.e. one hot encoded)\
\
\
       dummy\_y = np\_utils.to\_categorical(encoded\_Y)\
\
\
       \# define baseline model\
\
\
       def baseline\_model():\
\
\
       \# create model\
\
\
       model = Sequential()\
\
\
       model.add(Dense(8, input\_dim=4 , activation= “relu” ))\
\
\
       model.add(Dense(3, activation= “softmax” ))\
\
\
       # Compile model\
\
\
       model.compile(loss= “categorical\_crossentropy” , optimizer= “adam” , metrics=\[“accuracy”\])\
\
\
       return model\
\
\
       estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200, batch\_size=5, verbose=0)\
\
\
       kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
       results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
       print(“Accuracy: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
       Using Theano backend.\
\
\
       Accuracy: 64.67% (15.22%)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404368)\
064. ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
     NunuJuly 4, 2017 at 12:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404653 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason,\
\
\
     How can I increase the accuracy while training ? I am always getting an accuracy arround 68% and 70%!! even if i am chanching the optimizer, the loss function and the learning rate.\
\
\
     (I am using keras and CNN)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404653)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2017 at 10:02 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404870 "Direct link to this comment")\
\
\
\
\
\
       Here are many ideas:\
\
       [https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404870)\
\
     - ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
       NunuJuly 8, 2017 at 12:06 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405045 "Direct link to this comment")\
\
\
\
\
\
       Thanks a lot it is very useful 🙂\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405045)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)July 9, 2017 at 10:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405254 "Direct link to this comment")\
\
\
\
\
\
         Glad to hear it.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405254)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
           NunuJuly 12, 2017 at 7:27 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405665 "Direct link to this comment")\
\
\
\
\
\
           Dear Jason,\
\
\
           I have a question: my model should classify every image in one of the 4 classes that I have, should I use “categorical cross entropy” or I can use instead the “Binary cross entropy” ? because I read a lot that when there is n classes it is better to use categorical cross entropy, but also the binary one is used for the same cases. I am too much confused 🙁 can you help me in understanding this issue better!!\
\
\
           Thanks in advance,\
\
\
           Nunu\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405665)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)July 13, 2017 at 9:53 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405721 "Direct link to this comment")\
\
\
\
\
\
             When you have more than 2 classes, use categorical cross entropy.\
\
           - ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
             NunuJuly 19, 2017 at 12:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406398 "Direct link to this comment")\
\
\
\
\
\
             oh ok thanks a lot 🙂 I have another question : I used Rmsprop with different learning rates such that 0.0001, 0.001 and 0.01 and with softmax in the last dense layer everything was good so far. Then i changed from softmax to sigmoid and i tried to excuted the same program with the same learning rates used in the cas of softmax, and here i got the problem : using learning rate 0.001 i got loss and val loss NAN after 24 epochs !! In your opinion what is the reason of getting such values??\
\
\
             Thanks in advance,\
\
\
             have a nice day,\
\
\
             Nunu\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)July 19, 2017 at 8:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406442 "Direct link to this comment")\
\
\
\
\
\
             Ensure you have scaled your input/output data to the bounds of the input/output activation functions.\
\
           - ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
             NunuJuly 19, 2017 at 5:49 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406499 "Direct link to this comment")\
\
\
\
\
\
             Thanksssss 🙂\
065. ![](https://secure.gravatar.com/avatar/938d26930b69f2229fe0d252b06fc53cd903fa96287bab1ac854e64bf72ee4fc?s=40&d=mm&r=g)\
\
\
\
     SriramJuly 5, 2017 at 5:12 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404802 "Direct link to this comment")\
\
\
\
\
\
     HI Jason,\
\
\
\
     Thanks for the awesome tutorial. I have a question regarding your first hidden layer which has 8 neurons. Correct me if I’m wrong, but shouldn’t the number of neurons in a hidden layer be upperbounded by the number of inputs? (in this case 4).\
\
\
\
     Thanks,\
\
\
     Sriram\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404802)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2017 at 10:24 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404899 "Direct link to this comment")\
\
\
\
\
\
       No. There are no rules for the number of neurons in the hidden layer. Try different configurations and go with whatever robustly gives the best results on your problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404899)\
\
     - ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
       NunuJuly 13, 2017 at 8:17 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405777 "Direct link to this comment")\
\
\
\
\
\
       ok thanks a lot,\
\
\
\
       have a nice day 🙂\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405777)\
066. ![](https://secure.gravatar.com/avatar/c98cdc96074e9cf2b4368006321c82b953b2ab30562592f9334817e077fccaf4?s=40&d=mm&r=g)\
\
\
\
     riyaJuly 5, 2017 at 10:33 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404818 "Direct link to this comment")\
\
\
\
\
\
     i ran the above program and got error\
\
\
     Import error: bad magic numbers in ‘keras’:b’\\xf3\\r\\n’\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404818)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2017 at 10:25 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404901 "Direct link to this comment")\
\
\
\
\
\
       You may have a copy-paste example. Check your code file.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404901)\
067. ![](https://secure.gravatar.com/avatar/c98cdc96074e9cf2b4368006321c82b953b2ab30562592f9334817e077fccaf4?s=40&d=mm&r=g)\
\
\
\
     riyaJuly 6, 2017 at 9:43 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404949 "Direct link to this comment")\
\
\
\
\
\
     actually a pyc file was created in the same directory due to which this error occoured.After deleting the file,error was solved\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-404949)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 9, 2017 at 10:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405229 "Direct link to this comment")\
\
\
\
\
\
       Glad to hear it.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405229)\
068. ![](https://secure.gravatar.com/avatar/c98cdc96074e9cf2b4368006321c82b953b2ab30562592f9334817e077fccaf4?s=40&d=mm&r=g)\
\
\
\
     riyaJuly 7, 2017 at 9:44 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405037 "Direct link to this comment")\
\
\
\
\
\
     Hello jason,\
\
\
     how is the error calculated to adjust weights in neural network?does the classifier uses backpropgation or anything else for error correction and weight adjustment?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405037)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 9, 2017 at 10:44 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405250 "Direct link to this comment")\
\
\
\
\
\
       Yes, the backpropgation algorithm is used.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405250)\
069. ![](https://secure.gravatar.com/avatar/c98cdc96074e9cf2b4368006321c82b953b2ab30562592f9334817e077fccaf4?s=40&d=mm&r=g)\
\
\
\
     riyaJuly 9, 2017 at 7:15 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405302 "Direct link to this comment")\
\
\
\
\
\
     Thanks jason\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405302)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2017 at 10:15 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405461 "Direct link to this comment")\
\
\
\
\
\
       You’re welcome.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-405461)\
070. ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
     NunuJuly 19, 2017 at 6:27 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406504 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason,\
\
\
     In my classifier I have 4 classes and as I know the last Dense layer should also have 4 outputs correct me please if i am wrong :). Now I want to change the number of classes from 4 to 2 !! my dataset is labeled as follows :\
\
\
     1) BirdYES\_TreeNo\
\
\
     2) BirdNo\_TreeNo\
\
\
     3)BirdYES\_TreeYES\
\
\
     4)BirdNo\_TreeYES\
\
\
     At the begining my output vector that i did was \[0,0,0,0\] in such a way that it can take 1 in the first place and all the rest are zeros if the image labeled as BirdYES\_TreeNo and it can take 1 in the second place if it is labeled as BirdNo\_TreeNo and so on…\
\
\
\
     Can you give me any hint inorder to convert these 4 classes into only 2 ( is there a function in Python that can do this ?) class Bird and class Tree in which every class takes 2 values 1 and 0 ( 1 indicates the exsistance of a Bird/Tree and 0 indicates that there is no Bird/Tree). I hope that my explanation is clear.\
\
\
     I will appreciate so much any answer from your side.\
\
\
     Thanks in advance,\
\
\
     have a nice day,\
\
\
     Nunu\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406504)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 20, 2017 at 6:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406548 "Direct link to this comment")\
\
\
\
\
\
       Yes, the number of nodes in the output layer should match the number of classes.\
\
\
\
       Unless the number of classes is 2, in which case you can use a sigmoid activation function with a single neuron. Remember to change loss to binary\_crossentropy.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406548)\
071. ![](https://secure.gravatar.com/avatar/5aea53efac397710831066bcd4b8fbf592de3cee966c31827d093e6ec80c32c2?s=40&d=mm&r=g)\
\
\
\
     NunuJuly 20, 2017 at 6:07 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406619 "Direct link to this comment")\
\
\
\
\
\
     Thanks a lot for your help i will try it.\
\
\
\
     Have a nice day,\
\
\
     Nunu\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406619)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 21, 2017 at 9:32 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406712 "Direct link to this comment")\
\
\
\
\
\
       Good luck!\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-406712)\
\
     - ![](https://secure.gravatar.com/avatar/7f8b285e09cc73c440d20759e4054537629b931ebc0db673d3b1717ace3cafd4?s=40&d=mm&r=g)\
\
\
\
       Quang Huy ChuJune 7, 2020 at 9:46 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538479 "Direct link to this comment")\
\
\
\
\
\
       Hi Jason.\
\
\
\
       Can we use this baseline model to predict new data?\
\
\
\
       If yes, we use the function model.evaluate() or model.predict() ?\
\
\
\
       Thank you very much.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538479)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)June 7, 2020 at 1:13 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538496 "Direct link to this comment")\
\
\
\
\
\
         Yes, you can fit the model on all available data and use the predict() function from scikit-learn API.\
\
\
\
         If this is new to you, see this tutorial:\
\
         [https://machinelearningmastery.com/make-predictions-scikit-learn/](https://machinelearningmastery.com/make-predictions-scikit-learn/)\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538496)\
072. ![](https://secure.gravatar.com/avatar/a27929d5b751402303d088de2cb9ab7eeaa028d36d4559f8a1afd2f827c459a8?s=40&d=mm&r=g)\
\
\
\
     PrathmJuly 26, 2017 at 8:32 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-407344 "Direct link to this comment")\
\
\
\
\
\
     kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
\
     This line is giving me follwing error:\
\
\
\
     File “C:\\Users\\pratmerc\\AppData\\Local\\Continuum\\Anaconda3\\lib\\site-\
\
\
     packages\\pandas\\core\\indexing.py”, line 1231, in \_convert\_to\_indexer raise KeyError(‘%s\
\
\
     not in index’ % objarr\[mask\])\
\
\
\
     KeyError: ‘\[41421 7755 11349 16135 36853\] not in index’\
\
\
\
     Can you please help ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-407344)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 26, 2017 at 3:58 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-407385 "Direct link to this comment")\
\
\
\
\
\
       I’m sorry to hear that, perhaps check the data that you have loaded?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-407385)\
073. ![](https://secure.gravatar.com/avatar/d6790d5c890c79c3969778e7d025be236e03b800db5276cbb8bb86b713c2a2b5?s=40&d=mm&r=g)\
\
\
\
     Q. I.August 5, 2017 at 5:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-408565 "Direct link to this comment")\
\
\
\
\
\
     Hi,\
\
\
\
     Thanks for a great site. New visitor. I have a question. In line 38 in your code above, which is “print(encoder.inverse\_transform(predictions))”, don’t you have to do un-one-hot-encoded or reverse one-hot-encoded first to do encoder.inverse\_transform(predictions)?\
\
\
\
     Thanks.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-408565)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 5, 2017 at 5:49 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-408574 "Direct link to this comment")\
\
\
\
\
\
       Normally yes, here I would guess that the learn wrapper predicted integers directly (I don’t recall the specifics off hand).\
\
\
\
       Try printing the outcome of predict() to confirm.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-408574)\
074. ![](https://secure.gravatar.com/avatar/cf04fc75fc08cc603a9230dff2dbb81995b1e249857904c4eb660eb5ecc8367f?s=40&d=mm&r=g)\
\
\
\
     Hernando SalasAugust 11, 2017 at 5:16 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409258 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I really enjoy your tutorials awesome at presenting the material. I’m a little bit puzzled by the results of this project as I get %44 rather than %95 which is a huge difference. I have used your code as follows in ipython notebook online:\
\
\
\
     import numpy\
\
\
     import pandas\
\
\
     from keras.models import Sequential\
\
\
     from keras.layers import Dense\
\
\
     from keras.wrappers.scikit\_learn import KerasClassifier\
\
\
     from keras.utils import np\_utils\
\
\
     from sklearn.cross\_validation import cross\_val\_score, KFold\
\
\
     from sklearn.preprocessing import LabelEncoder\
\
\
     from sklearn.pipeline import Pipeline\
\
\
\
     \# fix random seed for reproducibility\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed)\
\
\
\
     \# load dataset\
\
\
     dataframe = pandas.read\_csv(“iris.csv”, header=None)\
\
\
     dataset = dataframe.values\
\
\
     X = dataset\[:,0:4\].astype(float)\
\
\
     Y = dataset\[:,4\]\
\
\
\
     #encode class values as integers\
\
\
     encoder = LabelEncoder()\
\
\
     encoder.fit(Y)\
\
\
     encoded\_Y = encoder.transform(Y)\
\
\
\
     \# convert integers to dummy variables (hot encoded)\
\
\
     dummy\_y = np\_utils.to\_categorical(encoded\_Y)\
\
\
\
     \# define baseline model\
\
\
     def baseline\_model():\
\
\
     # create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(4, input\_dim=4, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(3, init=’normal’, activation=’sigmoid’))\
\
\
     # Compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
\
     return model\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200, batch\_size=5, verbose=0)\
\
\
     kfold = KFold(n=len(X), n\_folds=10, shuffle=True, random\_state=seed)\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
\
     print(“Accuracy: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409258)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 11, 2017 at 6:46 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409277 "Direct link to this comment")\
\
\
\
\
\
       The algorithm is stochastic, so you will get different results each time it is run, try running it multiple times and take the average.\
\
\
\
       More about the stochastic nature of the algorithms here:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409277)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/cf04fc75fc08cc603a9230dff2dbb81995b1e249857904c4eb660eb5ecc8367f?s=40&d=mm&r=g)\
\
\
\
         Hernando SalasAugust 15, 2017 at 5:41 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409623 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
\
         Thanks for the reply. Run several times and got the same result. Any ideas?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409623)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/cf04fc75fc08cc603a9230dff2dbb81995b1e249857904c4eb660eb5ecc8367f?s=40&d=mm&r=g)\
\
\
\
           Hernando SalasAugust 15, 2017 at 5:43 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409624 "Direct link to this comment")\
\
\
\
\
\
           [https://notebooks.azure.com/hernandosalas/libraries/deeplearning/html/main.ipynb](https://notebooks.azure.com/hernandosalas/libraries/deeplearning/html/main.ipynb)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409624)\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2017 at 6:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409641 "Direct link to this comment")\
\
\
\
\
\
           You could try varying the configuration of the network to see if that has an effect?\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409641)\
075. ![](https://secure.gravatar.com/avatar/cf04fc75fc08cc603a9230dff2dbb81995b1e249857904c4eb660eb5ecc8367f?s=40&d=mm&r=g)\
\
\
\
     Hernando SalasAugust 16, 2017 at 5:02 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409743 "Direct link to this comment")\
\
\
\
\
\
     If I set it to:\
\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(4, input\_dim=4, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(3, init=’normal’, activation=’sigmoid’))\
\
\
\
     I get Accuracy: 44.00% (17.44%) everytime\
\
\
\
     If I set it to:\
\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(8, input\_dim=4, init=’normal’, activation=’relu’))\
\
\
     model.add(Dense(3, init=’normal’, activation=’softmax’))\
\
\
\
     I get Accuracy: 64.00% (10.83%) everytime\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409743)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 16, 2017 at 6:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409762 "Direct link to this comment")\
\
\
\
\
\
       Interesting. Thanks for sharing.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-409762)\
076. ![](https://secure.gravatar.com/avatar/96ee6e73d1c7c9e9842100ac8038ebbc9f0d3f640c808ba7c37829bf0fc1453e?s=40&d=mm&r=g)\
\
\
\
     AkashAugust 22, 2017 at 12:42 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-410465 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thank you for your wonderful tutorial and it was really helpful. I just want to ask if we can perform grid search cv also the similar way because I am not able to do it right now?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-410465)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 22, 2017 at 6:44 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-410507 "Direct link to this comment")\
\
\
\
\
\
       Yes, see this post:\
\
       [https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-410507)\
077. ![](https://secure.gravatar.com/avatar/1624767aea85bf3493d1dac77955f7f74a3f0316677e63d379702d3193652a71?s=40&d=mm&r=g)\
\
\
\
     AlexanderSeptember 9, 2017 at 6:56 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413167 "Direct link to this comment")\
\
\
\
\
\
     Hi, Jason. Thank you for beautiful work.\
\
\
     Help me please.\
\
\
     Where (in which folder, directory) should i save file “iris.csv” to use this code? Now system doesn’t see this file, when I write “dataframe=pandas.read\_csv….”\
\
\
\
     4\. Load The Dataset\
\
\
     The dataset can be loaded directly. Because the output variable contains strings, it is easiest to load the data using pandas. We can then split the attributes (columns) into input variables (X) and output variables (Y).\
\
\
     \# load dataset\
\
\
     dataframe = pandas.read\_csv(“iris.csv”, header=None)\
\
\
     dataset = dataframe.values\
\
\
     X = dataset\[:,0:4\].astype(float)\
\
\
     Y = dataset\[:,4\]\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413167)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 9, 2017 at 12:01 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413235 "Direct link to this comment")\
\
\
\
\
\
       Download it and place it in the same directory as your Python code file.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413235)\
078. ![](https://secure.gravatar.com/avatar/1624767aea85bf3493d1dac77955f7f74a3f0316677e63d379702d3193652a71?s=40&d=mm&r=g)\
\
\
\
     AlexanderSeptember 9, 2017 at 5:59 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413261 "Direct link to this comment")\
\
\
\
\
\
     Thank you, Jason. I’ll try.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-413261)\
\
079. ![](https://secure.gravatar.com/avatar/454a7093a02d834c3bf6ef2bb3cf4ae71911493040a175c19ae7c64ca24e0567?s=40&d=mm&r=g)\
\
\
\
     Tran MinhSeptember 18, 2017 at 4:29 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414189 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, thank you for your great instruction\
\
\
     I follow your code but unfortunately, I get only 68%~70% accuracy rate.\
\
\
     I use Tensorflow backend and modified seed as well as the number of hidden units but I still can’t reach to 90% of accuracy rate.\
\
\
\
     Do you have any idea how to improve it\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414189)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 19, 2017 at 7:32 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414234 "Direct link to this comment")\
\
\
\
\
\
       Perhaps try running the example a few times, see this post:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414234)\
080. ![](https://secure.gravatar.com/avatar/bca96e2b76ed725383f681e027bf1f4acea7aa224c5ab6d780c2bb2fcaa42e4d?s=40&d=mm&r=g)\
\
\
\
     GregSeptember 21, 2017 at 8:23 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414423 "Direct link to this comment")\
\
\
\
\
\
     Jason,\
\
\
\
     First thanks so much for a great post.\
\
\
\
     I cut and pasted the code above and got the following run times with a GTX 1060\
\
\
\
     real 2m49.436s\
\
\
     user 4m46.852s\
\
\
     sys 0m21.944s\
\
\
\
     and running without the GPU\
\
\
\
     124.93 user 25.74 system 1:04.90 elapsed 232% CPU\
\
\
\
     Is this reasonable? It seems slow for a toy problem.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414423)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 21, 2017 at 4:19 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414440 "Direct link to this comment")\
\
\
\
\
\
       Thanks for sharing.\
\
\
\
       Yes, LSTMs are slower than MLPs generally.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414440)\
081. ![](https://secure.gravatar.com/avatar/2f6d5ddd02fc106558ecce14ed9891ac7e3c3b95656ff3458dcd749570b80b01?s=40&d=mm&r=g)\
\
\
\
     BeeSeptember 27, 2017 at 1:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414873 "Direct link to this comment")\
\
\
\
\
\
     Hi Dr. Jason,\
\
\
\
     It’s a great tutorial. Do you have any similar tutorials for Unsupervised classification too?\
\
\
\
     Thanks,\
\
\
     Bee\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414873)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 27, 2017 at 5:43 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414898 "Direct link to this comment")\
\
\
\
\
\
       Unsupervised methods cannot be used for classification, only supervised methods.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-414898)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/75491ea9a309cca832ad1b56da88b595fd592b00f37aaa987b79775fedd69352?s=40&d=mm&r=g)\
\
\
\
         BeeOctober 2, 2017 at 5:08 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415286 "Direct link to this comment")\
\
\
\
\
\
         Sorry, it was my poor choice of words. What I meant was clustering data using unsupervised methods when I don’t have labels. Is that possible with Keras?\
\
\
\
         Thanks,\
\
\
         Bee\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415286)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)October 2, 2017 at 9:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415308 "Direct link to this comment")\
\
\
\
\
\
           It may be, but I do not have examples of working with unsupervised methods, sorry.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415308)\
082. ![](https://secure.gravatar.com/avatar/dee0eee1546eaaef32fd508038f1bba8848a977cd5160d93ebd043577ec9bdfe?s=40&d=mm&r=g)\
\
\
\
     MiqueiasOctober 3, 2017 at 8:48 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415391 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for you work describing in a very nice way how to use Keras! I’ve a question about the performance of categorical classification versus the binary one. Suppose you have a class for something you call your signal and, then, many other classes which you would call background. In that case, which way is more efficient to work on Keras: merging the different background classes and considering all of them as just one background class and then use binary classification or use a categorical one to account all the classes? In other words, is one way more sensible than the other for keras learn well the features from all the classes?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415391)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 3, 2017 at 3:44 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415431 "Direct link to this comment")\
\
\
\
\
\
       Great question.\
\
\
\
       It really depends on the specific data. I would recommend designing some experiments to see what works best.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415431)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/dee0eee1546eaaef32fd508038f1bba8848a977cd5160d93ebd043577ec9bdfe?s=40&d=mm&r=g)\
\
\
\
         MiqueiasOctober 3, 2017 at 10:52 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415476 "Direct link to this comment")\
\
\
\
\
\
         Thanks for fast replay Jason!\
\
\
         I’ll try that to see what I get.\
\
\
         I’m wondering if in categorical classification Keras can build up independent functions inside it. Because, since the background classes may exist in different phase space regions (what would be more truthfully described by separated functions), training the net with all of them together for binary classification may not extract all the features from each one. In principle, that could be done with a single net but, it would probably require more neurons (which increases the over-fitting issue).\
\
\
         By the way, what do you think about training different nets for signal vs. each background? Could they be combined in the end?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415476)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)October 4, 2017 at 5:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415513 "Direct link to this comment")\
\
\
\
\
\
           If the classes are separable I would encourage you to model them as separate problems.\
\
\
\
           Nevertheless, the best advice is always to test each idea and see what works best on your problem.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-415513)\
083. ![](https://secure.gravatar.com/avatar/a8330b65cdc7e252f9f332be3440a90ca729fc77b398e06ce68668ac2a504000?s=40&d=mm&r=g)\
\
\
\
     DaveOctober 11, 2017 at 5:22 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416286 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason! I have a question about multi classification\
\
\
\
     I would like to classify the 3 class of sleep disordered breathing.\
\
\
\
     I designed the LSTM network. but it works like under the table.\
\
\
\
     What is this situation?\
\
\
\
     Train matrix: precision recall f1-score support\
\
\
\
     0 0.00 0.00 0.00 1749\
\
\
     1 0.46 1.00 0.63 2979\
\
\
     2 0.00 0.00 0.00 1760\
\
\
\
     avg / total 0.21 0.46 0.29 6488\
\
\
\
     Train matrix: precision recall f1-score support\
\
\
\
     0 0.00 0.00 0.00 441\
\
\
     1 0.46 1.00 0.63 750\
\
\
     2 0.00 0.00 0.00 431\
\
\
\
     avg / total 0.21 0.46 0.29 1622\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416286)\
\
084. ![](https://secure.gravatar.com/avatar/75b1d156dd1fb698de382cc3cafc4f9ac035e7fe7cc548ae13caab39709912ff?s=40&d=mm&r=g)\
\
\
\
     sasiOctober 13, 2017 at 10:48 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416580 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Does this topic will match for this tutorial??\
\
\
     “Deep learning based multiclass classification tutorial”\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416580)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 14, 2017 at 5:46 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416623 "Direct link to this comment")\
\
\
\
\
\
       Yes.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416623)\
085. ![](https://secure.gravatar.com/avatar/9425845f96b964a1a02f1a09f90ac3af7c24edb91eba2eb6d35d2d555e4e1713?s=40&d=mm&r=g)\
\
\
\
     zaheerOctober 16, 2017 at 11:48 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416935 "Direct link to this comment")\
\
\
\
\
\
     This tutorial is awsom. thanks for your time.\
\
\
     My data is\
\
\
     404\. instances\
\
\
     2\. class label. A/B\
\
\
     20\. attribute columns.\
\
\
\
     i have tried the this example gives me 58% acc.\
\
\
     model = Sequential()\
\
\
     model.add(Dense(200, input\_dim=20, activation=’relu’))\
\
\
     model.add(Dense(2, activation=’softmax’))\
\
\
     # Compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     #Classifier invoking\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, epochs=200, batch\_size=5, verbose=0)\
\
\
\
     what should i do, how to increase the acc of the system\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416935)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 17, 2017 at 5:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416983 "Direct link to this comment")\
\
\
\
\
\
       See this post for a ton of ideas:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-416983)\
086. ![](https://secure.gravatar.com/avatar/c0ce3ac943598df25b0bb8750fd5bc7a337aa7e3adc0d5ef98a4321155118615?s=40&d=mm&r=g)\
\
\
\
     Curious\_KidOctober 24, 2017 at 1:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417661 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     My training data consists of lines of characters with each line corresponding to a label.\
\
\
\
     E.g. afhafajkfajkfhafkahfahk 6\
\
\
\
     fafafafafaftuiowjtwojdfafanfa 8\
\
\
\
     dakworfwajanfnafjajfahifqnfqqfnq 4\
\
\
\
     Here, 6,8 and 4 are labels for each line of the training data.\
\
\
     ……………………………………………………..\
\
\
\
     I have first done the integer encoding for each character and then done the one hot encoding. To keep the integer encoding consistent, I first looked for the unique letters in all the rows and then did the integer encoding. e.g. that’s why letter h will always be encoded as 7 in all the lines.\
\
\
\
     For a better understanding, consider a simple example where my training data has 3 lines(each line has some label):\
\
\
     af\
\
\
     fa\
\
\
     nf\
\
\
\
     It will be one hot encoded as:\
\
\
\
     0 \[\[1.0, 0.0, 0.0\], \[0.0, 1.0, 0.0\]\]\
\
\
     1 \[\[0.0, 1.0, 0.0\], \[1.0, 0.0, 0.0\]\]\
\
\
     2 \[\[0.0, 0.0, 1.0\], \[0.0, 1.0, 0.0\]\]\
\
\
\
     I wanted to do the classification for the unseen data(which label does the new line belong to) by training a neural network on this one hot encoded training data.\
\
\
\
     I am not able to understand how my model should look like as I want the model to learn from each one hot encoded character for each line. Could you please suggest me something in this case? Please let me know if you need more information to understand the problem.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417661)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 24, 2017 at 5:33 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417688 "Direct link to this comment")\
\
\
\
\
\
       This is a sequence classification task.\
\
\
\
       Perhaps this post will give you a template to get started:\
\
       [https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417688)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/c0ce3ac943598df25b0bb8750fd5bc7a337aa7e3adc0d5ef98a4321155118615?s=40&d=mm&r=g)\
\
\
\
         Curious\_KidOctober 24, 2017 at 6:10 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417695 "Direct link to this comment")\
\
\
\
\
\
         Thanks Jason for the reply.\
\
\
\
         However, I am not dealing with words. I just have characters in a line and I am doing one hot encoding for each character in a single line as I explained above. What I am confused with is the shapes that I have to give to the layers of my network.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417695)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)October 24, 2017 at 3:56 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417730 "Direct link to this comment")\
\
\
\
\
\
           I see, perhaps this post will help with reshaping your data:\
\
           [https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-417730)\
     - ![](https://secure.gravatar.com/avatar/34bdf2e3804b0442e0cac1025688be92a4b29d15d23b52a75ca4d700c3c3a909?s=40&d=mm&r=g)\
\
\
\
       Nrithya MuniswamyJanuary 26, 2018 at 1:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-427595 "Direct link to this comment")\
\
\
\
\
\
       @Curious\_Kid : did you find a workaround, I am dealing with same problem\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-427595)\
087. ![](https://secure.gravatar.com/avatar/5ee20ff7ad353c13204f30e184b8dbe65ddd9eff966dc3b6d931667e6d99cd9e?s=40&d=mm&r=g)\
\
\
\
     philippeNovember 6, 2017 at 9:10 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419147 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     very clear tutorial. one quick question, how do you decide on the number of hidden neurons (in classification case). it seems to follow (Hidden neurons = input \* 2) , how about \* 1 or \*3 is there a rule. same goes for epoch ; how do you choose nbr of iterations;\
\
\
\
     thanks.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419147)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 7, 2017 at 9:49 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419197 "Direct link to this comment")\
\
\
\
\
\
       There are no good rules, use trial and error or a robust test harness and a grid search.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419197)\
088. ![](https://secure.gravatar.com/avatar/03eac36696d5f3b60cb1b469d350de6b85b3ea976987110fe08aa54ef28b7054?s=40&d=mm&r=g)\
\
\
\
     Niklas WilkeNovember 13, 2017 at 11:26 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419918 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     great tutorial!\
\
\
     I’ve got a multi class classification problem. I try to classify different kind of bills into categorys (that are given !! no clustering!!) , like flight, train/bus, meal, hotels and so on.\
\
\
     I got a couple files in PDF which i transform in PNG to make it processable by MC Computer Vision using OCR.\
\
\
     After that i come out with a .txt or .csv file of the plain text.\
\
\
     Now i used skelarns vectorizers to create a bag of words and fit the single bills/documents.\
\
\
     Ending up with numpy-arrays looking like this (sample data i used to craete the code while i was gathering data):\
\
\
\
     \[\[3 0 1 1 0 0 0 0 2 0 2 2 1 3 1 1 0 3 0 0 3 2 1 0 1 3 1 0 0 5 0 0 1 1 0 1 0\
\
\
     0 1 1 1 1 0 1 0 1 0 1 0 2 0 2 1 0 1 0 1 1 1 1 1 0 0 1 0 1 1 1 1 0 0 1 1 1\
\
\
     0 1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 1 2 1 0 0 0 0 0 0 0 2 1 0 0 0 2 1 0 1 0 1\
\
\
     0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 2 0 0 0 0 0 0 2 0 1\
\
\
     0 0 1 1 0 0 1 1 1 0 0 1 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 1 0 0 1 1 1 0 2 0 0\
\
\
     0 1 4 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 2 3 0 1 0 0 0 0 0 1 0 3 0 1 0\
\
\
     1 1 0 0 0 0 0 0 1 2 0 0 0 3 0 0 0 1 0 0 0 1 1 0 2 0 0 0 0 1 0 1 1 0 0 1 0\
\
\
     1 1 0 1 0 0 1 0 0 0 0 1 0\]\]\
\
\
\
     How do i categoryze or transform this to something like the iris dataset ?\
\
\
     Isn’t it basically the same ? Just with way more numbers and bigger arrrays ?\
\
\
     I tried to iterate through the array to print every single number in a .csv-file and then just append the category at the back with some for loops but sadly you can’t iterate through numpy-arrays … + i can’t imagine that’s the intended way of labeling data …\
\
\
\
     Thanks for reading through this way too long comment , help is highly apreciated.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419918)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 14, 2017 at 10:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419951 "Direct link to this comment")\
\
\
\
\
\
       Yes, the vectorized documents become input to ML algorithms.\
\
\
\
       I’d love to hear how you go, post your results!\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-419951)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/03eac36696d5f3b60cb1b469d350de6b85b3ea976987110fe08aa54ef28b7054?s=40&d=mm&r=g)\
\
\
\
         Niklas WilkeNovember 30, 2017 at 1:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421631 "Direct link to this comment")\
\
\
\
\
\
         Finally solved all my preprocessing problems and today i was able to perform my first training trial runns with my actual dataset. (Btw : buffer\_y = dummy\_y)\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
         |     |     |\
         | --- | --- |\
         | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16 | def createModell():<br>#8137 words = input shape<br>#14 categorys = output shape<br>model=Sequential()<br>model.add(Dense(8137,input\_dim=8137,activation='relu'))<br>model.add(Dense(2250,activation='relu'))<br>#model.add(Dense(581,  activation='relu'))<br>model.add(Dense(14,activation='softmax'))<br>model.compile(loss='categorical\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>returnmodel<br>estimator=KerasClassifier(build\_fn=createModell,epochs=10,batch\_size=6)<br>crossvalidation\_data=KFold(n\_splits=39,shuffle=True)<br>results=cross\_val\_score(estimator,X,buffer\_y,cv=crossvalidation\_data) |\
\
\
\
\
\
\
\
\
\
\
\
         And hell am i overfitting.\
\
\
         0.98 acuraccy , which can’t be because my dataset is horribly unbalanced. (maybe thats the issue?)\
\
\
\
         Anyhow, i enabled the print option and for me it only displays 564/564 sample files for every epoche even though my dataset contains 579 … i check for you example and it also only displays 140/140 even though the iris dataset is 150 files big.\
\
\
\
         Are the splits to high ?\
\
\
         and what is a good amount of nodes for such a high input shape :/ tried to split it up to multiple layers so its not 8139 -> 4000-> 14\
\
\
\
         Cheers\
\
\
         Niklas\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421631)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)November 30, 2017 at 8:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421667 "Direct link to this comment")\
\
\
\
\
\
           Well done!\
\
\
\
           Consider the options in this post for imbalanced data:\
\
           [https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)\
\
\
\
           The count is wrong because you are using cross-validation (e.g. not all samples for each run).\
\
\
\
           You must use trial and error to explore alternative configurations, here are some ideas:\
\
           [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
           I hope that helps as a start.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421667)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/03eac36696d5f3b60cb1b469d350de6b85b3ea976987110fe08aa54ef28b7054?s=40&d=mm&r=g)\
\
\
\
             Niklas WilkeNovember 30, 2017 at 6:52 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421729 "Direct link to this comment")\
\
\
\
\
\
             Ah ok , good point. When i create 10 splits it only uses 521 files => 90% of 579\
\
\
\
             Will look into it and post my hopefully sucessfull results here.\
\
\
\
             Given that i had no issue with the imbalance of my dataset, is the general amount of nodes or layers alright ? I have literally no clue because all the tipps ive found so far refer to way smaller input shapes like 4 or 8.\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)December 1, 2017 at 7:28 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421796 "Direct link to this comment")\
\
\
\
\
\
             There are no good rules of thumb, I recommend testing a suite of configurations to see what works best for your problem.\
\
           - ![](https://secure.gravatar.com/avatar/03eac36696d5f3b60cb1b469d350de6b85b3ea976987110fe08aa54ef28b7054?s=40&d=mm&r=g)\
\
\
\
             Niklas WilkeNovember 30, 2017 at 7:34 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421732 "Direct link to this comment")\
\
\
\
\
\
             I read you mentioned other classifiers like decision trees performing well on imbalanced datasets.\
\
\
\
             Is there some way i can use other classifiers INSIDE of my NN ?\
\
\
\
             for example could i implement naive bayes into my NN ?\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)December 1, 2017 at 7:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421797 "Direct link to this comment")\
\
\
\
\
\
             Not that I am aware.\
\
\
\
             You could combine the predictions from multiple models into an ensemble though.\
089. ![](https://secure.gravatar.com/avatar/03eac36696d5f3b60cb1b469d350de6b85b3ea976987110fe08aa54ef28b7054?s=40&d=mm&r=g)\
\
\
\
     Niklas WilkeNovember 30, 2017 at 6:59 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421730 "Direct link to this comment")\
\
\
\
\
\
     Btw, even though i tell it to run 10 epoches , after the 10 epoches it just starts again with slightly different values. In your example it doesnt.\
\
\
\
     Epoch 1/10\
\
\
     521/521 \[==============================\] – 12s – loss: 2.0381 – acc: 0.4952\
\
\
     Epoch 2/10\
\
\
     521/521 \[==============================\] – 10s – loss: 0.3139 – acc: 0.9443\
\
\
     Epoch 3/10\
\
\
     521/521 \[==============================\] – 10s – loss: 0.0748 – acc: 0.9866\
\
\
     Epoch 4/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0578 – acc: 0.9942\
\
\
     Epoch 5/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0434 – acc: 0.9962\
\
\
     Epoch 6/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0352 – acc: 0.9962\
\
\
     Epoch 7/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0321 – acc: 0.9981\
\
\
     Epoch 8/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0314 – acc: 0.9981\
\
\
     Epoch 9/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0312 – acc: 0.9981\
\
\
     Epoch 10/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0311 – acc: 0.9981\
\
\
     58/58 \[==============================\] – 0s\
\
\
     Epoch 1/10\
\
\
     521/521 \[==============================\] – 13s – loss: 1.9028 – acc: 0.4722\
\
\
     Epoch 2/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.2883 – acc: 0.9463\
\
\
     Epoch 3/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.1044 – acc: 0.9770\
\
\
     Epoch 4/10\
\
\
     521/521 \[==============================\] – 11s – loss: 0.0543 – acc: 0.9942\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421730)\
\
090. ![](https://secure.gravatar.com/avatar/03eac36696d5f3b60cb1b469d350de6b85b3ea976987110fe08aa54ef28b7054?s=40&d=mm&r=g)\
\
\
\
     Niklas WilkeNovember 30, 2017 at 11:38 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421753 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     could you please comment on this blog entry :\
\
     [http://www.alfredo.motta.name/cross-validation-done-wrong/](http://www.alfredo.motta.name/cross-validation-done-wrong/)\
\
\
\
     Sounds pretty logical to me and isnt that exactly what we are doing here ?\
\
\
     If we ignore the feature selection part, we also split the data first and afterwards train the model ….\
\
\
\
     Thanks in advance\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-421753)\
\
091. ![](https://secure.gravatar.com/avatar/a55fb80141cebd84b610439b647bc479c863b967c36621dfb418f6ee318bde98?s=40&d=mm&r=g)\
\
\
\
     Summer CassidyDecember 16, 2017 at 12:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423272 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason Brownlee,\
\
\
     When I run the code I get an error. I have checked multiple times whether I have copied the code correctly. I am unable to trace why the error is occurring. Can you please help me out?\
\
\
     The error is:\
\
\
\
     Traceback (most recent call last):\
\
\
     File “F:/7th semester/machine language/thesis work/python/iris2.py”, line 36, in\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\model\_selection\\\_validation.py”, line 342, in cross\_val\_score\
\
\
     pre\_dispatch=pre\_dispatch)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\model\_selection\\\_validation.py”, line 206, in cross\_validate\
\
\
     for train, test in cv.split(X, y, groups))\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 779, in \_\_call\_\_\
\
\
     while self.dispatch\_one\_batch(iterator):\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 625, in dispatch\_one\_batch\
\
\
     self.\_dispatch(tasks)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 588, in \_dispatch\
\
\
     job = self.\_backend.apply\_async(batch, callback=cb)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\externals\\joblib\\\_parallel\_backends.py”, line 111, in apply\_async\
\
\
     result = ImmediateResult(func)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\externals\\joblib\\\_parallel\_backends.py”, line 332, in \_\_init\_\_\
\
\
     self.results = batch()\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 131, in \_\_call\_\_\
\
\
     return \[func(\*args, \*\*kwargs) for func, args, kwargs in self.items\]\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\externals\\joblib\\parallel.py”, line 131, in\
\
\
     return \[func(\*args, \*\*kwargs) for func, args, kwargs in self.items\]\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\sklearn\\model\_selection\\\_validation.py”, line 458, in \_fit\_and\_score\
\
\
     estimator.fit(X\_train, y\_train, \*\*fit\_params)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\keras\\wrappers\\scikit\_learn.py”, line 203, in fit\
\
\
     return super(KerasClassifier, self).fit(x, y, \*\*kwargs)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\keras\\wrappers\\scikit\_learn.py”, line 147, in fit\
\
\
     history = self.model.fit(x, y, \*\*fit\_args)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\keras\\models.py”, line 960, in fit\
\
\
     validation\_steps=validation\_steps)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\keras\\engine\\training.py”, line 1581, in fit\
\
\
     batch\_size=batch\_size)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\keras\\engine\\training.py”, line 1418, in \_standardize\_user\_data\
\
\
     exception\_prefix=’target’)\
\
\
     File “C:\\Users\\ratul\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\keras\\engine\\training.py”, line 153, in \_standardize\_input\_data\
\
\
     str(array.shape))\
\
\
     ValueError: Error when checking target: expected dense\_2 to have shape (None, 3) but got array with shape (90, 40)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423272)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 16, 2017 at 5:32 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423339 "Direct link to this comment")\
\
\
\
\
\
       Looks like you might be using different data.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423339)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/a55fb80141cebd84b610439b647bc479c863b967c36621dfb418f6ee318bde98?s=40&d=mm&r=g)\
\
\
\
         Summer CassidyDecember 16, 2017 at 6:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423360 "Direct link to this comment")\
\
\
\
\
\
         Thanks for looking into the problem. I downloaded the iris flower dataset but from a different source. Changing the source to UCI Machine Learning repository solved my problem.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423360)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)December 16, 2017 at 9:21 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423396 "Direct link to this comment")\
\
\
\
\
\
           Glad to hear it!\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-423396)\
092. ![](https://secure.gravatar.com/avatar/58d37ee1c61ff56be92557ddffb1e8a4953c46bf79b3efbe1a076cb20265f96d?s=40&d=mm&r=g)\
\
\
\
     PubuduJanuary 1, 2018 at 4:52 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425357 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason:\
\
\
\
     Thanks for the tute. BTW, how do you planning to void dummy variable trap. You don’t need all three types. Can you explain why you didn’t use train\_test\_split method?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425357)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 2, 2018 at 5:34 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425383 "Direct link to this comment")\
\
\
\
\
\
       The example uses k-fold cross validation instead of a train/test split.\
\
\
\
       The results are less biased with this method and I recommend it for smaller models.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425383)\
093. ![](https://secure.gravatar.com/avatar/ef1058dc7c63bc3821964937d8498eaf77d6e523ddab91e9ad434695770385ba?s=40&d=mm&r=g)\
\
\
\
     HieuJanuary 7, 2018 at 7:38 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425833 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason,\
\
\
     Thank you for your sharing.\
\
\
     I run your source code, now I want to replace “activation=’softmax'” – (model.add(Dense(3, activation=’softmax’)) with multi-class SVM to classify. How can I do it?\
\
\
     Coul you please help me? Thank you so much!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425833)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 8, 2018 at 5:42 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425850 "Direct link to this comment")\
\
\
\
\
\
       This is a neural network example, not SVM. Perhaps I don’t understand your question. Can you restate it?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425850)\
094. ![](https://secure.gravatar.com/avatar/ef1058dc7c63bc3821964937d8498eaf77d6e523ddab91e9ad434695770385ba?s=40&d=mm&r=g)\
\
\
\
     HieuJanuary 9, 2018 at 8:39 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425986 "Direct link to this comment")\
\
\
\
\
\
     Dear Jaso,\
\
\
     Thank you for your reply.\
\
\
     Because your example uses “Softmax regression” method to classify, Now I want to use “multi-class SVM” method to add to the neural network to classify. When using SVM method, the accuracy of training data doesn’t change in each iteration and I only got 9.5% after training.\
\
\
     This is my code\
\
\
     ……\
\
\
     model.add(Dense(1000, activation=’relu’))\
\
\
\
     #=======for softmax============\
\
\
     \# model.add(Dense(10, activation=’softmax’))\
\
\
     \# model.compile(loss=keras.losses.categorical\_crossentropy,\
\
\
     \# optimizer=keras.optimizers.Adam(),\
\
\
     \# metrics=\[‘accuracy’\])\
\
\
\
     #========for SVM ==============\
\
\
     model.add(Dense(10, kernel\_regularizer=regularizers.l2(0.01), activity\_regularizer=regularizers.l1(0.01)))\
\
\
     model.add(Activation(‘linear’))\
\
\
     model.compile(loss=’hinge’,\
\
\
     optimizer=’sgd’,\
\
\
     metrics=\[‘accuracy’\])\
\
\
\
     Thank you!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-425986)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 10, 2018 at 5:24 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426057 "Direct link to this comment")\
\
\
\
\
\
       Here are some ideas to try:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426057)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/ef1058dc7c63bc3821964937d8498eaf77d6e523ddab91e9ad434695770385ba?s=40&d=mm&r=g)\
\
\
\
         HieuJanuary 11, 2018 at 6:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426181 "Direct link to this comment")\
\
\
\
\
\
         Dear Jason,\
\
\
         Thank you for your help! I will read and try it.\
\
\
\
         Have a nice day.\
\
\
         Trung Hieu\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426181)\
095. ![](https://secure.gravatar.com/avatar/5d77a0ef94ce2512232267e88fd10698649261d699e89a3255b74bfed2cb3156?s=40&d=mm&r=g)\
\
\
\
     ArjunJanuary 10, 2018 at 9:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426086 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for the content.\
\
\
     Could you tell me how we could do grid search for a multi class classification problem?\
\
\
\
     I tried doing:\
\
\
     \# create model\
\
\
     model = KerasClassifier(build\_fn=neural, verbose=0)\
\
\
\
     \# define the grid search parameters\
\
\
     batch\_size = \[10, 20, 40, 60, 80, 100\]\
\
\
     epochs = \[10, 50, 100\]\
\
\
     param\_grid = dict(batch\_size=batch\_size, epochs=epochs)\
\
\
     grid = GridSearchCV(estimator=model, param\_grid=param\_grid, n\_jobs=-1)\
\
\
     grid\_result = grid.fit(X\_train, Y\_train)\
\
\
     \# summarize results\
\
\
     print(“Best: %f using %s” % (grid\_result.best\_score\_, grid\_result.best\_params\_))\
\
\
     means = grid\_result.cv\_results\_\[‘mean\_test\_score’\]\
\
\
     stds = grid\_result.cv\_results\_\[‘std\_test\_score’\]\
\
\
     params = grid\_result.cv\_results\_\[‘params’\]\
\
\
     for mean, stdev, param in zip(means, stds, params):\
\
\
     print(“%f (%f) with: %r” % (mean, stdev, param))\
\
\
\
     but its giving me an error saying :\
\
\
     ValueError: Invalid shape for y: ()\
\
\
\
     I had one hot encoded the Y variable( having 3 classes)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426086)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 10, 2018 at 3:41 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426107 "Direct link to this comment")\
\
\
\
\
\
       Looks like you might need to one hot encode your output data.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426107)\
096. ![](https://secure.gravatar.com/avatar/18586999cf3d1073098cea7939efd7ef42633a2e330cf4fa470509246a1f05b4?s=40&d=mm&r=g)\
\
\
\
     BudiJanuary 15, 2018 at 4:03 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426565 "Direct link to this comment")\
\
\
\
\
\
     Another nice result..\
\
\
\
     Using TensorFlow backend.\
\
\
     2018-01-15 00:01:58.609360: I tensorflow/core/platform/cpu\_feature\_guard.cc:137\] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX


     Baseline: 97.33% (4.42%)



     but, could you explain what the meaning of my CPU support instruction..



     thanks alot..



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426565)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 15, 2018 at 7:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426589 "Direct link to this comment")





       Well done.



       You can ignore that warning.



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426589)
097. ![](https://secure.gravatar.com/avatar/f65477121a1c68bd5bde814be27146cc3d1b4abba08724e6152ebf84389fc527?s=40&d=mm&r=g)



     kristiJanuary 18, 2018 at 3:14 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426826 "Direct link to this comment")





     I’m getting accuracy 0f 33.3% only.I’m using keras2



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426826)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 19, 2018 at 6:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426872 "Direct link to this comment")





       Perhaps try running the example again?



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426872)
098. ![](https://secure.gravatar.com/avatar/c22937e1a9792ca3c473f7722be02f21d0726dce2f224f1f26e66872aebbd191?s=40&d=mm&r=g)



     ShivangJanuary 19, 2018 at 1:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426855 "Direct link to this comment")





     Hey Jason,


     How would you handle the dummy variable trap? In this case, we have 3 categories by applying One hot encoding we get three columns but we can work with only two of them to avoid this dummy variable trap.


     Please tell how is it handled here?



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426855)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 19, 2018 at 6:34 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426887 "Direct link to this comment")





       What trap are you referring to?



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426887)




       - ![](https://secure.gravatar.com/avatar/c22937e1a9792ca3c473f7722be02f21d0726dce2f224f1f26e66872aebbd191?s=40&d=mm&r=g)



         ShivangJanuary 19, 2018 at 5:56 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426924 "Direct link to this comment")





         Please refer this:

         [http://www.algosome.com/articles/dummy-variable-trap-regression.html](http://www.algosome.com/articles/dummy-variable-trap-regression.html)



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426924)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)January 20, 2018 at 8:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426983 "Direct link to this comment")





           This is for inputs not outputs and is for linear models not non-linear models.



           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-426983)
099. ![](https://secure.gravatar.com/avatar/b2f5f3f709c418869c6c3747ac05a00e349dd583a23dc7cb1234195562ced4f3?s=40&d=mm&r=g)



     PradeepFebruary 2, 2018 at 3:23 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428210 "Direct link to this comment")





     Hello Jason !! Thanx for explaining in such a nice way.


     I am using the similar dataset, having multiple classes. But at the end, model give the accuracy.


     How can I visualize the individual class accuracy in terms of Precision and Recall?



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428210)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 3, 2018 at 8:33 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428265 "Direct link to this comment")





       You could collect the prediction in an array and compare them to the expected values using tools in sklearn:

       [http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428265)
100. ![](https://secure.gravatar.com/avatar/c51bfbee92a789231ca3735fafa854569cbc9dab64fb59f97d0bd0942a3cb316?s=40&d=mm&r=g)



     RahulFebruary 2, 2018 at 9:13 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428234 "Direct link to this comment")





     I want to plot confusion metrics to see the distribution of data in different classes. We got the value in the range of 0-1 for every data instances by using the softmax function.


     Out\[30\]:


     array(\[\[ 0.2284117 , 0.03548411, 0.0659482 , 0.63993007, 0.03022591\],\
\
\
     \[ 0.10440681, 0.11356669, 0.09002439, 0.63514292, 0.05685928\],\
\
\
     \[ 0.40078917, 0.11887287, 0.1319678 , 0.30179501, 0.04657512\],\
\
\
     …,\
\
\
     \[ 0.38920838, 0.09161357, 0.10990805, 0.37070984, 0.03856021\],\
\
\
     \[ 0.14154498, 0.53637242, 0.11574779, 0.18590394, 0.02043088\],\
\
\
     \[ 0.17462374, 0.02110649, 0.03105714, 0.6064955 , 0.16671705\]\], dtype=float32)



     I want to the result in only 0 and 1 format as the hight value is replaced by 1 and others are 0. How can I do this? For example, the above array should be converted into


     \[0,0,0,1,0\] and so on for different data. Please help



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428234)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 3, 2018 at 8:37 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428274 "Direct link to this comment")





       Perhaps apply the round() function?



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-428274)
101. ![](https://secure.gravatar.com/avatar/55030963f6bc34dc1a00b26bc9427b363259d3e28255d0ebfcbe6e06b7175e38?s=40&d=mm&r=g)



     CHIRANJEEVIFebruary 12, 2018 at 6:07 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429194 "Direct link to this comment")





     how can we predict output for new input values after validation ?



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429194)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 13, 2018 at 8:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429258 "Direct link to this comment")





       See this post:

       [https://machinelearningmastery.com/train-final-machine-learning-model/](https://machinelearningmastery.com/train-final-machine-learning-model/)



       Once you have a final model you can call:



       yhat = model.predict(X)



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429258)
102. ![](https://secure.gravatar.com/avatar/a43264367c740f7be89b6087672581267fa9932ba601744d6ba5dfe8d5c3a304?s=40&d=mm&r=g)



     Meroua DaoudiFebruary 19, 2018 at 12:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429863 "Direct link to this comment")





     Hi jason,



     in my problem i have multi class and one data object can belong to multiple class at time



     Do you know of any reference to this kind of problem



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429863)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 19, 2018 at 9:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429901 "Direct link to this comment")





       This is called multi-label classification:

       [https://en.wikipedia.org/wiki/Multi-label\_classification](https://en.wikipedia.org/wiki/Multi-label_classification)



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-429901)
103. ![](https://secure.gravatar.com/avatar/b91b5a0ab5ed6a962577c2e41731a7133ef403a1371f201a33b940d5fe54d1cc?s=40&d=mm&r=g)



     [Madhav Bhattarai](http://learnzone.info/)February 23, 2018 at 4:25 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430334 "Direct link to this comment")





     Hi Jason, as elegant as always. I am trying to solve the multiclass classification problem similar to this tutorial with the different dataset, where all my inputs are categorical. However, the accuracy of my model converges after achieving the accuracy of 57% and loss also converges after some point. My model doesn’t learn thereafter. Does this tutorial work for the dataset where all inputs are categorical? Is there some way to visualize and diagnose the issue?



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430334)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 24, 2018 at 9:10 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430394 "Direct link to this comment")





       This post should give you some good ideas to try:

       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430394)




       - ![](https://secure.gravatar.com/avatar/b91b5a0ab5ed6a962577c2e41731a7133ef403a1371f201a33b940d5fe54d1cc?s=40&d=mm&r=g)



         [Madhav Bhattarai](http://learnzone.info/)February 26, 2018 at 4:42 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430532 "Direct link to this comment")





         Thank you so much.



         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430532)
104. ![](https://secure.gravatar.com/avatar/e099519fc466828b2fb3d50fa8bfe9654ffa2e661f3b0638ff462b73c1b37ac6?s=40&d=mm&r=g)



     YodishFebruary 26, 2018 at 11:14 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430557 "Direct link to this comment")





     Is there a way I can print all the training epochs?



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430557)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 27, 2018 at 6:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430594 "Direct link to this comment")





       Yes, you can set the verbose=1 when calling fit().



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430594)
105. ![](https://secure.gravatar.com/avatar/55030963f6bc34dc1a00b26bc9427b363259d3e28255d0ebfcbe6e06b7175e38?s=40&d=mm&r=g)



     ankithaFebruary 27, 2018 at 4:42 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430631 "Direct link to this comment")





     HI Jason


     Is it possible to train a classifier dynamically ?


     if yes how can we implement that



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430631)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 28, 2018 at 6:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430677 "Direct link to this comment")





       Yes, it is called online learning where the model is updated after each pattern.



       You can achieve this directly in Keras by setting the batch size to 1.



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430677)
106. ![](https://secure.gravatar.com/avatar/9ae1fbe85eec49a2a3ab50416ef87b80dad91249c76cf0d3d8f99caf0fa8054b?s=40&d=mm&r=g)



     VaroonsMarch 2, 2018 at 5:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430846 "Direct link to this comment")





     Thanks for these great tutorials Jason.



     I had a question on multi label classification where the labels are one-hot encoded.



     When predicting new data, how do you map the one-hot encoded outputs to the actual class labels?



     Thanks!



     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430846)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2018 at 5:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430863 "Direct link to this comment")





       You can use argmax() on the vector to get the index with the highest probability.



       Also Keras has a predict\_classes() function on the model that does the same thing.



       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-430863)
107. ![](https://secure.gravatar.com/avatar/c6ffbcf4153038c4bd69fa870d592123e89d24801ffe7a24b5b21b559a9ad1c0?s=40&d=mm&r=g)



     Gledson MelottiMarch 4, 2018 at 5:16 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431059 "Direct link to this comment")





     Hi, how are you? I really enjoyed your example over sorting using iris dataset. I have some doubts. I use anaconda with python 3.6. I installed keras. In my algorithm and I would like to assign (include) more hidden layers. How should I do it?


     For example:


     4 inputs -> \[8 hidden nodes\] -> \[8 hidden nodes -> \[12 hidden nodes\] -> 3 outputs\
\
\
\
     Then you provided, as a response to a comment, a new prediction algorithm (where we split the dataset, train on 67% and make predictions on 33%). However, you included in the network model the following command: init = ‘normal’ (line 28). Why did you do this? When you’ve split the set into training and testing, you no longer use cross-validation. Could you use cross-validation together with the training and test set division?\
\
\
\
     Other questions: How to save the training template to use in the future with other test data? How to generate the ROC curves?\
\
\
\
     Thank you very much for your attention.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431059)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 4, 2018 at 6:08 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431081 "Direct link to this comment")\
\
\
\
\
\
       To add new lauyers, just add lines to the code as follows:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       |     |     |\
       | --- | --- |\
       | 1 | model.add(...) |\
\
\
\
\
\
\
\
\
\
\
\
       And replace … with the type of layer you want to add.\
\
\
\
       I used ‘normal’ to initialize the weights. I found it gave better skill with some trial and error.\
\
\
\
       Sorry, Id on’t have an example of generating roc curves for keras models.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431081)\
108. ![](https://secure.gravatar.com/avatar/c6ffbcf4153038c4bd69fa870d592123e89d24801ffe7a24b5b21b559a9ad1c0?s=40&d=mm&r=g)\
\
\
\
     Gledson MelottiMarch 4, 2018 at 5:41 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431062 "Direct link to this comment")\
\
\
\
\
\
     Hi, how are you? I’m using python by spider-anaconda. I use your iris dataset example for sorting. However, when I use the following commands:\
\
\
     import matplotlib.pyplot as plt\
\
\
     import keras.backend as K\
\
\
     from keras import preprocessing\
\
\
     from sklearn.model\_selection import cross\_val\_score\
\
\
     from sklearn.model\_selection import KFold\
\
\
     from sklearn.pipeline import Pipeline\
\
\
\
     I get the following message: imported but unused.\
\
\
     What should I do to not receive this message?\
\
\
\
     Thank you very much for your attention.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431062)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 4, 2018 at 6:08 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431082 "Direct link to this comment")\
\
\
\
\
\
       You can ignore it.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431082)\
109. ![](https://secure.gravatar.com/avatar/5ce520bd548482f1d201d3160c78d529d83e4f8776530e499afaf8f38938dafe?s=40&d=mm&r=g)\
\
\
\
     Mohannad RatebMarch 5, 2018 at 12:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431136 "Direct link to this comment")\
\
\
\
\
\
     HI jason ,\
\
\
\
     Excellent tutorial.\
\
\
\
     i have a question concerning on the number of hidden nodes , on which basis do we know it’s value .\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431136)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 5, 2018 at 6:25 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431174 "Direct link to this comment")\
\
\
\
\
\
       Use experimentation to estimate the number of hidden nodes that results in a model with the best skill on your dataset.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431174)\
110. ![](https://secure.gravatar.com/avatar/b81c5c551ab155b35a7399ac12356c007f7fbf5400cb7e070ed8aabd9d4f33ca?s=40&d=mm&r=g)\
\
\
\
     MoMarch 5, 2018 at 6:15 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431163 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     So after building the neural network from the training data, I want to test the network with the new set of test data. How can I do that?\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
     |     |     |\
     | --- | --- |\
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52<br>53<br>54<br>55<br>56<br>57<br>58<br>59<br>60<br>61<br>62<br>63<br>64<br>65 | \# import neural network libs<br>import numpy<br>import pandas<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from keras.wrappers.scikit\_learn import KerasClassifier<br>from keras.utils import np\_utils<br>from sklearn.model\_selection import cross\_val\_score<br>from sklearn.model\_selection import KFold<br>from sklearn.preprocessing import LabelEncoder<br>from sklearn.pipeline import Pipeline<br>from sklearn.cross\_validation import train\_test\_split<br>\# fix random seed for reproducibility<br>seed=7<br>numpy.random.seed(seed)<br>\# load pima indians dataset<br>trainset=numpy.loadtxt("optdigits.tra",delimiter=",")<br>testset=numpy.loadtxt("optdigits.tes",delimiter=",")<br>\# split into input (data) and output (labels) variables<br>data=trainset\[:,0:64\]<br>labels=trainset\[:,64\]<br>data\_testset=testset\[:,0:64\]<br>labels\_testset=testset\[:,64\]<br>\# encode class values as integers<br>encoder=LabelEncoder()<br>encoder.fit(labels)<br>encoded\_labels=encoder.transform(labels)<br>\# convert integers to OneHot variables (i.e. one hot encoded)<br>OneHot\_labels=np\_utils.to\_categorical(encoded\_labels)<br>\# define baseline model<br>def baseline\_model():<br>\# create model<br>model=Sequential()<br>model.add(Dense(36,input\_dim=64,init="uniform",activation='relu'))<br>model.add(Dense(10,activation='softmax'))<br>\# Compile model<br>model.compile(loss='categorical\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>returnmodel<br>\# Fit the model<br>estimator=KerasClassifier(build\_fn=baseline\_model,nb\_epoch=200,batch\_size=10,verbose=0)<br>\# evaluate the model using kFold cross validation with 20% of the data for testing and 80% for training<br>kfold=KFold(n\_splits=5,shuffle=True,random\_state=seed)<br>results=cross\_val\_score(estimator,data,OneHot\_labels,cv=kfold)<br>print("\\nOverall Validation accuracy: %.2f%% (%.2f%%)"%(results.mean()\*100,results.std()\*100))<br>\# build the neural network from all the training set<br>estimator.fit(data,labels)<br>predictions=estimator.predict(data\_testset)<br>print("\\nPredeiction: \\n",predictions)<br>print("\\nThe actual labels of the test set:\\n",labels\_testset)<br>\# build the confusion matrix after classifing the test data<br>from sklearn.metrics import confusion\_matrix<br>cm=confusion\_matrix(labels\_testset,predictions)<br>print("\\nThe confusion matrix when apply the test set on the trained nerual network:\\n",cm) |\
\
\
\
\
\
\
\
\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431163)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 5, 2018 at 6:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431176 "Direct link to this comment")\
\
\
\
\
\
       You must fit a final model.\
\
\
\
       This post will make the concept clear:\
\
       [https://machinelearningmastery.com/train-final-machine-learning-model/](https://machinelearningmastery.com/train-final-machine-learning-model/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-431176)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/2b996392f2e0758eedeb55cc65713fa6eb889bb05b23a5270fa4df97a09fdb4b?s=40&d=mm&r=g)\
\
\
\
         Partha Shankar NayakFebruary 19, 2019 at 3:52 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-469321 "Direct link to this comment")\
\
\
\
\
\
         Mo has done that in line 57 I believe.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-469321)\
111. ![](https://secure.gravatar.com/avatar/d36ea086e1a63200be167f1a3d2cf85960f5903a7983c0217a2590b42d9aaf9b?s=40&d=mm&r=g)\
\
\
\
     [Kashyap Raiyani](https://www.linkedin.com/in/kashyapraiyani/)March 21, 2018 at 9:24 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-432822 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     I did undergo the page and all the posts. I am having trouble with encoding label list. Please find the details as follows:\
\
\
\
     Problem:\
\
\
     Input data set file contain 3 columns in the following format unique\_id,text,aggression-level\
\
\
\
     The columns are separated by the comma and follow a minimal quoting pattern (such that only those columns are quoted which are in multiple lines or contain quotes in the text).\
\
\
\
     column 1: unique\_id facebook id\
\
\
     column 2: post/text\
\
\
     column 3: aggression-level: OAG, CAG, and NAG\
\
\
\
     There are 12000 records\
\
\
\
     Code as follows:\
\
\
\
     texts = \[\] # list of text samples\
\
\
     labels = \[\] # list of label ids\
\
\
     csvfile = pd.read\_csv(‘agr\_en\_train.csv’,names=\[‘id’, ‘post’, ‘label’\])\
\
\
     texts = csvfile\[‘post’\]\
\
\
     labels = csvfile\[‘label’\]\
\
\
     print(‘Found %s texts.’ % len(texts))\
\
\
\
     #label\_encoding\
\
\
     encoder = LabelEncoder()\
\
\
     encoder.fit(labels)\
\
\
     encoded\_Y = encoder.transform(labels)\
\
\
     dummy\_y = np\_utils.to\_categorical(encoded\_Y)\
\
\
     print(‘Shape of label tensor:’, dummy\_y.shape)\
\
\
\
     After Training model:\
\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     reduce\_lr = ReduceLROnPlateau(monitor=’val\_loss’, factor=0.5, patience=2, min\_lr=0.000001)\
\
\
     print(model.summary())\
\
\
     model.fit(x\_train, y\_train, batch\_size=256, epochs=25,validation\_data=(x\_val, y\_val), shuffle=True, callbacks=\[reduce\_lr\])\
\
\
\
     Below lines are giving Eorros:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
     |     |     |\
     | --- | --- |\
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9 | predict=model.predict(x\_val)<br>print(encoder.inverse\_transform(predict))<br>Traceback(most recent call last):<br>File"aggression\_analysis\_on\_facebookv2.py",line161,in<br>print(encoder.inverse\_transform(predict))<br>File"/usr/local/lib/python2.7/dist-packages/sklearn/preprocessing/label.py",line151,ininverse\_transform<br>ifdiff:<br>ValueError:The truth value of an arraywith more than one element isambiguous.Usea.any()ora.all() |\
\
\
\
\
\
\
\
\
\
\
\
     I am getting the predictions in np array but I am not able to convert back to the 3 classes (OAG, CAG, NAG) for test data.\
\
\
\
     Can you please have look at it?\
\
\
\
     Many thanks in advance.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-432822)\
\
112. ![](https://secure.gravatar.com/avatar/8a6def5f7e8f5985f9c22d01db10292e98c6636d8911c2c7b1a97d0e9cd332d5?s=40&d=mm&r=g)\
\
\
\
     Esteban VargasApril 5, 2018 at 12:54 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434200 "Direct link to this comment")\
\
\
\
\
\
     Jason this tutorial is just amazing! Thank you so much.\
\
\
\
     I want to ask you, how can this model be adapted for variables that measure different things? For example mixing lenghts, weights, etc.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434200)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 5, 2018 at 3:15 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434217 "Direct link to this comment")\
\
\
\
\
\
       Thanks.\
\
\
\
       Provide all the variables to the model, but rescale all variables to the range 0-1 prior to modeling.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434217)\
113. ![](https://secure.gravatar.com/avatar/58300f2a47f625ed3a90732340d40de4b8b66f06d81e6de4fb6e728d063d09d2?s=40&d=mm&r=g)\
\
\
\
     Stuart BlackApril 7, 2018 at 6:32 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434365 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for the really helpful tutorial!\
\
\
\
     Can you recommend a good way to normalise the data prior to feeding it into the model? Half of my columns have data values in the thousands and others have values no greater than 10.\
\
\
\
     Thanks\
\
\
\
     S\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434365)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 7, 2018 at 6:41 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434376 "Direct link to this comment")\
\
\
\
\
\
       Yes, use the sklearn MinMaxScaler. I have many tutorials on the topic:\
\
       [https://machinelearningmastery.com/?s=MinMaxScaler&submit=Search](https://machinelearningmastery.com/?s=MinMaxScaler&submit=Search)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-434376)\
114. ![](https://secure.gravatar.com/avatar/c3fe769296d92f97a076507cb1f2ae23eb016f5885b78de7abbde180f3dabd74?s=40&d=mm&r=g)\
\
\
\
     NikunjApril 19, 2018 at 8:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435412 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason! Thanks for the tutorial!\
\
\
     However I’m facing this problem –\
\
\
\
     Here is the code:\
\
\
\
     def baseline\_model():\
\
\
     model = Sequential()\
\
\
     model.add(Dense(256, input\_dim=90, activation=’relu’))\
\
\
     model.add(Dense(9, activation=’softmax’))\
\
\
     # learning rate is specified\
\
\
     keras.optimizers.Adam(lr=0.001)\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, epochs=50, batch\_size=500, verbose=1)\
\
\
     estimator.fit(X, dummy\_y)\
\
\
\
     Now, the output is :\
\
\
\
     150000/150000 \[==============================\] – 2s 12us/step – loss: 11.4893 – acc: 0.2870\
\
\
     Epoch 2/50\
\
\
     150000/150000 \[==============================\] – 2s 11us/step – loss: 11.4329 – acc: 0.2907\
\
\
     Epoch 3/50\
\
\
     150000/150000 \[==============================\] – 2s 10us/step – loss: 11.4329 – acc: 0.2907\
\
\
     Epoch 4/50\
\
\
     150000/150000 \[==============================\] – 2s 11us/step – loss: 11.4329 – acc: 0.2907\
\
\
     Epoch 5/50\
\
\
     150000/150000 \[==============================\] – 2s 11us/step – loss: 11.4329 – acc: 0.2907\
\
\
     Epoch 6/50\
\
\
     150000/150000 \[==============================\] – 2s 11us/step – loss: 11.4329 – acc: 0.2907\
\
\
     ………………..\
\
\
     ……………….\
\
\
\
     The loss and acc remain the same for the remaining epochs.\
\
\
     The no. of layers and activation type are specified.\
\
\
     Why is the loss remaining constant?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435412)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 19, 2018 at 2:46 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435430 "Direct link to this comment")\
\
\
\
\
\
       You may need to tune the model for your problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435430)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/c3fe769296d92f97a076507cb1f2ae23eb016f5885b78de7abbde180f3dabd74?s=40&d=mm&r=g)\
\
\
\
         NikunjApril 20, 2018 at 5:34 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435472 "Direct link to this comment")\
\
\
\
\
\
         How can I do that Jason?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435472)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)April 20, 2018 at 6:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435489 "Direct link to this comment")\
\
\
\
\
\
           I provide a long list of ideas here:\
\
           [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-435489)\
115. ![](https://secure.gravatar.com/avatar/f6786fc243fde0b1039e71455738d46795b4d78b084c3b27616add5990e0c798?s=40&d=mm&r=g)\
\
\
\
     anandMay 10, 2018 at 3:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437067 "Direct link to this comment")\
\
\
\
\
\
     hi jason thanks for this tutorial\
\
\
\
     when iam trying this tutorial iam getting an error message of\
\
\
\
     Using TensorFlow backend.\
\
\
     Traceback (most recent call last):\
\
\
     File “C:\\Users\\hp\\AppData\\Local\\Programs\\Python\\Python36\\keras example1.py”, line 29, in\
\
\
     model = KerasClassifier(built\_fn = baseline\_model,epochs=200, batch\_size=5,verbose=0)\
\
\
     File “C:\\Users\\hp\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\keras\\wrappers\\scikit\_learn.py”, line 61, in \_\_init\_\_\
\
\
     self.check\_params(sk\_params)\
\
\
     File “C:\\Users\\hp\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\keras\\wrappers\\scikit\_learn.py”, line 75, in check\_params\
\
\
     legal\_params\_fns.append(self.\_\_call\_\_)\
\
\
     AttributeError: ‘KerasClassifier’ object has no attribute ‘\_\_call\_\_’\
\
\
\
     and second what if i use numpy to load the dataset. “numpy.loadtxt(x.csv)”\
\
\
     and how to encode the labels\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437067)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 10, 2018 at 6:35 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437092 "Direct link to this comment")\
\
\
\
\
\
       I’m sorry to hear that, here are some ideas:\
\
       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437092)\
116. ![](https://secure.gravatar.com/avatar/3465137a367664567ff385caebd9eef972218533fbf5a301327b70dd472f8298?s=40&d=mm&r=g)\
\
\
\
     VoodoomonkeyMay 15, 2018 at 4:04 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437517 "Direct link to this comment")\
\
\
\
\
\
     Hello, Jason.\
\
\
     Been looking through some of your topics on deep learning with python.\
\
\
     They are very useful and give us a lot of information about using python with NN.\
\
\
     Thank you!\
\
\
\
     I’ve been trying to create a multi class classifier using your example but i can’t get it to work properly.\
\
\
\
     You see, i have approximately 20-80 classes and using your example i only get a really small accuracy rate.\
\
\
\
     My code looks like this (basically your code ) :\
\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed)\
\
\
     \# load dataset\
\
\
     dataframe = pandas.read\_csv(“csv1.csv”, header=None)\
\
\
     dataset = dataframe.values\
\
\
     X = dataset\[:,0:8\]\
\
\
     Y = dataset\[:,8:9\]\
\
\
     print(X.shape)\
\
\
     print(Y.shape)\
\
\
     encoder = LabelEncoder()\
\
\
     encoder.fit(Y)\
\
\
     encoded\_Y = encoder.transform(Y)\
\
\
     \# convert integers to dummy variables (i.e. one hot encoded)\
\
\
     dummy\_y = np\_utils.to\_categorical(encoded\_Y)\
\
\
     \# define baseline model\
\
\
     def baseline\_model():\
\
\
     # create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(8, input\_dim=8, activation=’relu’))\
\
\
     model.add(Dense(56, activation=’softmax’))\
\
\
     # Compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, epochs=200, batch\_size=5, verbose=0)\
\
\
     kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
     print(“Baseline: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     and my csv is :\
\
     [https://drive.google.com/open?id=1KmTpLHHd8apXrqOK8UcJfr3MbqWMe9ok](https://drive.google.com/open?id=1KmTpLHHd8apXrqOK8UcJfr3MbqWMe9ok)\
\
\
\
     Looking forward for your answer. This is very important for me and my future.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437517)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 15, 2018 at 8:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437539 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I cannot review your code, what problem are you having exactly?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437539)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/3465137a367664567ff385caebd9eef972218533fbf5a301327b70dd472f8298?s=40&d=mm&r=g)\
\
\
\
         VoodoomonkeyMay 15, 2018 at 11:39 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437599 "Direct link to this comment")\
\
\
\
\
\
         Would it be easier to review like this ?\
\
         [https://pastebin.com/hYa2cpmW](https://pastebin.com/hYa2cpmW)\
\
\
\
         The problem i’m having is that using the code you provided with my dataset i get\
\
\
         Baseline: 4.00% (6.63%)\
\
\
         Which is really low, and i don’t see any ways to fix that.\
\
\
         I’m trying to train it on 100 rows of data with 38 classes.\
\
\
\
         If i try to use it with more data, the baseline drops even more.\
\
\
\
         Is there a way to increase the percentage ? Maybe i’m doing something wrong ?\
\
\
         It always come’s down to – every example you provide works, but when i try my own data – it doesn’t work.\
\
\
\
         Can you please take a look at code and data, maybe ?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437599)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)May 16, 2018 at 6:05 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437632 "Direct link to this comment")\
\
\
\
\
\
           Here are some suggestions to lift model skill:\
\
           [https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437632)\
117. ![](https://secure.gravatar.com/avatar/de1373fbc8eba7aee88b146bb1890492606079d7af09073b451ef8bd33e5da32?s=40&d=mm&r=g)\
\
\
\
     ShyamMay 19, 2018 at 4:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437912 "Direct link to this comment")\
\
\
\
\
\
     I am running the code with the dependencies installed, but I am receiving this as an output.\
\
\
\
     C:\\Users\\shyam\\Anaconda3\\envs\\tensorflow\\lib\\site-packages\\h5py\\\_\_init\_\_.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\
\
\
     from .\_conv import register\_converters as \_register\_converters\
\
\
     Using TensorFlow backend.\
\
\
\
     Shouldn’t it be printing more than just “using TensorFlow backend”? Any help would be greatly appreciated\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437912)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 19, 2018 at 7:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437934 "Direct link to this comment")\
\
\
\
\
\
       You can ignore this warning.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-437934)\
118. ![](https://secure.gravatar.com/avatar/6da6539f066424669d949a339a359f8fd34a21f0aab90f7ea0b80a61f1a5d188?s=40&d=mm&r=g)\
\
\
\
     ChrisaMay 22, 2018 at 4:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-438185 "Direct link to this comment")\
\
\
\
\
\
     Hello and thanks for this excellent tutorial.\
\
\
\
     I have a dataset with 150 attributes per entry. If an attribute is unknown for an entry, then in the csv file it is represented with a “?”. I suppose this will be a problem in the training phase. Can you suggest a way to handle his?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-438185)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 22, 2018 at 6:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-438205 "Direct link to this comment")\
\
\
\
\
\
       This is a common question that I answer here:\
\
       [https://machinelearningmastery.com/faq/single-faq/how-do-i-handle-missing-data](https://machinelearningmastery.com/faq/single-faq/how-do-i-handle-missing-data)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-438205)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/6da6539f066424669d949a339a359f8fd34a21f0aab90f7ea0b80a61f1a5d188?s=40&d=mm&r=g)\
\
\
\
         ChrisaMay 22, 2018 at 7:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-438209 "Direct link to this comment")\
\
\
\
\
\
         Thank you very much\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-438209)\
119. ![](https://secure.gravatar.com/avatar/6da6539f066424669d949a339a359f8fd34a21f0aab90f7ea0b80a61f1a5d188?s=40&d=mm&r=g)\
\
\
\
     ChrisaJune 26, 2018 at 8:02 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442097 "Direct link to this comment")\
\
\
\
\
\
     Hello again. I finally narowed down which of the 150 attributes I need to use, but now there is another problem. The attributes I need are in specific columns and of different datatype. I tried working with numpy.loadtxt and numpy.genfromtxt but the format of the resulting arrays is not the right one. I get the mistake:\
\
\
     ValueError: Error when checking input: expected dense\_1\_input to have shape (5,) but got array with shape (1,)\
\
\
     where 5 are the attributes I m using\
\
\
     Can you help me?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442097)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/6da6539f066424669d949a339a359f8fd34a21f0aab90f7ea0b80a61f1a5d188?s=40&d=mm&r=g)\
\
\
\
       ChrisaJune 26, 2018 at 11:30 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442104 "Direct link to this comment")\
\
\
\
\
\
       I figured it out using:\
\
\
       dataframe = pandas.read\_csv(“IrisDataset.csv”, header=None, usecols = \[0,1,2,3,5\], dtype ={0:np.float32, 1:np.float32, 2:np.float32, 3:np.float32, 5: np.str })\
\
\
       where the fifth column is one I added in order to check the string attributes.\
\
\
       Now there is problem of how can I have strings as input for the neural\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442104)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)June 27, 2018 at 8:19 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442137 "Direct link to this comment")\
\
\
\
\
\
         Strings must be encoded, see this:\
\
         [https://machinelearningmastery.com/faq/single-faq/how-to-handle-categorical-data-with-string-values](https://machinelearningmastery.com/faq/single-faq/how-to-handle-categorical-data-with-string-values)\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442137)\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 27, 2018 at 8:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442135 "Direct link to this comment")\
\
\
\
\
\
       Perhaps this post will help you load your data:\
\
       [https://machinelearningmastery.com/load-machine-learning-data-python/](https://machinelearningmastery.com/load-machine-learning-data-python/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442135)\
120. ![](https://secure.gravatar.com/avatar/bb29f9ffed8ed07b06715d9341948ebb5ce61b7836f4a8a0a4e0f2e9623f38da?s=40&d=mm&r=g)\
\
\
\
     KushagraJuly 2, 2018 at 12:14 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442496 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason!\
\
\
     Thank you for such awesome posts. Do you have tutorials or recommendations for classifying raw time series data using RNN GRU or LSTM?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442496)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 2, 2018 at 2:59 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442501 "Direct link to this comment")\
\
\
\
\
\
       1D CNNs are very effective for time series classification in my experience.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-442501)\
121. ![](https://secure.gravatar.com/avatar/b467fb0480d588fa528ca5a0a1c1f2dc2fc8ee73ddc0d64f4f483b59da8571ec?s=40&d=mm&r=g)\
\
\
\
     Sanjeev RanjanJuly 9, 2018 at 3:46 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443021 "Direct link to this comment")\
\
\
\
\
\
     Please help:\
\
\
     Error when checking target: expected dense\_6 to have shape (10,) but got array with shape (1,)\
\
\
\
     I have to do a multi-class classification to predict value ranging between 1 to 5\
\
\
     there are total of 46 columns. All columns have numerical values only.\
\
\
\
     model = Sequential()\
\
\
     model.add(Dense(64, activation=’relu’, input\_dim=46)) #there are 46 feature in my dataset to be trained\
\
\
     model.add(Dropout(0.5))\
\
\
     model.add(Dense(64, activation=’relu’))\
\
\
     model.add(Dropout(0.5))\
\
\
     model.add(Dense(10, activation=’softmax’))\
\
\
\
     model.compile(optimizer=’rmsprop’, loss=’categorical\_crossentropy’, metrics=\[‘accuracy’\])\
\
\
\
     model.fit(X\_train, Y\_train, epochs=20, batch\_size=128)\
\
\
\
     I got error in last line.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443021)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 10, 2018 at 6:42 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443074 "Direct link to this comment")\
\
\
\
\
\
       This might help:\
\
       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443074)\
122. ![](https://secure.gravatar.com/avatar/6da6539f066424669d949a339a359f8fd34a21f0aab90f7ea0b80a61f1a5d188?s=40&d=mm&r=g)\
\
\
\
     ChrisaJuly 10, 2018 at 11:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443111 "Direct link to this comment")\
\
\
\
\
\
     I tried adding this block of code in the end in order to test the model on new data,\
\
\
\
     estimator.fit(X, dummy\_y)\
\
\
     predictions=estimator.predict(X)\
\
\
     correct=0\
\
\
\
     for i in range(np.size(X,0)):\
\
\
     if predictions\[i\].argmax()==dummy\_y\[i\].argmax():\
\
\
     print (“%d well predicted\\n” %i)\
\
\
     correct+=1\
\
\
     print (“Correct predicted: %d” %correct)\
\
\
\
     In fact, there is no new data. The test array X is the same as the training one, so I expected a very big number of corrects.. However the corrects are 50. After printing the predictions, I realized that all indexes are predicted as “Iris-setosa” which is the first label, so the rate is approximately 33.3%. Am I doing something wrong?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443111)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 10, 2018 at 2:29 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443122 "Direct link to this comment")\
\
\
\
\
\
       I explain how to make predictions on new data here:\
\
       [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443122)\
123. ![](https://secure.gravatar.com/avatar/8478a960d8b387f98eedb441039dbf7731f7ac5737ac2714ea20504ebe5d3633?s=40&d=mm&r=g)\
\
\
\
     AlexJuly 11, 2018 at 1:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443159 "Direct link to this comment")\
\
\
\
\
\
     Thanks for the awesome tutorial\
\
\
\
     One question, now that I have the model, how can I predict new data.\
\
\
\
     Imagine I have now this scenario\
\
\
\
     1\. flowers.csv with 4 rows of collected data (without the labels)\
\
\
\
     Now I want to feed the csv to the model to have the predictions for every data\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443159)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2018 at 5:59 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443181 "Direct link to this comment")\
\
\
\
\
\
       This post explains more on how to make predictions:\
\
       [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443181)\
124. ![](https://secure.gravatar.com/avatar/8478a960d8b387f98eedb441039dbf7731f7ac5737ac2714ea20504ebe5d3633?s=40&d=mm&r=g)\
\
\
\
     AlexJuly 11, 2018 at 1:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443160 "Direct link to this comment")\
\
\
\
\
\
     I tried this for predictions\
\
\
     \# load dataset\
\
\
     dataframe2 = pandas.read\_csv(“flowers-pred.csv”, header=None)\
\
\
     dataset2 = dataframe.values\
\
\
     \# new instance where we do not know the answer\
\
\
     Xnew = dataset2\[:,0:4\].astype(float)\
\
\
     \# make a prediction\
\
\
     ynew = model.predict\_classes(Xnew)\
\
\
     \# show the inputs and predicted outputs\
\
\
     print(“X=%s, Predicted=%s” % (Xnew\[0\], ynew\[0\]))\
\
\
\
     And I get the result\
\
\
     X=\[4.6 3.1 1.5 0.2\], Predicted=1\
\
\
\
     Sometimes the values of X does not correspond to the real values of the file and always the prediction is 1.\
\
\
\
     Because is one hot encoding I supouse the prediccion should be 0 0 1 or 1 0 0 or 0 1 0\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443160)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2018 at 6:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443182 "Direct link to this comment")\
\
\
\
\
\
       All models have error. You can try improving the performance of the model.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443182)\
125. ![](https://secure.gravatar.com/avatar/8478a960d8b387f98eedb441039dbf7731f7ac5737ac2714ea20504ebe5d3633?s=40&d=mm&r=g)\
\
\
\
     AlexJuly 11, 2018 at 1:49 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443161 "Direct link to this comment")\
\
\
\
\
\
     I found what I was doing wrong,\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, epochs=200, batch\_size=5, verbose=0)\
\
\
     kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
     print(“Baseline: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     model = baseline\_model()\
\
\
\
     \# load dataset\
\
\
     dataframe2 = pandas.read\_csv(“flores-pred.csv”, header=None)\
\
\
     dataset2 = dataframe.values\
\
\
     \# new instance where we do not know the answer\
\
\
     Xnew = dataset2\[:,0:4\].astype(float)\
\
\
     \# make a prediction\
\
\
     ynew = model.predict(Xnew)\
\
\
     \# show the inputs and predicted outputs\
\
\
     print(“X=%s, Predicted=%s” % (Xnew\[2\], ynew\[2\]))\
\
\
\
     Now this works, but all the predictions are almost the same\
\
\
     X=\[4.7 3.2 1.3 0.2\], Predicted=\[0.13254479 0.7711002 0.09635501\]\
\
\
\
     NO matter wich flower is in the row, I always gets 0 1 0\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443161)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2018 at 6:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443183 "Direct link to this comment")\
\
\
\
\
\
       Perhaps there’s a bug in the way you are making predictions?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-443183)\
126. ![](https://secure.gravatar.com/avatar/7c384c1f5c8cbf6c3de4a0140a451e663620360de72bbf3fea0c2f73c1607a6b?s=40&d=mm&r=g)\
\
\
\
     AnshJuly 22, 2018 at 12:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444076 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I was just wondering. rather than one hot encoding 3 categories as shown below.\
\
\
\
     Iris-setosa, Iris-versicolor, Iris-virginica\
\
\
     1, 0, 0\
\
\
     0, 1, 0\
\
\
     0, 0, 1\
\
\
\
     Can’t we change the three categories.\
\
\
     Y Y1\
\
\
     Iris-setosa 0 0\
\
\
     Iris-versicolor 0 1\
\
\
     Iris-virginica 1 0\
\
\
\
     and if we could what will be the core difference in training the models using the above two mentioned ways.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444076)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 22, 2018 at 6:25 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444097 "Direct link to this comment")\
\
\
\
\
\
       I don’t follow, what would a model predict?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444097)\
127. ![](https://secure.gravatar.com/avatar/daa8320469a93c5b174811b1097bab950546098504d57220415087a1d2c49d12?s=40&d=mm&r=g)\
\
\
\
     [Felipe](http://erfelipe.com.br/)July 25, 2018 at 12:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444303 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, great tutorial, thanks.\
\
\
     Do you know some path to use ontology (OWL or RDF) like input data to improve a best analise?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444303)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 25, 2018 at 6:19 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444341 "Direct link to this comment")\
\
\
\
\
\
       I don’t sorry.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-444341)\
128. ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
     ShooterAugust 9, 2018 at 7:35 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445675 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, what if X data contains numbers as well as multiple classes?\
\
\
\
     Thanks in advance.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445675)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 10, 2018 at 6:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445714 "Direct link to this comment")\
\
\
\
\
\
       X is the input only, y contains the output or the classes.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445714)\
129. ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
     ShooterAugust 10, 2018 at 3:14 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445760 "Direct link to this comment")\
\
\
\
\
\
     I mean what if X contains multiple labels like “High and Low”? We need to use one hot encoding on that X data too and continue other steps in the same way?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445760)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 11, 2018 at 6:05 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445806 "Direct link to this comment")\
\
\
\
\
\
       If you are working with categorical inputs, you will need to encode them in some way.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445806)\
130. ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
     ShooterAugust 10, 2018 at 7:00 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445771 "Direct link to this comment")\
\
\
\
\
\
     Hi jason, It seems you have already answered my question in one of the comments. I need to convert the categorical value into one hot encoding then create dummy variable and then input it. Thanks.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445771)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 11, 2018 at 6:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445810 "Direct link to this comment")\
\
\
\
\
\
       Yes.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445810)\
131. ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
     ShooterAugust 10, 2018 at 8:23 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445778 "Direct link to this comment")\
\
\
\
\
\
     Hi, I wanted to ask again that using K-fold validation like this\
\
\
\
     kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
\
     or using train/test split and validation data like this\
\
\
\
     x\_train,x\_test,y\_train,y\_test=train\_test\_split(X,dummy\_y,test\_size=0.33,random\_state=seed)\
\
\
\
     estimator.fit(x\_train,y\_train,validation\_data=(x\_test,y\_test))\
\
\
\
     These are just sampling techniques, we can use any one of them according to the availability and size of data right?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445778)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 11, 2018 at 6:08 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445813 "Direct link to this comment")\
\
\
\
\
\
       Yes.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-445813)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
         ShooterAugust 17, 2018 at 6:17 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446346 "Direct link to this comment")\
\
\
\
\
\
         Thanks.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446346)\
132. ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
     ShooterAugust 17, 2018 at 10:03 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446367 "Direct link to this comment")\
\
\
\
\
\
     Can u please provide one example of multilabel multi-class classification too?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446367)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 18, 2018 at 5:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446384 "Direct link to this comment")\
\
\
\
\
\
       Thanks for the suggestion.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446384)\
133. ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
     ShooterAugust 18, 2018 at 3:48 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446404 "Direct link to this comment")\
\
\
\
\
\
     All examples i have seen so far in LSTM are related to classifiying imdb datasets or vocabulary like that. There are no simple examples to describe classification using LSTM. Can u please provide one example doing the same above iris classification using LSTM so that we can have a general idea.\
\
\
\
     Thanks in advance.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446404)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 19, 2018 at 6:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446448 "Direct link to this comment")\
\
\
\
\
\
       LSTMs are for sequence data. For classification, this means sequence classification or time series classification.\
\
\
\
       Does that help?\
\
\
\
       You cannot use LSTMs on the Iris flowers dataset for example. Learn more here:\
\
       [https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/](https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446448)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
         ShooterAugust 24, 2018 at 5:49 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446865 "Direct link to this comment")\
\
\
\
\
\
         Thanks Jason. I have another question. I have total of 1950 data. Will it be enough if i train/test split into 90:10 ratio i.e 1560 data for training,195 for validation and 195 for testing. If i decrease training data, accuracy starts decreasing.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446865)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 25, 2018 at 5:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446895 "Direct link to this comment")\
\
\
\
\
\
           It is impossible for me to say, try it and see.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446895)\
134. ![](https://secure.gravatar.com/avatar/014f4996ddfbc1e40406ad13555e572f04a74c8830a12359c65ae1abc059a785?s=40&d=mm&r=g)\
\
\
\
     ShooterAugust 25, 2018 at 7:08 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446938 "Direct link to this comment")\
\
\
\
\
\
     Ok thanks, I’ll try it. Another question, How can i calculate accuracy of the model using sum of squared errors. I need to compare a model that gives sum of squared errors in regression with my model that gives output in accuracy that is a classification problem.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446938)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 26, 2018 at 6:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446980 "Direct link to this comment")\
\
\
\
\
\
       Sum squared errors is for regression, not classification.\
\
\
\
       For metrics, you can use sklearn to calculate anything you wish:\
\
       [http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-446980)\
135. ![](https://secure.gravatar.com/avatar/720a692b850232eaf5005d7683acf2a7544caf9051fb2fcd5da61953a3fcd799?s=40&d=mm&r=g)\
\
\
\
     JetSeptember 4, 2018 at 5:41 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-447792 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     awesome page & tutorials!\
\
\
\
     Is there a way to do stratified k-fold cross-validation on multi-label classification, or at least k-fold cross-validation?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-447792)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 5, 2018 at 6:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-447870 "Direct link to this comment")\
\
\
\
\
\
       There may be, I don’t have any multi-label examples though, sorry.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-447870)\
\
     - ![](https://secure.gravatar.com/avatar/92a9a18c3e0e517171f59b9ab95fedf2415973cabf76becdab0187c263a57b56?s=40&d=mm&r=g)\
\
\
\
       Manuel GonçalvesNovember 23, 2018 at 6:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-455878 "Direct link to this comment")\
\
\
\
\
\
       Use k-Fold on your Y and put the indexes on your one-hot-encodered. Something like this:\
\
\
\
       df = pandas.read\_csv, slice, blah blah blah\
\
\
       X = slice df etc..etc..\
\
\
       y = slice df etc..etc..\
\
\
\
       dum\_y = np\_utils.to\_categorical(y) #from keras\
\
\
\
       #now you have y and dum\_y that is one-hot-encodered\
\
\
\
       skfold = StratifiedKFold(n\_splits=10, random\_state=0) #create a stratified Kfold\
\
\
       for train, test in skfold.split(X, y): #note that you are spliting the y without one-hot just to get indexes\
\
\
       model = Sequential()\
\
\
       model.add(Dense(blah blah blah)\
\
\
       …\
\
\
       #compile\
\
\
       model.compile(blah blah blah)\
\
\
       #now the magic, use indexes on one-hot-encodered, since the indexes are the same\
\
\
       model.fit(X\[train\], dum\_y\[train\], validation\_data=(X\[test\], dum\_y\[test\]), epochs=250, batch\_size=50,verbose=False)\
\
\
       #do the rest of your code\
\
\
       #the model will be created and fitted 10 times\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-455878)\
136. ![](https://secure.gravatar.com/avatar/608b9c2b757842a226a8e5ce71200c061a5ef7345e526b200656a1c110d5a10c?s=40&d=mm&r=g)\
\
\
\
     sathvikSeptember 21, 2018 at 2:24 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-449484 "Direct link to this comment")\
\
\
\
\
\
     That was really an excellent article..\
\
\
     can i implement CNN for feature Extraction from images then save the extracted features and apply SVM or XG boost for binary classification..please share the code to serve the purpose..thanks a lot..\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-449484)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 22, 2018 at 6:24 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-449530 "Direct link to this comment")\
\
\
\
\
\
       Yes! I show how to use a VGG model to extraction features for describing the contents of photos. For example, the last part of this tutorial:\
\
       [https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/](https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-449530)\
137. ![](https://secure.gravatar.com/avatar/1482e09c1f0fa23c7163fabadc12c5c95a2e5812ad89bcf64345f2e300f8225d?s=40&d=mm&r=g)\
\
\
\
     GeorgeOctober 22, 2018 at 10:07 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-452368 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason and many thanks for your helpful posts.\
\
\
\
     I haven’t find any multilabel classification post, so I am posting on this.\
\
\
\
     I have a problem to interpret the results of multi label classification.\
\
\
\
     Let’s say I have this problem.I have images with structures (ex building)\
\
\
\
     structure: 0 is there is no structure , 1 if it is\
\
\
     type: 3 different types of structures (1,2,3)\
\
\
     nb of structure\
\
\
\
     So, I have data:\
\
\
\
     labels = np.array(\[\[0,’nan’, ‘nan’\],\
\
\
     \[1, 2, 2\],\
\
\
     \[1, 3, 1\],\
\
\
     \[1, 1, 1\]\])\
\
\
\
     When I have no structure all rest values are nan.\
\
\
\
     The second line means I have a structure of type 2 and also have 2 structures.\
\
\
     The third line means, I have a structure of type 3 and it is just one.\
\
\
     The fourth means I have a structure of type 1, just one.\
\
\
\
     I am applying the mlb:\
\
\
\
     mlb = MultiLabelBinarizer()\
\
\
     labels = mlb.fit\_transform(labels)\
\
\
\
     and the mlb classes is :\
\
\
\
     array(\[‘0’, ‘1’, ‘2’, ‘3’, ‘nan’\], dtype=object)\
\
\
\
     My test data is for example: \[1, 2, 2\] // 1: there is a structure, 2: of type 2, 2: there are 2 structures in the image\
\
\
\
     And my result predict array is : \[20,10,2,4,50\]\
\
\
\
     The problem is what 20 means.Is there a structure or not?Test data has the value 1 which means there is structure.So,\
\
\
     I 20% means possibility to have structure?\
\
\
\
     The 10 means that we have 10% possibility to be of type 1, then 2% to be of type 2 and 4% to be of type 3.\
\
\
     The 50% means that there is a possibility 50% to have how number of faces??? Two faces?as the test data says?\
\
\
\
     If there is no structure, the test array will be (\[0, ‘nan’, ‘nan’\])\
\
\
     So, the same prediction : \[20,10,2,4,50\]\
\
\
\
     What 20% means?There is , or there is not a structure?\
\
\
     The 10,2,4 are the possibilities of type 1,2,3\
\
\
     The 50% is for the number of structures.But, is 50% for no structures , or for some number?\
\
\
\
     So, I have problem with first and last indices.\
\
\
\
     Thank you very much!\
\
\
\
     George\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-452368)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 23, 2018 at 6:24 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-452399 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I don’t have material on multi-label classification, so I can’t give useful off the cuff advice on the topic. I hope to cover it in the future.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-452399)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/1482e09c1f0fa23c7163fabadc12c5c95a2e5812ad89bcf64345f2e300f8225d?s=40&d=mm&r=g)\
\
\
\
         GeorgeOctober 23, 2018 at 7:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-452415 "Direct link to this comment")\
\
\
\
\
\
         Ok, thanks maybe I’ll post on stackoverflow if someone can help.Thanks.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-452415)\
138. ![](https://secure.gravatar.com/avatar/0d2aaeb2da2275694a9563c9fec1539246c67e969b724728c658f054150e162b?s=40&d=mm&r=g)\
\
\
\
     chrisDecember 22, 2018 at 4:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-459537 "Direct link to this comment")\
\
\
\
\
\
     hi Jason ,thanks for this amazing article.I want to predict the number of passengers in diferent airports.i am given the date ,airport departure,airport arrival , city ,longitude etc.I want to use neural network since the problem is not linear but i am having dificulty finding the right model.Everything that i uses gives me acc 0.42 max.Any suggestions?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-459537)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 22, 2018 at 6:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-459558 "Direct link to this comment")\
\
\
\
\
\
       I have some suggestions here:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-459558)\
139. ![](https://secure.gravatar.com/avatar/9d84bb9fdd50e7ad8f9b852a6e63e9703d5937e894c4c3abffbd67f0ab094ca0?s=40&d=mm&r=g)\
\
\
\
     [DeeB](http://www.activereservoir.com/)December 29, 2018 at 12:16 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460541 "Direct link to this comment")\
\
\
\
\
\
     Dr J,\
\
\
\
     Thanks for all your hard work and contribution. They are immensely useful.\
\
\
\
     One quick question, how to cross plot y\_pred (which is a vector) and dummy\_y (which is a tuple etc,) to test how good the prediction is? It gives obvious error msg of size mismatch.\
\
\
\
     Thanks,\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460541)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 30, 2018 at 5:34 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460645 "Direct link to this comment")\
\
\
\
\
\
       Perhaps change both pieces of data to have the same dimensionality first?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460645)\
140. ![](https://secure.gravatar.com/avatar/3cb0e20a3ac79890d0e69f3e4e6ccfed4e39a1d97f5a50ad4f8e0ab973a16de0?s=40&d=mm&r=g)\
\
\
\
     FlávioJFPereiraDecember 31, 2018 at 7:19 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460850 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason! What a nice tutorial! I’m needing some advice for an academic project. Instead of classification between 3 classes, like in your problem, I got 5 classes and my target has a probability of belonging to each of these 5 classes!\
\
\
\
     What are you advices for my network implementation? I mean, how should my output layer be to return the probabilities?\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460850)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 31, 2018 at 11:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460864 "Direct link to this comment")\
\
\
\
\
\
       Use a softmax activation function on the output layer.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460864)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/3cb0e20a3ac79890d0e69f3e4e6ccfed4e39a1d97f5a50ad4f8e0ab973a16de0?s=40&d=mm&r=g)\
\
\
\
         FlávioJFPereiraDecember 31, 2018 at 12:04 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460870 "Direct link to this comment")\
\
\
\
\
\
         still using categorical\_crossentropy as loss function? or something like mse?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460870)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)January 1, 2019 at 6:11 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460964 "Direct link to this comment")\
\
\
\
\
\
           Yes, categorical cross entropy loss is used for multi-class classification.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-460964)\
141. ![](https://secure.gravatar.com/avatar/c9d742ec99867c63822ba2b597860e8e323554b2d31e7a5178252d26e4571e8c?s=40&d=mm&r=g)\
\
\
\
     SajadJanuary 20, 2019 at 12:54 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-464244 "Direct link to this comment")\
\
\
\
\
\
     Thanks for your great explanation\
\
\
     Is there any code for getting back from ‘dummy y’ one hot matrix to the actual ‘y’ vector?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-464244)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 20, 2019 at 5:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-464271 "Direct link to this comment")\
\
\
\
\
\
       Yes, you can use the argmax() function.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-464271)\
142. ![](https://secure.gravatar.com/avatar/f1c0de6edb3403c06a562deb6692acca553784f85483111f60bbe83217e325cf?s=40&d=mm&r=g)\
\
\
\
     EmmenTrapFebruary 7, 2019 at 12:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-467278 "Direct link to this comment")\
\
\
\
\
\
     HI , Thanks for your great tutorial sir, I have used this code for my project to classification of rise seed varieties, the classifier has 15 classes and i have received the 90% accuracy. now i need to get prediction with the trained model, so can you help me that ho to get the prediction with unknown data for multi-class classification\
\
\
     thank you\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-467278)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 7, 2019 at 6:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-467329 "Direct link to this comment")\
\
\
\
\
\
       Yes, I explain how here:\
\
       [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-467329)\
143. ![](https://secure.gravatar.com/avatar/d21007e8daad94298247230a165e66349b3800fe858d847aabb6bf1513693ded?s=40&d=mm&r=g)\
\
\
\
     rio sunethFebruary 26, 2019 at 2:36 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470723 "Direct link to this comment")\
\
\
\
\
\
     is there an example of a classification model for networking traffic to detect botnets on a computer network package, thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470723)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 27, 2019 at 7:24 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470876 "Direct link to this comment")\
\
\
\
\
\
       There might be, I’m not aware of it sorry. Perhaps try a search on scholar.google.com.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470876)\
144. ![](https://secure.gravatar.com/avatar/7ee5d47a06e9d29a3358d18db94d95becc5b4a20c10154544cb80975cd9b846d?s=40&d=mm&r=g)\
\
\
\
     ZAINAB SHEERIN M SFebruary 27, 2019 at 12:10 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470805 "Direct link to this comment")\
\
\
\
\
\
     Hey!!!\
\
\
     I’m working on medical data, with the same model done here.\
\
\
     I have data in 3 different files that is normal, bacterial pneumonia and viral pneumonia with images in it.\
\
\
     instead of using csv file in the directory, how can I do it with my data.\
\
\
     kindly do the needful.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470805)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/7ee5d47a06e9d29a3358d18db94d95becc5b4a20c10154544cb80975cd9b846d?s=40&d=mm&r=g)\
\
\
\
       ZAINAB SHEERIN M SFebruary 27, 2019 at 12:14 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470807 "Direct link to this comment")\
\
\
\
\
\
       That 3 different files is in train,test and validation categories\
\
\
       each.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470807)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)February 27, 2019 at 7:31 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470888 "Direct link to this comment")\
\
\
\
\
\
         Yes, you can use the Keras flow\_from\_directory() function:\
\
         [https://keras.io/preprocessing/image/](https://keras.io/preprocessing/image/)\
\
\
\
         I hope to have an example of this very soon.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-470888)\
145. ![](https://secure.gravatar.com/avatar/4ea9f7be3df9012f0231873d017914f09f6802fa1a8df7fb074f5a941af7f2a2?s=40&d=mm&r=g)\
\
\
\
     pabloMarch 5, 2019 at 12:02 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-472308 "Direct link to this comment")\
\
\
\
\
\
     Jason\
\
\
\
     Run perfectly¡…thank you very much for you time and interesting for helping us¡.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-472308)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 5, 2019 at 6:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-472352 "Direct link to this comment")\
\
\
\
\
\
       Well done!\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-472352)\
146. ![](https://secure.gravatar.com/avatar/51b1e26c7c8efc614b8ddea61b3eecb9640e3661c4b26a3aeecc95fbc475f232?s=40&d=mm&r=g)\
\
\
\
     AndrewMarch 9, 2019 at 2:51 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473255 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason,\
\
\
\
     Your guides have been a tremendous help to me. Unfortunately, I’m coming from an applied science background and don’t quite fully understand LSTMs. I’ve run a Random Forest classifier on my data and already gotten a 92% accuracy, but my accuracy is absolutely awful with my LSTM (~11%, 9 classes so basically random chance). My data is 4500 trials of triaxial data at 3 joints (9 inputs), time series data, padded with 0s to match sequence length.\
\
\
\
     This is my code:\
\
\
     model.add(Masking(mask\_value=0., input\_shape=(366,9)))\
\
\
     model.add(LSTM(10,input\_shape=(366,9),return\_sequences=True, activation=’tanh’))\
\
\
     model.add(LSTM(10,return\_sequences=False,activation=’tanh’))\
\
\
     model.add(Dense(units=9,activation=’softmax’))\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer = ‘adam’, metrics=\[‘accuracy’\])\
\
\
     history = model.fit(xtrain\_nots,ytrain, epochs=400, batch\_size=100)\
\
\
\
     This is what my training accuracy looks like:\
\
     [https://i.imgur.com/tCZUlNi.png](https://i.imgur.com/tCZUlNi.png)\
\
\
\
     Is it possible that I just don’t have enough data? Would greatly appreciate some help on figuring out how to improve accuracy.\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473255)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 9, 2019 at 6:31 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473285 "Direct link to this comment")\
\
\
\
\
\
       It is possible that the LSTM is just not a good fit for your data.\
\
\
\
       Compare an MLP and CNN, as well as hybrids like CNN-LSTM and ConvLSTM. You can get started here:\
\
       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473285)\
147. ![](https://secure.gravatar.com/avatar/0504f28463b8acb65d7e61e8af5b0f5035d6ba9a93257737a89f118ef36cda32?s=40&d=mm&r=g)\
\
\
\
     ismetbMarch 9, 2019 at 6:51 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473291 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason. First of all, thanks for all the great effort you put in ML. I’ve learnt a great deal of things from you.\
\
\
\
     My question is, after using LabelEncoder to assign integers to our target instead of String, do we have to use OHE? I mean, after “encoded\_Y = encoder.transform(Y)” code, I have a target of single column and 3 classes all of which are integer. Why do we go further and make the target 3 columns?\
\
\
\
     Is there any difference between; a) using single column as target and using 1 neuron at output layer along with softmax and b) using 3 columns as target and using 3 neurons at output layer along with softmax.\
\
\
\
     I know OHE is mainly used for String labels but if my target is labeled with integers only (such as 1 for flower\_1, 2 for flower\_2 and 3 for flower\_3), I should be able to use it as is, am I wrong?\
\
\
\
     Regards\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473291)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 10, 2019 at 8:09 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473401 "Direct link to this comment")\
\
\
\
\
\
       The idea of a OHE is to treat the labels separately, rather than a linear continuum on one variable (which might not make sense, e.g. what is 1.5?).\
\
\
\
       You don’t have to OHE, try it and see if it improves performance.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473401)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/0504f28463b8acb65d7e61e8af5b0f5035d6ba9a93257737a89f118ef36cda32?s=40&d=mm&r=g)\
\
\
\
         ismetbMarch 11, 2019 at 7:11 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473604 "Direct link to this comment")\
\
\
\
\
\
         Thanks Jason for the reply. I’ve always thought, predicting 1.5 was equal to \[0, 0.5, 0.5\] categorical prediction which means 50-50 chance for classes 1 and 2.\
\
\
\
         Then what about binary classification (BC)? Is there any difference between 0 and 1 labelling (linear conitnuum of one variable) and categorical labelling? I have never seen anyone try categorical labelling for BC (and I intend to try) but I would like to learn your thought on this.\
\
\
\
         And for BC, would you suggest \[0, 1\] or \[-1, 1\] for labels? Would it make any difference?\
\
\
\
         Regards\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473604)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 12, 2019 at 6:48 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473684 "Direct link to this comment")\
\
\
\
\
\
           Typically, a one hot encoding for binary classification is equivalent to predicting a probability 0-1.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-473684)\
148. ![](https://secure.gravatar.com/avatar/1d2d2547120726532b653a6e7dd38af7997cb9ca16963a92f6fdcd8a7f88b8f2?s=40&d=mm&r=g)\
\
\
\
     TimMarch 18, 2019 at 3:56 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-474851 "Direct link to this comment")\
\
\
\
\
\
     Hello, Jason!\
\
\
\
     How can I do step-by-step debugging for functions (Kfold, KerasClassifier, hidden layer) to see intermediate values?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-474851)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 18, 2019 at 6:08 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-474863 "Direct link to this comment")\
\
\
\
\
\
       Good question.\
\
\
\
       It might be easier to use the Keras API and the KFold class directly so that you can see what is happening.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-474863)\
149. ![](https://secure.gravatar.com/avatar/1d2d2547120726532b653a6e7dd38af7997cb9ca16963a92f6fdcd8a7f88b8f2?s=40&d=mm&r=g)\
\
\
\
     MtimILMarch 25, 2019 at 9:37 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-476308 "Direct link to this comment")\
\
\
\
\
\
     Hello, Jason.\
\
\
\
     1) After learning the neural network I get the following weights:\
\
\
\
     \[\[-0.04067891 -0.01663 0.01646814 -0.07344743\]\
\
\
     \[ 0.02537021 -0.03948928 0.00033538 -0.1734132 \]\
\
\
     \[ 0.06725066 0.07520587 0.04672117 0.03763839\]\
\
\
     \[ 0.02950417 0.02176755 -0.023499 0.05072991\]\] \[0. 0. 0. 0.\]\
\
\
\
     \[\[ 0.00432587 -0.04444616 0.02091608\]\
\
\
     \[ 0.01232713 -0.02063667 -0.07363331\]\
\
\
     \[ 0.04093491 -0.0216442 -0.05544085\]\
\
\
     \[ 0.08577123 -0.03977689 0.02796889\]\] \[0. 0. 0.\]\
\
\
\
     Why is bias zero and the weights values are very small ?\
\
\
\
     Code:\
\
\
     \# define baseline model\
\
\
     def baseline\_model():\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(4, input\_dim=4, kernel\_initializer=’normal’, activation=’relu’))\
\
\
     model.add(Dense(3, kernel\_initializer=’normal’, activation=’sigmoid’))\
\
\
     \# Compile model\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, epochs=200, batch\_size=5, verbose=0)\
\
\
\
     kfold = KFold(n\_splits=10, shuffle=True, random\_state=seed)\
\
\
\
     model = baseline\_model()\
\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
\
     print(model.layers\[0\].get\_weights()\[0\], model.layers\[0\].get\_weights()\[1\])\
\
\
     print(model.layers\[1\].get\_weights()\[0\], model.layers\[1\].get\_weights()\[1\])\
\
\
\
     print(“Accuracy: %.2f%% (%.2f%%)” % (results.mean()\*100, results.std()\*100))\
\
\
\
     2) How can I get (output on screen) the values as a result of the activation function for the hidden and output layer ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-476308)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 26, 2019 at 8:05 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-476376 "Direct link to this comment")\
\
\
\
\
\
       Why? That is what the model learned – that’s the best we can say.\
\
\
\
       You can make each layer an output layer via the functional API, then collect all of the activations. This might help as a start:\
\
       [https://machinelearningmastery.com/keras-functional-api-deep-learning/](https://machinelearningmastery.com/keras-functional-api-deep-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-476376)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/1d2d2547120726532b653a6e7dd38af7997cb9ca16963a92f6fdcd8a7f88b8f2?s=40&d=mm&r=g)\
\
\
\
         MtimILMarch 27, 2019 at 4:04 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-476525 "Direct link to this comment")\
\
\
\
\
\
         Thank You very much!\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-476525)\
150. ![](https://secure.gravatar.com/avatar/383c5d1fdcaa660d4141c963f5a286464f3bbe62d1f15b02e8a3e4bde5f82a08?s=40&d=mm&r=g)\
\
\
\
     James LeeMay 23, 2019 at 3:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-486316 "Direct link to this comment")\
\
\
\
\
\
     Thank you for the excellent tutorial as always!\
\
\
     Do you mind clarifying what output activation and loss function should be used for multilabel problems?\
\
\
     For example, tagging movie genres with comedy, thriller, crime, scifi. They are not mutually exclusive. A movie can be tagged with all 4.\
\
\
     Then I could hot encode like \[1, 0, 0, 0\], \[1, 1, 0, 0\], \[1, 1, 1, 0\] \[1, 0, 1, 0\], and so on.\
\
\
     What would be the best combination in this case: activation (softmax vs sigmoid) and loss (binary\_crossentropy vs categorical\_crossentropy)?\
\
\
     What makes sense most to me is sigmoid activation (not exclusive) + binary\_crossentropy (treat each output neuron as binary problem), but I’ve read multiple stackoverflow and other articles suggesting conflicting informations.\
\
\
     Thank you!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-486316)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 23, 2019 at 6:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-486349 "Direct link to this comment")\
\
\
\
\
\
       Yes, I given an example of multi-label classification here:\
\
       [https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-486349)\
151. ![](https://secure.gravatar.com/avatar/61f4401f656dfb7a195e6b86680a4d500edd6c23fa4388eaead699a4c18b0860?s=40&d=mm&r=g)\
\
\
\
     Somaye Hamedi BazazJune 15, 2019 at 4:57 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-488961 "Direct link to this comment")\
\
\
\
\
\
     very well Thanks a lot\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-488961)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 15, 2019 at 6:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-488990 "Direct link to this comment")\
\
\
\
\
\
       Thanks, I’m happy to hear that.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-488990)\
152. ![](https://secure.gravatar.com/avatar/992eb263b7e909b46a290aeab97be47f5b15d86dcfa08319483d19b86a535b29?s=40&d=mm&r=g)\
\
\
\
     EmmaJune 17, 2019 at 9:46 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-489242 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, how can I add “none of the above” class in neural network? I searched on the net but didn’t find anything useful.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-489242)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 18, 2019 at 6:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-489286 "Direct link to this comment")\
\
\
\
\
\
       It would be one more class, e.g.: apple, orange, none.\
\
\
\
       You would then need to add examples of this new “none” class.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-489286)\
153. ![](https://secure.gravatar.com/avatar/78c0c673f3e0cb82effe5f809317c0ba03d3aab567bbf4fa46f88bee860e572b?s=40&d=mm&r=g)\
\
\
\
     Esra KarasuJuly 10, 2019 at 6:10 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492313 "Direct link to this comment")\
\
\
\
\
\
     Hi, I have a question for you. For this study, I wrote code of performance measures such as confusion matrix, precision, recall and f-score. But it gave me the following error. I’d be very happy if you could help.\
\
\
\
     Code:\
\
\
     confusion matrix\
\
\
     Y\_pred = baseline\_model.predict(X)\
\
\
     Y\_pred\_classes=np.argmax(Y\_pred, axis=1)\
\
\
     Y\_true= np.argmax(Y, axis=1)\
\
\
     confusion\_mtx= confusion\_matrix (Y\_true, Y\_pred\_classes)\
\
\
     fig,ax= plt.subplots(figsize=(8,8))\
\
\
     sns.heatmap(confusion\_mtx, annot=True, linewidths=0.01, cmap=’Greens’, linecolor=’gray’, fmt=’.1f’, ax=ax)\
\
\
     plt.ylabel(‘Gerçek Sınıf’)\
\
\
     plt.xlabel(‘Tahmin Edilen Sınıf’)\
\
\
\
     \# accuracy: (tp + tn) / (p + n)\
\
\
     accuracy = accuracy\_score(Y\_true, Y\_pred\_classes)\
\
\
     print(‘Accuracy: %f’ % accuracy)\
\
\
     \# precision tp / (tp + fp)\
\
\
     precision = precision\_score(Y\_true, Y\_pred\_classes, average=”macro”)\
\
\
     print(‘Precision: %f’ % precision)\
\
\
     \# recall: tp / (tp + fn)\
\
\
     recall = recall\_score(Y\_true, Y\_pred\_classes, average=”macro”)\
\
\
     print(‘Recall: %f’ % recall)\
\
\
     \# f1: 2 tp / (2 tp + fp + fn)\
\
\
     f1 = f1\_score(Y\_true, Y\_pred\_classes, average=”macro”)\
\
\
     print(‘F1 score: %f’ % f1)\
\
\
     plt.show()\
\
\
\
     ERROR:\
\
\
     AttributeError: ‘function’ object has no attribute ‘predict’\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492313)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 10, 2019 at 8:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492353 "Direct link to this comment")\
\
\
\
\
\
       This is a common question that I answer here:\
\
       [https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code](https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492353)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/78c0c673f3e0cb82effe5f809317c0ba03d3aab567bbf4fa46f88bee860e572b?s=40&d=mm&r=g)\
\
\
\
         Esra KarasuJuly 10, 2019 at 5:56 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492388 "Direct link to this comment")\
\
\
\
\
\
         Thank you. Code is running. It gives accuracy. But it doesn’t give the confusion matrix. I can’t find my mistake.\
\
\
\
         Y\_pred = baseline\_model.predict(X)\
\
\
         Y\_pred\_classes=np.argmax(Y\_pred, axis=1)\
\
\
         Y\_true= np.argmax(Y, axis=1)\
\
\
\
         There’s an error here.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492388)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2019 at 9:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492436 "Direct link to this comment")\
\
\
\
\
\
           Perhaps use the sklearn function:\
\
           [https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492436)\
154. ![](https://secure.gravatar.com/avatar/cea81d27afd94e48471dfe65a4a0ec44522853af2913f41eb9d0f35750b5856a?s=40&d=mm&r=g)\
\
\
\
     Jiawei ZhangJuly 12, 2019 at 12:18 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492616 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason!\
\
\
\
     Much thanks to your tutorials (I finished my first fully functional lstm classification project)\
\
\
\
     I have a simple question about keras LSTM binary classification, it might sounds stupid but I am stuck.\
\
\
     My train\_y and test\_y are now values of {0,1,2,4}. I want to set the binary output label 0 if{0,1} 1 if {2,4}. Could you give me some advice on how to do the data preprocessing please ?\
\
\
\
     Thank you so much!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492616)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 13, 2019 at 6:51 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492698 "Direct link to this comment")\
\
\
\
\
\
       Perhaps try defining your data manually?\
\
\
       Perhaps try defining your data programatically?\
\
\
       Perhaps try defining your data in excel?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-492698)\
155. ![](https://secure.gravatar.com/avatar/89f9e429b1108ddd8eaff5313ef2cfcfc358272dd77618b497210f4ff38dbfdd?s=40&d=mm&r=g)\
\
\
\
     anes ouadouAugust 6, 2019 at 12:49 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495690 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     thank you for your posts, I learned a lot from them. I have a multi class classification problem with three classes. I am currently trying to create the data for the training. the problem is the items belonging to each class are very close to each other to the point that when I extract one element that belongs to class 1 I will have a part of another element that belongs to class 2 or class 3. my question is, is it okay to have a part of an element that belong to one class appear in an instance that belongs to another class.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495690)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 6, 2019 at 2:06 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495705 "Direct link to this comment")\
\
\
\
\
\
       Perhaps you can locate or devise additional features that help to separate the instances/samples?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495705)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/89f9e429b1108ddd8eaff5313ef2cfcfc358272dd77618b497210f4ff38dbfdd?s=40&d=mm&r=g)\
\
\
\
         anes ouadouAugust 7, 2019 at 12:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495756 "Direct link to this comment")\
\
\
\
\
\
         the instances are extracted from a 3-D density map. Each instance is a type of atom that are located close to each other. I am trying to create a model to detect each atom (3 atoms) so that I can later find the optimal path between them (distance between atoms matters). I am treating the problem as multi-class classification. what do you recommend I do\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495756)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 7, 2019 at 7:59 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495801 "Direct link to this comment")\
\
\
\
\
\
           Intersting. Not sure what you’re trying to achieve exactly, optimal paths in n-dimensional space (e.g. 3d) sounds like a spanning tree or kd tree or similar would be more appropriate.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495801)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/89f9e429b1108ddd8eaff5313ef2cfcfc358272dd77618b497210f4ff38dbfdd?s=40&d=mm&r=g)\
\
\
\
             anes ouadouAugust 7, 2019 at 11:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495823 "Direct link to this comment")\
\
\
\
\
\
             The problem is for protein tertiary structure prediction. a protein is a series of amino acids. There are these three atoms that appear in each amino acid. So I am trying to detect them so that later on I can find the optimal path. Predicting the correct location of these atoms facilitate the building of the path.\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)August 7, 2019 at 2:21 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495840 "Direct link to this comment")\
\
\
\
\
\
             Perhaps distance between points, e.g predict membership of new point based on a distance measure, like euclidean distance?\
156. ![](https://secure.gravatar.com/avatar/6d0ef0698c4850ae0c214059033072434d25f57c6a48f5e24e5c3eb64ee3b3d4?s=40&d=mm&r=g)\
\
\
\
     DoronAugust 6, 2019 at 6:57 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495724 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I love reading your posts. Extremely helpful and well detailed.\
\
\
\
     I am currently working on a multiclass-multivariate-multistep time series forecasting project using LSTM’s and its other variations using Keras with Tensorflow backend. I was wondering perhaps you posted an article about it/something similar that I can use as a reference.\
\
\
\
     The closest one I have found (over the internet) was a post by you:\
\
\
\
     [https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)\
\
\
\
     However that did not include this specific problem statement. Any advice?\
\
\
\
     Much appreciated!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495724)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 7, 2019 at 7:44 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495786 "Direct link to this comment")\
\
\
\
\
\
       I do have examples of multi step, multivariate and time series classification, but not all together.\
\
\
\
       You can draw together the elements needed from the tutorials here:\
\
       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495786)\
157. ![](https://secure.gravatar.com/avatar/e075f8837e81e6fe90d2dcf9b82f9325f5d4f38c2da8f8cec602a25490b1605a?s=40&d=mm&r=g)\
\
\
\
     JGAugust 6, 2019 at 9:54 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495743 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Great (2016 old Tutorial), that I used to explore new facts. Let me share with you.\
\
\
\
     1) When I used KerasClassifier (within cross\_val\_score for Kfold partition) I repeat your results of 97.3% Acc and 4.4 for sigma (std deviation), but I also train a model (manually and I obtain Acc = 100%. So it is clear the effect of Kfold statistical partition that average results of many cases. do you agree?\
\
\
\
     2) I changed the module ‘keras.utils.np.utils.to\_categorical’ to more direct ‘keras.utils.to\_categorical’.same results. And using now ‘model Api keras’ instead of ‘sequential’ for more versatility.\
\
\
\
     3) I applied the Pipeline module to include ‘standardize’ options such as MinMaxScaler, StandardScaler, for Iris Input X data preprocessing. But I always get a little be worst results (96% Acc and 5.3 Sigma)…I am surprised about it! any idea why?\
\
\
\
     4) The most sensitive analysis I perform in comparison with your results is when apply ‘validation-split’ e.g. 0.2 instead of your default of 0.0 as argument of KerasClassifier…in that case Acc Kfol d(average) get down to 94.7% . I guess subtracting sample from training to allocate unsee validation sample must be the cause…do you agree?\
\
\
\
     5) I also confirme that if instead of using binary matrix of Iris Output (‘onehotencoding’) I use integer class values of Iris for training…I get worse results, as you anticipated it (i get down from 97% Acc to 88.7% Acc). OK.\
\
\
\
     6) I also implement ‘GaussianNoise’ function of keras layer to get better performance (some kind of data augmentation that simulate more sample data of Iris)…But always get ‘little’ be worst results or equal as maximum in some cases…any explanation?\
\
\
\
     Jason one more time thank you for your ‘scriplet’ fully codes that are inside any tutorial, as case study, that could be explore right away, numerically and conceptually, in many ways.\
\
\
\
     JG\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495743)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 7, 2019 at 7:56 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495796 "Direct link to this comment")\
\
\
\
\
\
       Good questions!\
\
\
\
       I would go with the k-fold result, in practice data samples are noisy, you want a robust score to reflect that.\
\
\
\
       Scaling is not a silver bullet, always good to check with and without, especially when using relu activations.\
\
\
\
       Changing the form of the output would require a change to loss function as well. categorical cross entropy for categorical distribution is a gold standard for a reason – it works really well.\
\
\
\
       Try shrinking the amount of noise down so that the samples don’t overlap too much across classes.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-495796)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/e075f8837e81e6fe90d2dcf9b82f9325f5d4f38c2da8f8cec602a25490b1605a?s=40&d=mm&r=g)\
\
\
\
         JGAugust 14, 2019 at 7:59 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496747 "Direct link to this comment")\
\
\
\
\
\
         wise answers Jason I appreciate your continuous engagement to share and give support to these tutorials…\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496747)\
158. ![](https://secure.gravatar.com/avatar/a8a545a072bd1c1e6b43280233bfb1930d17c6dffe88a5be48779928e622a3ba?s=40&d=mm&r=g)\
\
\
\
     [joker](https://tracekelston1986.wordpress.com/)August 13, 2019 at 5:06 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496539 "Direct link to this comment")\
\
\
\
\
\
     This site was… how do you say it? Relevant!! Finally I’ve found something that helped me.\
\
\
     Many thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496539)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 13, 2019 at 6:14 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496557 "Direct link to this comment")\
\
\
\
\
\
       Thanks, I’m glad it helps.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-496557)\
159. ![](https://secure.gravatar.com/avatar/32dd070b9a3066f0ba19fbc3d8db64074f29dbe42a1a5a0fa7f4f3b27bf4fe3a?s=40&d=mm&r=g)\
\
\
\
     PoojaAugust 19, 2019 at 1:22 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-497273 "Direct link to this comment")\
\
\
\
\
\
     hi,\
\
\
     I m doing work on EMG classification where I have 3 different types of EMG time series data named as myopathy, neuropathy, healthy data. my task is to build a model that classifies different EMG.\
\
\
\
     so my question is can I classify my data without attribute of data .if no then please let me how I can find the different attribute of my data and feed to network\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-497273)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 19, 2019 at 6:11 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-497293 "Direct link to this comment")\
\
\
\
\
\
       To train a supervised learning model, you must have input data and a label or real value as output.\
\
\
\
       If you are working with time series classification data, you can get started here:\
\
       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-497293)\
160. ![](https://secure.gravatar.com/avatar/0a0aca4ee14fda43971109879739120d15cc6ac976e5298ad83e3269edb4b362?s=40&d=mm&r=g)\
\
\
\
     [Layne](https://www.linkedin.com/in/laynesadler/)August 30, 2019 at 1:24 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499074 "Direct link to this comment")\
\
\
\
\
\
     Thank you so much! I love it.\
\
\
\
     10 is a lot of cv folds for such a small dataset. Feels like the folds would be too small to get 10 good chunks that represent the data. I went with 3 and got `Baseline: 98.00% (1.63%)`.\
\
\
\
     Have you written other more advanced keras classification tutorials?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499074)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 30, 2019 at 2:17 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499082 "Direct link to this comment")\
\
\
\
\
\
       Thanks. Yes, you could be right, 15 examples per fold is small.\
\
\
\
       Yes, some of the computer vision examples are more advanced:\
\
       [https://machinelearningmastery.com/start-here/#dlfcv](https://machinelearningmastery.com/start-here/#dlfcv)\
\
\
\
       The time series examples are more advanced as well:\
\
       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)\
\
\
\
       What are you looking for exactly?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499082)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/0a0aca4ee14fda43971109879739120d15cc6ac976e5298ad83e3269edb4b362?s=40&d=mm&r=g)\
\
\
\
         LayneAugust 30, 2019 at 2:29 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499086 "Direct link to this comment")\
\
\
\
\
\
         Thanks Jason! Well, I am gearing up for a project to automatically classify DNA mutations (MB of labeled data, not GB). There are 4 categories of the `impact` column with subcategories of each\
\
         [https://useast.ensembl.org/info/genome/variation/prediction/predicted\_data.html](https://useast.ensembl.org/info/genome/variation/prediction/predicted_data.html)\
\
\
\
         So I am looking to learn things like “how many layers and nodes should i have” and “what are other important feature engineering tools aside from StandardScaler().”\
\
\
\
         Here is a slice of the data (not the real dataset)\
\
         [https://www.kaggle.com/kevinarvai/clinvar-conflicting](https://www.kaggle.com/kevinarvai/clinvar-conflicting)\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499086)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 31, 2019 at 6:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499143 "Direct link to this comment")\
\
\
\
\
\
           This might help:\
\
           [https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network](https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499143)\
     - ![](https://secure.gravatar.com/avatar/0a0aca4ee14fda43971109879739120d15cc6ac976e5298ad83e3269edb4b362?s=40&d=mm&r=g)\
\
\
\
       Layne SadlerAugust 30, 2019 at 2:20 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499084 "Direct link to this comment")\
\
\
\
\
\
       A second run with the same settings `98.67% (0.94%)`\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-499084)\
161. ![](https://secure.gravatar.com/avatar/ad417c1a2d5c1ae2596179d466186a37f070f58be86110161e24ac802efa6a68?s=40&d=mm&r=g)\
\
\
\
     MehrabSeptember 9, 2019 at 4:59 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500695 "Direct link to this comment")\
\
\
\
\
\
     in this model, how i can generate classification report like precision & recall value\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500695)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 10, 2019 at 5:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500782 "Direct link to this comment")\
\
\
\
\
\
       See this post:\
\
       [https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-500782)\
162. ![](https://secure.gravatar.com/avatar/f161c1a2bf4bd7bd10fbcf068c2e45750d71d47b4a0bb22e5ca5662664e7af60?s=40&d=mm&r=g)\
\
\
\
     wanSeptember 24, 2019 at 5:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-502717 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Sorry I’m new to this. I have 2 question.\
\
\
\
     Is it important for the dataset in CSV file?\
\
\
     If i have set of dataset image in .png, how to modify the coding?\
\
\
\
     Thank you =)\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-502717)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 24, 2019 at 7:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-502751 "Direct link to this comment")\
\
\
\
\
\
       Yes, this tutorial will show you how to load images:\
\
       [https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/](https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/)\
\
\
\
       And this:\
\
       [https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-502751)\
163. ![](https://secure.gravatar.com/avatar/0505834b68ca01bcea2895a99934ad64aaf5601bdb5f3b8c8ab16f8c086654ea?s=40&d=mm&r=g)\
\
\
\
     majimomiOctober 25, 2019 at 12:25 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-507319 "Direct link to this comment")\
\
\
\
\
\
     What is a point for introducing scikit-learn here? We could just stick to Keras to train our model using Keras?\
\
\
\
     \# Define loss function and optimization technique\
\
\
     model.compile(\
\
\
     optimizer=’adam’,\
\
\
     loss=’categorical\_crossentropy’,\
\
\
     metrics=\[‘accuracy’\],\
\
\
     )\
\
\
\
     \# Train the model\
\
\
     history = model.fit(X,dummy\_Y,epochs=200, batch\_size=5, verbose=0)\
\
\
\
     \# evaluate the keras model\
\
\
     \_, accuracy = model.evaluate(X, dummy\_Y)\
\
\
     print(‘Accuracy: %.2f’ % (accuracy\*100))\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-507319)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 25, 2019 at 6:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-507365 "Direct link to this comment")\
\
\
\
\
\
       Yes, you can use Keras directly.\
\
\
\
       The wrapper helps if you want to use a pipeline or cross validation.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-507365)\
164. ![](https://secure.gravatar.com/avatar/0d4cf252ce05cd1724b486ce860e615efd3f62d1a6b214f893fe6f250b067c58?s=40&d=mm&r=g)\
\
\
\
     naniNovember 26, 2019 at 7:55 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512494 "Direct link to this comment")\
\
\
\
\
\
     i have a data in 40001 rows and 8 columns in that how to take input layer size and hidden layer layers\
\
\
     i’m taking\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(12, input\_dim=8, activation=’relu’))\
\
\
     model.add(Dense(8, activation=’relu’))\
\
\
     model.add(Dense(1, activation=’sigmoid’))\
\
\
     this is correct t worng?\
\
\
     how?\
\
\
     i did n’t understanding neural network?\
\
\
     plz help me?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512494)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 27, 2019 at 6:03 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512553 "Direct link to this comment")\
\
\
\
\
\
       Sounds like a good start, perhaps then try tuning the model in order to get the most out of it.\
\
\
\
       There are some suggestions here:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512553)\
165. ![](https://secure.gravatar.com/avatar/0d4cf252ce05cd1724b486ce860e615efd3f62d1a6b214f893fe6f250b067c58?s=40&d=mm&r=g)\
\
\
\
     naniNovember 27, 2019 at 5:19 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512651 "Direct link to this comment")\
\
\
\
\
\
     thank you for valuable time….\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512651)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 28, 2019 at 6:32 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512720 "Direct link to this comment")\
\
\
\
\
\
       You’re welcome.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512720)\
166. ![](https://secure.gravatar.com/avatar/0d4cf252ce05cd1724b486ce860e615efd3f62d1a6b214f893fe6f250b067c58?s=40&d=mm&r=g)\
\
\
\
     naniNovember 28, 2019 at 3:51 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512796 "Direct link to this comment")\
\
\
\
\
\
     i have a data training data 40001 rows and 8 columns and testing data 40001 x 8 how to take input layer size and hidden layer layers\
\
\
     i did n’t understanding neural network?\
\
\
     how to classify the one class neural network\
\
\
     send me neural network programming code??\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512796)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 29, 2019 at 6:43 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512884 "Direct link to this comment")\
\
\
\
\
\
       Perhaps start with this tutorial to better understand how to develop a small neural network:\
\
       [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-512884)\
167. ![](https://secure.gravatar.com/avatar/818d16d400f3ec0d0f2fb0b2823087c6c888c82e945fa07319dd8cae8fb4e13a?s=40&d=mm&r=g)\
\
\
\
     Rana SaleemDecember 6, 2019 at 4:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-513831 "Direct link to this comment")\
\
\
\
\
\
     Dear\
\
\
     Please how can i handle output desecrate value 0,25,50,75,100 and the data also in numeric form. you have any example code please share the link. and guide me.does any need of classification? how can i handle?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-513831)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 6, 2019 at 5:27 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-513864 "Direct link to this comment")\
\
\
\
\
\
       Perhaps you can post-process the predictions?\
\
\
\
       Perhaps you can map the discrete values to an ordinal, e.g. 1,2,3?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-513864)\
168. ![](https://secure.gravatar.com/avatar/9425845f96b964a1a02f1a09f90ac3af7c24edb91eba2eb6d35d2d555e4e1713?s=40&d=mm&r=g)\
\
\
\
     zaheer Ullah KhanJanuary 3, 2020 at 8:10 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-517005 "Direct link to this comment")\
\
\
\
\
\
     Hello, Jason, Your articles and post are really awesome, would you please a post about multi-class multi-label problem. and brief about some evaluation metrics used in measuring the model output.\
\
\
     would be very thankful.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-517005)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 4, 2020 at 8:29 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-517051 "Direct link to this comment")\
\
\
\
\
\
       See this tutorial:\
\
       [https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-517051)\
169. ![](https://secure.gravatar.com/avatar/89594c3b0fb8c4cd9fcca2f9bb50ce1ef799fcbbe7c239a17330dae9d82cef72?s=40&d=mm&r=g)\
\
\
\
     Shiva Ram DamFebruary 8, 2020 at 4:05 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-520809 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Thanks for the great post. I am taking reference from your post for my masters thesis.\
\
\
\
     How can we print the individual confusion matrix for each fold of cross validation set (here 10 folds in your tutorial). And also the confusion matrix for overall validation set.\
\
\
\
     Looking forward for your prompt response.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-520809)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 8, 2020 at 7:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-520843 "Direct link to this comment")\
\
\
\
\
\
       No, confusion matrix is used for one test set only.\
\
\
\
       Use a different metric across the folds.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-520843)\
170. ![](https://secure.gravatar.com/avatar/b1bb8b4ab00b865878207eef128daf8cb74dfb21df7049e31a4f03a8b7d847cb?s=40&d=mm&r=g)\
\
\
\
     Cameron WilsonFebruary 13, 2020 at 11:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521491 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
\
     I am trying to implement a CNN for classifying images. I have been following bits of a couple of different tutorials on how to do each section.\
\
\
\
     I have a convolutional model I think I am happy with, however, my problem arises that I want to do k-fold validation as shown in your tutorial here. The other tutorial I have been following uses ImageDataGenerator().flow\_from\_directory() but I see no way to use this and then perform k-fold validation on the data.\
\
\
\
     My data set is a total of 50,000 images split into 24 respective folders of each class of image.\
\
\
\
     Any pointers would be appreciated.\
\
\
\
     Thanks\
\
\
\
     Cameron\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521491)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 13, 2020 at 1:24 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521503 "Direct link to this comment")\
\
\
\
\
\
       Yes, perhaps enumerate the k-fold manually, this shows you how:\
\
       [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521503)\
171. ![](https://secure.gravatar.com/avatar/9ce7f8ed4fee9127d04e7a48ecfbe5d2d37f79f9aa39409cafd83ce0b146e3a4?s=40&d=mm&r=g)\
\
\
\
     adamFebruary 13, 2020 at 3:47 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521531 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Appreciate your hard work on these tutorials.It really helps.\
\
\
     I have a question at high level:\
\
\
\
     I’ve done multiple multi-class classification projects. Some of them I can transfer the problem to be building multiple binomial classification model.Some are not.\
\
\
\
     For a multi-class classification problem with let’s say 100 classes. It is usually very hard for the model to make prediction. I’ve been trying to build tree-based models, but the accuracy or confusion metrics dont seem good enough.\
\
\
     My question is: Is neural network (deep learning) models a better fit for this problem? How should we approach classification problem with a large number of classes?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521531)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 14, 2020 at 6:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521611 "Direct link to this comment")\
\
\
\
\
\
       Thanks.\
\
\
\
       It really depends on the specifics of the data. I recommend testing a suite of different algorithms in order to discover what works best for your dataset.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-521611)\
172. ![](https://secure.gravatar.com/avatar/ee4c20ca014a408cd28bb02f77a765132307831ffda3e311d42eddd8f26753dd?s=40&d=mm&r=g)\
\
\
\
     Alex RamirezMarch 16, 2020 at 7:05 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-525708 "Direct link to this comment")\
\
\
\
\
\
     Hello! Amazing explanaition. I have a question.\
\
\
\
     In the example where you add the following code:\
\
\
\
     \# fix random seed for reproducibility\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed)\
\
\
\
     My question is If I add\
\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed) ; numpy.random.rand(4)\
\
\
\
     to restart the random seed, do you think its a good idea?\
\
\
\
     If so, what number would you use for this example?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-525708)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 16, 2020 at 10:22 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-525713 "Direct link to this comment")\
\
\
\
\
\
       Thanks.\
\
\
\
       I would recommend removing random seed stuff these days and use repeated cross-validation to evaluate your model:\
\
       [https://machinelearningmastery.com/evaluate-skill-deep-learning-models/](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-525713)\
173. ![](https://secure.gravatar.com/avatar/ee4c20ca014a408cd28bb02f77a765132307831ffda3e311d42eddd8f26753dd?s=40&d=mm&r=g)\
\
\
\
     Alex RamirezMarch 16, 2020 at 7:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-525709 "Direct link to this comment")\
\
\
\
\
\
     I forgot to ask. How many baseline scores would you consider as minimum to obtain the average?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-525709)\
\
174. ![](https://secure.gravatar.com/avatar/fa7c3c2818ac92cc9097e3fad0a70a1a76cde2e1a246a1d51ce8cb65161e6aae?s=40&d=mm&r=g)\
\
\
\
     Mbonu ChineduApril 24, 2020 at 2:37 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-531504 "Direct link to this comment")\
\
\
\
\
\
     Thank you very much for this topic jason.\
\
\
     it really helped me in solving a huge problem for Multi Label classification.\
\
\
\
     Thanks J…..\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-531504)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 25, 2020 at 6:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-531573 "Direct link to this comment")\
\
\
\
\
\
       You’re welcome, I’m happy to hear that!\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-531573)\
175. ![](https://secure.gravatar.com/avatar/dc97f96e92f9b9ace5d2b2c45799d3c5cc6985d9fa83e6bae0bf335d1e5fef48?s=40&d=mm&r=g)\
\
\
\
     Sankar RajApril 30, 2020 at 2:44 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-532427 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
     How to find the number of neurons for hidden layer(s)? Is there any specific method or approach?\
\
\
\
     Thanks in advance!!\
\
\
     Sankar R\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-532427)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 30, 2020 at 6:51 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-532486 "Direct link to this comment")\
\
\
\
\
\
       Good question, I answer it here:\
\
       [https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network](https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-532486)\
176. ![](https://secure.gravatar.com/avatar/11948f58805d77d2a4e8b4b0b78725ed28cabd2f837383d635088075250be1c4?s=40&d=mm&r=g)\
\
\
\
     AchinthaMay 16, 2020 at 1:54 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-534887 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     my project have 3 inputs and 1 output this output I mean predicted value. so my question is this tutorial can I use my situation??\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-534887)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 16, 2020 at 6:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-534926 "Direct link to this comment")\
\
\
\
\
\
       If the output is a class label and there are more than 2 labels, this might be a useful tutorial for your problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-534926)\
177. ![](https://secure.gravatar.com/avatar/11948f58805d77d2a4e8b4b0b78725ed28cabd2f837383d635088075250be1c4?s=40&d=mm&r=g)\
\
\
\
     AchinthaMay 21, 2020 at 1:22 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535718 "Direct link to this comment")\
\
\
\
\
\
     Thanks a lot,\
\
\
\
     Hi. Jason,\
\
\
     ValueError: Error when checking input: expected dense\_3\_input to have shape (4,) but got array with shape (2,) – when I input the last two lines in this tutorial come up this error. why error like this??\
\
\
\
     X = dataset\[:,0:4\].astype(float)\
\
\
     Y = dataset\[:,4\] these your code lines I changed like this,\
\
\
\
     X = dataset\[:,1:3\].astype(float)\
\
\
     Y = dataset\[:,4\]\
\
\
\
     Thank you.\
\
\
     Achintha\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535718)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 21, 2020 at 1:42 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535735 "Direct link to this comment")\
\
\
\
\
\
       Sorry to hear that, this will help:\
\
       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535735)\
178. ![](https://secure.gravatar.com/avatar/183ff68a35f9aa0b422ecbf23b5109ce5d94b0adc65df296b0705891d5362b57?s=40&d=mm&r=g)\
\
\
\
     sanaMay 22, 2020 at 4:02 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535911 "Direct link to this comment")\
\
\
\
\
\
     how can i convert image dataset to csv file and how can I differentiate species of fruit fly\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535911)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 23, 2020 at 6:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535997 "Direct link to this comment")\
\
\
\
\
\
       We do not convert images to CVS, we load them directly as numpy arrays:\
\
       [https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-535997)\
179. ![](https://secure.gravatar.com/avatar/7f8b285e09cc73c440d20759e4054537629b931ebc0db673d3b1717ace3cafd4?s=40&d=mm&r=g)\
\
\
\
     QUANG HUY CHUMay 26, 2020 at 12:40 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-536455 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, I have run the model for several time and noticed that as my dataset (which is 5 input, 3 classes) I got standard deviation result about over 40%.\
\
\
\
     Can you have any suggestions how we can optimize this value or it is come from my dataset value?\
\
\
\
     Thank you vary much\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-536455)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 26, 2020 at 1:22 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-536466 "Direct link to this comment")\
\
\
\
\
\
       Yes, the tutorials here will help you lift the performance of your deep learning model:\
\
       [https://machinelearningmastery.com/start-here/#better](https://machinelearningmastery.com/start-here/#better)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-536466)\
180. ![](https://secure.gravatar.com/avatar/7f8b285e09cc73c440d20759e4054537629b931ebc0db673d3b1717ace3cafd4?s=40&d=mm&r=g)\
\
\
\
     QUANG HUY CHUJune 5, 2020 at 12:56 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538033 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, as I see your code I have noticed this line:\
\
\
\
     estimator = KerasClassifier(build\_fn=baseline\_model, epochs=200, batch\_size=5, verbose=0)\
\
\
     results = cross\_val\_score(estimator, X, dummy\_y, cv=kfold)\
\
\
\
     Also in another post I also see you use this code:\
\
\
\
     history = model.fit(trainX, trainy, validation\_data=(testX, testy), epochs=100, verbose=0)\
\
\
\
     What is different aim of those 2 code line since the model is constructed in the same way.\
\
\
\
     The Baseline is the same with Accuracy ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538033)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 5, 2020 at 8:16 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538114 "Direct link to this comment")\
\
\
\
\
\
       The first line defines the model then evaluates it using cross-validation.\
\
\
\
       The second fits the model on a train dataset and evaluates it each epoch using a validation dataset.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538114)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/7f8b285e09cc73c440d20759e4054537629b931ebc0db673d3b1717ace3cafd4?s=40&d=mm&r=g)\
\
\
\
         QUANG HUY CHUJune 5, 2020 at 10:46 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538157 "Direct link to this comment")\
\
\
\
\
\
         So as I understand the First model is used when we want to check how good the model with Training dataset with KFold Cross-Validation\
\
\
\
         The Seccond Model is used when we check how good the model with the validation data (which is split from the train data), also the training data of this model just trained one time only and use the parameter from that train and predict the validation data (Its like one time Kfold validation if k=1).\
\
\
\
         Sorry if what i am saying confused you, I am new to Keras and also Deep Learning, I am read many your post and figuring how the difference when we want to build a model and test the model from the beginning.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538157)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 5, 2020 at 1:41 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538177 "Direct link to this comment")\
\
\
\
\
\
           You can do it that way if you like. Whatever gives you confidence in evaluating the models performance in making predictions on new data.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538177)\
181. ![](https://secure.gravatar.com/avatar/2e5470513200fcbf58bb7b2e781f1b4ffec1d5e02b4dab106c7914c045bc642b?s=40&d=mm&r=g)\
\
\
\
     AasthaJune 8, 2020 at 7:56 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538669 "Direct link to this comment")\
\
\
\
\
\
     Could you please let me know what would be the best approach for image classification in case we have an extremely large number of Labels and there might be overlapping in some labels i.e. not all are extremely distinguishable.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538669)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 9, 2020 at 6:01 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538730 "Direct link to this comment")\
\
\
\
\
\
       Perhaps try using transfer learning and tune a model to your dataset.\
\
\
\
       This may help as a starting point that you can adapt to your problem:\
\
       [https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-538730)\
182. ![](https://secure.gravatar.com/avatar/01afc1be285a2dd73897a0958f601f7f497f9c01cb81b9da18ba649b1105ec26?s=40&d=mm&r=g)\
\
\
\
     KaushikJune 20, 2020 at 5:06 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540205 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason, Does this classification work if there are let’s say 10 classes and all 9 classes are integers and one class is a string. Does the encoding work in this case?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540205)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 20, 2020 at 6:19 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540241 "Direct link to this comment")\
\
\
\
\
\
       All classes must be encoded as numbers first.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540241)\
183. ![](https://secure.gravatar.com/avatar/7d1c41ff2126b1337695ddddeb605c98d15403d03bb15923f4d93fa2d426dca0?s=40&d=mm&r=g)\
\
\
\
     [MD MAHMUDUL HASAN](https://research.qut.edu.au/carrsq/people/md-mahmudul-hasan/)June 24, 2020 at 10:36 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540921 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason Brownlee, Thanks. Very helpful tutorial. How can I find the sensitivity & specificity in the case of 10 fold cross-validation instead of scores?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540921)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 25, 2020 at 6:18 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540976 "Direct link to this comment")\
\
\
\
\
\
       Good question, this library implements sensitivity and specificity:\
\
       [https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.metrics](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.metrics)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-540976)\
184. ![](https://secure.gravatar.com/avatar/079f1bb8b142b37ecffa6db004e9c496dfa4dace5aa165b391a43536430833d9?s=40&d=mm&r=g)\
\
\
\
     NicolasAugust 12, 2020 at 1:59 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-548751 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, sorry I have a question, if I want to use this model to predict the categorical class of some new data, lets say:\
\
\
\
     import numpy as np\
\
\
     new\_data = np.array(\[\[5.7, 2.5, 5. , 2. \]\])\
\
\
\
     How can I do that? since result that the baseline\_model () function returns does not have the .predict() function.\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-548751)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 12, 2020 at 6:11 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-548771 "Direct link to this comment")\
\
\
\
\
\
       This will show you how to make a single prediction:\
\
       [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-548771)\
185. ![](https://secure.gravatar.com/avatar/ac2c92f20da6e15a513894f4322e4423f81a67a9b3966a984ecfe286bfeb3db9?s=40&d=mm&r=g)\
\
\
\
     SergioAugust 27, 2020 at 1:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-551097 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason, I followed up and got similar results regarding the Iris multi-class problem, but then I tried to implement a similar solution to another multiclassification problem of my own and I’m getting less than 50% accuracy in the crossvalidation, I have already tried plenty of batch sizes, epochs and also added extra hiddien layers or change the number of neurons and I got from 30% to 50%, but I can’t seem to get any higher, can you please tell me what should I try, or why can this be happening? Or the way that I should troubleshoot it?\
\
\
\
     PD: I have also changed the sized of the input data and its features, to see if that was maybe the problem but it remains the same.\
\
\
\
     Thanks for your time, I’ll be waiting for a response.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-551097)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 27, 2020 at 6:20 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-551134 "Direct link to this comment")\
\
\
\
\
\
       Debugging and tuning neural nets is a big topic, you can get started here:\
\
       [https://machinelearningmastery.com/start-here/#better](https://machinelearningmastery.com/start-here/#better)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-551134)\
186. ![](https://secure.gravatar.com/avatar/9b033bb1148e6f0faaf3113a83ebe034784c44ef07f2e03392e7daf611f49cf0?s=40&d=mm&r=g)\
\
\
\
     BobAugust 29, 2020 at 10:14 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-553274 "Direct link to this comment")\
\
\
\
\
\
     How would you setup a 2, 3, 4 classification model? For instance if you have an NLP multi classification problem, where you have 4 labels \[agree, disagree, discuss, unrelated\], where related = \[agree, disagree, discuss\] this is also true so that: \[related, unrelated\].\
\
\
\
     How would you do:\
\
\
\
     1st. Model\
\
\
     \[related, unrelated\] — (classification model, but only grab the things classified as related) –>\
\
\
\
     2nd. Model ( gree = \[agree, disagree\] )\
\
\
     \[gree, unrelated\] –( classification model, but only grabs the gree)->\
\
\
\
     3rd Model\
\
\
     \[agree, disagree) –(classification model, that now classifies only these two) –> output would be all 4 original classifications without ‘related’. So it would be \[agree, disagree, discuss, unrelated\]\
\
\
\
     Really, I just don’t know how to divert the Keras results to a different model. Would I make multiple Y-columns that are one-hot encode like\
\
\
     \[agree\| disagree\| discuss\| unrelated\| related\]\
\
\
     0 1 0 0 1\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-553274)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 29, 2020 at 1:02 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-553407 "Direct link to this comment")\
\
\
\
\
\
       Probably start off treating the labels as nominal, one hot encoding, 4 nodes in the output layer.\
\
\
\
       Then perhaps try encoding them in the range 0-1, try modeling as a regression problem and see if the ordinal relationship can be harnessed.\
\
\
\
       Yes, to get started with one hot encoding, see this:\
\
       [https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)\
\
\
\
       Keras has the to\_categorical() function to make things very easy:\
\
       [https://keras.io/api/utils/python\_utils/#to\_categorical-function](https://keras.io/api/utils/python_utils/#to_categorical-function)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-553407)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/3903be43f77e8856bda87940cc61d6b73505dd6fd3a96ff01c3601b3dc7b06d1?s=40&d=mm&r=g)\
\
\
\
         VonkaSeptember 1, 2020 at 5:05 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-556675 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason, thank you for this wonderful article. I dit not see where to post a comment, I only see the reply button, so I post my comment here.\
\
\
         I would like to know how I could get the confusion matrix from this Multi-Class Classification model. Thank you in advance.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-556675)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)September 2, 2020 at 6:24 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-557155 "Direct link to this comment")\
\
\
\
\
\
           You’re welcome.\
\
\
\
           Here is an example:\
\
           [https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-557155)\
187. ![](https://secure.gravatar.com/avatar/85c7af48b0a2a6dd1e1cb21a2e47381b35d47a3082cbc9db58b84e50c0fa4131?s=40&d=mm&r=g)\
\
\
\
     AndresSeptember 20, 2020 at 9:30 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-564072 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, very good article.\
\
\
     I have a question, on this website [https://unipython.com/clasificacion-multiclase-de-especies-de-flores/](https://unipython.com/clasificacion-multiclase-de-especies-de-flores/)\
\
\
     They use your article, have they asked your permission? because I think they charge money because it is within a more general course…\
\
\
     A greeting\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-564072)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 21, 2020 at 8:09 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-564115 "Direct link to this comment")\
\
\
\
\
\
       They do not have permission!\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-564115)\
188. ![](https://secure.gravatar.com/avatar/7e1cf165a2ba42a1c5f20eaaa8cddb920c21d1b8f44c146331d53a12e71afdbc?s=40&d=mm&r=g)\
\
\
\
     K DOctober 27, 2020 at 3:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-570969 "Direct link to this comment")\
\
\
\
\
\
     The code did not run. I got the following message:\
\
\
     No module named ‘scipy.sparse’\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-570969)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 27, 2020 at 6:48 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-571003 "Direct link to this comment")\
\
\
\
\
\
       Perhaps you need to update your version of scipy.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-571003)\
189. ![](https://secure.gravatar.com/avatar/616181abb2afd355250ab7da48051316c9f8f8dc42ab5b5cf3059680b46fdd15?s=40&d=mm&r=g)\
\
\
\
     A ADecember 16, 2020 at 5:57 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-585230 "Direct link to this comment")\
\
\
\
\
\
     Do I also have to one-hot encode the class labels even if I use the loss parameter sparse\_categorical\_crossentropy as an argument to model.compile function?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-585230)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 16, 2020 at 7:54 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-585265 "Direct link to this comment")\
\
\
\
\
\
       No.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-585265)\
190. ![](https://secure.gravatar.com/avatar/262781573baaa39efdf3298317db083240162d637fc07ed4468f54e61509c6b7?s=40&d=mm&r=g)\
\
\
\
     Strivathsav Ashwin RamamoorthyDecember 28, 2020 at 7:10 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-589429 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I need your opinion on two questions which I have.\
\
\
\
     1) First one is that, I have been trying to implement a MLP model for multi-classification based on your post “Multi-class classification tutorial with keras deep learning library”. The input dimension is \[34000,33\] and output is \[34000,64\] where 64 is the total number of classes. I have defined an architecture as follows:\
\
\
     model = Sequential()\
\
\
     model.add(Dense(100, input\_dim = 33, activation = ‘relu’))\
\
\
     model.add(Dense(64, activation = ‘softmax’))\
\
\
     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’,metrics = \[‘accuracy’\])\
\
\
\
     I think I have defined one input layer, one hidden layer and one output layer. Could you validate the python lines which I have written?\
\
\
\
     2) Finally, after training the model how could we use the model to predict on some examples after training?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-589429)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 29, 2020 at 5:12 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-589563 "Direct link to this comment")\
\
\
\
\
\
       Looks like a good start.\
\
\
\
       You can make predictions by calling model.predict(), here are some examples:\
\
       [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-589563)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/262781573baaa39efdf3298317db083240162d637fc07ed4468f54e61509c6b7?s=40&d=mm&r=g)\
\
\
\
         Strivathsav Ashwin RamamoorthyJanuary 1, 2021 at 3:02 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-590358 "Direct link to this comment")\
\
\
\
\
\
         Thanks Jason for the response.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-590358)\
191. ![](https://secure.gravatar.com/avatar/ac943e5ed8f0af08a7698cee57946d6c4be9fe7b5f512a94e144b702bf6e5365?s=40&d=mm&r=g)\
\
\
\
     ahmed ben mohamedJanuary 1, 2021 at 9:13 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-590416 "Direct link to this comment")\
\
\
\
\
\
     how to download file csv this project ?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-590416)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 1, 2021 at 9:22 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-590419 "Direct link to this comment")\
\
\
\
\
\
       Here is the direct link:\
\
       [http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-590419)\
192. ![](https://secure.gravatar.com/avatar/ac943e5ed8f0af08a7698cee57946d6c4be9fe7b5f512a94e144b702bf6e5365?s=40&d=mm&r=g)\
\
\
\
     ahmed ben mohamedJanuary 3, 2021 at 9:50 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-591119 "Direct link to this comment")\
\
\
\
\
\
     ImportError: Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-591119)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 4, 2021 at 6:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-591203 "Direct link to this comment")\
\
\
\
\
\
       The error suggest you need to update your version of the tensorflow library.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-591203)\
193. ![](https://secure.gravatar.com/avatar/736728e61cb9098faa454ab0a8c977a1d8abd4385b4e1d893bad9540fd8a1a55?s=40&d=mm&r=g)\
\
\
\
     Muhammad Usama ZahidJanuary 9, 2021 at 8:59 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-592456 "Direct link to this comment")\
\
\
\
\
\
     I really love your tutorials. Very neatly explained.Kudos to u sir!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-592456)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 10, 2021 at 5:40 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-592495 "Direct link to this comment")\
\
\
\
\
\
       Thanks!\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-592495)\
194. ![](https://secure.gravatar.com/avatar/038b8b9004da8acf4e5d61860e3f4880dee9b9e9d14031a9dfdb2b7c83afd315?s=40&d=mm&r=g)\
\
\
\
     Katia laghaJanuary 24, 2021 at 9:56 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-594242 "Direct link to this comment")\
\
\
\
\
\
     First of all, thank you for this tutorial. I really appreciate it.\
\
\
     Then i have a question if you can help me !\
\
\
\
     I build a model that will predict 3 outputs. But if I want to predict on another dataset that contains just 2 of these values I can’t use the previous model since this one will have just 2 outputs, this is my problem!\
\
\
\
     To be more clearer, I’ll explain again.\
\
\
     We suppose that the IRIS database is divided into two datasets:\
\
\
     dataset1: Iris-setosa, Iris-versicolor, Iris-virginica\
\
\
     dataset2: Iris-setosa, Iris-versicolor\
\
\
\
     If I do my training on the 1st one. Then, if I want to reload the model and improve it but this time I do the training on the 2nd dataset. This is where I block because the number of outputs is not the same.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-594242)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 25, 2021 at 5:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-594275 "Direct link to this comment")\
\
\
\
\
\
       You’re welcome.\
\
\
\
       A model for dataset1 can be used to make predictions for dataset2 directly without change.\
\
\
\
       It could be fined tuned on dataset2, perhaps with a small learning rate.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-594275)\
195. ![](https://secure.gravatar.com/avatar/95fd4dd83e098a4b9ca440eddd4f15e1e4dc23c61e1420c2d0e223821d7cf483?s=40&d=mm&r=g)\
\
\
\
     Arjun SatishFebruary 17, 2021 at 6:07 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597730 "Direct link to this comment")\
\
\
\
\
\
     How can I display learning curves in the above python code? I am a novice.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597730)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 17, 2021 at 7:49 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597738 "Direct link to this comment")\
\
\
\
\
\
       This tutorial will show you how:\
\
       [https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597738)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/95fd4dd83e098a4b9ca440eddd4f15e1e4dc23c61e1420c2d0e223821d7cf483?s=40&d=mm&r=g)\
\
\
\
         Arjun SatishFebruary 18, 2021 at 6:43 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597934 "Direct link to this comment")\
\
\
\
\
\
         Is there a way to classify a single data point into more than one class? For, example, I have data about a flower and I need the model to predicts its presence in more than one class like it comes under plant, green-leafy, red-coloured.\
\
\
         So after we encode it, its may look like \[1,1,0,1,0\]. Hope you get the idea of what I am trying to project.\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597934)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)February 19, 2021 at 5:57 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597996 "Direct link to this comment")\
\
\
\
\
\
           Yes, predict() returns probabilities for each class.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-597996)\
196. ![](https://secure.gravatar.com/avatar/ed459d651110dcdce03d2b9426443319cd634460361bbf46bf6a18f133753a63?s=40&d=mm&r=g)\
\
\
\
     FarhatFebruary 28, 2021 at 7:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-599194 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     thanks for your awesome contents.\
\
\
\
     I wanted to know, what if I have multiple columns as outputs and all of them are categorical?\
\
\
\
     I understand that for one categorical output column I have to use n\_outputs for output layer with softmax activation.\
\
\
\
     But how to modify the model when I have suppose 5 columns with categorical values and I have say 3 categories in each.\
\
\
\
     Do i have to use 5 output layers with n\_output=3 and softmax activation or is there any way where i can do this in one layer?\
\
\
\
     Thanks in advance\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-599194)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 28, 2021 at 1:54 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-599222 "Direct link to this comment")\
\
\
\
\
\
       You’re welcome.\
\
\
\
       One approach would be to use the functional API and define 3 output models, each outputs a vector with softmax activation.\
\
\
\
       This will give you ideas:\
\
       [https://machinelearningmastery.com/keras-functional-api-deep-learning/](https://machinelearningmastery.com/keras-functional-api-deep-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-599222)\
197. ![](https://secure.gravatar.com/avatar/1014a23e0086345b363c2d09b46a9ae39d130ac532277c59bd3a869157222e4a?s=40&d=mm&r=g)\
\
\
\
     ShikharApril 1, 2021 at 10:52 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-603054 "Direct link to this comment")\
\
\
\
\
\
     If i am using only label encoder then my y\_train data will only contain3 different values and it would be of the shape ( -1 , 1) . Then can you please tell me what would be the last dense layer shape and what loss would be used\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-603054)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 2, 2021 at 5:39 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-603105 "Direct link to this comment")\
\
\
\
\
\
       When using a one hot encoding, the shape of y should be the number of samples (rows) and the number of classes (columns).\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-603105)\
198. ![](https://secure.gravatar.com/avatar/ee59dc685e00cd6a1958251b809ba8688e569256eec478993b6564c4c32163f6?s=40&d=mm&r=g)\
\
\
\
     MohammadrezaApril 21, 2021 at 2:26 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606287 "Direct link to this comment")\
\
\
\
\
\
     Hi\
\
\
\
     Don’t we need a DenseFeatures layer as the first layer for multi-class classification?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606287)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 22, 2021 at 5:36 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606394 "Direct link to this comment")\
\
\
\
\
\
       Sure, we do.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606394)\
199. ![](https://secure.gravatar.com/avatar/ee59dc685e00cd6a1958251b809ba8688e569256eec478993b6564c4c32163f6?s=40&d=mm&r=g)\
\
\
\
     MohammadrezaApril 22, 2021 at 9:54 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606437 "Direct link to this comment")\
\
\
\
\
\
     Thanks! I noticed you don’t have it in your code and the code still works. How would you add this layer to your code? Does it bring any advantages?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606437)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 23, 2021 at 4:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606593 "Direct link to this comment")\
\
\
\
\
\
       If you’re referring to Dense MLP layers, these can be created with the Dense() layer object.\
\
       [https://keras.io/api/layers/core\_layers/dense/](https://keras.io/api/layers/core_layers/dense/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-606593)\
200. ![](https://secure.gravatar.com/avatar/01303d3a3b166f26fad98495a741546a89684bf1187a3b70afed133f7b409f4f?s=40&d=mm&r=g)\
\
\
\
     Sowmya KrishnanApril 28, 2021 at 9:00 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607325 "Direct link to this comment")\
\
\
\
\
\
     Thanks for this helpful tutorial on multi-class classification! I’m working on a similar problem with a large number of classes (100+) and my metrics deteriorate with larger batch sizes (above 5). I noticed that you have used a batch size of 5 in the tutorial. Does the batch size depend in any way on the number of classes we have to predict at the end? How can I find the reason for my model metrics deteriorating for larger batch sizes?\
\
\
\
     For the model I have used a 1D-CNN which takes a string as input and predicts the classes as output. Any suggestions to troubleshoot this problem will be really helpful. And finally a naive question – Are smaller batch sizes such as 1 or 5 acceptable for publications? I’m really new to machine learning and I’m not aware if there is a general trend for batch sizes in publications.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607325)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 29, 2021 at 6:26 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607387 "Direct link to this comment")\
\
\
\
\
\
       Batch size may have to be tuned for your model and dataset, this may help:\
\
       [https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607387)\
201. ![](https://secure.gravatar.com/avatar/0484d85b46f879f36a5883097f81f34e17888a282817bf1ae48399b4768f4f68?s=40&d=mm&r=g)\
\
\
\
     Abraham LinApril 29, 2021 at 12:53 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607350 "Direct link to this comment")\
\
\
\
\
\
     Hi, Jason, Thank for for the tutorial. I am a beginner of deep learning. This model worked well on my computer. I assumed this model was “trained” by running this model existing iris data. My question is how to test the performance of this “trained” model on new datasets? Do you have codes to perform that?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607350)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 29, 2021 at 6:30 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607395 "Direct link to this comment")\
\
\
\
\
\
       We estimate the performance a model on new data using k-fold cross-validation:\
\
       [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/)\
\
\
\
       We then choose a model/config (based on k-fold cross-validation estiamtes), train it on all data and use it to start making predictions on new data:\
\
       [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-607395)\
202. ![](https://secure.gravatar.com/avatar/95fd4dd83e098a4b9ca440eddd4f15e1e4dc23c61e1420c2d0e223821d7cf483?s=40&d=mm&r=g)\
\
\
\
     Arjun SatishMay 29, 2021 at 8:29 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-611683 "Direct link to this comment")\
\
\
\
\
\
     Is this a feedback or feedforward algorithm?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-611683)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 30, 2021 at 5:50 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-611715 "Direct link to this comment")\
\
\
\
\
\
       Not sure what you mean exactly?\
\
\
\
       MLPs are a feed-forward neural network once trained. Their trained using backprop under SGD.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-611715)\
203. ![](https://secure.gravatar.com/avatar/8299a4d0f22e0e873d1738d99639ca2d170a74c0eae309bd6a5dc191e18a97e3?s=40&d=mm&r=g)\
\
\
\
     MaiteJune 25, 2021 at 11:08 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614594 "Direct link to this comment")\
\
\
\
\
\
     I’m working on a similar problem of multi-class clasification. How can I deal with adding “None of the above” in Image Classification? Is there any resourse I can check?\
\
\
\
     You website has always help me so much. Thank you!\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614594)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 26, 2021 at 4:55 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614630 "Direct link to this comment")\
\
\
\
\
\
       None of the above would be all zeros (e.g. no class).\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614630)\
204. ![](https://secure.gravatar.com/avatar/b6c165f9536e58eb564a7cc7b80db126a774337e5682e43cc095cb7effeeafdd?s=40&d=mm&r=g)\
\
\
\
     sinferJune 27, 2021 at 10:08 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614797 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     A pretty interesting learning !\
\
\
     I have a question if i use this MLP model in scenario like fault detection and diagnosis for a time series telemetry data with labeled as fault and normal data can i predict the possible class label based on the multiple input samples.\
\
\
\
     as per your model implemented here it will predict \[5,3,4,5\] –>>> iris setosa\
\
\
\
     I want to have something like this \[\[5,3,4,5\],\[4.9,3,4,5.2\]\[4.8,2.9,4,5.1\]\] –>>> iris setosa\
\
\
\
     Thank you\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614797)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 28, 2021 at 7:58 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614843 "Direct link to this comment")\
\
\
\
\
\
       Thanks!\
\
\
\
       Perhaps explore an RNN like an lstm:\
\
       [https://machinelearningmastery.com/start-here/#lstm](https://machinelearningmastery.com/start-here/#lstm)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614843)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/b6c165f9536e58eb564a7cc7b80db126a774337e5682e43cc095cb7effeeafdd?s=40&d=mm&r=g)\
\
\
\
         sinferJune 28, 2021 at 12:43 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614861 "Direct link to this comment")\
\
\
\
\
\
         Thanks 😉\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614861)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 29, 2021 at 4:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614925 "Direct link to this comment")\
\
\
\
\
\
           You’re welcome.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614925)\
205. ![](https://secure.gravatar.com/avatar/b6c165f9536e58eb564a7cc7b80db126a774337e5682e43cc095cb7effeeafdd?s=40&d=mm&r=g)\
\
\
\
     sinferJune 28, 2021 at 2:37 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614819 "Direct link to this comment")\
\
\
\
\
\
     One more thing Jason,\
\
\
     Have you got any multi class classification done with a Deep Neural Network, since this is a MLP implemented here\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614819)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 28, 2021 at 7:59 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614844 "Direct link to this comment")\
\
\
\
\
\
       Yes, you can see some LSTM examples for HAR here:\
\
       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)\
\
\
\
       There are also many CNN examples for multi-class classification on the blog.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614844)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/b6c165f9536e58eb564a7cc7b80db126a774337e5682e43cc095cb7effeeafdd?s=40&d=mm&r=g)\
\
\
\
         sinferJune 28, 2021 at 11:38 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614855 "Direct link to this comment")\
\
\
\
\
\
         Cool.I saw that somewhere it had been mentioned, applying LSTM on time series data for multi class classification is not doing that good if my memory is correct. Is there any truth about that?\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614855)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 29, 2021 at 4:44 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614923 "Direct link to this comment")\
\
\
\
\
\
           It really depends on the specifics of the data. Perhaps try it and compare the results to other methods.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614923)\
206. ![](https://secure.gravatar.com/avatar/b6c165f9536e58eb564a7cc7b80db126a774337e5682e43cc095cb7effeeafdd?s=40&d=mm&r=g)\
\
\
\
     sinferJune 28, 2021 at 2:47 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614820 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Is there any way to convert to this MLP to DNN model by adding back propagation? if yes how.\
\
\
\
     Thanks a lot\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614820)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 28, 2021 at 7:59 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614845 "Direct link to this comment")\
\
\
\
\
\
       MLPs are trained using backprop in keras.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614845)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/b6c165f9536e58eb564a7cc7b80db126a774337e5682e43cc095cb7effeeafdd?s=40&d=mm&r=g)\
\
\
\
         sinferJune 28, 2021 at 11:51 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614857 "Direct link to this comment")\
\
\
\
\
\
         Oh thanks Jason, didn’t know that earlier.\
\
\
         Can you please check my following code i have added for my KERAS model to predict the label out of 8 classes for time series classification.\
\
\
\
         def create\_dnn\_model():\
\
\
         #create sequential model\
\
\
         model = Sequential()\
\
\
         model.add(Dense(64,input\_dim=10,activation=’relu’))\
\
\
         model.add(Dense(32,activation=’relu’))\
\
\
         model.add(Dense(16,activation=’relu’))\
\
\
         model.add(Dense(8,activation=’softmax’))\
\
\
         # Compile model\
\
\
         model.compile(loss=’categorical\_crossentropy’,optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
         return model\
\
\
\
         estimator = KerasClassifier(build\_fn=create\_dnn\_model, epochs=200, batch\_size=5, verbose =1)\
\
\
         estimator.fit(X\_train,y\_train)\
\
\
\
         There are like more than 100,000 records from all the 8 class labels.\
\
\
         So this is a MLP model. and I followed this IRIS tutorial to adhere the model into my implementation.\
\
\
\
         And more specially I have performed Standardization on input train data and input test data. And this model gives me a train accuracy of 99.05 % and test accuracy of 97%. but it hardly predicts a label correctly and i find this a bit confusing since my accuracy takes a high value.\
\
\
\
         and i got the train accuracy based on my standardized train data and test accuracy based on standardized test data like in the following\
\
\
\
         estimator.score(Scaled \_X\_train,y\_train)\
\
\
         estimator.score(Scaled\_X\_test,y\_test)\
\
\
\
         when i call the predict function i don’t have to call the exact same standardize function i have called on the input data on my sample inputs riight\
\
\
\
         [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614857)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 29, 2021 at 4:45 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614924 "Direct link to this comment")\
\
\
\
\
\
           I recommend testing a suite of model configurations in order to discover what works well or best for your specific dataset.\
\
\
\
           Yes, you must prepare any new data in an identical manner as the training set, e.g. the same data prep object.\
\
\
\
           [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614924)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/b6c165f9536e58eb564a7cc7b80db126a774337e5682e43cc095cb7effeeafdd?s=40&d=mm&r=g)\
\
\
\
             sinferJune 29, 2021 at 3:49 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-614959 "Direct link to this comment")\
\
\
\
\
\
             Can i call my model a deep neural network since i have used (more than one) hidden layers instead of calling this a MLP? And also since i have used an keras classifier this does back propagation as well right\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)June 30, 2021 at 5:17 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-615009 "Direct link to this comment")\
\
\
\
\
\
             Sure. It’s just marketing.\
\
\
\
             Yes, all neural nets are fit with backprop.\
207. ![](https://secure.gravatar.com/avatar/f531955ce529cc257b2f8537ecf5582da2a70f111370ccbcfce700c506d75cf4?s=40&d=mm&r=g)\
\
\
\
     AmandaJuly 5, 2021 at 11:44 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-615551 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
     Thank you in advance for the material that has been submitted\
\
\
\
     I want to ask for classes on cnn, if there are 213 classes, can I use a confusion matrix?\
\
\
     or is there anything other than confusion matrix?\
\
\
     thank you\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-615551)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2021 at 5:48 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-615573 "Direct link to this comment")\
\
\
\
\
\
       Yes, but it may not be useful – e.g. too large to read effectively.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-615573)\
208. ![](https://secure.gravatar.com/avatar/e470383d67365b7e322aa5b44f8fe41a9a3cc66596395f3e365531e6a1196ff4?s=40&d=mm&r=g)\
\
\
\
     WillJuly 28, 2021 at 2:03 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-618603 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason.\
\
\
\
     Thanks for the above article.\
\
\
\
     I have a query regarding the OHE aspect of the above.\
\
\
\
     Should one hot encoding (OHE) always be performed on the output variables as good practice? Even, for example, on a multi-output regression problem? (I have taken a look here: [https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/](https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/) but still unsure).\
\
\
\
     In the example I’m looking at, my dataframe/array has 8 separate columns which are the output variables the model is trying to predict. Should OHE be performed on these?\
\
\
\
     If so, this transforms the shape of my input/output data from:\
\
\
\
     Input (200664, 8)\
\
\
     Output (200664, 8)\
\
\
\
     To:\
\
\
\
     Input (200664, 8)\
\
\
     Output (200664, 8, 8)\
\
\
\
     Would this then mean I have to reshape my input variables? I appreciate it’s hard without visibility of the data/code, but any guidance would be more than welcomed.\
\
\
\
     Thanks again.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-618603)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 28, 2021 at 5:28 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-618636 "Direct link to this comment")\
\
\
\
\
\
       OHE is only needed for the output variable if there are more than two classes.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-618636)\
209. ![](https://secure.gravatar.com/avatar/333a252b45737f3a46d51ee4b8efd5da0f6373024696be151fe4e248b270307b?s=40&d=mm&r=g)\
\
\
\
     LenaAugust 19, 2021 at 10:06 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-622617 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thank you for the great tutorial. I am wondering if there is a way to label the outputs here like in a multi output Keras Functional API model. I am hoping to use KerasClassifier for 100+ categories (possible chronic conditions in medicare data), and having them labeled throughout would limit the possibility of mixing them up along the way.\
\
\
\
     Thanks again!\
\
\
     Lena\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-622617)\
\
\
\
\
     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)\
\
\
\
       Adrian TamAugust 20, 2021 at 1:28 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-622647 "Direct link to this comment")\
\
\
\
\
\
       Are you looking for the argmax() function in numpy?\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-622647)\
210. ![](https://secure.gravatar.com/avatar/7e1593922625442904fed7a410faed7e8d80e712a38564c4479796ae7fb3778c?s=40&d=mm&r=g)\
\
\
\
     sama samaanAugust 31, 2021 at 6:28 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-624965 "Direct link to this comment")\
\
\
\
\
\
     Hello\
\
\
     I have a dataset with 4000 rows. The deep learning model work well with 14 to 20 rows. when I put all the 4000 rows it takes too long time in execution and no results occur. what is the reason behind this?\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-624965)\
\
\
\
\
     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)\
\
\
\
       Adrian TamSeptember 1, 2021 at 8:31 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-625352 "Direct link to this comment")\
\
\
\
\
\
       Because you train with entire dataset in each iteration? You can use SGD and train with only a small subset of the rows in each iteration, but sampling from the entire 4000 rows each time.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-625352)\
211. ![](https://secure.gravatar.com/avatar/4214303737b8eddc15855c86f56d968048a6c4fc2e7277137314f6f86ec7fae9?s=40&d=mm&r=g)\
\
\
\
     KevinSeptember 1, 2021 at 3:16 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-625230 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     When I have created this model, how do I see which features were the strongest predictors in the model? I want to present this model to stakeholders, but how do I interpret the model? In a logistic regression I would look at the p-values of the model.\
\
\
\
     Many thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-625230)\
\
\
\
\
     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)\
\
\
\
       Adrian TamSeptember 1, 2021 at 9:00 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-625369 "Direct link to this comment")\
\
\
\
\
\
       Deep learning is difficult to see this, at least difficult to see from the model. One way to heuristically verify, however, is to do the prediction over and over with each feature replaced (e.g., zero out all features, or replace with random number) and expect the prediction went bad. An unimportant feature would not change the prediction much.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-625369)\
212. ![](https://secure.gravatar.com/avatar/47fc99db67ea0d0fa983d4311b48cb7c3991663c4878a1770e39de00b24a4d9c?s=40&d=mm&r=g)\
\
\
\
     benFebruary 6, 2022 at 4:19 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-653621 "Direct link to this comment")\
\
\
\
\
\
     Came across this article looking for ways to address class imbalance for a Multi-Class problem as this will affect the scores.\
\
\
\
     If the classes are say 90/40/20 accuracy will not be 97%.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-653621)\
\
213. ![](https://secure.gravatar.com/avatar/821af00b77b51b7a2d397891d979b1f11d23cce0332c0c2746d9bf8051cc451b?s=40&d=mm&r=g)\
\
\
\
     Saif SohelApril 8, 2022 at 8:18 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-663505 "Direct link to this comment")\
\
\
\
\
\
     Hi jason\
\
\
     \# encode class values as integers\
\
\
     encoder = LabelEncoder()\
\
\
     encoder.fit(Y)\
\
\
     encoded\_Y = encoder.transform(Y)\
\
\
     \# convert integers to dummy variables (i.e. one hot encoded)\
\
\
     dummy\_y = np\_utils.to\_categorical(encoded\_Y)\
\
\
\
     Um facing below problem\
\
\
\
     —————————————————————————\
\
\
     NameError Traceback (most recent call last)\
\
\
     in ()\
\
\
     1 # encode class values as integers\
\
\
     —-\> 2 encoder = LabelEncoder()\
\
\
     3 encoder.fit(Y)\
\
\
     4 encoded\_Y = encoder.transform(Y)\
\
\
     5 # convert integers to dummy variables (i.e. one hot encoded)\
\
\
\
     NameError: name ‘LabelEncoder’ is not defined\
\
\
\
     could you plz help me to resolve. For your kind information um a beginner\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-663505)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)\
\
\
\
       James CarmichaelApril 9, 2022 at 8:44 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-663652 "Direct link to this comment")\
\
\
\
\
\
       Hi Saif…Without seeing a complete code listing, it is not clear how you defined all of the variables and functions.\
\
\
\
       Please provide a complete listing so that we can better assist you.\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-663652)\
214. ![](https://secure.gravatar.com/avatar/64e6561d4c4e45ca516bc66672a6d02d5c6e186d45ac788739b8b86dc1266c55?s=40&d=mm&r=g)\
\
\
\
     Jannadi KhemaisOctober 4, 2022 at 7:38 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-682399 "Direct link to this comment")\
\
\
\
\
\
     Thank you, much appreciated , sharply and clearly explained, it helps me to make the implementation of neural networks easy for multiclass or multinomial classification.\
\
\
\
     Question: Could you please elaborate more the L2 and L1 regularization techniques and what is the difference between L1 and L2 regularization for linear regression? Please share tutorial link ,if possible \_ Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-682399)\
\
215. ![](https://secure.gravatar.com/avatar/b775ab6a67896068611bb6a8ce5d6941a06db43a19df8d1855712afa815e193a?s=40&d=mm&r=g)\
\
\
\
     MaryamMarch 30, 2023 at 11:02 pm[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-689549 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for this great tutorial. Could you kindly let me know how to get the confusion matrix and all metrics reports (precision, recall, f1) for all features/predictors using your exact code? I really appreciate it.\
\
\
\
     [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-689549)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)\
\
\
\
       James CarmichaelMarch 31, 2023 at 7:09 am[#](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-689565 "Direct link to this comment")\
\
\
\
\
\
       Hi Maryam…The following resource may be of interest:\
\
\
\
       [https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)\
\
\
\
       [Reply](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/#comment-689565)\
\
### Leave a Reply [Click here to cancel reply.](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/\#respond)\
\
Comment \*\
\
Name (required)\
\
Email (will not be published) (required)\
\
Δ