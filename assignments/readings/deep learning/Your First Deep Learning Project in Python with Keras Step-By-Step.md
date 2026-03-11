### [Navigation](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/\#navigation)

[![Airia Enterprise AI](https://www.kdnuggets.com/wp-content/uploads/t1-airia-2602.png)\\
\\
Learn more](https://airia.com/resources/2026-state-of-ai/?utm_source=GTM+2&utm_medium=Email&utm_campaign=Newsletter)

By[Jason Brownlee](https://machinelearningmastery.com/author/jasonb/ "Posts by Jason Brownlee")onAugust 16, 2022in[Deep Learning](https://machinelearningmastery.com/category/deep-learning/ "View all items in Deep Learning")[1,172](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comments)

Share _Post_Share

**Keras** is a powerful and easy-to-use free open source Python library for developing and evaluating _**[deep learning](https://machinelearningmastery.com/what-is-deep-learning/) models**_.

It is part of the [TensorFlow](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/) library and allows you to define and train neural network models in just a few lines of code.

In this tutorial, you will discover how to create your first deep learning neural network model in Python using Keras.

**Kick-start your project** with my new book [Deep Learning With Python](https://machinelearningmastery.com/deep-learning-with-python/), including _step-by-step tutorials_ and the _Python source code_ files for all examples.

_Let’s get started._

- **Update Feb/2017**: Updated prediction example, so rounding works in Python 2 and 3.
- **Update Mar/2017**: Updated example for the latest versions of Keras and TensorFlow.
- **Update Mar/2018**: Added alternate link to download the dataset.
- **Update Jul/2019**: Expanded and added more useful resources.
- **Update Sep/2019**: Updated for Keras v2.2.5 API.
- **Update Oct/2019**: Updated for Keras v2.3.0 API and TensorFlow v2.0.0.
- **Update Aug/2020**: Updated for Keras v2.4.3 and TensorFlow v2.3.
- **Update Oct/2021**: Deprecated predict\_class syntax
- **Update Jun/2022**: Updated to modern TensorFlow syntax

![Tour of Deep Learning Algorithms](https://machinelearningmastery.com/wp-content/uploads/2016/04/Tour-of-Deep-Learning-Algorithms.jpg)

Develop your first neural network in Python with Keras step-by-step

Photo by Phil Whitehouse, some rights reserved.

## Keras Tutorial Overview

There is not a lot of code required, but we will go over it slowly so that you will know how to create your own models in the future.

_The steps you will learn in this tutorial are as follows:_

1. Load Data
2. Define Keras Model
3. Compile Keras Model
4. Fit Keras Model
5. Evaluate Keras Model
6. Tie It All Together
7. Make Predictions

**This Keras tutorial makes a few assumptions. You will need to have:**

1. Python 2 or 3 installed and configured
2. SciPy (including NumPy) installed and configured
3. Keras and a backend (Theano or TensorFlow) installed and configured

If you need help with your environment, see the tutorial:

- [How to Setup a Python Environment for Deep Learning](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

Create a new file called **keras\_first\_network.py** and type or copy-and-paste the code into the file as you go.

### Need help with Deep Learning in Python?

Take my free 2-week email course and discover MLPs, CNNs and LSTMs (with code).

Click to sign-up now and also get a free PDF Ebook version of the course.

Start Your FREE Mini-Course Now

## 1\. Load Data

The first step is to define the functions and classes you intend to use in this tutorial.

You will use the [NumPy library](https://www.numpy.org/) to load your dataset and two classes from the [Keras library](https://www.tensorflow.org/api_docs/python/tf/keras) to define your model.

The imports required are listed below.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5 | \# first neural network with keras tutorial<br>from numpy import loadtxt<br>from tensorflow.keras.models import Sequential<br>from tensorflow.keras.layers import Dense<br>... |

You can now load our dataset.

In this Keras tutorial, you will use the Pima Indians onset of diabetes dataset. This is a standard machine learning dataset from the UCI Machine Learning repository. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

As such, it is a binary classification problem (onset of diabetes as 1 or not as 0). All of the input variables that describe each patient are numerical. This makes it easy to use directly with neural networks that expect numerical input and output values and is an ideal choice for our first neural network in Keras.

The dataset is available here:

- [Dataset CSV File (pima-indians-diabetes.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- [Dataset Details](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)

Download the dataset and place it in your local working directory, the same location as your Python file.

Save it with the filename:

|     |     |
| --- | --- |
| 1 | pima-indians-diabetes.csv |

Take a look inside the file; you should see rows of data like the following:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6 | 6,148,72,35,0,33.6,0.627,50,1<br>1,85,66,29,0,26.6,0.351,31,0<br>8,183,64,0,0,23.3,0.672,32,1<br>1,89,66,23,94,28.1,0.167,21,0<br>0,137,40,35,168,43.1,2.288,33,1<br>... |

You can now load the file as a matrix of numbers using the NumPy function [loadtxt()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html).

There are eight input variables and one output variable (the last column). You will be learning a model to map rows of input variables (X) to an output variable (y), which is often summarized as _y = f(X)_.

The variables can be summarized as follows:

Input Variables (X):

1. Number of times pregnant
2. Plasma glucose concentration at 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)

Output Variables (y):

1. Class variable (0 or 1)

Once the CSV file is loaded into memory, you can split the columns of data into input and output variables.

The data will be stored in a 2D array where the first dimension is rows and the second dimension is columns, e.g., \[rows, columns\].

You can split the array into two arrays by selecting subsets of columns using the standard NumPy [slice operator](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/) or “:”. You can select the first eight columns from index 0 to index 7 via the slice 0:8. We can then select the output column (the 9th variable) via index 8.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | ...<br>\# load the dataset<br>dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')<br>\# split into input (X) and output (y) variables<br>X=dataset\[:,0:8\]<br>y=dataset\[:,8\]<br>... |

You are now ready to define your neural network model.

**Note:** The dataset has nine columns, and the range 0:8 will select columns from 0 to 7, stopping before index 8. If this is new to you, then you can learn more about array slicing and ranges in this post:

- [How to Index, Slice, and Reshape NumPy Arrays for Machine Learning in Python](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)

## 2\. Define Keras Model

Models in Keras are defined as a sequence of layers.

We create a _[Sequential model](https://keras.io/models/sequential/)_ and add layers one at a time until we are happy with our network architecture.

The first thing to get right is to ensure the input layer has the correct number of input features. This can be specified when creating the first layer with the **input\_shape** argument and setting it to `(8,)` for presenting the eight input variables as a vector.

How do we know the number of layers and their types?

This is a tricky question. There are heuristics that you can use, and often the best network structure is found through a process of trial and error experimentation ( [I explain more about this here](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)). Generally, you need a network large enough to capture the structure of the problem.

In this example, let’s use a fully-connected network structure with three layers.

Fully connected layers are defined using the [Dense class](https://keras.io/layers/core/). You can specify the number of neurons or nodes in the layer as the first argument and the activation function using the **activation** argument.

Also, you will use the [rectified linear unit activation function](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) referred to as ReLU on the first two layers and the Sigmoid function in the output layer.

It used to be the case that Sigmoid and Tanh activation functions were preferred for all layers. These days, better performance is achieved using the ReLU activation function. Using a sigmoid on the output layer ensures your network output is between 0 and 1 and is easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

You can piece it all together by adding each layer:

- The model expects rows of data with 8 variables (the _input\_shape=(8,)_ argument).
- The first hidden layer has 12 nodes and uses the relu activation function.
- The second hidden layer has 8 nodes and uses the relu activation function.
- The output layer has one node and uses the sigmoid activation function.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | ...<br>\# define the keras model<br>model=Sequential()<br>model.add(Dense(12,input\_shape=(8,),activation='relu'))<br>model.add(Dense(8,activation='relu'))<br>model.add(Dense(1,activation='sigmoid'))<br>... |

**Note:** The most confusing thing here is that the shape of the input to the model is defined as an argument on the first hidden layer. This means that the line of code that adds the first Dense layer is doing two things, defining the input or visible layer and the first hidden layer.

## 3\. Compile Keras Model

Now that the model is defined, _you can compile it_.

Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as Theano or TensorFlow. The backend automatically chooses the best way to represent the network for training and making predictions to run on your hardware, such as CPU, GPU, or even distributed.

When compiling, you must specify some additional properties required when training the network. Remember training a network means finding the best set of weights to map inputs to outputs in your dataset.

You must specify the loss function to use to evaluate a set of weights, the optimizer used to search through different weights for the network, and any optional metrics you want to collect and report during training.

In this case, use cross entropy as the **loss** argument. This loss is for a binary classification problems and is defined in Keras as “ **binary\_crossentropy**“. You can learn more about choosing loss functions based on your problem here:

- [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

We will define the **optimizer** as the efficient stochastic gradient descent algorithm “ **adam**“. This is a popular version of gradient descent because it automatically tunes itself and gives good results in a wide range of problems. To learn more about the Adam version of stochastic gradient descent, see the post:

- [Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

Finally, because it is a classification problem, you will collect and report the classification accuracy defined via the **metrics** argument.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4 | ...<br>\# compile the keras model<br>model.compile(loss='binary\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>... |

## 4\. Fit Keras Model

You have defined your model and compiled it to get ready for efficient computation.

Now it is time to execute the model on some data.

You can train or fit your model on your loaded data by calling the **fit()** function on the model.

Training occurs over epochs, and each epoch is split into batches.

- **Epoch**: One pass through all of the rows in the training dataset
- **Batch**: One or more samples considered by the model within an epoch before weights are updated

One epoch comprises one or more batches, based on the chosen batch size, and the model is fit for many epochs. For more on the difference between epochs and batches, see the post:

- [What is the Difference Between a Batch and an Epoch in a Neural Network?](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)

The training process will run for a fixed number of epochs (iterations) through the dataset that you must specify using the **epochs** argument. You must also set the number of dataset rows that are considered before the model weights are updated within each epoch, called the batch size, and set using the **batch\_size** argument.

This problem will run for a small number of epochs (150) and use a relatively small batch size of 10.

These configurations can be chosen experimentally by trial and error. You want to train the model enough so that it learns a good (or good enough) mapping of rows of input data to the output classification. The model will always have some error, but the amount of error will level out after some point for a given model configuration. This is called model convergence.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4 | ...<br>\# fit the keras model on the dataset<br>model.fit(X,y,epochs=150,batch\_size=10)<br>... |

This is where the work happens on your CPU or GPU.

No GPU is required for this example, but if you’re interested in how to run large models on GPU hardware cheaply in the cloud, see this post:

- [How to Setup Amazon AWS EC2 GPUs to Train Keras Deep Learning Models](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

## 5\. Evaluate Keras Model

You have trained our neural network on the entire dataset, and you can evaluate the performance of the network on the same dataset.

This will only give you an idea of how well you have modeled the dataset (e.g., train accuracy), but no idea of how well the algorithm might perform on new data. This was done for simplicity, but ideally, you could separate your data into train and test datasets for training and evaluation of your model.

You can evaluate your model on your training dataset using the **evaluate()** function and pass it the same input and output used to train the model.

This will generate a prediction for each input and output pair and collect scores, including the average loss and any metrics you have configured, such as accuracy.

The **evaluate()** function will return a list with two values. The first will be the loss of the model on the dataset, and the second will be the accuracy of the model on the dataset. You are only interested in reporting the accuracy so ignore the loss value.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4 | ...<br>\# evaluate the keras model<br>\_,accuracy=model.evaluate(X,y)<br>print('Accuracy: %.2f'%(accuracy\*100)) |

## 6\. Tie It All Together

You have just seen how you can easily create your first neural network model in Keras.

Let’s tie it all together into a complete code example.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21 | \# first neural network with keras tutorial<br>from numpy import loadtxt<br>from tensorflow.keras.models import Sequential<br>from tensorflow.keras.layers import Dense<br>\# load the dataset<br>dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')<br>\# split into input (X) and output (y) variables<br>X=dataset\[:,0:8\]<br>y=dataset\[:,8\]<br>\# define the keras model<br>model=Sequential()<br>model.add(Dense(12,input\_shape=(8,),activation='relu'))<br>model.add(Dense(8,activation='relu'))<br>model.add(Dense(1,activation='sigmoid'))<br>\# compile the keras model<br>model.compile(loss='binary\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>\# fit the keras model on the dataset<br>model.fit(X,y,epochs=150,batch\_size=10)<br>\# evaluate the keras model<br>\_,accuracy=model.evaluate(X,y)<br>print('Accuracy: %.2f'%(accuracy\*100)) |

You can copy all the code into your Python file and save it as “ **keras\_first\_network.py**” in the same directory as your data file “ **pima-indians-diabetes.csv**“. You can then run the Python file as a script from your command line (command prompt) as follows:

|     |     |
| --- | --- |
| 1 | python keras\_first\_network.py |

Running this example, you should see a message for each of the 150 epochs, printing the loss and accuracy, followed by the final evaluation of the trained model on the training dataset.

It takes about 10 seconds to execute on my workstation running on the CPU.

Ideally, you would like the loss to go to zero and the accuracy to go to 1.0 (e.g., 100%). This is not possible for any but the most trivial machine learning problems. Instead, you will always have some error in your model. The goal is to choose a model configuration and training configuration that achieve the lowest loss and highest accuracy possible for a given dataset.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12 | ...<br>768/768 \[==============================\] - 0s 63us/step - loss: 0.4817 - acc: 0.7708<br>Epoch 147/150<br>768/768 \[==============================\] - 0s 63us/step - loss: 0.4764 - acc: 0.7747<br>Epoch 148/150<br>768/768 \[==============================\] - 0s 63us/step - loss: 0.4737 - acc: 0.7682<br>Epoch 149/150<br>768/768 \[==============================\] - 0s 64us/step - loss: 0.4730 - acc: 0.7747<br>Epoch 150/150<br>768/768 \[==============================\] - 0s 63us/step - loss: 0.4754 - acc: 0.7799<br>768/768 \[==============================\] - 0s 38us/step<br>Accuracy: 76.56 |

**Note:** If you try running this example in an IPython or Jupyter notebook, you may get an error.

The reason is the output progress bars during training. You can easily turn these off by setting **verbose=0** in the call to the **fit()** and **evaluate()** functions; for example:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6 | ...<br>\# fit the keras model on the dataset without progress bars<br>model.fit(X,y,epochs=150,batch\_size=10,verbose=0)<br>\# evaluate the keras model<br>\_,accuracy=model.evaluate(X,y,verbose=0)<br>... |

**Note**: Your [results may vary](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

**What score did you get?**

Post your results in the comments below.

Neural networks are stochastic algorithms, meaning that the same algorithm on the same data can train a different model with different skill each time the code is run. This is a feature, not a bug. You can learn more about this in the post:

- [Embrace Randomness in Machine Learning](https://machinelearningmastery.com/randomness-in-machine-learning/)

The variance in the performance of the model means that to get a reasonable approximation of how well your model is performing, you may need to fit it many times and calculate the average of the accuracy scores. For more on this approach to evaluating neural networks, see the post:

- [How to Evaluate the Skill of Deep Learning Models](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

For example, below are the accuracy scores from re-running the example five times:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5 | Accuracy: 75.00<br>Accuracy: 77.73<br>Accuracy: 77.60<br>Accuracy: 78.12<br>Accuracy: 76.17 |

You can see that all accuracy scores are around 77%, and the average is 76.924%.

## 7\. Make Predictions

The number one question I get asked is:

> “After I train my model, how can I use it to make predictions on new data?”

Great question.

You can adapt the above example and use it to generate predictions on the training dataset, pretending it is a new dataset you have not seen before.

Making predictions is as easy as calling the **predict()** function on the model. You are using a sigmoid activation function on the output layer, so the predictions will be a probability in the range between 0 and 1. You can easily convert them into a crisp binary prediction for this classification task by rounding them.

For example:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5 | ...<br>\# make probability predictions with the model<br>predictions=model.predict(X)<br>\# round predictions <br>rounded=\[round(x\[0\])forxinpredictions\] |

Alternately, you can convert the probability into 0 or 1 to predict crisp classes directly; for example:

|     |     |
| --- | --- |
| 1<br>2<br>3 | ...<br>\# make class predictions with the model<br>predictions=(model.predict(X)>0.5).astype(int) |

The complete example below makes predictions for each example in the dataset, then prints the input data, predicted class, and expected class for the first five examples in the dataset.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23 | \# first neural network with keras make predictions<br>from numpy import loadtxt<br>from tensorflow.keras.models import Sequential<br>from tensorflow.keras.layers import Dense<br>\# load the dataset<br>dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')<br>\# split into input (X) and output (y) variables<br>X=dataset\[:,0:8\]<br>y=dataset\[:,8\]<br>\# define the keras model<br>model=Sequential()<br>model.add(Dense(12,input\_shape=(8,),activation='relu'))<br>model.add(Dense(8,activation='relu'))<br>model.add(Dense(1,activation='sigmoid'))<br>\# compile the keras model<br>model.compile(loss='binary\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>\# fit the keras model on the dataset<br>model.fit(X,y,epochs=150,batch\_size=10,verbose=0)<br>\# make class predictions with the model<br>predictions=(model.predict(X)>0.5).astype(int)<br>\# summarize the first 5 cases<br>foriinrange(5):<br>print('%s => %d (expected %d)'%(X\[i\].tolist(),predictions\[i\],y\[i\])) |

Running the example does not show the progress bar as before, as the verbose argument has been set to 0.

After the model is fit, predictions are made for all examples in the dataset, and the input rows and predicted class value for the first five examples is printed and compared to the expected class value.

You can see that most rows are correctly predicted. In fact, you can expect about 76.9% of the rows to be correctly predicted based on your estimated performance of the model in the previous section.

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5 | \[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0\] => 0 (expected 1)<br>\[1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0\] => 0 (expected 0)<br>\[8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0\] => 1 (expected 1)<br>\[1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0\] => 0 (expected 0)<br>\[0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0\] => 1 (expected 1) |

If you would like to know more about how to make predictions with Keras models, see the post:

- [How to Make Predictions with Keras](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)

## Keras Tutorial Summary

In this post, you discovered how to create your first neural network model using the powerful Keras Python library for deep learning.

Specifically, you learned the six key steps in using Keras to create a neural network or deep learning model step-by-step, including:

1. How to load data
2. How to define a neural network in Keras
3. How to compile a Keras model using the efficient numerical backend
4. How to train a model on data
5. How to evaluate a model on data
6. How to make predictions with the model

Do you have any questions about Keras or about this tutorial?

Ask your question in the comments, and I will do my best to answer.

## Keras Tutorial Extensions

Well done, you have successfully developed your first neural network using the Keras deep learning library in Python.

This section provides some extensions to this tutorial that you might want to explore.

- **Tune the Model.** Change the configuration of the model or training process and see if you can improve the performance of the model, e.g., achieve better than 76% accuracy.
- **Save the Model**. Update the tutorial to save the model to a file, then load it later and use it to make predictions ( [see this tutorial](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)).
- **Summarize the Model**. Update the tutorial to summarize the model and create a plot of model layers ( [see this tutorial](https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/)).
- **Separate, Train, and Test Datasets**. Split the loaded dataset into a training and test set (split based on rows) and use one set to train the model and the other set to estimate the performance of the model on new data.
- **Plot Learning Curves**. The fit() function returns a history object that summarizes the loss and accuracy at the end of each epoch. Create line plots of this data, called [learning curves](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) ( [see this tutorial](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)).
- **Learn a New Dataset**. Update the tutorial to use a different tabular dataset, perhaps from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
- **Use Functional API**. Update the tutorial to use the Keras Functional API for defining the model ( [see this tutorial](https://machinelearningmastery.com/keras-functional-api-deep-learning/)).

## Further Reading

Are you looking for some more Deep Learning tutorials with Python and Keras?

Take a look at some of these:

### Related Tutorials

- [5 Step Life-Cycle for Neural Network Models in Keras](https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/)
- [Multi-Class Classification Tutorial with the Keras Deep Learning Library](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)
- [Regression Tutorial with the Keras Deep Learning Library in Python](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)
- [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

### Books

- [Deep Learning](https://amzn.to/2Fn9d3U) (Textbook), 2016.
- [Deep Learning with Python](https://machinelearningmastery.com/deep-learning-with-python/) (my book).

### APIs

- [Keras Deep Learning Library Homepage](https://keras.io/)
- [Keras API Documentation](https://keras.io/api/)

**How did you go? Do you have any questions about deep learning?**

Post your questions in the comments below, and I will do my best to help.

Share _Post_Share

### More On This Topic

- [![daniel-bernard-GIMgnwHSxdI-unsplash](https://machinelearningmastery.com/wp-content/uploads/2019/02/daniel-bernard-GIMgnwHSxdI-unsplash-150x150.jpg)Your First Machine Learning Project in Python Step-By-Step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
- [![mlm-first-local-llm-project-step-by-step](https://machinelearningmastery.com/wp-content/uploads/2025/06/mlm-first-local-llm-project-step-by-step-200x200.png)Your First Local LLM API Project in Python Step-By-Step](https://machinelearningmastery.com/your-first-local-llm-api-project-in-python-step-by-step/)
- [![mlm-first-openai-llm-project-step-by-step](https://machinelearningmastery.com/wp-content/uploads/2025/06/mlm-first-openai-llm-project-step-by-step-200x200.png)Your First OpenAI API Project in Python Step-By-Step](https://machinelearningmastery.com/your-first-openai-api-project-in-python-step-by-step/)
- [![Yuor First Machine Learning Project in R Step-by-Step](https://machinelearningmastery.com/wp-content/uploads/2016/01/Yuor-First-Machine-Learning-Project-in-R-Step-by-Step.jpg)Your First Machine Learning Project in R Step-By-Step](https://machinelearningmastery.com/machine-learning-in-r-step-by-step/)
- [![Amazon Web Services](https://machinelearningmastery.com/wp-content/uploads/2016/05/Amazon-Web-Services.jpg)How to Train Keras Deep Learning Models on AWS EC2…](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)
- [![drown_-in_city-V2DylCx9kkc-unsplash](https://machinelearningmastery.com/wp-content/uploads/2023/01/drown_-in_city-V2DylCx9kkc-unsplash-150x150.jpg)Develop Your First Neural Network with PyTorch, Step by Step](https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/)

[Using Normalization Layers to Improve Deep Learning Models](https://machinelearningmastery.com/using-normalization-layers-to-improve-deep-learning-models/)

[How to Save and Load Your Keras Deep Learning Model](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)

### 1,172 Responses to _Your First Deep Learning Project in Python with Keras Step-by-Step_

001. ![](https://secure.gravatar.com/avatar/95138f901bda618aa4bc4258907772af88b5195ee582b301ee63071ed93db133?s=40&d=mm&r=g)



     SauravMay 27, 2016 at 11:08 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-352098 "Direct link to this comment")





     The input layer doesn’t have any activation function, but still activation=”relu” is mentioned in the first layer of the model. Why?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-352098)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 28, 2016 at 6:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-352136 "Direct link to this comment")





       Hi Saurav,



       The first layer in the network here is technically a hidden layer, hence it has an activation function.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-352136)




       - ![](https://secure.gravatar.com/avatar/3ef95fa03527343459528154bce8ff847bf3be639047712dfcc9274a03f1a02e?s=40&d=mm&r=g)



         sam JohnsonDecember 21, 2016 at 2:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376532 "Direct link to this comment")





         Why have you made it a hidden layer though? the input layer is not usually represented as a hidden layer?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376532)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)December 21, 2016 at 8:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376587 "Direct link to this comment")





           Hi sam,



           Note this line:







































































           |     |     |
           | --- | --- |
           | 1 | model.add(Dense(12,input\_dim=8,init='uniform',activation='relu')) |











           It does a few things.



           - It defines the input layer as having 8 inputs.
           - It defines a hidden layer with 12 neurons, connected to the input layer that use relu activation function.
           - It initializes all weights using a sample of uniform random numbers.

Does that help?

[Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376587)

           - ![](https://secure.gravatar.com/avatar/c1f9fa5ea706bb270ff255d5b1d66c19ab6bc1626be53d7513d7b5ab992ca2b5?s=40&d=mm&r=g)



             PavideviMay 17, 2017 at 2:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399830 "Direct link to this comment")





             Hi Jason,



             U have used two different activation functions so how can we know which activation function fit the model?

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)May 17, 2017 at 8:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399860 "Direct link to this comment")





             Sorry, I don’t understand the question.

           - ![](https://secure.gravatar.com/avatar/c7f411c056085a3ca03031baad0b7e0ad50471fc346228c38174c65d03bbbabc?s=40&d=mm&r=g)



             Marco CheungAugust 23, 2017 at 12:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-410619 "Direct link to this comment")





             Hi Jason,



             I am interested in deep learning and machine learning. You mentioned “It defines a hidden layer with 12 neurons, connected to the input layer that use relu activation function.” I wonder how can we determine the number of neurons in order to achieve a high accuracy rate of the model?



             Thanks a lot!!!

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)August 23, 2017 at 6:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-410662 "Direct link to this comment")





             Use trial and error. We cannot specify the “best” number of neurons analytically. We must test.

           - ![](https://secure.gravatar.com/avatar/fc2faebc8febf7e6a1bae100a88be46a2e26395be169c059daeffaab0e7a9199?s=40&d=mm&r=g)



             Ramzan ShahidNovember 10, 2017 at 4:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419557 "Direct link to this comment")





             Sir, thanks for your tutorial. Would you like to make tutorial on stock Data Prediction through Neural Network Model and training this on any stock data. If you have on this so please share the link. Thanks

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)November 10, 2017 at 10:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419589 "Direct link to this comment")





             I am reticent to post tutorials on stock market prediction given the random walk hypothesis of security prices:

             [https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/](https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/)

           - ![](https://secure.gravatar.com/avatar/b53bf8ea65526e461f5dc2e7971aeb8d59e123b65954001660bc3a97ed21745d?s=40&d=mm&r=g)



             Dhara BhavsarAugust 28, 2019 at 9:54 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-498849 "Direct link to this comment")





             Hi,



             I would like to know more about activation function. How it is working? How many activation functions? Using different activation function How much affect the output of the model?



             I would like to also know about the Hidden Layer. How the size of the hidden layer affect the model?

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)August 29, 2019 at 6:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-498902 "Direct link to this comment")





             In this tutorial, we use relu in the hidden layers, learn more here:

             [https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)



             The size of the layer impacts the capacity of the model, learn more here:

             [https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/](https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/)
         - ![](https://secure.gravatar.com/avatar/538624411d4a2f68c9706554ce32c7be73aea254b0291bd843262e91bae276a8?s=40&d=mm&r=g)



           Ryder CarterAugust 16, 2024 at 9:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-721716 "Direct link to this comment")





           \> ```model.add(Dense(12, input_shape = (8,), activation = 'relu'))```


           Why does the input layer have 12 neurons when only 8 input variables exist? Isn’t the input layer supposed to have the same number of neurons as the number of variables so that every input goes into exactly one neuron? Am I misunderstanding anything?



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-721716)
       - ![](https://secure.gravatar.com/avatar/a8d82b3e2a59c30f46135ffc4b480496dc1b9ae32094ba40a804edb4a9410249?s=40&d=mm&r=g)



         dhaniJune 28, 2018 at 2:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442183 "Direct link to this comment")





         hi how use cnn for pixel classification on mhd images



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442183)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)June 28, 2018 at 6:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442208 "Direct link to this comment")





           What is pixel classification? What are mhd images?



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442208)




           - ![](https://secure.gravatar.com/avatar/99b384d7d4cb74e1bde40c297f035d339a576edfd19ca10535e1507174cd9ccc?s=40&d=mm&r=g)



             Seth HammockMarch 6, 2024 at 7:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-710869 "Direct link to this comment")





             Are you talking about neural style transfer?

           - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



             James CarmichaelMarch 6, 2024 at 10:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-710881 "Direct link to this comment")





             Hi Seth…That is an important topic. More can be found here:



             [https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa](https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa)
       - ![](https://secure.gravatar.com/avatar/8b131416d6422ba7d4bbf50c5ae2b7de8151b3116d182c78f73c8fe8c1f61a47?s=40&d=mm&r=g)



         Tanmay KulkarniFebruary 11, 2020 at 5:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-521222 "Direct link to this comment")





         Hello! I want to know if there’s a way to know the values of all weights after each updation?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-521222)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)February 11, 2020 at 5:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-521223 "Direct link to this comment")





           Yes, you can save them to file or review them manually.



           Often saving is achieved using a checkpoint:

           [https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-521223)
     - ![](https://secure.gravatar.com/avatar/ac8f08480dc64436ad1d703c3e32ddb3bd5327ab07ab19a7f9c67dbe15af9834?s=40&d=mm&r=g)



       BlackBookKeeperAugust 18, 2018 at 10:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446421 "Direct link to this comment")





       runfile(‘C:/Users/Owner/Documents/untitled1.py’, wdir=’C:/Users/Owner/Documents’)


       Traceback (most recent call last):



       File “”, line 1, in


       runfile(‘C:/Users/Owner/Documents/untitled1.py’, wdir=’C:/Users/Owner/Documents’)



       File “C:\\Users\\Owner\\Anaconda3\\lib\\site-packages\\spyder\\utils\\site\\sitecustomize.py”, line 705, in runfile


       execfile(filename, namespace)



       File “C:\\Users\\Owner\\Anaconda3\\lib\\site-packages\\spyder\\utils\\site\\sitecustomize.py”, line 102, in execfile


       exec(compile(f.read(), filename, ‘exec’), namespace)



       File “C:/Users/Owner/Documents/untitled1.py”, line 13, in


       model.add(Dense(12, input\_dim=8, activation=’relu’))



       File “C:\\Users\\Owner\\Anaconda3\\lib\\site-packages\\keras\\engine\\sequential.py”, line 160, in add


       name=layer.name + ‘\_input’)



       File “C:\\Users\\Owner\\Anaconda3\\lib\\site-packages\\keras\\engine\\input\_layer.py”, line 177, in Input


       input\_tensor=tensor)



       File “C:\\Users\\Owner\\Anaconda3\\lib\\site-packages\\keras\\legacy\\interfaces.py”, line 91, in wrapper


       return func(\*args, \*\*kwargs)



       File “C:\\Users\\Owner\\Anaconda3\\lib\\site-packages\\keras\\engine\\input\_layer.py”, line 86, in \_\_init\_\_


       name=self.name)



       File “C:\\Users\\Owner\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow\_backend.py”, line 515, in placeholder


       x = tf.placeholder(dtype, shape=shape, name=name)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\ops\\array\_ops.py”, line 1530, in placeholder


       return gen\_array\_ops.\_placeholder(dtype=dtype, shape=shape, name=name)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\ops\\gen\_array\_ops.py”, line 1954, in \_placeholder


       name=name)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\op\_def\_library.py”, line 767, in apply\_op


       op\_def=op\_def)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\ops.py”, line 2508, in create\_op


       set\_shapes\_for\_outputs(ret)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\ops.py”, line 1894, in set\_shapes\_for\_outputs


       output.set\_shape(s)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\ops.py”, line 443, in set\_shape


       self.\_shape = self.\_shape.merge\_with(shape)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\tensor\_shape.py”, line 550, in merge\_with


       stop = key.stop



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\tensor\_shape.py”, line 798, in as\_shape


       “””Returns this shape as a `TensorShapeProto`.”””



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\tensor\_shape.py”, line 431, in \_\_init\_\_


       size for one or more dimension. e.g. `TensorShape([None, 256])`



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\tensor\_shape.py”, line 376, in as\_dimension


       other = as\_dimension(other)



       File “C:\\Users\\Owner\\AppData\\Roaming\\Python\\Python36\\site-packages\\tensorflow\\python\\framework\\tensor\_shape.py”, line 32, in \_\_init\_\_


       if value is None:



       TypeError: int() argument must be a string, a bytes-like object or a number, not ‘TensorShapeProto’



       this error occurs when {model.add(Dense(12, input\_dim=8, activation=’relu’))} this command is run



       any help?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446421)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)August 19, 2018 at 6:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446453 "Direct link to this comment")





         Save all code into a file and run it as follows:

         [https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line](https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line)



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446453)
     - ![](https://secure.gravatar.com/avatar/2dab983acd6ac3254c649e9e70ff44fcd2271dc53a47aa8caef965c70b175d67?s=40&d=mm&r=g)



       PenchalaiahDecember 8, 2019 at 6:24 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514217 "Direct link to this comment")





       Fantastic tutorial. The explanation is simple and precise. Thanks a lot



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514217)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)December 9, 2019 at 6:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514276 "Direct link to this comment")





         Thanks!



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514276)
     - ![](https://secure.gravatar.com/avatar/3f4e5e6c25199c08e3a1eda08241329a0056cc50f74d31e90bf2190277b06423?s=40&d=mm&r=g)



       LocJune 29, 2022 at 1:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-674130 "Direct link to this comment")





       great arttist



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-674130)
002. ![](https://secure.gravatar.com/avatar/b9b6457dd1333707e062e089c2a76358e9adc343c06857d52f784a279596c3cf?s=40&d=mm&r=g)



     GeoffMay 29, 2016 at 6:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-352220 "Direct link to this comment")





     Can you explain how to implement weight regularization into the layers?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-352220)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 15, 2016 at 5:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-353891 "Direct link to this comment")





       Yep, see here:

       [http://keras.io/regularizers/](http://keras.io/regularizers/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-353891)




       - ![](https://secure.gravatar.com/avatar/5aa7c8968dc224ac022d272b2240bcb78a8740c2b686b4e5ddbe321f5ff5ce89?s=40&d=mm&r=g)



         [afthab](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)October 5, 2018 at 8:32 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450778 "Direct link to this comment")





         hey yo!!! how u r start coding in python



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450778)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)October 6, 2018 at 5:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450811 "Direct link to this comment")





           Start here:

           [https://machinelearningmastery.com/faq/single-faq/how-do-i-get-started-with-python-programming](https://machinelearningmastery.com/faq/single-faq/how-do-i-get-started-with-python-programming)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450811)
003. ![](https://secure.gravatar.com/avatar/815bff860827df3e75ea01df3864ef54771579137c8875c37a90ae706c641977?s=40&d=mm&r=g)



     KWCJune 14, 2016 at 12:08 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-353775 "Direct link to this comment")





     Import statements if others need them:



     from keras.models import Sequential


     from keras.layers import Dense, Activation



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-353775)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 15, 2016 at 5:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-353890 "Direct link to this comment")





       Thanks.



       I had them in Part 6, but I have also added them to Part 1.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-353890)




       - ![](https://secure.gravatar.com/avatar/d7846422c7aa0fff52854859911460bfed08b2d96a4a5acedfed0ee02ecf9e4e?s=40&d=mm&r=g)



         ShiranJanuary 20, 2020 at 11:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518535 "Direct link to this comment")





         Great post!


         Is it possible to train a neural network that receives as input a vector x and tries to predict another vector y where both x and y are floats?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518535)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)January 20, 2020 at 2:07 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518542 "Direct link to this comment")





           Yes, this is called regression:

           [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518542)
004. ![](https://secure.gravatar.com/avatar/7a27627e116f4cba0488887ce6fa2a5466acbef50eb67c1d390e26b7b97a7061?s=40&d=mm&r=g)



     Aakash NainJune 29, 2016 at 6:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355321 "Direct link to this comment")





     If there are 8 inputs for the first layer then why we have taken them as ’12’ in the following line :



     model.add(Dense(12, input\_dim=8, init=’uniform’, activation=’relu’))



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355321)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 30, 2016 at 6:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355358 "Direct link to this comment")





       Hi Aakash.



       The input layer is defined by the input\_dim parameter, here set to 8.



       The first hidden layer has 12 neurons.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355358)
005. ![](https://secure.gravatar.com/avatar/86720fdbb48834cd2f5f2da59e1cff0744c301ae18457f617c6de6d31710c88c?s=40&d=mm&r=g)



     JoshuaJuly 2, 2016 at 12:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355496 "Direct link to this comment")





     I ran your program and i have an error:


     ValueError: could not convert string to float:


     what could be the reason for this, and how may I solve it.


     thanks.


     great post by the way.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355496)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 2, 2016 at 6:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355510 "Direct link to this comment")





       It might be a copy-paste error. Perhaps try to copy and run the whole example listed in section 6?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355510)




       - ![](https://secure.gravatar.com/avatar/018ebf313b3c57a8ef53de10a35833b994e5b1aa4b4e2843418884bfbb1375b4?s=40&d=mm&r=g)



         AkashSeptember 28, 2018 at 11:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450037 "Direct link to this comment")





         Hello sir, I am facing the same problem valueError: could not convert string to float: ‘”6’


         also I am running the example from section 6.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450037)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)September 28, 2018 at 3:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450062 "Direct link to this comment")





           I have some suggestions here:

           [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450062)
       - ![](https://secure.gravatar.com/avatar/264da69a4c3dfb77d8d20235cdbdd0fd58b0dc8420b8aee451f51d699b854e19?s=40&d=mm&r=g)



         [yashu](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)October 5, 2018 at 8:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450777 "Direct link to this comment")





         jason can u plzz help me how to code



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450777)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)October 6, 2018 at 5:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450810 "Direct link to this comment")





           Sorry, I cannot help you to write code.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450810)
     - ![](https://secure.gravatar.com/avatar/22fda331fff54e21dab4cf61cc48e0764832d63e68d011e07c589af279d7056c?s=40&d=mm&r=g)



       KeyChyJuly 3, 2019 at 5:45 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491410 "Direct link to this comment")





       Maybe when you set all parameters in an extra column in your \*.csv file. Than you schould replace the delimiter from , to ; like:


       dataset = numpy.loadtxt(“pima-indians-diabetes.csv”, delimiter=”;”)


       This solved the Problem for me.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491410)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)July 4, 2019 at 7:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491481 "Direct link to this comment")





         Thanks for sharing.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491481)
006. ![](https://secure.gravatar.com/avatar/5d8480676426556cded762f8c2522fd2f2fcec2cc9aafb3bd22796d481bda298?s=40&d=mm&r=g)



     cheikh brahimJuly 5, 2016 at 7:40 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355726 "Direct link to this comment")





     thank you for your simple and useful example.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355726)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2016 at 6:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355767 "Direct link to this comment")





       You’re welcome cheikh.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355767)
007. ![](https://secure.gravatar.com/avatar/2ea3cdb5cbd2fe53c8025dfdb88bc4ec3020f047c210fb391266f06f9d82dba2?s=40&d=mm&r=g)



     Nikhil ThakurJuly 6, 2016 at 6:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355826 "Direct link to this comment")





     Hello Sir, I am trying to use Keras for NLP , specifically sentence classification. I have given the model building part below. It’s taking quite a lot time to execute. I am using Pycharm IDE.



     batch\_size = 32


     nb\_filter = 250


     filter\_length = 3


     nb\_epoch = 2


     pool\_length = 2


     output\_dim = 5


     hidden\_dims = 250



     \# Build the model



     model1 = Sequential()



     model1.add(Convolution1D(nb\_filter, filter\_length ,activation=’relu’,border\_mode=’valid’,


     input\_shape=(len(embb\_weights),dim), weights=\[embb\_weights\]))



     model1.add(Dense(hidden\_dims))


     model1.add(Dropout(0.2))


     model1.add(Activation(‘relu’))



     model1.add(MaxPooling1D(pool\_length=pool\_length))



     model1.add(Dense(output\_dim, activation=’sigmoid’))



     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)



     model1.compile(loss=’mean\_squared\_error’,


     optimizer=sgd,


     metrics=\[‘accuracy’\])



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355826)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 7, 2016 at 7:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355899 "Direct link to this comment")





       You may want a larger network. You may also want to use a standard repeating structure like CNN->CNN->Pool->Dense.



       See this post on using a CNN:

       [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)



       Later, you may also want to try some stacked LSTMs.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-355899)
008. ![](https://secure.gravatar.com/avatar/54e56a276f722c1074fc413f1ecf37f8d56fc0784f933758157522d6aa973d33?s=40&d=mm&r=g)



     Andre NormanJuly 15, 2016 at 10:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-356877 "Direct link to this comment")





     Hi Jason, thanks for the awesome example. Given that the accuracy of this model is 79.56%. From here on, what steps would you take to improve the accuracy?



     Given my nascent understanding of Machine Learning, my initial approach would have been:



     Implement forward propagation, then compute the cost function, then implement back propagation, use gradient checking to evaluate my network (disable after use), then use gradient descent.



     However, this approach seems arduous compared to using Keras. Thanks for your response.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-356877)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 15, 2016 at 10:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-356879 "Direct link to this comment")





       Hi Andre, indeed Keras makes working with neural nets so much easier. Fun even!



       We may be maxing out on this problem, but here is some general advice for lifting performance.


       – data prep – try lots of different views of the problem and see which is best at exposing the structure of the problem to the learning algorithm (data transforms, feature engineering, etc.)


       – algorithm selection – try lots of algorithms and see which one or few are best on the problem (try on all views)


       – algorithm tuning – tune well performing algorithms to get the most out of them (grid search or random search hyperparameter tuning)


       – ensembles – combine predictions from multiple algorithms (stacking, boosting, bagging, etc.)



       For neural nets, there are a lot of things to tune, I think there are big gains in trying different network topologies (layers and number of neurons per layer) in concert with training epochs and learning rate (bigger nets need more training).



       I hope that helps as a start.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-356879)




       - ![](https://secure.gravatar.com/avatar/54e56a276f722c1074fc413f1ecf37f8d56fc0784f933758157522d6aa973d33?s=40&d=mm&r=g)



         Andre NormanJuly 18, 2016 at 7:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357254 "Direct link to this comment")





         Awesome! Thanks Jason =)



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357254)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)July 18, 2016 at 8:03 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357266 "Direct link to this comment")





           You’re welcome Andre.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357266)
     - ![](https://secure.gravatar.com/avatar/5ca4abd8736d5669183dda487c80ba9c322825e6e1cbc9701a202d4d3219604f?s=40&d=mm&r=g)



       quentinAugust 7, 2017 at 8:41 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408851 "Direct link to this comment")





       Some interesting stuff here

       [https://youtu.be/vq2nnJ4g6N0](https://youtu.be/vq2nnJ4g6N0)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408851)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)August 8, 2017 at 7:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408904 "Direct link to this comment")





         Thanks for sharing. What did you like about it?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408904)
009. ![](https://secure.gravatar.com/avatar/c95f7d60bcfe7c5de4d8be996dfed23bcbb9db08ee003363d5eb1d63d6d5b095?s=40&d=mm&r=g)



     [Romilly Cocking](http://blog.rareschool.com/)July 21, 2016 at 12:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357560 "Direct link to this comment")





     Hi Jason, it’s a great example but if anyone runs it in an IPython/Jupyter notebook they are likely to encounter an I/O error when running the fit step. This is due to a known bug in IPython.



     The solution is to set verbose=0 like this



     \# Fit the model


     model.fit(X, Y, nb\_epoch=40, batch\_size=10, verbose=0)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357560)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 21, 2016 at 5:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357574 "Direct link to this comment")





       Great, thanks for sharing Romilly.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357574)
010. ![](https://secure.gravatar.com/avatar/3d9f754d62c191ef4f8d055c3f732f3c3c4431999a30db0918730d8b9591f3f0?s=40&d=mm&r=g)



     AnirbanJuly 23, 2016 at 10:20 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357819 "Direct link to this comment")





     Great example. Have a query though. How do I now give a input and get the output (0 or 1). Can you pls give the cmd for that.


     Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357819)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 24, 2016 at 6:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357854 "Direct link to this comment")





       You can call model.predict() to get predictions and round on each value to snap to a binary value.



       For example, below is a complete example showing you how to round the predictions and print them to console.











































































       |     |     |
       | --- | --- |
       | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26 | \# Create first network with Keras<br>from keras.models import Sequential<br>from keras.layers import Dense<br>import numpy<br>\# fix random seed for reproducibility<br>seed=7<br>numpy.random.seed(seed)<br>\# load pima indians dataset<br>dataset=numpy.loadtxt("pima-indians-diabetes.csv",delimiter=",")<br>\# split into input (X) and output (Y) variables<br>X=dataset\[:,0:8\]<br>Y=dataset\[:,8\]<br>\# create model<br>model=Sequential()<br>model.add(Dense(12,input\_dim=8,init='uniform',activation='relu'))<br>model.add(Dense(8,init='uniform',activation='relu'))<br>model.add(Dense(1,init='uniform',activation='sigmoid'))<br>\# Compile model<br>model.compile(loss='binary\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>\# Fit the model<br>model.fit(X,Y,nb\_epoch=150,batch\_size=10,verbose=2)<br>\# calculate predictions<br>predictions=model.predict(X)<br>\# round predictions<br>rounded=\[round(x)forxinpredictions\]<br>print(rounded) |











       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357854)




       - ![](https://secure.gravatar.com/avatar/8beed5973ff412ec755aa0077a2f2d1bbce740b88a855e155824fb902e42da36?s=40&d=mm&r=g)



         DebanjanMarch 27, 2017 at 12:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394314 "Direct link to this comment")





         Hi, Why you are not using any test set? You are predicting from the training set , I think.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394314)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)March 28, 2017 at 8:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394401 "Direct link to this comment")





           Correct, it is just an example to get you started with Keras.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394401)
       - ![](https://secure.gravatar.com/avatar/312bb8d9716be32a6a00d882825c4d97f87f550882e957f0026b733275c4d274?s=40&d=mm&r=g)



         DavidJune 26, 2017 at 12:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403782 "Direct link to this comment")





         Jason, I’m not quite understanding how the predicted values (\[1.0, 0.0, 1.0, 0.0, 1.0,…) map to the real world problem. For instance, what does that first “1.0” in the results indicate?\
\
\
\
         I get that it’s a prediction of ‘true’ for diabetes…but to which patient is it predicting that—the first in the list? So then the second result, “0.0,” is the prediction for the second patient/row in the dataset?\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403782)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 26, 2017 at 6:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403811 "Direct link to this comment")\
\
\
\
\
\
           Remember the original file has 0 and 1 values in the final class column where 0 is no onset of diabetes and 1 is an onset of diabetes.\
\
\
\
           We are predicting new values in this column.\
\
\
\
           We are making predictions for special rows, we pass in their medical info and predict the onset of diabetes. We just happen to do this for a number of rows at a time.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403811)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/ce9b8f1c9cd693f045fe5a54c0b310ebed6330199464c10874106941c5ec83c9?s=40&d=mm&r=g)\
\
\
\
             amiJuly 16, 2018 at 4:30 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443605 "Direct link to this comment")\
\
\
\
\
\
             hello jason\
\
\
\
             i am getting this error while calculating the predictions.\
\
\
\
             #calculate predictions\
\
\
\
             predictions = model.predict(X)\
\
\
\
             #round predictions\
\
\
\
             rounded = \[round(x) for x in predictions\]\
\
\
\
             print(rounded)\
\
\
\
             —————————————————————————\
\
\
             TypeError Traceback (most recent call last)\
\
\
             in ()\
\
\
             2 predictions = model.predict(X)\
\
\
             3 #round predictions\
\
\
             —-\> 4 rounded = \[round(x) for x in predictions\]\
\
\
             5 print(rounded)\
\
\
\
             in (.0)\
\
\
             2 predictions = model.predict(X)\
\
\
             3 #round predictions\
\
\
             —-\> 4 rounded = \[round(x) for x in predictions\]\
\
\
             5 print(rounded)\
\
\
\
             TypeError: type numpy.ndarray doesn’t define \_\_round\_\_ method\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)July 17, 2018 at 6:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443650 "Direct link to this comment")\
\
\
\
\
\
             Try removing the call to round().\
       - ![](https://secure.gravatar.com/avatar/9984a31649528835eeda3e5ad1cda8e19a89bc86a94c74cdaf5f798351c75898?s=40&d=mm&r=g)\
\
\
\
         RachelJune 28, 2017 at 8:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404110 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
         Can I ask why you use the same data X you fit the model to do the prediction?\
\
\
\
         \# Fit the model\
\
\
         model.fit(X, Y, epochs = 150, batch\_size = 10, verbose = 2)\
\
\
\
         \# calculate predictions\
\
\
         predictions = model.predict(X)\
\
\
\
         Rachel\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404110)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 29, 2017 at 6:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404175 "Direct link to this comment")\
\
\
\
\
\
           It is all I have at hand. X means data matrix.\
\
\
\
           Replace X in predict() with Xprime or whatever you like.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404175)\
       - ![](https://secure.gravatar.com/avatar/db658d62d6d46cb9d7662443586ef2fd1bf0b47b1e9fe417b60b0a04576d2a09?s=40&d=mm&r=g)\
\
\
\
         [jitendra](http://na/)March 27, 2018 at 7:20 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433385 "Direct link to this comment")\
\
\
\
\
\
         hii, how will i feed the input (8,125,96,0,0,0.0,0.232,54) to get our output.\
\
\
\
         predictions = model.predict(X)\
\
\
         i mean insead of X i want to get output of 8,125,96,0,0,0.0,0.232,54.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433385)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 28, 2018 at 6:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433426 "Direct link to this comment")\
\
\
\
\
\
           Wrap your input in an array, n-columns with one row, then pass that to the model.\
\
\
\
           Does that help?\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433426)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/524129c80bc96c1b42537c97c5de0945bc4e902672589bfb79b4e0e1a9c99f00?s=40&d=mm&r=g)\
\
\
\
             RomanOctober 5, 2018 at 11:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450788 "Direct link to this comment")\
\
\
\
\
\
             Hello, trying to use predictions on similar neural network but keep getting errors that input dimension has other shape.\
\
\
\
             Can you say how array must look on exampled neural network?\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)October 6, 2018 at 5:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450815 "Direct link to this comment")\
\
\
\
\
\
             For an MLP, data must be organized into a 2d array of samples x features\
011. ![](https://secure.gravatar.com/avatar/3d9f754d62c191ef4f8d055c3f732f3c3c4431999a30db0918730d8b9591f3f0?s=40&d=mm&r=g)\
\
\
\
     AnirbanJuly 23, 2016 at 10:52 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357821 "Direct link to this comment")\
\
\
\
\
\
     I am not able to get to the last epoch. Getting error before that:\
\
\
     Epoch 11/150\
\
\
     390/768 \[==============>……………\]Traceback (most recent call last):.6921\
\
\
\
     ValueError: I/O operation on closed file\
\
\
\
     I could resolve this by varying the epoch and batch size.\
\
\
\
     Now to predict a unknown value, i loaded a new dataset and used predict cmd as below :\
\
\
     dataset\_test = numpy.loadtxt(“pima-indians-diabetes\_test.csv”,delimiter=”,”) –has only one row\
\
\
\
     X = dataset\_test\[:,0:8\]\
\
\
     model.predict(X)\
\
\
\
     But I am getting error :\
\
\
     X = dataset\_test\[:,0:8\]\
\
\
\
     IndexError: too many indices for array\
\
\
\
     Can you help pls.\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357821)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 24, 2016 at 6:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357855 "Direct link to this comment")\
\
\
\
\
\
       I see problems like this when you run from a notebook or from an IDE.\
\
\
\
       Consider running examples from the console to ensure they work.\
\
\
\
       Consider tuning off verbose output (verbose=0 in the call to fit()) to disable the progress bar.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-357855)\
012. ![](https://secure.gravatar.com/avatar/33fd683ccf0a0969d06be831ec67c9e0d5daf0016bf4dc5b391e49d6dd918035?s=40&d=mm&r=g)\
\
\
\
     David KluszczynskiJuly 28, 2016 at 12:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358192 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason!\
\
\
     Loved the tutorial! I have a question however.\
\
\
     Is there a way to save the weights to a file after the model is trained for uses, such as kaggle?\
\
\
     Thanks,\
\
\
     David\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358192)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 28, 2016 at 5:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358207 "Direct link to this comment")\
\
\
\
\
\
       Thanks David.\
\
\
\
       You can save the network weights to file by calling model.save\_weights(“model.h5”)\
\
\
\
       You can learn more in this post:\
\
       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358207)\
013. ![](https://secure.gravatar.com/avatar/5f77b2d57d0f659b52749236508fdb22bf1adfae8b2a9caaa012bc8c6c037ea5?s=40&d=mm&r=g)\
\
\
\
     Alex HopperJuly 29, 2016 at 5:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358265 "Direct link to this comment")\
\
\
\
\
\
     Hey, Jason! Thank you for the awesome tutorial! I’ve use your tutorial to learn about CNN. I have one question for you… Supposing I want to use Keras to classicate images and I have 3 or more classes to classify, How could my algorithm know about this classes? You know, I have to code what is a cat, a dog and a horse. Is there any way to code this? I’ve tried it:\
\
\
\
     target\_names = \[‘class 0(Cats)’, ‘class 1(Dogs)’, ‘class 2(Horse)’\]\
\
\
     print(classification\_report(np.argmax(Y\_test,axis=1), y\_pred,target\_names=target\_names))\
\
\
\
     But my results are not classifying correctly.\
\
\
\
     precision recall f1-score support\
\
\
     class 0(Cat) 0.00 0.00 0.00 17\
\
\
     class 1(Dog) 0.00 0.00 0.00 14\
\
\
     class 2(Horse) 0.99 1.00 0.99 2526\
\
\
\
     avg / total 0.98 0.99 0.98 2557\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358265)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 29, 2016 at 6:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358272 "Direct link to this comment")\
\
\
\
\
\
       Great question Alex.\
\
\
\
       This is an example of a multi-class classification problem. You must use a one hot encoding on the output variable to be able to model it with a neural network and specify the number of classes as the number of outputs on the final layer of your network.\
\
\
\
       I provide a tutorial with the famous iris dataset that has 3 output classes here:\
\
       [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358272)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f77b2d57d0f659b52749236508fdb22bf1adfae8b2a9caaa012bc8c6c037ea5?s=40&d=mm&r=g)\
\
\
\
         Alex HopperAugust 1, 2016 at 1:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358454 "Direct link to this comment")\
\
\
\
\
\
         Thank you.\
\
\
         I’ll check it.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358454)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 1, 2016 at 6:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358468 "Direct link to this comment")\
\
\
\
\
\
           No problem Alex.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358468)\
014. ![](https://secure.gravatar.com/avatar/18b5dcd7f8c6e15a6c865b0232e2f46e81f304a3ce468eb30df857fa259b2802?s=40&d=mm&r=g)\
\
\
\
     AnonymouseAugust 2, 2016 at 11:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358637 "Direct link to this comment")\
\
\
\
\
\
     This was really useful, thank you\
\
\
\
     I’m using keras (with CNNs) for sentiment classification of documents and I’d like to improve the performance, but I’m completely at a loss when it comes to tuning the parameters in a non-arbitrary way. Could you maybe point me somewhere that will help me go about this in a more systematic fashion? There must be some heuristics or rules-of-thumb that could guide me.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358637)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 3, 2016 at 8:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358671 "Direct link to this comment")\
\
\
\
\
\
       I have a tutorial coming out soon (next week) that provide lots of examples of tuning the hyperparameters of a neural network in Keras, but limited to MLPs.\
\
\
\
       For CNNs, I would advise tuning the number of repeating layers (conv + max pool), the number of filters in repeating block, and the number and size of dense layers at the predicting part of your network. Also consider using some fixed layers from pre-trained models as the start of your network (e.g. VGG) and try just training some input and output layers around it for your problem.\
\
\
\
       I hope that helps as a start.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-358671)\
015. ![](https://secure.gravatar.com/avatar/29b9039e225c979deaab18e76a3c4b3cfca67594e37a2edf24f60549f136e73f?s=40&d=mm&r=g)\
\
\
\
     ShoponAugust 14, 2016 at 5:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-359933 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason , My Accuracy is : 0.0104 , but yours is 0.7879 and my loss is : -9.5414 . Is there any problem with the dataset ? I downloaded the dataset from a different site .\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-359933)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2016 at 12:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360050 "Direct link to this comment")\
\
\
\
\
\
       I think there might be something wrong with your implementation or your dataset. Your numbers are way out.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360050)\
016. ![](https://secure.gravatar.com/avatar/fb79412c4e9a782a2314d65cad708726eb57b7689401e7a728a51543bb2f82c9?s=40&d=mm&r=g)\
\
\
\
     mohamedAugust 15, 2016 at 9:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360016 "Direct link to this comment")\
\
\
\
\
\
     after training, how i can use the trained model on new sample\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360016)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2016 at 12:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360051 "Direct link to this comment")\
\
\
\
\
\
       You can call model.predict()\
\
\
\
       See an above comment for a specific code example.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360051)\
017. ![](https://secure.gravatar.com/avatar/907bbde36f7d3ce16812c3b37969ca731f393888769e075da2e42f2258d11391?s=40&d=mm&r=g)\
\
\
\
     Omachi OkoloAugust 16, 2016 at 10:21 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360223 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     i’m a student conducting a research on how to use artificial neural network to predict the business viability of potential software projects.\
\
\
     I intend to use python as a programming language. The application of ANN fascinates me but i’m new to machine learning and python. Can you help suggest how to go about this.\
\
\
     Many thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360223)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 17, 2016 at 9:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360270 "Direct link to this comment")\
\
\
\
\
\
       Consider getting a good grounding in how to work through a machine learning problem end to end in python first.\
\
\
\
       Here is a good tutorial to get you started:\
\
       [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360270)\
018. ![](https://secure.gravatar.com/avatar/e9e3c4c4612825a7acc7b7859fe49ede4cac34f4b1993aec2923c3c5d27179c1?s=40&d=mm&r=g)\
\
\
\
     AgniAugust 17, 2016 at 6:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360254 "Direct link to this comment")\
\
\
\
\
\
     Dear Jeson, this is a great tutorial for beginners. It will satisfy the need of many students who are looking for the initial help. But I have a question. Could you please light on a few things: i) how to test the trained model using test dataset (i.e., loading of test dataset and applied the model and suppose the test file name is test.csv) ii) print the accuracy obtained on test dataset iii) the o/p has more than 2 class (suppose 4-class classification problem).\
\
\
     Please show the whole program to overcome any confusion.\
\
\
     Thanks a lot.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360254)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 17, 2016 at 10:03 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360276 "Direct link to this comment")\
\
\
\
\
\
       I provide an example elsewhere in the comments, you can also see how to make predictions on new data in this post:\
\
       [https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/](https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/)\
\
\
\
       For an example of multi-class classification, you can see this tutorial:\
\
       [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360276)\
019. ![](https://secure.gravatar.com/avatar/4dc479af46e98ecb2bbaac86cbc204915990df6b2962f3aa906a428607f447b3?s=40&d=mm&r=g)\
\
\
\
     Doron VetlzerAugust 17, 2016 at 9:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360265 "Direct link to this comment")\
\
\
\
\
\
     I am trying to build a Neural Network with some recursive connections but not a full recursive layer, how do I do this in Keras?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360265)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/4dc479af46e98ecb2bbaac86cbc204915990df6b2962f3aa906a428607f447b3?s=40&d=mm&r=g)\
\
\
\
       Doron VetlzerAugust 17, 2016 at 9:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360266 "Direct link to this comment")\
\
\
\
\
\
       I could print a diagram of the network but what I want Basically is that each neuron in the current time frame to know only its own previous output and not the output of all the neurons in the output layer.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360266)\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 17, 2016 at 10:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360278 "Direct link to this comment")\
\
\
\
\
\
       I don’t know off hand Doron.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-360278)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/4dc479af46e98ecb2bbaac86cbc204915990df6b2962f3aa906a428607f447b3?s=40&d=mm&r=g)\
\
\
\
         Doron VeltzerAugust 23, 2016 at 2:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-361067 "Direct link to this comment")\
\
\
\
\
\
         Thanks for replying though, have a good day.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-361067)\
020. ![](https://secure.gravatar.com/avatar/61492814f5682d40c276b4eeffd2f2eea6de492cacaebe2b6615beb687ef088b?s=40&d=mm&r=g)\
\
\
\
     sairamAugust 30, 2016 at 8:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-362638 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     This is a great tutorial . Thanks for sharing.\
\
\
\
     I am having a dataset of 100 finger prints and i want to extract minutiae of 100 finger prints using python ( Keras). Can you please advise where to start? I am really confused.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-362638)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 31, 2016 at 8:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-362842 "Direct link to this comment")\
\
\
\
\
\
       If your fingerprints are images, you may want to consider using convolutional neural networks (CNNs) that are much better at working image data.\
\
\
\
       See this tutorial on digit recognition for a start:\
\
       [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-362842)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/68fb7ded630b45987761fd127561ac83bc34e3ca80005bc8cbe3070d940dcbe6?s=40&d=mm&r=g)\
\
\
\
         padmashriJuly 6, 2017 at 10:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404952 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason\
\
\
         Thanks for this great tutorial, i am new to machine learning i went through your basic tutorial on keras and also handwritten-digit-recognition. I would like to understand how i can train a set of image data, for eg. the set of image data can be some thing like square, circle, pyramid.\
\
\
         pl. let me know how the input data needs to fed to the program and how we need to export the model.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404952)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 9, 2017 at 10:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405230 "Direct link to this comment")\
\
\
\
\
\
           Start by preparing a high-quality dataset.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405230)\
021. ![](https://secure.gravatar.com/avatar/b9a6292b7240845f923b51bdc68b99df7daa0e646dc60ea15cae8df0d09c4240?s=40&d=mm&r=g)\
\
\
\
     CMSeptember 1, 2016 at 4:23 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363101 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for the great article. But I had 1 query.\
\
\
\
     Are there any inbuilt functions in keras that can give me the feature importance for the ANN model?\
\
\
\
     If not, can you suggest a technique I can use to extract variable importance from the loss function? I am considering an approach similar to that used in RF which involves permuting the values of the selected variable and calculating the relative increase in loss.\
\
\
\
     Regards,\
\
\
     CM\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363101)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 2, 2016 at 8:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363222 "Direct link to this comment")\
\
\
\
\
\
       I don’t believe so CM.\
\
\
\
       I would suggest using a wrapper method and evaluate subsets of features to develop a feature importance/feature selection report.\
\
\
\
       I talk a lot more about feature selection in this post:\
\
       [https://machinelearningmastery.com/an-introduction-to-feature-selection/](https://machinelearningmastery.com/an-introduction-to-feature-selection/)\
\
\
\
       I provide an example of feature selection in scikit-learn here:\
\
       [https://machinelearningmastery.com/feature-selection-machine-learning-python/](https://machinelearningmastery.com/feature-selection-machine-learning-python/)\
\
\
\
       I hope that helps as a start.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363222)\
\
     - ![](https://secure.gravatar.com/avatar/bac4ee6d25010266125af41ce0d0b36eb5d39763cea167accc39f40a1a54209e?s=40&d=mm&r=g)\
\
\
\
       Minesh JethvaMay 15, 2017 at 7:49 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399721 "Direct link to this comment")\
\
\
\
\
\
       have you develop any progress for this approach? I also have same problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399721)\
022. ![](https://secure.gravatar.com/avatar/81eedc0ec855fc13967e7d844602e1e45714b40167e9192bcd678b029714741e?s=40&d=mm&r=g)\
\
\
\
     KamalSeptember 7, 2016 at 2:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363819 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason, I am new to Deep learning. Being a novice, I am asking you a technical question which may seem silly. My question is that- can we use features (for example length of the sentence etc.) of a sentence while classifying a sentence ( suppose the o/p are +ve sentence and -ve sentence) using deep neural network?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363819)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 7, 2016 at 10:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363856 "Direct link to this comment")\
\
\
\
\
\
       Great question Kamal, yes you can. I would encourage you to include all such features and see which give you a bump in performance.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-363856)\
023. ![](https://secure.gravatar.com/avatar/35a884d9f0638dc35b416f4c4ecc0a4b83008866b24bdc9d5303f2b67dae554a?s=40&d=mm&r=g)\
\
\
\
     SaurabhSeptember 11, 2016 at 12:42 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364196 "Direct link to this comment")\
\
\
\
\
\
     Hi, How would I use this on a dataset that has multiple outputs? For example a dataset with output A and B where A could be 0 or 1 and B could be 3 or 4 ?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364196)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 12, 2016 at 8:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364259 "Direct link to this comment")\
\
\
\
\
\
       You could use two neurons in the output layer and normalize the output variables to both be in the range of 0 to 1.\
\
\
\
       This tutorial on multi-class classification might give you some ideas:\
\
       [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364259)\
024. ![](https://secure.gravatar.com/avatar/7e3798eb81c3d371f3191a0ddede028792109b254dc8edc81cc743a1a1f82671?s=40&d=mm&r=g)\
\
\
\
     Tom\_PSeptember 17, 2016 at 1:47 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364714 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     The tutorial looks really good but unfortunately I keep getting an error when importing Dense from keras.layers, I get the error : AttributeError: module ‘theano’ has no attribute ‘gof’\
\
\
     I have tried reinstalling Theano but it has not fixed the issue.\
\
\
\
     Best wishes\
\
\
     Tom\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364714)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 18, 2016 at 7:57 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364837 "Direct link to this comment")\
\
\
\
\
\
       Hi Tom, sorry to hear that. I have not seen this problem before.\
\
\
\
       Have you searched google? I can see a few posts and it might be related to your version of scipy or similar.\
\
\
\
       Let me know how you go.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-364837)\
025. ![](https://secure.gravatar.com/avatar/99ad0d32a4a93df2d5fe3c1ec73dee31352e4f7d9cb08f0a512c7054623f4087?s=40&d=mm&r=g)\
\
\
\
     shudhanSeptember 21, 2016 at 5:54 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365180 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason,\
\
\
\
     Can you please make a tutorial on how to add additional train data into the already trained model? This will be helpful for the bigger data sets. I read that warm start is used for random forest. But not sure how to implement as algorithm. A generalised version of how to implement would be good. Thank You!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365180)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 22, 2016 at 8:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365231 "Direct link to this comment")\
\
\
\
\
\
       Great question Shudhan!\
\
\
\
       Yes, you could save your weights, load them later into a new network topology and start training on new data again.\
\
\
\
       I’ll work out an example in coming weeks, time permitting.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365231)\
026. ![](https://secure.gravatar.com/avatar/dbf523faec5573a648ccd01d94274c296d6b6da7a3c36c2c1da913df14a9a6ae?s=40&d=mm&r=g)\
\
\
\
     JoannaSeptember 22, 2016 at 1:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365204 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     first of all congratulations for this amazing work that you have done!\
\
\
     Here is my question:\
\
\
     What about if my .csv file includes also both nominal and numerical attributes?\
\
\
     Should I change my nominal values to numerical?\
\
\
\
     Thank you in advance\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365204)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 22, 2016 at 8:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365238 "Direct link to this comment")\
\
\
\
\
\
       Hi Joanna, yes.\
\
\
\
       You can use a label encoder to convert nominal to integer, and then even convert the integer to one hot encoding.\
\
\
\
       This post will give you code you can use:\
\
       [https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/](https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-365238)\
027. ![](https://secure.gravatar.com/avatar/59c5c5835ac9cdba2a6858ad86f2484b29b7c6209c71f2ead85b9f5b1b243da6?s=40&d=mm&r=g)\
\
\
\
     ATMOctober 2, 2016 at 5:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366048 "Direct link to this comment")\
\
\
\
\
\
     A small bug:-\
\
\
     Line 25 : rounded = \[round(x) for x in predictions\]\
\
\
\
     should have numpy.round instead, for the code to run!\
\
\
     Great tutorial, regardless. The best i’ve seen for intro to ANN in python. Thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366048)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 2, 2016 at 8:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366058 "Direct link to this comment")\
\
\
\
\
\
       Perhaps it’s your version of Python or environment?\
\
\
\
       In Python 2.7 the round() function is built-in.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366058)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/6485d75125cd811c99a301dceb74aeae47e8b200f4d312911b9ddaf9cf8a6cc9?s=40&d=mm&r=g)\
\
\
\
         ACJanuary 14, 2017 at 2:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-381921 "Direct link to this comment")\
\
\
\
\
\
         If there is comment for python3, should be better.\
\
\
         #use unmpy.round instead, if using python3,\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-381921)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)January 15, 2017 at 5:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382141 "Direct link to this comment")\
\
\
\
\
\
           Thanks for the note AC.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382141)\
028. ![](https://secure.gravatar.com/avatar/3c774c7ea4b48030a5d4d23e6529618da3f2361c05fe7d768cff501e7acd6fa6?s=40&d=mm&r=g)\
\
\
\
     AshOctober 9, 2016 at 1:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366635 "Direct link to this comment")\
\
\
\
\
\
     This is simple to grasp! Great post! How can we perform dropout in keras?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366635)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 9, 2016 at 6:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366651 "Direct link to this comment")\
\
\
\
\
\
       Thanks Ash.\
\
\
\
       You can learn about drop out with Keras here:\
\
       [https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-366651)\
029. ![](https://secure.gravatar.com/avatar/41058d1b837772d3d8ada9fd0e49402f3117560aa10fc6fad6465212093218f1?s=40&d=mm&r=g)\
\
\
\
     Homagni SahaOctober 14, 2016 at 4:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367094 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
     You are using model.predict in the end to predict the results. Is it possible to save the model somewhere in the harddisk and transfer it to another machine(turtlebot running on ROS for my instance) and then use the model directly on turtlebot to predict the results?\
\
\
     Please tell me how\
\
\
     Thanking you\
\
\
     Homagni Saha\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367094)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 14, 2016 at 9:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367116 "Direct link to this comment")\
\
\
\
\
\
       Hi Homagni, great question.\
\
\
\
       Absolutely!\
\
\
\
       Learn exactly how in this tutorial I wrote:\
\
       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367116)\
030. ![](https://secure.gravatar.com/avatar/d9350e12cfc90c6c404594ff1bc80d3c76ae2295a664c2c807457a91c1343fb2?s=40&d=mm&r=g)\
\
\
\
     RimiOctober 16, 2016 at 8:21 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367261 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     I implemented you code to begin with. But I am getting an accuracy of 45.18% with the same parameters and everything.\
\
\
     Cant figure out why.\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367261)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 17, 2016 at 10:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367304 "Direct link to this comment")\
\
\
\
\
\
       There does sound like a problem there Rimi.\
\
\
\
       Confirm the code and data match exactly.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-367304)\
031. ![](https://secure.gravatar.com/avatar/c420c0f62d24f69fa2161277b94a67ab2401928e0b5a29e07d37924e80e499c1?s=40&d=mm&r=g)\
\
\
\
     AnkitOctober 26, 2016 at 8:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368300 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     I am little confused with first layer parameters. You said that first layer has 12 neurons and expects 8 input variables.\
\
\
\
     Why there is a difference between number of neurons, input\_dim for first layer.\
\
\
\
     Regards,\
\
\
     Ankit\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368300)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 27, 2016 at 7:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368357 "Direct link to this comment")\
\
\
\
\
\
       Hi Ankit,\
\
\
\
       The problem has 8 input variables and the first hidden layer has 12 neurons. Inputs are the columns of data, these are fixed. The Hidden layers in general are whatever we design based on whatever capacity we think we need to represent the complexity of the problem. In this case, we have chosen 12 neurons for the first hidden layer.\
\
\
\
       I hope that is clearer.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368357)\
032. ![](https://secure.gravatar.com/avatar/d7aea65780b951404b99221b62503a03e04e88b26f1934ad6ce48fbaa8a89d3b?s=40&d=mm&r=g)\
\
\
\
     TomOctober 27, 2016 at 3:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368339 "Direct link to this comment")\
\
\
\
\
\
     Hi,\
\
\
     I have a data , IRIS like data but with more colmuns.\
\
\
     I want to use MLP and DBN/CNNClassifier (or any other Deep Learning classificaiton algorithm) on my data to see how correctly it does classified into 6 groups.\
\
\
\
     Previously using DEEP LEARNING FOR J, today first time see KERAS.\
\
\
     does KERAS has examples (code examples) of DL Classification algorithms?\
\
\
\
     Kindly,\
\
\
     Tom\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368339)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 27, 2016 at 7:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368361 "Direct link to this comment")\
\
\
\
\
\
       Yes Tom, the example in this post is an example of a neural network (deep learning) applied to a classification problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368361)\
033. ![](https://secure.gravatar.com/avatar/e143849ee624af6bed3d72e4e8b3d76f61bc0ce2ad834042fc53b5feaf9971ab?s=40&d=mm&r=g)\
\
\
\
     [Rumesa](http://none/)October 30, 2016 at 1:57 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368638 "Direct link to this comment")\
\
\
\
\
\
     I have installed theano but it gives me the error of tensorflow.is it mendatory to install both packages? because tensorflow is not supported on wndows.the only way to get it on windows is to install virtual machine\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368638)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 30, 2016 at 8:57 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368679 "Direct link to this comment")\
\
\
\
\
\
       Keras will work just fine with Theano.\
\
\
\
       Just install Theano, and configure Keras to use the Theano backend.\
\
\
\
       More information about configuring the Keras backend here:\
\
       [https://machinelearningmastery.com/introduction-python-deep-learning-library-keras/](https://machinelearningmastery.com/introduction-python-deep-learning-library-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368679)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/e143849ee624af6bed3d72e4e8b3d76f61bc0ce2ad834042fc53b5feaf9971ab?s=40&d=mm&r=g)\
\
\
\
         RumesaOctober 31, 2016 at 4:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368749 "Direct link to this comment")\
\
\
\
\
\
         hey jason I have run your code but got the following error.Although I have aready installed theano backend.help me out.I just stuck.\
\
\
\
         Using TensorFlow backend.\
\
\
         Traceback (most recent call last):\
\
\
         File “C:\\Users\\pc\\Desktop\\first.py”, line 2, in\
\
\
         from keras.models import Sequential\
\
\
         File “C:\\Users\\pc\\Anaconda3\\lib\\site-packages\\keras\\\_\_init\_\_.py”, line 2, in\
\
\
         from . import backend\
\
\
         File “C:\\Users\\pc\\Anaconda3\\lib\\site-packages\\keras\\backend\\\_\_init\_\_.py”, line 64, in\
\
\
         from .tensorflow\_backend import \*\
\
\
         File “C:\\Users\\pc\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow\_backend.py”, line 1, in\
\
\
         import tensorflow as tf\
\
\
         ImportError: No module named ‘tensorflow’\
\
\
         >>>\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368749)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)October 31, 2016 at 5:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368758 "Direct link to this comment")\
\
\
\
\
\
           Change the backend used by Keras from TensorFlow to Theano.\
\
\
\
           You can do this either by using the command line switch or changing the Keras config file.\
\
\
\
           See the link I posted in the previous post for instructions.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368758)\
     - ![](https://secure.gravatar.com/avatar/c2f968bcb9dac15a04fa9747675eece82ea15443c31bf7d646cb2185c18402b9?s=40&d=mm&r=g)\
\
\
\
       MariaJanuary 6, 2017 at 1:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-380544 "Direct link to this comment")\
\
\
\
\
\
       Hello Rumesa!\
\
\
       Have you solved your problem? I have the same one. Everywhere is the same answer with keras.json file or envirinment variable but it doesn’t work. Can you tell me what have worked for you?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-380544)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)January 7, 2017 at 8:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-380737 "Direct link to this comment")\
\
\
\
\
\
         Interesting.\
\
\
\
         Maybe there is an issue with the latest version and a tight coupling to tensorflow? I have not seen this myself.\
\
\
\
         Perhaps it might be worth testing prior versions of Keras, such as 1.1.0?\
\
\
\
         Try this:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
         | 1 | pip install--upgrade--no-deps keras==1.1.0 |\
\
\
\
\
\
\
\
\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-380737)\
034. ![](https://secure.gravatar.com/avatar/4ca6a7e5a2632a1ed6b47a1cf82b2745c28f00114f50a2698ce8969dc7dd4c28?s=40&d=mm&r=g)\
\
\
\
     AlexonNovember 1, 2016 at 6:54 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368895 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     First off, thanks so much for creating these resources, I have been keeping an eye on your newsletter for a while now, and I finally have the free time to start learning more about it myself, so your work has been really appreciated.\
\
\
\
     My question is: How can I set/get the weights of each hidden node?\
\
\
\
     I am planning to create several arrays randomized weights, then use a genetic algorithm to see which weight array performs the best and improve over generations. How would be the best way to go about this, and if I use a “relu” activation function, am I right in thinking these randomly generated weights should be between 0 and 0.05?\
\
\
\
     Many thanks for your help 🙂\
\
\
     Alexon\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368895)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 1, 2016 at 8:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368909 "Direct link to this comment")\
\
\
\
\
\
       Thanks Alexon,\
\
\
\
       You can get and set the weights from a network.\
\
\
\
       You can learn more about how to do this in the context of saving the weights to file here:\
\
       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
       I hope that helps as a start, I’d love to hear how you go.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-368909)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/4ca6a7e5a2632a1ed6b47a1cf82b2745c28f00114f50a2698ce8969dc7dd4c28?s=40&d=mm&r=g)\
\
\
\
         AlexonNovember 6, 2016 at 6:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-369623 "Direct link to this comment")\
\
\
\
\
\
         Thats great, thanks for pointing me in the right direction.\
\
\
         I’d be happy to let you know how it goes, but might take a while as this is very much a “when I can find the time” project between jobs 🙂\
\
\
\
         Cheers!\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-369623)\
035. ![](https://secure.gravatar.com/avatar/e25438c6d94925cd99a4a8ca5c5f301c88fcca1edd815a6f66d416449df7a7a7?s=40&d=mm&r=g)\
\
\
\
     [Arnaldo Gunzi](http://ideiasesquecidas.com/)November 2, 2016 at 10:17 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-369156 "Direct link to this comment")\
\
\
\
\
\
     Nice introduction, thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-369156)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 3, 2016 at 7:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-369229 "Direct link to this comment")\
\
\
\
\
\
       I’m glad you found it useful Arnaldo.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-369229)\
036. ![](https://secure.gravatar.com/avatar/45c78e04c70604db2cebe343acae394cd7ca67389d5511af11efde400ca64e47?s=40&d=mm&r=g)\
\
\
\
     AbbeyNovember 14, 2016 at 11:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370561 "Direct link to this comment")\
\
\
\
\
\
     Good day\
\
\
\
     I have a question, how can I represent a character as a vector that could be an input for the neural network to predict the word meaning and trained using LSTM\
\
\
\
     For instance, I have bf to predict boy friend or best friend and similarly I have 2mor to predict tomorrow. I need to encode all the input as a character represented as vector, so that it can be train with RNN/LSTM to predict the output.\
\
\
\
     Thank you.\
\
\
\
     Kind Regards\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370561)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 15, 2016 at 7:54 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370608 "Direct link to this comment")\
\
\
\
\
\
       Hi Abbey, You can map characters to integers to get integer vectors.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370608)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/45c78e04c70604db2cebe343acae394cd7ca67389d5511af11efde400ca64e47?s=40&d=mm&r=g)\
\
\
\
         AbbeyNovember 15, 2016 at 6:17 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370652 "Direct link to this comment")\
\
\
\
\
\
         Thank you Jason, if i map characters to integers value to get vectors using English Alphabets, numbers and special characters\
\
\
\
         The question is how will LSTM predict the character. Please example in more details for me.\
\
\
\
         Regards\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370652)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)November 16, 2016 at 9:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370785 "Direct link to this comment")\
\
\
\
\
\
           Hi Abbey,\
\
\
\
           If your output values are also characters, you can map them onto integers, and reverse the mapping to convert the predictions back to text.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370785)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/45c78e04c70604db2cebe343acae394cd7ca67389d5511af11efde400ca64e47?s=40&d=mm&r=g)\
\
\
\
             AbbeyNovember 16, 2016 at 8:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370855 "Direct link to this comment")\
\
\
\
\
\
             The output value of the characters encoding will be text\
       - ![](https://secure.gravatar.com/avatar/45c78e04c70604db2cebe343acae394cd7ca67389d5511af11efde400ca64e47?s=40&d=mm&r=g)\
\
\
\
         AbbeyNovember 15, 2016 at 6:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370653 "Direct link to this comment")\
\
\
\
\
\
         Thank you, Jason, if I map characters to integers value to get vectors representation of the informal text using English Alphabets, numbers and special characters\
\
\
\
         The question is how will LSTM predict the character or words that have close meaning to the input value. Please example in more details for me. I understand how RNN/LSTM work based on your tutorial example but the logic in designing processing is what I am stress with.\
\
\
\
         Regards\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-370653)\
037. ![](https://secure.gravatar.com/avatar/6615bedfd08d824dd215a53738c63dc52571e8460b080953ce7331b72bcfc5dd?s=40&d=mm&r=g)\
\
\
\
     AmmarNovember 27, 2016 at 10:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372532 "Direct link to this comment")\
\
\
\
\
\
     hi Jason,\
\
\
     i am trying to implement CNN one dimention on my data. so, i bluit my network.\
\
\
     the issue is:\
\
\
     def train\_model(model, X\_train, y\_train, X\_test, y\_test):\
\
\
     X\_train = X\_train.reshape(-1, 1, 41)\
\
\
     X\_test = X\_test.reshape(-1, 1, 41)\
\
\
\
     numpy.random.seed(seed)\
\
\
     model.fit(X\_train, y\_train, validation\_data=(X\_test, y\_test), nb\_epoch=100, batch\_size=64)\
\
\
     # Final evaluation of the model\
\
\
     scores = model.evaluate(X\_test, y\_test, verbose=0)\
\
\
     print(“Accuracy: %.2f%%” % (scores\[1\] \* 100))\
\
\
     this method above does not work and does not give me any error message.\
\
\
     could you help me with this please?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372532)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 28, 2016 at 8:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372662 "Direct link to this comment")\
\
\
\
\
\
       Hi Ammar, I’m surprised that there is no error message.\
\
\
\
       Perhaps run from the command line and add some print() statements to see exactly where it stops.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372662)\
038. ![](https://secure.gravatar.com/avatar/45adea083fddcdcff6c572d7800f325cba7123ebeb72fdaeb083abc8fe636af1?s=40&d=mm&r=g)\
\
\
\
     KKNovember 28, 2016 at 6:55 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372738 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
     Great work. I have another doubt. How can we apply this to text mining. I have a csv file containing review document and label. I want to apply classify the documents based on the text available. Can U do this favor.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372738)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 29, 2016 at 8:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372815 "Direct link to this comment")\
\
\
\
\
\
       I would recommend converting the chars to ints and then using an Embedding layer.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-372815)\
039. ![](https://secure.gravatar.com/avatar/f1930e8107845659a7b36ef136355d65a100f809e4dc907e16ea397598b70870?s=40&d=mm&r=g)\
\
\
\
     Alex MNovember 30, 2016 at 10:52 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373111 "Direct link to this comment")\
\
\
\
\
\
     Mr Jason, this is great tutorial but I am stack with some errors.\
\
\
\
     First I can’t load data set correctly, tried to correct error but can’t make it. ( FileNotFoundError: \[Errno 2\] No such file or directory: ‘pima-indians-diabetes.csv’ ).\
\
\
\
     Second: While trying to evaluate the model it says (X is not defined) May be this is because uploading failed.\
\
\
\
     Thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373111)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 1, 2016 at 7:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373161 "Direct link to this comment")\
\
\
\
\
\
       You need to download the file and place it in your current working directory Alex.\
\
\
\
       Does that help?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373161)\
040. ![](https://secure.gravatar.com/avatar/f1930e8107845659a7b36ef136355d65a100f809e4dc907e16ea397598b70870?s=40&d=mm&r=g)\
\
\
\
     Alex MDecember 1, 2016 at 6:45 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373233 "Direct link to this comment")\
\
\
\
\
\
     Sir, it is now successful….\
\
\
     Thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373233)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 2, 2016 at 8:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373342 "Direct link to this comment")\
\
\
\
\
\
       Glad to hear it Alex.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373342)\
041. ![](https://secure.gravatar.com/avatar/aa3d9f9ae3d32d62371cee99385dbff276741000e6377e073fc79cc70cc9cbe3?s=40&d=mm&r=g)\
\
\
\
     BappadityaDecember 2, 2016 at 7:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373423 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     First of all a special thanks to you for providing such a great tutorial. I am very new to machine learning and truly speaking i had no background in data science. The concept of ML overwhelmed me and now i have a desire to be an expert of this field. I need your advice to start from a scratch. Also i am a PhD student in Computer Engineering ( computer hardware )and i want to apply it as a tool for fault detection and testing for ICs.Can you provide me some references on this field?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373423)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 3, 2016 at 8:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373511 "Direct link to this comment")\
\
\
\
\
\
       Hi Bappaditya,\
\
\
\
       My best advice for getting started is here:\
\
       [https://machinelearningmastery.com/start-here/#getstarted](https://machinelearningmastery.com/start-here/#getstarted)\
\
\
\
       I believe machine learning and deep learning are good tools for use on problems in fault detection. A good place to find references is here [http://scholar.google.com](https://scholar.google.com/)\
\
\
\
       Best of luck with your project.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373511)\
042. ![](https://secure.gravatar.com/avatar/f1930e8107845659a7b36ef136355d65a100f809e4dc907e16ea397598b70870?s=40&d=mm&r=g)\
\
\
\
     Alex MDecember 3, 2016 at 8:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373580 "Direct link to this comment")\
\
\
\
\
\
     Well as usual in our daily coding life errors happen, now I have this error how can I correct it? Thanks!\
\
\
\
     ” —————————————————————————\
\
\
     NoBackendError Traceback (most recent call last)\
\
\
     in ()\
\
\
     16 import librosa.display\
\
\
     17 audio\_path = (‘/Users/MA/Python Notebook/OK.mp3’)\
\
\
     —\> 18 y, sr = librosa.load(audio\_path)\
\
\
\
     C:\\Users\\MA\\Anaconda3\\lib\\site-packages\\librosa\\core\\audio.py in load(path, sr, mono, offset, duration, dtype)\
\
\
     107\
\
\
     108 y = \[\]\
\
\
     –\> 109 with audioread.audio\_open(os.path.realpath(path)) as input\_file:\
\
\
     110 sr\_native = input\_file.samplerate\
\
\
     111 n\_channels = input\_file.channels\
\
\
\
     C:\\Users\\MA\\Anaconda3\\lib\\site-packages\\audioread\\\_\_init\_\_.py in audio\_open(path)\
\
\
     112\
\
\
     113 # All backends failed!\
\
\
     –\> 114 raise NoBackendError()\
\
\
\
     NoBackendError:\
\
\
\
     ”\
\
\
\
     That is the error I am getting just when trying to load a song into librosa…\
\
\
     Thanks!! @Jason Brownlee\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373580)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 4, 2016 at 5:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373652 "Direct link to this comment")\
\
\
\
\
\
       Sorry, this looks like an issue with your librosa library, not a machine learning issue. I can’t give you expert advice, sorry.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373652)\
043. ![](https://secure.gravatar.com/avatar/f1930e8107845659a7b36ef136355d65a100f809e4dc907e16ea397598b70870?s=40&d=mm&r=g)\
\
\
\
     Alex MDecember 4, 2016 at 10:30 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373746 "Direct link to this comment")\
\
\
\
\
\
     Thanks I have managed to correct the error…\
\
\
\
     Happy Sunday to you all……\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373746)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 5, 2016 at 6:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373806 "Direct link to this comment")\
\
\
\
\
\
       Glad to hear it Alex.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373806)\
\
     - ![](https://secure.gravatar.com/avatar/f40719a1ab8ff5bbb0c4133d438b53f5170f968618d731ad43cf7f1ace63f0c0?s=40&d=mm&r=g)\
\
\
\
       ayushJune 19, 2018 at 3:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441201 "Direct link to this comment")\
\
\
\
\
\
       how did you solved the problem?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441201)\
044. ![](https://secure.gravatar.com/avatar/22fbfdf70c466a5bfe44b7b4076ab63c218f2fdfdd9905dc59437855b219bdd5?s=40&d=mm&r=g)\
\
\
\
     LeiDecember 4, 2016 at 10:52 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373749 "Direct link to this comment")\
\
\
\
\
\
     Hi, Jason, thank you for your amazing examples.\
\
\
     I run the same code on my laptop. But I did not get the same results. What could be the possible reasons?\
\
\
     I am using windows 8.1 64bit+eclipse+anaconda 4.2+theano 0.9.4+CUDA7.5\
\
\
     I got results like follows.\
\
\
\
     … …\
\
\
     Epoch 145/150\
\
\
\
     10/768 \[…………………………\] – ETA: 0s – loss: 0.3634 – acc: 0.8000\
\
\
     80/768 \[==>………………………\] – ETA: 0s – loss: 0.4066 – acc: 0.7750\
\
\
     150/768 \[====>…………………….\] – ETA: 0s – loss: 0.4059 – acc: 0.8067\
\
\
     220/768 \[=======>………………….\] – ETA: 0s – loss: 0.4047 – acc: 0.8091\
\
\
     300/768 \[==========>……………….\] – ETA: 0s – loss: 0.4498 – acc: 0.7867\
\
\
     380/768 \[=============>…………….\] – ETA: 0s – loss: 0.4595 – acc: 0.7895\
\
\
     450/768 \[================>………….\] – ETA: 0s – loss: 0.4568 – acc: 0.7911\
\
\
     510/768 \[==================>………..\] – ETA: 0s – loss: 0.4553 – acc: 0.7882\
\
\
     580/768 \[=====================>……..\] – ETA: 0s – loss: 0.4677 – acc: 0.7776\
\
\
     660/768 \[========================>…..\] – ETA: 0s – loss: 0.4697 – acc: 0.7788\
\
\
     740/768 \[===========================>..\] – ETA: 0s – loss: 0.4611 – acc: 0.7838\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4614 – acc: 0.7799\
\
\
     Epoch 146/150\
\
\
\
     10/768 \[…………………………\] – ETA: 0s – loss: 0.3846 – acc: 0.8000\
\
\
     90/768 \[==>………………………\] – ETA: 0s – loss: 0.5079 – acc: 0.7444\
\
\
     170/768 \[=====>……………………\] – ETA: 0s – loss: 0.4500 – acc: 0.7882\
\
\
     250/768 \[========>…………………\] – ETA: 0s – loss: 0.4594 – acc: 0.7840\
\
\
     330/768 \[===========>………………\] – ETA: 0s – loss: 0.4574 – acc: 0.7818\
\
\
     400/768 \[==============>……………\] – ETA: 0s – loss: 0.4563 – acc: 0.7775\
\
\
     470/768 \[=================>…………\] – ETA: 0s – loss: 0.4654 – acc: 0.7723\
\
\
     540/768 \[====================>………\] – ETA: 0s – loss: 0.4537 – acc: 0.7870\
\
\
     620/768 \[=======================>……\] – ETA: 0s – loss: 0.4615 – acc: 0.7806\
\
\
     690/768 \[=========================>….\] – ETA: 0s – loss: 0.4631 – acc: 0.7739\
\
\
     750/768 \[============================>.\] – ETA: 0s – loss: 0.4649 – acc: 0.7733\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4636 – acc: 0.7734\
\
\
     Epoch 147/150\
\
\
\
     10/768 \[…………………………\] – ETA: 0s – loss: 0.3561 – acc: 0.9000\
\
\
     90/768 \[==>………………………\] – ETA: 0s – loss: 0.4167 – acc: 0.8556\
\
\
     170/768 \[=====>……………………\] – ETA: 0s – loss: 0.4824 – acc: 0.8059\
\
\
     250/768 \[========>…………………\] – ETA: 0s – loss: 0.4534 – acc: 0.8080\
\
\
     330/768 \[===========>………………\] – ETA: 0s – loss: 0.4679 – acc: 0.7848\
\
\
     400/768 \[==============>……………\] – ETA: 0s – loss: 0.4590 – acc: 0.7950\
\
\
     460/768 \[================>………….\] – ETA: 0s – loss: 0.4619 – acc: 0.7913\
\
\
     530/768 \[===================>……….\] – ETA: 0s – loss: 0.4562 – acc: 0.7868\
\
\
     600/768 \[======================>…….\] – ETA: 0s – loss: 0.4497 – acc: 0.7883\
\
\
     680/768 \[=========================>….\] – ETA: 0s – loss: 0.4525 – acc: 0.7853\
\
\
     760/768 \[============================>.\] – ETA: 0s – loss: 0.4568 – acc: 0.7803\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4561 – acc: 0.7812\
\
\
     Epoch 148/150\
\
\
\
     10/768 \[…………………………\] – ETA: 0s – loss: 0.4183 – acc: 0.9000\
\
\
     80/768 \[==>………………………\] – ETA: 0s – loss: 0.3674 – acc: 0.8750\
\
\
     160/768 \[=====>……………………\] – ETA: 0s – loss: 0.4340 – acc: 0.8250\
\
\
     240/768 \[========>…………………\] – ETA: 0s – loss: 0.4799 – acc: 0.7583\
\
\
     320/768 \[===========>………………\] – ETA: 0s – loss: 0.4648 – acc: 0.7719\
\
\
     400/768 \[==============>……………\] – ETA: 0s – loss: 0.4596 – acc: 0.7775\
\
\
     470/768 \[=================>…………\] – ETA: 0s – loss: 0.4475 – acc: 0.7809\
\
\
     540/768 \[====================>………\] – ETA: 0s – loss: 0.4545 – acc: 0.7778\
\
\
     620/768 \[=======================>……\] – ETA: 0s – loss: 0.4590 – acc: 0.7742\
\
\
     690/768 \[=========================>….\] – ETA: 0s – loss: 0.4769 – acc: 0.7652\
\
\
     760/768 \[============================>.\] – ETA: 0s – loss: 0.4748 – acc: 0.7658\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4734 – acc: 0.7669\
\
\
     Epoch 149/150\
\
\
\
     10/768 \[…………………………\] – ETA: 0s – loss: 0.3043 – acc: 0.9000\
\
\
     90/768 \[==>………………………\] – ETA: 0s – loss: 0.4913 – acc: 0.7111\
\
\
     170/768 \[=====>……………………\] – ETA: 0s – loss: 0.4779 – acc: 0.7588\
\
\
     250/768 \[========>…………………\] – ETA: 0s – loss: 0.4794 – acc: 0.7640\
\
\
     320/768 \[===========>………………\] – ETA: 0s – loss: 0.4957 – acc: 0.7562\
\
\
     370/768 \[=============>…………….\] – ETA: 0s – loss: 0.4891 – acc: 0.7703\
\
\
     450/768 \[================>………….\] – ETA: 0s – loss: 0.4737 – acc: 0.7867\
\
\
     520/768 \[===================>……….\] – ETA: 0s – loss: 0.4675 – acc: 0.7865\
\
\
     600/768 \[======================>…….\] – ETA: 0s – loss: 0.4668 – acc: 0.7833\
\
\
     680/768 \[=========================>….\] – ETA: 0s – loss: 0.4677 – acc: 0.7809\
\
\
     760/768 \[============================>.\] – ETA: 0s – loss: 0.4648 – acc: 0.7803\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4625 – acc: 0.7826\
\
\
     Epoch 150/150\
\
\
\
     10/768 \[…………………………\] – ETA: 0s – loss: 0.2751 – acc: 1.0000\
\
\
     100/768 \[==>………………………\] – ETA: 0s – loss: 0.4501 – acc: 0.8100\
\
\
     170/768 \[=====>……………………\] – ETA: 0s – loss: 0.4588 – acc: 0.8059\
\
\
     250/768 \[========>…………………\] – ETA: 0s – loss: 0.4299 – acc: 0.8200\
\
\
     310/768 \[===========>………………\] – ETA: 0s – loss: 0.4298 – acc: 0.8129\
\
\
     380/768 \[=============>…………….\] – ETA: 0s – loss: 0.4365 – acc: 0.8053\
\
\
     460/768 \[================>………….\] – ETA: 0s – loss: 0.4469 – acc: 0.7957\
\
\
     540/768 \[====================>………\] – ETA: 0s – loss: 0.4436 – acc: 0.8000\
\
\
     620/768 \[=======================>……\] – ETA: 0s – loss: 0.4570 – acc: 0.7871\
\
\
     690/768 \[=========================>….\] – ETA: 0s – loss: 0.4664 – acc: 0.7783\
\
\
     760/768 \[============================>.\] – ETA: 0s – loss: 0.4617 – acc: 0.7789\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4638 – acc: 0.7773\
\
\
\
     32/768 \[>………………………..\] – ETA: 0s\
\
\
     448/768 \[================>………….\] – ETA: 0sacc: 79.69%\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373749)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 5, 2016 at 6:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373807 "Direct link to this comment")\
\
\
\
\
\
       There is randomness in the learning process that we cannot control for yet.\
\
\
\
       See this post:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-373807)\
045. ![](https://secure.gravatar.com/avatar/c7f9ac8c98adb1aed54e96be6c464798a065dcd4ae8d991ac5bc36b576e268f5?s=40&d=mm&r=g)\
\
\
\
     [Nanya](https://xiaomenglnan.github.io/)December 10, 2016 at 2:55 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-374651 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason Brownlee,Thx for sharing~\
\
\
     I’m new in deep learning.And I am wondering can what you dicussed here:”Keras” be used to build a CNN in tensorflow and train some csv fiels for classification.May be this is a stupid question,but waiting for you reply.I’m working on my graduation project for Word sense disambiguation with cnn,and just can’t move on.Hope for your heip~Bese wishes!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-374651)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 11, 2016 at 5:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-374745 "Direct link to this comment")\
\
\
\
\
\
       Sorry Nanya, I’m not sure I understand your question. Are you able to rephrase it?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-374745)\
046. ![](https://secure.gravatar.com/avatar/7f9e08851059143a3981377f7e72598102d602481b04c4ab49a474f39937b569?s=40&d=mm&r=g)\
\
\
\
     AnonDecember 16, 2016 at 12:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-375349 "Direct link to this comment")\
\
\
\
\
\
     I’ve just installed Anaconda with Keras and am using python 3.5.\
\
\
     It seems there’s an error with the rounding using Py3 as opposed to Py2. I think it’s because of this change: [https://github.com/numpy/numpy/issues/5700](https://github.com/numpy/numpy/issues/5700)\
\
\
\
     I removed the rounding and just used print(predictions) and it seemed to work outputting floats instead.\
\
\
\
     Does this look correct?\
\
\
\
     …\
\
\
     Epoch 150/150\
\
\
     0s – loss: 0.4593 – acc: 0.7839\
\
\
     \[\[ 0.79361773\]\
\
\
     \[ 0.10443526\]\
\
\
     \[ 0.90862554\]\
\
\
     …,\
\
\
     \[ 0.33652252\]\
\
\
     \[ 0.63745886\]\
\
\
     \[ 0.11704451\]\]\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-375349)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 16, 2016 at 5:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-375377 "Direct link to this comment")\
\
\
\
\
\
       Nice, it does look good!\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-375377)\
047. ![](https://secure.gravatar.com/avatar/f3d177a6f5e56fdee55ba225e10a9b0fe7b89b4414b5128324dc97c53b1fe466?s=40&d=mm&r=g)\
\
\
\
     Florin Claudiu MihalacheDecember 19, 2016 at 2:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-375891 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason Brownlee\
\
\
     I tried to modified your exemple for my problem (Letter Recognition , [http://archive.ics.uci.edu/ml/datasets/Letter+Recognition](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition)).\
\
\
     My data set look like [http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data](http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data) (T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8) .I try to split the data in input and ouput like this :\
\
\
\
     X = dataset\[:,1:17\]\
\
\
     Y = dataset\[:,0\]\
\
\
     but a have some error (something related that strings are not recognized) .\
\
\
     I tried to modified each letter whit the ASCII code (A became 65 and so on).The string error disappeared.\
\
\
     The program compiles now but the output look like this :\
\
\
\
     17445/20000 \[=========================>….\] – ETA: 0s – loss: -1219.4768 – acc:0.0000e+00\
\
\
     17605/20000 \[=========================>….\] – ETA: 0s – loss: -1219.4706 – acc:0.0000e+00\
\
\
     17730/20000 \[=========================>….\] – ETA: 0s – loss: -1219.4566 – acc:0.0000e+00\
\
\
     17890/20000 \[=========================>….\] – ETA: 0s – loss: -1219.4071 – acc:0.0000e+00\
\
\
     18050/20000 \[==========================>…\] – ETA: 0s – loss: -1219.4599 – acc:0.0000e+00\
\
\
     18175/20000 \[==========================>…\] – ETA: 0s – loss: -1219.3972 – acc:0.0000e+00\
\
\
     18335/20000 \[==========================>…\] – ETA: 0s – loss: -1219.4642 – acc:0.0000e+00\
\
\
     18495/20000 \[==========================>…\] – ETA: 0s – loss: -1219.5032 – acc:0.0000e+00\
\
\
     18620/20000 \[==========================>…\] – ETA: 0s – loss: -1219.4391 – acc:0.0000e+00\
\
\
     18780/20000 \[===========================>..\] – ETA: 0s – loss: -1219.5652 – acc:0.0000e+00\
\
\
     18940/20000 \[===========================>..\] – ETA: 0s – loss: -1219.5520 – acc:0.0000e+00\
\
\
     19080/20000 \[===========================>..\] – ETA: 0s – loss: -1219.5381 – acc:0.0000e+00\
\
\
     19225/20000 \[===========================>..\] – ETA: 0s – loss: -1219.5182 – acc:0.0000e+00\
\
\
     19385/20000 \[============================>.\] – ETA: 0s – loss: -1219.6742 – acc:0.0000e+00\
\
\
     19535/20000 \[============================>.\] – ETA: 0s – loss: -1219.7030 – acc:0.0000e+00\
\
\
     19670/20000 \[============================>.\] – ETA: 0s – loss: -1219.7634 – acc:0.0000e+00\
\
\
     19830/20000 \[============================>.\] – ETA: 0s – loss: -1219.8336 – acc:0.0000e+00\
\
\
     19990/20000 \[============================>.\] – ETA: 0s – loss: -1219.8532 – acc:0.0000e+00\
\
\
     20000/20000 \[==============================\] – 1s – loss: -1219.8594 – acc: 0.0000e+00\
\
\
     18880/20000 \[===========================>..\] – ETA: 0sacc: 0.00%\
\
\
\
     I do not understand why. Can you please help me\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-375891)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/7f9e08851059143a3981377f7e72598102d602481b04c4ab49a474f39937b569?s=40&d=mm&r=g)\
\
\
\
       AnonDecember 26, 2016 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-377637 "Direct link to this comment")\
\
\
\
\
\
       What version of Python are you running?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-377637)\
048. ![](https://secure.gravatar.com/avatar/737a3b471f3ca18791b75ff13396bc184e04de604f5d8edbe1ba1a28e78baaca?s=40&d=mm&r=g)\
\
\
\
     karishma sharmaDecember 22, 2016 at 10:03 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376750 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Since the epoch is set to 150 and batch size is 10, does the training algorithm pick 10 training examples at random in each iteration, given that we had only 768 total in X. Or does it sample randomly after it has finished covering all.\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376750)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 23, 2016 at 5:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376943 "Direct link to this comment")\
\
\
\
\
\
       Good question,\
\
\
\
       It iterates over the dataset 150 times and within one epoch it works through 10 rows at a time before doing an update to the weights. The patterns are shuffled before each epoch.\
\
\
\
       I hope that helps.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-376943)\
049. ![](https://secure.gravatar.com/avatar/bf9e26146fc7554e43404757dbc8e0309ff9dcc196af26aef3657fa7b52510fa?s=40&d=mm&r=g)\
\
\
\
     KaustuvJanuary 9, 2017 at 4:57 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-381108 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
     Thanks a lot for this blog. It really helps me to start learning deep learning which was in a planning state for last few months. Your simple enrich blogs are awsome. No questions from my side before completing all tutorials.\
\
\
     One question regarding availability of your book. How can I buy those books from India ?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-381108)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 9, 2017 at 7:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-381137 "Direct link to this comment")\
\
\
\
\
\
       All my books and training are digital, you can purchase them from here:\
\
       [https://machinelearningmastery.com/products](https://machinelearningmastery.com/products)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-381137)\
050. ![](https://secure.gravatar.com/avatar/31502e670bfd140dbaca7ba3b72a7477a4240493c3142f337f2727f872059f6b?s=40&d=mm&r=g)\
\
\
\
     Stephen WilsonJanuary 15, 2017 at 4:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382225 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, firstly your work here is a fantastic resource and I am very thankful for the effort you put in.\
\
\
     I am a slightly-better-than-beginner at python and an absolute novice at ML, I wonder if you could help me classify my problem and find an angle to work at it from.\
\
\
\
     My data is thus:\
\
\
     Column Names: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, Result\
\
\
     Values: 4, 4, 6, 6, 3, 2, 5, 5, 0, 0, 0, 0, 0, 0, 0, 4\
\
\
\
     I want to find the percentage chance of each Column Names category being the Result based off the configuration of all the values present from 1-15. Then if need be compare the configuration of Values with another row of values to find the same, Resulting in the total needed calculation as:\
\
\
\
     Column Names: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, Result\
\
\
     Values: 4, 4, 6, 6, 3, 2, 5, 5, 0, 0, 0, 0, 0, 0, 0, 4\
\
\
     Values2: 7, 3, 5, 1, 4, 8, 6, 2, 9, 9, 9, 9, 9, 9, 9\
\
\
\
     I apologize if my explanation is not clear, and appreciate any help you can give me thank you.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382225)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 16, 2017 at 10:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382380 "Direct link to this comment")\
\
\
\
\
\
       Hi Stephen,\
\
\
\
       This process might help you work through your problem:\
\
       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)\
\
\
\
       Specifically the first step in defining your problem.\
\
\
\
       Let me know how you go.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382380)\
051. ![](https://secure.gravatar.com/avatar/cd1432c6d3b4cddcfbee720493292634501dbde48ecfc6b66a0cad49140d18ad?s=40&d=mm&r=g)\
\
\
\
     RohitJanuary 16, 2017 at 10:37 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382469 "Direct link to this comment")\
\
\
\
\
\
     Thanks Jason for such a nice and concise example.\
\
\
\
     Just wanted to ask if it is possible to save this model in a file and port it to may be an Android or iOS device? If so, what are the libraries available for the same?\
\
\
\
     Thanks\
\
\
\
     Rohit\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382469)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 17, 2017 at 7:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382531 "Direct link to this comment")\
\
\
\
\
\
       Thanks Rohit,\
\
\
\
       Here’s an example of saving a Keras model to file:\
\
       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
       I don’t know about running Keras on an Android or iOS device. Let me know how you go.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382531)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/db0173fa2b9225aaeedd0a90655056f23080703cf299945c00e15b78c90899bb?s=40&d=mm&r=g)\
\
\
\
         zaheer khanJune 16, 2017 at 7:17 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402754 "Direct link to this comment")\
\
\
\
\
\
         Dear Jason, Thanks for sharing this article.\
\
\
         I am novice to the deep learning, and my apology if my question is not clear. my question is could we call all that functions and program from any .php,.aspx, or .html webpage. i mean i load the variables and other files selection from user interface and then make them input to this functions.\
\
\
\
         will be waiting for your kind reply.\
\
\
         thanks in advance.\
\
\
         zaheer\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402754)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)June 17, 2017 at 7:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402832 "Direct link to this comment")\
\
\
\
\
\
           Perhaps, this sounds like a systems design question, not really machine learning.\
\
\
\
           I would suggest you gather requirements, assess risks like any software engineering project.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402832)\
052. ![](https://secure.gravatar.com/avatar/447d8366a09672cc256f2ef72f7745577629a5a1927abbbad220137e34996a58?s=40&d=mm&r=g)\
\
\
\
     [Hsiang](http://www.hsianghung.tech/)January 18, 2017 at 3:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382776 "Direct link to this comment")\
\
\
\
\
\
     Hi, Jason\
\
\
\
     Thank you for your blog! It is wonderful!\
\
\
\
     I used tensorflow as backend, and implemented the procedures using Jupyter.\
\
\
     I did “source activate tensorflow” -> “ipython notebook”.\
\
\
     I can successfully use Keras and import tensorflow.\
\
\
\
     However, it seems that such environment doesn’t support pandas and sklearn.\
\
\
     Do you have any way to incorporate pandas, sklearn and keras?\
\
\
     (I wish to use sklearn to revisit the classification problem and compare the accuracy with the deep learning method. But I also wish to put the works together in the same interface.)\
\
\
\
     Thanks!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382776)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 19, 2017 at 7:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382891 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I do not use notebooks myself. I cannot offer you good advice.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382891)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/447d8366a09672cc256f2ef72f7745577629a5a1927abbbad220137e34996a58?s=40&d=mm&r=g)\
\
\
\
         HsiangJanuary 19, 2017 at 12:53 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382930 "Direct link to this comment")\
\
\
\
\
\
         Thanks, Jason!\
\
\
         Actually the problem is not on notebooks. Even I used the terminal mode, i.e. doing “source activate tensorflow” only. It failed to import sklearn. Does that mean tensorflow library is not compatible with sklearn? Thanks again!\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-382930)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)January 20, 2017 at 10:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383064 "Direct link to this comment")\
\
\
\
\
\
           Sorry Hsiang, I don’t have experience using sklearn and tensorflow with virtual environments.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383064)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/447d8366a09672cc256f2ef72f7745577629a5a1927abbbad220137e34996a58?s=40&d=mm&r=g)\
\
\
\
             HsiangJanuary 21, 2017 at 12:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383177 "Direct link to this comment")\
\
\
\
\
\
             Thank you!\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)January 21, 2017 at 10:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383249 "Direct link to this comment")\
\
\
\
\
\
             You’re welcome Hsiang.\
053. ![](https://secure.gravatar.com/avatar/386841d73410b1afd0a5a5c90f291f87125cbd4abec165bdb5acbf66e2a160d7?s=40&d=mm&r=g)\
\
\
\
     keshav bansalJanuary 24, 2017 at 12:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383693 "Direct link to this comment")\
\
\
\
\
\
     hello sir,\
\
\
     A very informative post indeed . I know my question is a very trivial one but can you please show me how to predict on a explicitly mentioned data tuple say v=\[6,148,72,35,0,33.6,0.627,50\]\
\
\
     thanks for the tutorial anyway\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383693)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 24, 2017 at 11:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383796 "Direct link to this comment")\
\
\
\
\
\
       Hi keshav,\
\
\
\
       You can make predictions by calling model.predict()\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383796)\
054. ![](https://secure.gravatar.com/avatar/f593558e211a43d7112762596e4bd02fbaaf05568d2e6209b7ca547deb7467b3?s=40&d=mm&r=g)\
\
\
\
     CATRINA WEBBJanuary 25, 2017 at 9:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383995 "Direct link to this comment")\
\
\
\
\
\
     When I rerun the file (without predictions) does it reset the model and weights?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-383995)\
\
055. ![](https://secure.gravatar.com/avatar/a9b0a2eed76597dff5d9cb0a4ad7db22b99261cc8973f5918a35b5f7d25b9ca2?s=40&d=mm&r=g)\
\
\
\
     EricsonJanuary 30, 2017 at 8:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-385061 "Direct link to this comment")\
\
\
\
\
\
     excuse me sir, i wanna ask you a question about this paragraph”dataset = numpy.loadtxt(“pima-indians-diabetes.csv”,delimiter=’,’)”, i used the mac and downloaded the dataset,then i exchanged the text into csv file. Running the program\
\
\
\
     ,hen i got:{Python 2.7.13 (v2.7.13:a06454b1afa1, Dec 17 2016, 12:39:47)\
\
\
     \[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)\] on darwin\
\
\
     Type “copyright”, “credits” or “license()” for more information.\
\
\
     >>>\
\
\
     ============ RESTART: /Users/luowenbin/Documents/database\_test.py ============\
\
\
     Using TensorFlow backend.\
\
\
\
     Traceback (most recent call last):\
\
\
     File “/Users/luowenbin/Documents/database\_test.py”, line 9, in\
\
\
     dataset = numpy.loadtxt(“pima-indians-diabetes.csv”,delimiter=’,’)\
\
\
     File “/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/lib/npyio.py”, line 985, in loadtxt\
\
\
     items = \[conv(val) for (conv, val) in zip(converters, vals)\]\
\
\
     File “/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/lib/npyio.py”, line 687, in floatconv\
\
\
     return float(x)\
\
\
     ValueError: could not convert string to float: book\
\
\
     >>\> }\
\
\
     How can i solve this problem? give me a hand thank you!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-385061)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 1, 2017 at 10:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-385398 "Direct link to this comment")\
\
\
\
\
\
       Hi Ericson,\
\
\
\
       Confirm that the contents of “pima-indians-diabetes.csv” meet your expectation of a list of CSV lines.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-385398)\
056. ![](https://secure.gravatar.com/avatar/c7ce40186e6030c6cd631991460a52f31a1510365b727a0d0d055fe6d7aa24d3?s=40&d=mm&r=g)\
\
\
\
     SukhpalFebruary 7, 2017 at 9:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-386441 "Direct link to this comment")\
\
\
\
\
\
     excuse me sir,when i run this code for my data set ,I encounter this problem…please help me finding solution to this problem\
\
\
     runfile(‘C:/Users/sukhpal/.spyder/temp.py’, wdir=’C:/Users/sukhpal/.spyder’)\
\
\
     Using TensorFlow backend.\
\
\
     Traceback (most recent call last):\
\
\
\
     File “”, line 1, in\
\
\
     runfile(‘C:/Users/sukhpal/.spyder/temp.py’, wdir=’C:/Users/sukhpal/.spyder’)\
\
\
\
     File “C:\\Users\\sukhpal\\Anaconda2\\lib\\site-packages\\spyder\\utils\\site\\sitecustomize.py”, line 866, in runfile\
\
\
     execfile(filename, namespace)\
\
\
\
     File “C:\\Users\\sukhpal\\Anaconda2\\lib\\site-packages\\spyder\\utils\\site\\sitecustomize.py”, line 87, in execfile\
\
\
     exec(compile(scripttext, filename, ‘exec’), glob, loc)\
\
\
\
     File “C:/Users/sukhpal/.spyder/temp.py”, line 1, in\
\
\
     from keras.models import Sequential\
\
\
\
     File “C:\\Users\\sukhpal\\Anaconda2\\lib\\site-packages\\keras\\\_\_init\_\_.py”, line 2, in\
\
\
     from . import backend\
\
\
\
     File “C:\\Users\\sukhpal\\Anaconda2\\lib\\site-packages\\keras\\backend\\\_\_init\_\_.py”, line 67, in\
\
\
     from .tensorflow\_backend import \*\
\
\
\
     File “C:\\Users\\sukhpal\\Anaconda2\\lib\\site-packages\\keras\\backend\\tensorflow\_backend.py”, line 1, in\
\
\
     import tensorflow as tf\
\
\
\
     ImportError: No module named tensorflow\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-386441)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 8, 2017 at 9:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-386530 "Direct link to this comment")\
\
\
\
\
\
       This is a change with the most recent version of tensorflow, I will investigate and change the example.\
\
\
\
       For now, consider installing and using an older version of tensorflow.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-386530)\
057. ![](https://secure.gravatar.com/avatar/30b044122b02614c43e3fc4d88884eba1e1abef37432993132dffbd10b19385a?s=40&d=mm&r=g)\
\
\
\
     [Will](http://www.willkriski.com/)February 14, 2017 at 5:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387845 "Direct link to this comment")\
\
\
\
\
\
     Great tutorial! Amazing amount of work you’ve put in and great marketing skills (I also have an email list, ebooks and sequence, etc). I ran this in Jupyter notebook… I noticed the 144th epoch (acc .7982) had more accuracy than at 150. Why is that?\
\
\
\
     P.S. i did this for the print: print(numpy.round(predictions))\
\
\
     It seems to avoid a list of arrays which when printing includes the dtype (messy)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387845)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 14, 2017 at 10:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387886 "Direct link to this comment")\
\
\
\
\
\
       Thanks Will.\
\
\
\
       The model will fluctuate in performance while learning. You can configure triggered check points to save the model if/when conditions like a decrease in train/validation performance is detected. Here’s an example:\
\
       [https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387886)\
058. ![](https://secure.gravatar.com/avatar/c7ce40186e6030c6cd631991460a52f31a1510365b727a0d0d055fe6d7aa24d3?s=40&d=mm&r=g)\
\
\
\
     SukhpalFebruary 14, 2017 at 3:50 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387936 "Direct link to this comment")\
\
\
\
\
\
     Please help me to find out this error\
\
\
     runfile(‘C:/Users/sukhpal/.spyder/temp.py’, wdir=’C:/Users/sukhpal/.spyder’)ERROR: execution aborted\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387936)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 15, 2017 at 11:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388114 "Direct link to this comment")\
\
\
\
\
\
       I’m not sure Sukhpal.\
\
\
\
       Consider getting code working from the command line, I don’t use IDEs myself.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388114)\
059. ![](https://secure.gravatar.com/avatar/fb4a17e8abedd7ace37f859c5b70ccadab4ef4c9e4b0065a33b603785db80266?s=40&d=mm&r=g)\
\
\
\
     KamalFebruary 14, 2017 at 5:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387951 "Direct link to this comment")\
\
\
\
\
\
     please help me to find this error find this error\
\
\
     Epoch 194/195\
\
\
     195/195 \[==============================\] – 0s – loss: 0.2692 – acc: 0.8667\
\
\
     Epoch 195/195\
\
\
     195/195 \[==============================\] – 0s – loss: 0.2586 – acc: 0.8667\
\
\
     195/195 \[==============================\] – 0s\
\
\
     Traceback (most recent call last):\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-387951)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 15, 2017 at 11:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388115 "Direct link to this comment")\
\
\
\
\
\
       What was the error exactly Kamal?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388115)\
060. ![](https://secure.gravatar.com/avatar/fb4a17e8abedd7ace37f859c5b70ccadab4ef4c9e4b0065a33b603785db80266?s=40&d=mm&r=g)\
\
\
\
     KamalFebruary 15, 2017 at 3:24 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388170 "Direct link to this comment")\
\
\
\
\
\
     sir when i run the code on my data set\
\
\
     then it doesnot show overall accuracy although it shows the accuracy and loss for the whole iterations\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388170)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 16, 2017 at 11:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388332 "Direct link to this comment")\
\
\
\
\
\
       I’m not sure I understand your question Kamal, please you could restate it?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388332)\
061. ![](https://secure.gravatar.com/avatar/9b3ebf8e6bceef4b825ca9f38f0a30c6f201d0fb71780fd13618435bdec31235?s=40&d=mm&r=g)\
\
\
\
     ValFebruary 15, 2017 at 9:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388217 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, im just starting deep learning in python using keras and theano. I have followed the installation instructions without a hitch. Tested some examples but when i run this one line by line i get a lot of exceptions and errors once i run the “model.fit(X,Y, nb\_epochs=150, batch\_size=10”\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388217)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 16, 2017 at 11:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388333 "Direct link to this comment")\
\
\
\
\
\
       What errors are you getting?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388333)\
062. ![](https://secure.gravatar.com/avatar/d775b381958936f99b895efe7377aaa082c12b5085ecf2a7588ff612e93da5af?s=40&d=mm&r=g)\
\
\
\
     CrisHFebruary 17, 2017 at 8:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388662 "Direct link to this comment")\
\
\
\
\
\
     Hi, how do I know what number to use for random.seed() ? I mean you use 7, is there any reason for that? Also is it enough to use it only once, in the beginning of the code?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388662)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 18, 2017 at 8:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388792 "Direct link to this comment")\
\
\
\
\
\
       You can use any number CrisH. The fixed random seed makes the example reproducible.\
\
\
\
       You can learn more about randomness and random seeds in this post:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388792)\
063. ![](https://secure.gravatar.com/avatar/27c95db00e19d02d0ec2c0e0d4c2d8e5d758e989626e0ec059b1f8981b950638?s=40&d=mm&r=g)\
\
\
\
     kkFebruary 18, 2017 at 1:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388717 "Direct link to this comment")\
\
\
\
\
\
     am new to deep learning and found this great tutorial. keep it up and look forward!!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388717)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 18, 2017 at 8:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388795 "Direct link to this comment")\
\
\
\
\
\
       Thanks!\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-388795)\
064. ![](https://secure.gravatar.com/avatar/bfbdb772db76ef56fd2fbaf96a4993559fc43094a345658ce72672f7540f7228?s=40&d=mm&r=g)\
\
\
\
     [Iqra Ameer](https://google/)February 21, 2017 at 5:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389313 "Direct link to this comment")\
\
\
\
\
\
     HI, I have a problem in execution the above example as it. It seems that it’s not running properly and stops at Using TensorFlow backend.\
\
\
\
     Epoch 147/150\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4709 – acc: 0.7878\
\
\
     Epoch 148/150\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4690 – acc: 0.7812\
\
\
     Epoch 149/150\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4711 – acc: 0.7721\
\
\
     Epoch 150/150\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4731 – acc: 0.7747\
\
\
     32/768 \[>………………………..\] – ETA: 0sacc: 76.43%\
\
\
\
     I am new in this field, could you please guide me about this error.\
\
\
     I also executed on another data set, it stops with the same behavior.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389313)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 21, 2017 at 9:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389340 "Direct link to this comment")\
\
\
\
\
\
       What is the error exactly? The example hangs?\
\
\
\
       Maybe try the Theano backend and see if that makes a difference. Also make sure all of your libraries are up to date.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389340)\
065. ![](https://secure.gravatar.com/avatar/bfbdb772db76ef56fd2fbaf96a4993559fc43094a345658ce72672f7540f7228?s=40&d=mm&r=g)\
\
\
\
     [Iqra Ameer](https://google/)February 22, 2017 at 5:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389450 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason,\
\
\
     Thank you so much for your valuable suggestions. I tried Theano backend and also updated all my libraries, but again it hanged at:\
\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4656 – acc: 0.7799\
\
\
     Epoch 149/150\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4589 – acc: 0.7826\
\
\
     Epoch 150/150\
\
\
     768/768 \[==============================\] – 0s – loss: 0.4611 – acc: 0.7773\
\
\
     32/768 \[>………………………..\] – ETA: 0sacc: 78.91%\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389450)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 22, 2017 at 10:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389472 "Direct link to this comment")\
\
\
\
\
\
       I’m sorry to hear that, I have not seen this issue before.\
\
\
\
       Perhaps a RAM issue or a CPU overheating issue? Are you able to try different hardware?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389472)\
\
     - ![](https://secure.gravatar.com/avatar/9ce67fc56f47341b9074e25dc49b560aff6f0bd5a16c07a384689fb5e920cac9?s=40&d=mm&r=g)\
\
\
\
       frdMarch 8, 2017 at 2:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391727 "Direct link to this comment")\
\
\
\
\
\
       Hi!\
\
\
\
       Were you able to find a solution for that?\
\
\
\
       I’m having exactly the same problem\
\
\
\
       ( … )\
\
\
       Epoch 149/150\
\
\
       768/768 \[==============================\] – 0s – loss: 0.4593 – acc: 0.7773\
\
\
       Epoch 150/150\
\
\
       768/768 \[==============================\] – 0s – loss: 0.4586 – acc: 0.7891\
\
\
       32/768 \[>………………………..\] – ETA: 0sacc: 76.69%\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391727)\
066. ![](https://secure.gravatar.com/avatar/fb4a17e8abedd7ace37f859c5b70ccadab4ef4c9e4b0065a33b603785db80266?s=40&d=mm&r=g)\
\
\
\
     BhanuFebruary 23, 2017 at 1:51 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389674 "Direct link to this comment")\
\
\
\
\
\
     Hello sir,\
\
\
     i want to ask wether we can convert this code to deep learning wid increasing number of layers..\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389674)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 24, 2017 at 10:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389818 "Direct link to this comment")\
\
\
\
\
\
       Sure you can increase the number of layers, try it and see.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-389818)\
067. ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
     Ananya MohapatraFebruary 28, 2017 at 6:40 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390595 "Direct link to this comment")\
\
\
\
\
\
     hello sir,\
\
\
     could you please tell me how do i determine the no.of neurons in each layer, because i am using a different datset and am unable to know the no.of neurons in each layer\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390595)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 1, 2017 at 8:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390705 "Direct link to this comment")\
\
\
\
\
\
       Hi Ananya, great question.\
\
\
\
       Sorry, there is no good theory on how to configure a neural net.\
\
\
\
       You can configure the number of neurons in a layer by trial and error. Also consider tuning the number of epochs and batch size at the same time.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390705)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
         Ananya MohapatraMarch 1, 2017 at 4:42 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390766 "Direct link to this comment")\
\
\
\
\
\
         thank you so much sir. It worked ! 🙂\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390766)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2017 at 8:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390867 "Direct link to this comment")\
\
\
\
\
\
           Glad to here it Ananya.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390867)\
068. ![](https://secure.gravatar.com/avatar/a6f2857edd46aede4c4dcebc7d0d9a96035176cd9bac4c552b08d5cf24769153?s=40&d=mm&r=g)\
\
\
\
     Jayant SahewalFebruary 28, 2017 at 8:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390608 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     really helpful blog. I have a question about how much time does it take to converge?\
\
\
\
     I have a dataset with around 4000 records, 3 input columns and 1 output column. I came up with the following model\
\
\
\
     def create\_model(dropout\_rate=0.0, weight\_constraint=0, learning\_rate=0.001, activation=’linear’):\
\
\
     # create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(6, input\_dim=3, init=’uniform’, activation=activation, W\_constraint=maxnorm(weight\_constraint)))\
\
\
     model.add(Dropout(dropout\_rate))\
\
\
     model.add(Dense(1, init=’uniform’, activation=’sigmoid’))\
\
\
     # Optimizer\
\
\
     optimizer = Adam(lr=learning\_rate)\
\
\
     # Compile model\
\
\
     model.compile(loss=’binary\_crossentropy’, optimizer=optimizer, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
\
     \# create model\
\
\
     model = KerasRegressor(build\_fn=create\_model, verbose=0)\
\
\
     \# define the grid search parameters\
\
\
     batch\_size = \[10\]\
\
\
     epochs = \[100\]\
\
\
     weight\_constraint = \[3\]\
\
\
     dropout\_rate = \[0.9\]\
\
\
     learning\_rate = \[0.01\]\
\
\
     activation = \[‘linear’\]\
\
\
     param\_grid = dict(batch\_size=batch\_size, nb\_epoch=epochs, dropout\_rate=dropout\_rate, \\
\
\
     weight\_constraint=weight\_constraint, learning\_rate=learning\_rate, activation=activation)\
\
\
     grid = GridSearchCV(estimator=model, param\_grid=param\_grid, n\_jobs=-1, cv=5)\
\
\
     grid\_result = grid.fit(X\_train, Y\_train)\
\
\
\
     I have a 32 core machine with 64 GB RAM and it does not converge even in more than an hour. I can see all the cores busy, so it is using all the cores for training. However, if I change the input neurons to 3 then it converges in around 2 minutes.\
\
\
\
     Keras version: 1.1.1\
\
\
     Tensorflow version: 0.10.0rc0\
\
\
     theano version: 0.8.2.dev-901275534cbfe3fbbe290ce85d1abf8bb9a5b203\
\
\
\
     It’s using Tensorflow backend. Can you help me understand what is going on or point me in the right direction? Do you think switching to theano will help?\
\
\
\
     Best,\
\
\
     Jayant\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390608)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 1, 2017 at 8:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390709 "Direct link to this comment")\
\
\
\
\
\
       This post might help you tune your deep learning model:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       I hope that helps as a start.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390709)\
069. ![](https://secure.gravatar.com/avatar/ba1e4b539ec730b9c63117a55d8d5f840569207ce1122a610cc2163c8a190769?s=40&d=mm&r=g)\
\
\
\
     Animesh MohantyMarch 1, 2017 at 9:21 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390804 "Direct link to this comment")\
\
\
\
\
\
     hello sir,\
\
\
     could you please tell me how can i plot the results of the code on a graph . I made a few adjustments to the code so as to run it on a different dataset.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390804)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2017 at 8:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390870 "Direct link to this comment")\
\
\
\
\
\
       What do you want to plot exactly Animesh?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390870)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/ba1e4b539ec730b9c63117a55d8d5f840569207ce1122a610cc2163c8a190769?s=40&d=mm&r=g)\
\
\
\
         Animesh MohantyMarch 2, 2017 at 4:56 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390943 "Direct link to this comment")\
\
\
\
\
\
         Accuracy vs no.of neurons in the input layer and the no.of neurons in the hidden layer\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390943)\
070. ![](https://secure.gravatar.com/avatar/18ce64a54d848417a1ea7a24ffa83da26c1e27b7378317c306be46d8260d7e11?s=40&d=mm&r=g)\
\
\
\
     paramMarch 2, 2017 at 12:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390826 "Direct link to this comment")\
\
\
\
\
\
     sir can u plz explain\
\
\
     the different attributes used in this statement\
\
\
     print(“%s: %.2f%%” % (model.metrics\_names\[1\], scores\[1\]\*100))\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390826)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/18ce64a54d848417a1ea7a24ffa83da26c1e27b7378317c306be46d8260d7e11?s=40&d=mm&r=g)\
\
\
\
       paramMarch 2, 2017 at 12:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390827 "Direct link to this comment")\
\
\
\
\
\
       precisely,what is model.metrics\_names\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390827)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2017 at 8:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390873 "Direct link to this comment")\
\
\
\
\
\
         model.metrics\_names is a list of names of the metrics collected during training.\
\
\
\
         More details here:\
\
         [https://keras.io/models/sequential/](https://keras.io/models/sequential/)\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390873)\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2017 at 8:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390872 "Direct link to this comment")\
\
\
\
\
\
       Hi param,\
\
\
\
       It is using string formatting. %s formats a string, %.2f formats a floating point value with 2 decimal places, %% includes a percent symbol.\
\
\
\
       You can learn more about the print function here:\
\
       [https://docs.python.org/3/library/functions.html#print](https://docs.python.org/3/library/functions.html#print)\
\
\
\
       More info on string formatting here:\
\
       [https://pyformat.info/](https://pyformat.info/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390872)\
071. ![](https://secure.gravatar.com/avatar/a02ad67690d50bcaa27777db8758f4969107df6c440af2f9ddbcf30cb0fbf7c7?s=40&d=mm&r=g)\
\
\
\
     Vijin K PMarch 2, 2017 at 4:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390842 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     It was an awesome post. Could you please tell me how to we decide the following in a DNN 1. number of neurons in the hidden layers\
\
\
     2\. number of hidden layers\
\
\
\
     Thanks.\
\
\
     Vijin\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390842)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2017 at 8:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390875 "Direct link to this comment")\
\
\
\
\
\
       Great question Vijin.\
\
\
\
       Generally, trial and error. There are no good theories on how to configure a neural network.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390875)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/a02ad67690d50bcaa27777db8758f4969107df6c440af2f9ddbcf30cb0fbf7c7?s=40&d=mm&r=g)\
\
\
\
         Vijin K PMarch 3, 2017 at 5:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391000 "Direct link to this comment")\
\
\
\
\
\
         We do cross validation, grid search etc to find the hyper parameters in machine algorithms. Similarly can we do anything to identify the above parameters??\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391000)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 3, 2017 at 7:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391016 "Direct link to this comment")\
\
\
\
\
\
           Yes, we can use grid search and tuning for neural nets.\
\
\
\
           The stochastic nature of neural nets means that each experiment (set of configs) will have to be run many times (30? 100?) so that you can take the mean performance.\
\
\
\
           More general info on tuning neural nets here:\
\
           [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
           More on randomness and stochastic algorithms here:\
\
           [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391016)\
072. ![](https://secure.gravatar.com/avatar/70d42627b0b5f7aec7329b9b2ff9a741b1f5071ccd3c82f7501376b68cfc7121?s=40&d=mm&r=g)\
\
\
\
     BogdanMarch 2, 2017 at 11:48 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390971 "Direct link to this comment")\
\
\
\
\
\
     Jason, Please tell me about these lines in your code:\
\
\
\
     seed = 7\
\
\
     numpy.random.seed(seed)\
\
\
\
     What do they do? And why do they do it?\
\
\
\
     One more question is why do you call the last section Bonus:Make a prediction?\
\
\
     I thought this what ANN was created for. What the point if your network’s output is just what you have already know?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-390971)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 3, 2017 at 7:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391015 "Direct link to this comment")\
\
\
\
\
\
       They seed the random number generator so that it produces the same sequence of random numbers each time the code is run. This is to ensure you get the same result as me.\
\
\
\
       I’m not convinced it works with Keras though.\
\
\
\
       More on randomness in machine learning here:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       I was showing how to build and evaluate the model in this tutorial. The part about standalone prediction was an add-on.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391015)\
073. ![](https://secure.gravatar.com/avatar/8b633aec5b5e3f878c01fb1998381df727e27ec21b691881caaa7cdf1f882603?s=40&d=mm&r=g)\
\
\
\
     Sounak sahooMarch 3, 2017 at 7:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391085 "Direct link to this comment")\
\
\
\
\
\
     what exactly is the work of “seed” in the neural network code? what does it do?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391085)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 6, 2017 at 10:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391450 "Direct link to this comment")\
\
\
\
\
\
       Seed refers to seeding the random number generator so that the same sequence of random numbers is generated each time the example is run.\
\
\
\
       The aim is to make the examples 100% reproducible, but this is hard with symbolic math libs like Theano and TensorFlow backends.\
\
\
\
       For more on randomness in machine learning, see this post:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391450)\
074. ![](https://secure.gravatar.com/avatar/d55a95794bff58df3dbe11b76e592dfff5a0c31431fcb6d19034655151391a89?s=40&d=mm&r=g)\
\
\
\
     Priya SundariMarch 3, 2017 at 10:19 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391107 "Direct link to this comment")\
\
\
\
\
\
     hello sir\
\
\
     could you plz tell me what is the role of optimizer and binary\_crossentropy exactly? it is written that optimizer is used to search through the weights of the network which weights are we talking about exactly?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391107)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 6, 2017 at 10:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391454 "Direct link to this comment")\
\
\
\
\
\
       Hi Priya,\
\
\
\
       You can learn more about the fundamentals of neural nets here:\
\
       [https://machinelearningmastery.com/neural-networks-crash-course/](https://machinelearningmastery.com/neural-networks-crash-course/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391454)\
075. ![](https://secure.gravatar.com/avatar/70d42627b0b5f7aec7329b9b2ff9a741b1f5071ccd3c82f7501376b68cfc7121?s=40&d=mm&r=g)\
\
\
\
     BogdanMarch 3, 2017 at 10:23 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391108 "Direct link to this comment")\
\
\
\
\
\
     If I am not mistaken, those lines I commented about used when we write\
\
\
\
     init = ‘uniform’\
\
\
\
     ?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391108)\
\
076. ![](https://secure.gravatar.com/avatar/70d42627b0b5f7aec7329b9b2ff9a741b1f5071ccd3c82f7501376b68cfc7121?s=40&d=mm&r=g)\
\
\
\
     BogdanMarch 3, 2017 at 10:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391111 "Direct link to this comment")\
\
\
\
\
\
     Could you explain in more details what is the batch size?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391111)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 6, 2017 at 10:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391455 "Direct link to this comment")\
\
\
\
\
\
       Hi Bogdan,\
\
\
\
       Batch size is how many patterns to show to the network before the weights are updated with the accumulated errors. The smaller the batch, the faster the learning, but also the more noisy the learning (higher variance).\
\
\
\
       Try exploring different batch sizes and see the effect on the train and test performance over each epoch.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391455)\
077. ![](https://secure.gravatar.com/avatar/a305fbbb522dadd8f41564e32395c07b1980e87f41a0046363abf35c95883b60?s=40&d=mm&r=g)\
\
\
\
     MohammadMarch 7, 2017 at 6:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391591 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason\
\
\
     Firstly, thanks for your great tutorials.\
\
\
     I am trying to classify computer networks packets using first 500 bytes of every packet to identify its protocol. I am trying to use 1d convolution. for simpler task,I just want to do binary classification and then tackle multilabel classification for 10 protocols. Here is my code but the accuracy which is like .63. how can I improve the performance? should I Use RNNs?\
\
\
     ########\
\
\
     model=Sequential()\
\
\
     model.add(Convolution1D(64,10,border\_mode=’valid’,\
\
\
     activation=’relu’,subsample\_length=1, input\_shape=(500, 1)))\
\
\
     #model.add(Convolution2D(32,5,5,border\_mode=’valid’,input\_shape=(1,28,28),))\
\
\
     model.add(MaxPooling1D(2))\
\
\
     model.add(Flatten())\
\
\
     model.add(Dense(200,activation=’relu’))\
\
\
     model.add(Dense(1,activation=’sigmoid’))\
\
\
     model.compile(loss=’binary\_crossentropy’,\
\
\
     optimizer=’adam’,metrics=\[‘accuracy’\])\
\
\
     model.fit(train\_set, y\_train,\
\
\
     batch\_size=250,\
\
\
     nb\_epoch=30,\
\
\
     show\_accuracy=True)\
\
\
     #x2= get\_activations(model, 0,xprim )\
\
\
     #score = model.evaluate(t, y\_test, show\_accuracy = True, verbose = 0)\
\
\
     #print(score\[0\])\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391591)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 7, 2017 at 9:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391618 "Direct link to this comment")\
\
\
\
\
\
       This post lists some ideas to try an lift performance:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391618)\
078. ![](https://secure.gravatar.com/avatar/95fc3646116a3bc3e242527fa4cdbc83ac17523fc389f6f6c9634f90bf4d7f4d?s=40&d=mm&r=g)\
\
\
\
     DamianoMarch 7, 2017 at 10:13 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391702 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, thank you so much for this awesome tutorial. I have just started with python and machine learning.\
\
\
     I am joking with the code doing few changes, for example i have changed..\
\
\
\
     this:\
\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(250, input\_dim=8, init=’uniform’, activation=’relu’))\
\
\
     model.add(Dense(200, init=’uniform’, activation=’relu’))\
\
\
     model.add(Dense(200, init=’uniform’, activation=’relu’))\
\
\
     model.add(Dense(1, init=’uniform’, activation=’sigmoid’))\
\
\
\
     and this:\
\
\
\
     model.fit(X, Y, nb\_epoch=250, batch\_size=10)\
\
\
\
     then i would like to pass some arrays for prediction so…\
\
\
\
     new\_input = numpy.array(\[\[3,88,58,11,54,24.8,267,22\],\[6,92,92,0,0,19.9,188,28\], \[10,101,76,48,180,32.9,171,63\], \[2,122,70,27,0,36.8,0.34,27\], \[5,121,72,23,112,26.2,245,30\]\])\
\
\
\
     predictions = model.predict(new\_input)\
\
\
     print predictions # \[1.0, 1.0, 1.0, 0.0, 1.0\]\
\
\
\
     is this correct? In this example i used the same series of training (that have 0 class), but i am getting wrong results. Only one array is correctly predicted.\
\
\
\
     Thank you so much!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391702)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 8, 2017 at 9:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391772 "Direct link to this comment")\
\
\
\
\
\
       Looks good. Perhaps you could try changing the configuration of your model to make it more skillful?\
\
\
\
       See this post:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-391772)\
079. ![](https://secure.gravatar.com/avatar/999b0108a4dc5d21aa79e9a2cb622a3729bac6a9f3c75ff9954461a77d4cc915?s=40&d=mm&r=g)\
\
\
\
     ANJIMarch 13, 2017 at 8:48 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392512 "Direct link to this comment")\
\
\
\
\
\
     hello sir,\
\
\
     could you please tell me to rectify my error below it is raised while model is training:\
\
\
\
     str(array.shape))\
\
\
     ValueError: Error when checking model input: expected convolution2d\_input\_1 to have 4 dimensions, but got array with shape (68, 28, 28).\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392512)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 14, 2017 at 8:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392580 "Direct link to this comment")\
\
\
\
\
\
       It looks like you are working with CNN, not related to this tutorial.\
\
\
\
       Consider trying this tutorial to get familiar with CNNs:\
\
       [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392580)\
080. ![](https://secure.gravatar.com/avatar/08573e7dfc6a78f0d9f83ea7baefbdb45a0bd7fb10b1b554e2ce02787e2b5542?s=40&d=mm&r=g)\
\
\
\
     RimjhimMarch 14, 2017 at 8:21 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392648 "Direct link to this comment")\
\
\
\
\
\
     I want a neural that can predict sin values. Further from a given data set i need to determine the function(for example if the data is of tan or cos, then how to determine that data is of tan only or cos only)\
\
\
\
     Thanks in advance\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392648)\
\
081. ![](https://secure.gravatar.com/avatar/7acb23ce5fd33ad95b33baea64d77933def1ceea3de2e75982180152f5724e93?s=40&d=mm&r=g)\
\
\
\
     SudarshanMarch 15, 2017 at 11:19 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392808 "Direct link to this comment")\
\
\
\
\
\
     Keras just updated to Keras 2.0. I have an updated version of this code here: [https://github.com/sudarshan85/keras-projects/tree/master/mlm/pima\_indians](https://github.com/sudarshan85/keras-projects/tree/master/mlm/pima_indians)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392808)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 16, 2017 at 7:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392855 "Direct link to this comment")\
\
\
\
\
\
       Nice work.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392855)\
082. ![](https://secure.gravatar.com/avatar/aae8c00df17f18a85e3bbfc46fd37eda9f085c37f847e78b09f161c45cbb9eb3?s=40&d=mm&r=g)\
\
\
\
     subhasishMarch 16, 2017 at 5:09 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392898 "Direct link to this comment")\
\
\
\
\
\
     hello sir,\
\
\
     can we use PSO (particle swarm optimisation) in this? if so can you tell how?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392898)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 17, 2017 at 8:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392979 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I don’t have an example of PSO for fitting neural network weights.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392979)\
083. ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
     Ananya MohapatraMarch 16, 2017 at 10:03 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392923 "Direct link to this comment")\
\
\
\
\
\
     hello sir,\
\
\
     what type of neural network is used in this code? as there are 3 types of Neural network that are… feedforward, radial basis function and recurrent neurak network.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392923)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 17, 2017 at 8:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392986 "Direct link to this comment")\
\
\
\
\
\
       A multilayer perceptron (MLP) neural network. A classic type from the 1980s.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392986)\
084. ![](https://secure.gravatar.com/avatar/9e241d74ea199842edf534c9d7bc8277d4969ddb770053f092b3f7c3b3a54c04?s=40&d=mm&r=g)\
\
\
\
     DiegoMarch 17, 2017 at 3:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392952 "Direct link to this comment")\
\
\
\
\
\
     got this error while compiling..\
\
\
\
     sigmoid\_cross\_entropy\_with\_logits() got an unexpected keyword argument ‘labels’\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392952)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 17, 2017 at 8:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392989 "Direct link to this comment")\
\
\
\
\
\
       Perhaps confirm that your libraries are all up to date (Keras, Theano or TensorFlow)?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-392989)\
085. ![](https://secure.gravatar.com/avatar/26b9f30ef07152eca410a8dd83927285f86408c1997aa70ea8a5d9cfc2623921?s=40&d=mm&r=g)\
\
\
\
     RohanMarch 20, 2017 at 5:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393356 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason!\
\
\
\
     I am trying to use two odd frames of a video to predict the even one. Thus I need to give two images as input to the network and get one image as output. Can you help me with the syntax for the first model.add()? I have X\_train of dimension (190, 2, 240, 320, 3) where 190 are the number of odd pairs, 2 are the two odd images, and (240,320,3) are the (height, width, depth) of each image.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393356)\
\
086. ![](https://secure.gravatar.com/avatar/58d3b0d5ef14658415b828a7e8621f51344494b0538c2722b5dbfc363be88b60?s=40&d=mm&r=g)\
\
\
\
     Herli MenezesMarch 21, 2017 at 8:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393508 "Direct link to this comment")\
\
\
\
\
\
     Hello, Jason,\
\
\
     Thanks for your good tutorial. However i found some issues:\
\
\
     Warnings like these:\
\
\
\
     1 – Warning (from warnings module):\
\
\
     File “/usr/lib/python2.7/site-packages/keras/legacy/interfaces.py”, line 86\
\
\
     ‘` call to the Keras 2 API: ' + signature)\
\
     UserWarning: Update your`Dense` call to the Keras 2 API:`Dense(12, activation=”relu”, kernel\_initializer=”uniform”, input\_dim=8)``\
\
     `\
     `\
\
     `2 - Warning (from warnings module):\
\
     File "/usr/lib/python2.7/site-packages/keras/legacy/interfaces.py", line 86\
\
         '` call to the Keras 2 API: ‘ + signature)\
\
\
     UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(8, activation="relu", kernel_initializer="uniform")`\
\
\
\
     3 – Warning (from warnings module):\
\
\
     File “/usr/lib/python2.7/site-packages/keras/legacy/interfaces.py”, line 86\
\
\
     ‘` call to the Keras 2 API: ' + signature)\
\
     UserWarning: Update your`Dense` call to the Keras 2 API:`Dense(1, activation=”sigmoid”, kernel\_initializer=”uniform”)``\
\
     `\
     `\
\
     `3 - Warning (from warnings module):\
\
     File "/usr/lib/python2.7/site-packages/keras/models.py", line 826\
\
         warnings.warn('The`nb\_epoch` argument in`fit` '\
\
     UserWarning: The`nb\_epoch` argument in`fit` has been renamed`epochs\`.\
\
\
\
     I think these are due to some package update..\
\
\
\
     But, the output of predictions was an array of zeros…\
\
\
     such as: \[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ….0.0\]\
\
\
\
     I am running in a Linux Machine, Fedora 24,\
\
\
     Python 2.7.13 (default, Jan 12 2017, 17:59:37)\
\
\
     \[GCC 6.3.1 20161221 (Red Hat 6.3.1-1)\] on linux2\
\
\
\
     Why?\
\
\
\
     Thank you!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393508)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 21, 2017 at 8:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393525 "Direct link to this comment")\
\
\
\
\
\
       These look like warnings related to the recent Keras 2.0 release.\
\
\
\
       They look like just warning and that you can still run the example.\
\
\
\
       I do not know why you are getting all zeros. I will investigate.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393525)\
087. ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
     Ananya MohapatraMarch 21, 2017 at 6:21 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393588 "Direct link to this comment")\
\
\
\
\
\
     hello sir,\
\
\
     can you please help me build a recurrent neural network with the above given dataset. i am having a bit trouble in building the layers…\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393588)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 22, 2017 at 7:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393702 "Direct link to this comment")\
\
\
\
\
\
       Hi Ananya ,\
\
\
\
       The Pima Indian diabetes dataset is a binary classification problem. It is not appropriate for a Recurrent Neural Network as there is no sequence information to learn.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393702)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
         Ananya MohapatraMarch 22, 2017 at 8:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393786 "Direct link to this comment")\
\
\
\
\
\
         sir so could you tell on which type of dataset would the recurrent neural network accurately work? i have the dataset of EEG signals of epileptic patients…will recurrent network work on this?\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393786)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 23, 2017 at 8:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393848 "Direct link to this comment")\
\
\
\
\
\
           It may if it is regular enough.\
\
\
\
           LSTMs are excellent at sequence problems that have regularity or clear signals to detect.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393848)\
088. ![](https://secure.gravatar.com/avatar/69bf185ea36e9e3f8365a4ba879c614df9f22903fcb47bf8a69bddb25b7d5e5b?s=40&d=mm&r=g)\
\
\
\
     ShaneMarch 22, 2017 at 5:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393686 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, I have a quick question related to an error I am receiving when running the code in the tutorial…\
\
\
\
     When I run\
\
\
\
     \# Compile model\
\
     `model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])`\
\
\
\
     Python returns the following error:\
\
\
\
     sigmoid\_cross\_entropy\_with\_logits() got an unexpected keyword argument ‘labels’\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393686)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 22, 2017 at 8:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393711 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I have not seen this error Shane.\
\
\
\
       Perhaps check that your environment is up to date with the latest versions of the deep learning libraries?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393711)\
089. ![](https://secure.gravatar.com/avatar/b5146599af402a55841df880305f246c6045a8310d0f3d3ee2f510f2675c0aa3?s=40&d=mm&r=g)\
\
\
\
     TejesMarch 24, 2017 at 1:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393926 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Thanks for this awesome post.\
\
\
     I ran your code with tensorflow back end, just out of curiosity. The accuracy returned was different every time I ran the code. That didn’t happen with Theano. Can you tell me why?\
\
\
\
     Thanks in advance!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393926)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 24, 2017 at 7:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393972 "Direct link to this comment")\
\
\
\
\
\
       You will get different accuracy each time you run the code because neural networks are stochastic.\
\
\
\
       This is not related to the backend (I expect).\
\
\
\
       More on randomness in machine learning here:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-393972)\
090. ![](https://secure.gravatar.com/avatar/b8ecdad7d11d8e1d92c6a3d290b73ae507829bd4823979b9b4bf8c21ab460aec?s=40&d=mm&r=g)\
\
\
\
     Saurabh BhagvatulaMarch 27, 2017 at 9:49 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394354 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     I’m new to deep learning and learning it from your tutorials, which previously helped me understand Machine Learning very well.\
\
\
     In the following code, I want to know why the number of neurons differ from input\_dim in first layer of Nueral Net.\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(12, input\_dim=8, init=’uniform’, activation=’relu’))\
\
\
     model.add(Dense(8, init=’uniform’, activation=’relu’))\
\
\
     model.add(Dense(1, init=’uniform’, activation=’sigmoid’))\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394354)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 28, 2017 at 8:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394406 "Direct link to this comment")\
\
\
\
\
\
       You can specify the number of inputs via “input\_dim”, you can specify the number of neurons in the first hidden layer as the first parameter to Dense().\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394406)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/b8ecdad7d11d8e1d92c6a3d290b73ae507829bd4823979b9b4bf8c21ab460aec?s=40&d=mm&r=g)\
\
\
\
         Saurabh BhagvatulaMarch 28, 2017 at 4:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394456 "Direct link to this comment")\
\
\
\
\
\
         Thanx a lot.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394456)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 29, 2017 at 9:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394555 "Direct link to this comment")\
\
\
\
\
\
           You’re welcome.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394555)\
091. ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
     NaliniMarch 29, 2017 at 2:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394521 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
\
     while running this code for k fold cross validation it is not working.please give the code for k fold cross validation in binary class\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394521)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 29, 2017 at 9:10 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394563 "Direct link to this comment")\
\
\
\
\
\
       Generally neural nets are too slow/large for k-fold cross validation.\
\
\
\
       Nevertheless, you can use a sklearn wrapper for a keras model and use it with any sklearn resampling method:\
\
       [https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/](https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394563)\
092. ![](https://secure.gravatar.com/avatar/a7f7d2b7908a31204944bc552f4fbb3c8f21da49518677e5acb7ddeb26bbbd3e?s=40&d=mm&r=g)\
\
\
\
     trangtruongMarch 29, 2017 at 7:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394626 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, why i use function evaluate to get accuracy score my model with test dataset, it return result >1, i can’t understand.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-394626)\
\
093. ![](https://secure.gravatar.com/avatar/b2f53223fe85f296a0765b37732b4f9f280b7ece005ab35bc74776ceda09dfc6?s=40&d=mm&r=g)\
\
\
\
     enixonApril 3, 2017 at 3:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395187 "Direct link to this comment")\
\
\
\
\
\
     Hey Jason, thanks for this great article! I get the following error when running the code above:\
\
\
\
     TypeError: Received unknown keyword arguments: {‘epochs’: 150}\
\
\
\
     Any ideas on why that might be? I can’t get ‘epochs’, nb\_epochs, etc to work…\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395187)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 4, 2017 at 9:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395324 "Direct link to this comment")\
\
\
\
\
\
       You need to update to Keras version 2.0 or higher.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395324)\
094. ![](https://secure.gravatar.com/avatar/947cc974c03e8216162839aa7eda73f546246a4868825f445c95bb2c7e4b421f?s=40&d=mm&r=g)\
\
\
\
     Ananya MohapatraApril 5, 2017 at 9:30 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395452 "Direct link to this comment")\
\
\
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
     model.add(Dense(10, input\_dim=25, init=’normal’, activation=’softplus’))\
\
\
     model.add(Dense(3, init=’normal’, activation=’softmax’))\
\
\
     # Compile model\
\
\
     model.compile(loss=’mean\_squared\_error’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     return model\
\
\
     sir here mean\_square\_error has been used for loss calculation. Is it the same as LMS algorithm. If not, can we use LMS , NLMS or RLS to calculate the loss?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395452)\
\
095. ![](https://secure.gravatar.com/avatar/71451fbdd83246c87ab9b0dd38768637964b721a98409b535dbbe47068c181c2?s=40&d=mm&r=g)\
\
\
\
     Ahmad HijaziApril 5, 2017 at 10:19 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395458 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason, thank you a lot for this example.\
\
\
\
     My question is, after I trained the model and an accuracy of 79.2% for example is obtained successfully, how can I test this model on new data?\
\
\
\
     for example if a new patient with new records appear, I want to guess the result (0 or 1) for him, how can I do that in the code?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395458)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 9, 2017 at 2:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395785 "Direct link to this comment")\
\
\
\
\
\
       You can fit your model on all available training data then make predictions on new data as follows:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395785)\
096. ![](https://secure.gravatar.com/avatar/71451fbdd83246c87ab9b0dd38768637964b721a98409b535dbbe47068c181c2?s=40&d=mm&r=g)\
\
\
\
     Perick FlausApril 6, 2017 at 12:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395463 "Direct link to this comment")\
\
\
\
\
\
     Thanks Jason, how can we test if new patient will be diabetic or no (0 or 1) ?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395463)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 9, 2017 at 2:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395786 "Direct link to this comment")\
\
\
\
\
\
       Fit the model on all training data and call:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-395786)\
097. ![](https://secure.gravatar.com/avatar/aa6d618de15439650a4b2a888ff033f43d205ab35e41906f8e2b92c44a6f1631?s=40&d=mm&r=g)\
\
\
\
     GangadharApril 12, 2017 at 1:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396129 "Direct link to this comment")\
\
\
\
\
\
     Dr Jason,\
\
\
\
     In compiling the model i got below error\
\
\
\
     TypeError: compile() got an unexpected keyword argument ‘metrics’\
\
\
\
     unable to resolve the below error\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396129)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 12, 2017 at 7:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396170 "Direct link to this comment")\
\
\
\
\
\
       Ensure you have the latest version of Keras, v2.0 or higher.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396170)\
098. ![](https://secure.gravatar.com/avatar/49edbbd906961ac5935f8d2a1d6cc1f404119bf6e8b31a95f3704e7ac7e68225?s=40&d=mm&r=g)\
\
\
\
     Omogbehin AzeezApril 13, 2017 at 1:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396247 "Direct link to this comment")\
\
\
\
\
\
     Hello sir,\
\
\
     Thank you for the post. A quick question, my dataset has 24 input and 1 binary output( 170 instances, 100 epoch , hidden layer=6 and 10 batch, kernel\_initializer=’normal’) . I adapted your code using Tensor flow and keras. I am having an accuracy of 98 to 100 percent. I am scared of over-fitting in my model. I need your candid advice. Kind regards sir\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396247)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 13, 2017 at 10:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396292 "Direct link to this comment")\
\
\
\
\
\
       Yes, evaluate your model using k-fold cross-validation to ensure you are not tricking yourself.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396292)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/49edbbd906961ac5935f8d2a1d6cc1f404119bf6e8b31a95f3704e7ac7e68225?s=40&d=mm&r=g)\
\
\
\
         Omogbehin AzeezApril 14, 2017 at 1:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396360 "Direct link to this comment")\
\
\
\
\
\
         Thank you sir\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396360)\
099. ![](https://secure.gravatar.com/avatar/8a264f271bcf3d2bed582d5552a9311b1081e8fc8ab21836edd3198f41fb6815?s=40&d=mm&r=g)\
\
\
\
     Sethu BakthaApril 13, 2017 at 5:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396263 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     If I want to use the diabetes dataset (NOT Pima) [https://archive.ics.uci.edu/ml/datasets/Diabetes](https://archive.ics.uci.edu/ml/datasets/Diabetes) to predict Blood Glucose which tutorials and e-books of yours would I need to start with…. Also, the data in its current format with time, code and value is it usable as is or do I need to convert the data in another format to be able to use it.\
\
\
\
     Thanks for your help\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396263)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 13, 2017 at 10:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396299 "Direct link to this comment")\
\
\
\
\
\
       This process will help you frame and work through your dataset:\
\
       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)\
\
\
\
       I hope that helps as a start.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396299)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/8a264f271bcf3d2bed582d5552a9311b1081e8fc8ab21836edd3198f41fb6815?s=40&d=mm&r=g)\
\
\
\
         Sethu BakthaApril 13, 2017 at 10:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396301 "Direct link to this comment")\
\
\
\
\
\
         Dr. Jason,\
\
\
         The data is time series(time based data) with categorical(20) with two numbers one for insulin level and another for blood sugar level… Each time series data does not have every categorical data… For example one category is blood sugar before breakfast, another category is blood sugar after breakfast, before lunch and after lunch… Some times some of these category data is missing… I read through the above link, but does not talk about time series, categorical data with some category of data missing what to do in those cases…. Please let me know if any of your books will help clarify these points?\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396301)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)April 14, 2017 at 8:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396387 "Direct link to this comment")\
\
\
\
\
\
           Hi Sethu,\
\
\
\
           I have many posts on time series that will help. Get started here:\
\
           [https://machinelearningmastery.com/start-here/#timeseries](https://machinelearningmastery.com/start-here/#timeseries)\
\
\
\
           With categorical data, I would recommend an integer encoding perhaps followed by a one-hot encoding. You can learn more about these encodings here:\
\
           [https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/](https://machinelearningmastery.com/data-preparation-gradient-boosting-xgboost-python/)\
\
\
\
           I hope that helps.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396387)\
100. ![](https://secure.gravatar.com/avatar/49edbbd906961ac5935f8d2a1d6cc1f404119bf6e8b31a95f3704e7ac7e68225?s=40&d=mm&r=g)\
\
\
\
     Omogbehin AzeezApril 14, 2017 at 9:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396401 "Direct link to this comment")\
\
\
\
\
\
     Hello sir,\
\
\
\
     Is it compulsory to normalize the data before using ANN model. I read it somewhere I which the author insisted that each attribute be comparable on the scale of \[0,1\] for a meaningful model. What is your take on that sir. Kind regards.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396401)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 15, 2017 at 9:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396495 "Direct link to this comment")\
\
\
\
\
\
       Yes. You must scale your data to the bounds of the activation used.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396495)\
101. ![](https://secure.gravatar.com/avatar/9a152e7dfc889b4f329403c64e71a54535129300e9b6d55656c4038cc966a1e8?s=40&d=mm&r=g)\
\
\
\
     shivaApril 14, 2017 at 10:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396403 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, You are simply awesome. I’m one of the many who got benefited from your book “machine learning mastery with python”. I’m working with a medical image classification problem. I have two classes of medical images (each class having 1000 images of 32\*32) to be worked upon by the convolutional neural networks. Could you guide me how to load this data to the keras dataset? Or how to use my data while following your simple steps? kindly help.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396403)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 15, 2017 at 9:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396496 "Direct link to this comment")\
\
\
\
\
\
       Load the data as numpy arrays and then you can use it with Keras.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396496)\
102. ![](https://secure.gravatar.com/avatar/49edbbd906961ac5935f8d2a1d6cc1f404119bf6e8b31a95f3704e7ac7e68225?s=40&d=mm&r=g)\
\
\
\
     Omogbehin AzeezApril 18, 2017 at 12:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396725 "Direct link to this comment")\
\
\
\
\
\
     Hello sir,\
\
\
\
     I adapted your code with the cross validation pipelined with ANN (Keras) for my model. It gave me 100% still. I got the data from UCI ( Chronic Kidney Disease). It was 400 instances, 24 input attributes and 1 binary attribute. When I removed the rows with missing data I was left with 170 instances. Is my dataset too small for (24 input layer, 24 hidden layer and 1 output layer ANN, using adam and kernel initializer as uniform )?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396725)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 18, 2017 at 8:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396781 "Direct link to this comment")\
\
\
\
\
\
       It is not too small.\
\
\
\
       Generally, the size of the training dataset really depends on how you intend to use the model.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396781)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/49edbbd906961ac5935f8d2a1d6cc1f404119bf6e8b31a95f3704e7ac7e68225?s=40&d=mm&r=g)\
\
\
\
         Omogbehin AzeezApril 18, 2017 at 11:10 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396843 "Direct link to this comment")\
\
\
\
\
\
         Thank you sir for the response, I guess I have to contend with the over-fitting of my model.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396843)\
103. ![](https://secure.gravatar.com/avatar/090a56d37a40637f3f47ff483b818d9b34c71363ad0507545f3255c6de331c14?s=40&d=mm&r=g)\
\
\
\
     Padmanabhan KrishnamurthyApril 19, 2017 at 6:26 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396916 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Great tutorial. Love the site 🙂\
\
\
     Just a quick query : why have you used adam as an optimizer over sgd? Moreover, when do we use sgd optimization, and what exactly does it involve?\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396916)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 20, 2017 at 9:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396985 "Direct link to this comment")\
\
\
\
\
\
       ADAM seems to consistently work well with little or no customization.\
\
\
\
       SGD requires configuration of at least the learning rate and momentum.\
\
\
\
       Try a few methods and use the one that works best for your problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-396985)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/090a56d37a40637f3f47ff483b818d9b34c71363ad0507545f3255c6de331c14?s=40&d=mm&r=g)\
\
\
\
         Padmanabhan KrishnamurthyApril 20, 2017 at 4:32 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397025 "Direct link to this comment")\
\
\
\
\
\
         Thanks 🙂\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397025)\
104. ![](https://secure.gravatar.com/avatar/49edbbd906961ac5935f8d2a1d6cc1f404119bf6e8b31a95f3704e7ac7e68225?s=40&d=mm&r=g)\
\
\
\
     Omogbehin AzeezApril 25, 2017 at 8:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397534 "Direct link to this comment")\
\
\
\
\
\
     Hello sir,\
\
\
\
     Good day sir, how can I get all the weights and biases of the keras ANN. Kind regards.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397534)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 26, 2017 at 6:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397626 "Direct link to this comment")\
\
\
\
\
\
       You can save the network weights, see this post:\
\
       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
       You can also use the API to access the weights directly.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397626)\
105. ![](https://secure.gravatar.com/avatar/9a152e7dfc889b4f329403c64e71a54535129300e9b6d55656c4038cc966a1e8?s=40&d=mm&r=g)\
\
\
\
     ShivaApril 27, 2017 at 5:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397717 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     I am currently working with the IMDB sentiment analysis problem as mentioned in your book. Am using Anaconda 3 with Python 3.5.2. In an attempt to summarize the review length as you have mentioned in your book, When i try to execute the command:\
\
\
\
     result = map(len, X)\
\
\
     print(“Mean %.2f words (%f)” % (numpy.mean(result), numpy.std(result)))\
\
\
\
     it returns the error: unsupported operand type(s) for /: ‘map’ and ‘int’\
\
\
\
     kindly help with the modified syntax. looking forward…\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397717)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 27, 2017 at 8:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397746 "Direct link to this comment")\
\
\
\
\
\
       I’m sorry to hear that. Perhaps comment out that line?\
\
\
       Or change it to remove the formatting and just print the raw mean and stdev values for you to review?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-397746)\
106. ![](https://secure.gravatar.com/avatar/c1fdefa6856de0e464df0399632323334357a1cb3a6e31b16df7ea7b19943aae?s=40&d=mm&r=g)\
\
\
\
     ElikplimMay 1, 2017 at 1:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398112 "Direct link to this comment")\
\
\
\
\
\
     Hello, quite new to Python, Numpy and Keras(background in PHP, MYSQL etc). If there are 8 input variables and 1 output varable(9 total), and the Array indexing starts from zero(from what I’ve gathered it’s a Numpy Array, which is built on Python lists) and the order is \[rows, columns\], then shouldn’t our input variable(X) be X = dataset\[:,0:7\] (where we select from the 1st to 8th columns, ie. 0th to 7th indices) and output variable(Y) be Y = dataset\[:,8\] (where we the 9th column, ie. 8th index)?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398112)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 1, 2017 at 5:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398138 "Direct link to this comment")\
\
\
\
\
\
       You can learn more about array indexing in numpy here:\
\
       [https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398138)\
107. ![](https://secure.gravatar.com/avatar/611c6f46655646ee33911a0082bd998b571534b08cd202c7217adb59f1e1ff4a?s=40&d=mm&r=g)\
\
\
\
     Jackie LeeMay 1, 2017 at 12:47 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398168 "Direct link to this comment")\
\
\
\
\
\
     I’m having troubles with the predictions part. It saves ValueError: Error when checking model input: expected dense\_1\_input to have shape (None, 502) but got array with shape (170464, 502)\
\
\
\
     \### MAKE PREDICTIONS ###\
\
\
     testset = numpy.loadtxt(“right\_stim\_FD1.csv”, delimiter=”,”)\
\
\
     A = testset\[:,0:502\]\
\
\
     B = testset\[:,502\]\
\
\
     probabilities = model.predict(A, batch\_size=10, verbose=1)\
\
\
     predictions = float(round(a) for a in probabilities)\
\
\
     accuracy = numpy.mean(predictions == B)\
\
\
     #round predictions\
\
\
     #rounded = \[round(x\[0\]) for x in predictions\]\
\
\
     print(predictions)\
\
\
     print(“Prediction Accuracy: %.2f%%” % (accuracy\*100))\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398168)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 2, 2017 at 5:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398223 "Direct link to this comment")\
\
\
\
\
\
       It looks like you might be giving the entire dataset as the output (y) rather than just the output variable.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398223)\
108. ![](https://secure.gravatar.com/avatar/e040a30b80801ceb1de4e76d569874c185b9cb51b985c4e14437001c81be3d5b?s=40&d=mm&r=g)\
\
\
\
     Anastasios SelalmazidisMay 2, 2017 at 12:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398196 "Direct link to this comment")\
\
\
\
\
\
     Hi there,\
\
\
\
     I have a question regarding deep learning. In this tutorial we build a MLP with Keras. Is this Deep Learning or is it just a MLP Backpropagation ?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398196)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 2, 2017 at 5:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398228 "Direct link to this comment")\
\
\
\
\
\
       Deep learning is MLP backprop these days:\
\
       [https://machinelearningmastery.com/what-is-deep-learning/](https://machinelearningmastery.com/what-is-deep-learning/)\
\
\
\
       Generally, deep learning refers to MLPs with lots of layers.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398228)\
109. ![](https://secure.gravatar.com/avatar/0c7d67d4a790ed23b09ba4efb28df724909e84e7e6333644c54442787b5363ae?s=40&d=mm&r=g)\
\
\
\
     Eric TMay 2, 2017 at 8:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398299 "Direct link to this comment")\
\
\
\
\
\
     Hi,\
\
\
     Would you mind if I use this code as an example of a simple network in a school project of mine?\
\
\
     Need to ask before using it, since I cannot find anywhere in this tutorial that you are OK with anyone using the code, and the ethics moment of my course requires me to ask (and of course give credit where credit is due).\
\
\
     Kind regards\
\
\
     Eric T\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398299)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 3, 2017 at 7:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398358 "Direct link to this comment")\
\
\
\
\
\
       Yes it’s fine but I take no responsibility and you must credit the source.\
\
\
\
       I answer this question in my FAQ:\
\
       [https://machinelearningmastery.com/start-here/#faq](https://machinelearningmastery.com/start-here/#faq)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398358)\
110. ![](https://secure.gravatar.com/avatar/1e8e638429975ce5d9b6290336009870f66dbdb6313466e3281fa72273a1737f?s=40&d=mm&r=g)\
\
\
\
     BinhLNMay 7, 2017 at 3:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398857 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason\
\
\
     I have a problem\
\
\
     My Dataset have 500 record. But My teacher want my dataset have 100.000 record. I must have a new algorithm for data generation. Please help me\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-398857)\
\
111. ![](https://secure.gravatar.com/avatar/6f6209c73eaac8dd297b6782d615e93c5d45fad34eb178d92ca65ab4d19ea4c0?s=40&d=mm&r=g)\
\
\
\
     DpMay 11, 2017 at 2:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399316 "Direct link to this comment")\
\
\
\
\
\
     Can you give a deep cnn code which includes 25 layers , in the first conv layer the filter sizs should be 39×39 woth a total lf 64 filters , in the 2nd conv layer , 21 ×21 with 32 filters , in the 3rd conv layer 11×11 with 64 filters , 4th Conv layer 7×7 with 32 layers . For a input size of image 256×256. Im Competely new in this Deep learning Thing but if you can code that for me it would be a great help. Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399316)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 11, 2017 at 8:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399359 "Direct link to this comment")\
\
\
\
\
\
       Consider using an off-the-shelf model like VGG:\
\
       [https://keras.io/applications/](https://keras.io/applications/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399359)\
112. ![](https://secure.gravatar.com/avatar/7ab8ce1ebab0a466d9c9ae7e0c17652b5f3492c0e7d718c58f702c2d2916a4e6?s=40&d=mm&r=g)\
\
\
\
     MapleMay 13, 2017 at 12:58 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399561 "Direct link to this comment")\
\
\
\
\
\
     I have to follow with the facebook metrics. But the result is very low. Help me.\
\
\
     I changed the input but did not improve\
\
     [http://archive.ics.uci.edu/ml/datasets/Facebook+metrics](http://archive.ics.uci.edu/ml/datasets/Facebook+metrics)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399561)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 14, 2017 at 7:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399621 "Direct link to this comment")\
\
\
\
\
\
       I have a list of suggestions that may help as a start:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399621)\
113. ![](https://secure.gravatar.com/avatar/adcc970ed10281a9247a7eef178f975cfd271267ede3a2ba8de8333c640d525a?s=40&d=mm&r=g)\
\
\
\
     AlessandroMay 14, 2017 at 1:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399595 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Great Tutorial and thanks for your effort.\
\
\
\
     I have a question, since I am beginner with keras and tensorflow.\
\
\
     I have installed both of them, keras and tensorflow, the latest version and I have run your example but I get always the same error:\
\
\
\
     Traceback (most recent call last):\
\
\
     File “CNN.py”, line 18, in\
\
\
     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     File “/Users/MacBookPro1/.virtualenvs/keras\_tf/lib/python2.7/site-packages/keras/models.py”, line 777, in compile\
\
\
     \*\*kwargs)\
\
\
     File “/Users/MacBookPro1/.virtualenvs/keras\_tf/lib/python2.7/site-packages/keras/engine/training.py”, line 910, in compile\
\
\
     sample\_weight, mask)\
\
\
     File “/Users/MacBookPro1/.virtualenvs/keras\_tf/lib/python2.7/site-packages/keras/engine/training.py”, line 436, in weighted\
\
\
     score\_array = fn(y\_true, y\_pred)\
\
\
     File “/Users/MacBookPro1/.virtualenvs/keras\_tf/lib/python2.7/site-packages/keras/losses.py”, line 51, in binary\_crossentropy\
\
\
     return K.mean(K.binary\_crossentropy(y\_pred, y\_true), axis=-1)\
\
\
     File “/Users/MacBookPro1/.virtualenvs/keras\_tf/lib/python2.7/site-packages/keras/backend/tensorflow\_backend.py”, line 2771, in binary\_crossentropy\
\
\
     logits=output)\
\
\
     TypeError: sigmoid\_cross\_entropy\_with\_logits() got an unexpected keyword argument ‘labels’\
\
\
\
     Could you help? Thanks\
\
\
\
     Alessandro\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399595)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)May 14, 2017 at 7:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399628 "Direct link to this comment")\
\
\
\
\
\
       Ouch, I have not seen this error before.\
\
\
\
       Some ideas:\
\
\
       – Consider trying the theano backend and see if that makes a difference.\
\
\
       – Try searching/posting on the keras user group and slack channel.\
\
\
       – Try searching/posting on stackoverflow or cross validated.\
\
\
\
       Let me know how you go.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399628)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/adcc970ed10281a9247a7eef178f975cfd271267ede3a2ba8de8333c640d525a?s=40&d=mm&r=g)\
\
\
\
         AlessandroMay 14, 2017 at 9:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399643 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
\
         I found the issue. The tensorflow installation was outdated; so I have updated it and everything\
\
\
         is working nicely.\
\
\
\
         Good night,\
\
\
         Alessandro\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399643)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)May 15, 2017 at 5:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399680 "Direct link to this comment")\
\
\
\
\
\
           I’m glad to hear it Alessandro.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-399680)\
114. ![](https://secure.gravatar.com/avatar/0cc09d946f2f3dc267b2df57be03bcfaeae469b2bff417f1347b9f94a2770ef9?s=40&d=mm&r=g)\
\
\
\
     Sheikh Rafiul IslamMay 25, 2017 at 3:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-400500 "Direct link to this comment")\
\
\
\
\
\
     Thank you Mr. Brownlee for your wonderful easy to understand explanation\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-400500)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2017 at 11:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401122 "Direct link to this comment")\
\
\
\
\
\
       Thnaks.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401122)\
115. ![](https://secure.gravatar.com/avatar/c871cf91c0419f81e1394e56412e4f12ba743a33a638e36fe5f4608abdb23390?s=40&d=mm&r=g)\
\
\
\
     WAZEDMay 29, 2017 at 12:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-400698 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Thank you very much for your wonderful tutorial. I have a question regarding the metrices.Is there default way to declare metrices “Precision” and “Recall” in addtion with the “Accurace”.\
\
\
\
     Br\
\
\
     WAZED\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-400698)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2017 at 12:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401168 "Direct link to this comment")\
\
\
\
\
\
       Yes, see here:\
\
       [https://keras.io/metrics/](https://keras.io/metrics/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401168)\
116. ![](https://secure.gravatar.com/avatar/05fd091766ce837cfe253980d97567bd5e255ac84f15cce437f1cd8b9d4850c4?s=40&d=mm&r=g)\
\
\
\
     chiranjib konwarMay 29, 2017 at 4:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-400709 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     please send me a small note containing resources from where i can learn deep learning from scratch. thanks for the wonderful read you had prepared.\
\
\
\
     Thanks in advance\
\
\
\
     yes, my email id is [chiranjib.konwar@gmail.com](mailto:chiranjib.konwar@gmail.com)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-400709)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2017 at 12:16 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401171 "Direct link to this comment")\
\
\
\
\
\
       Here:\
\
       [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401171)\
117. ![](https://secure.gravatar.com/avatar/e7644bf9f17e29550f9c831e3785bccb2ada2cf16d0fc885fcaa151711113db6?s=40&d=mm&r=g)\
\
\
\
     JeffJune 1, 2017 at 11:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401008 "Direct link to this comment")\
\
\
\
\
\
     Why the NN have mistakes many times?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401008)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2017 at 12:54 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401228 "Direct link to this comment")\
\
\
\
\
\
       What do you mean exactly?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401228)\
118. ![](https://secure.gravatar.com/avatar/cc4bc25b37d1040714778d80b888d3169a1652932b8603cc2be5c014cfa85301?s=40&d=mm&r=g)\
\
\
\
     kevinJune 2, 2017 at 5:53 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401267 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I seem to be getting an error when applying the fit method:\
\
\
\
     ValueError: Error when checking input: expected dense\_1\_input to have shape (None, 12) but got array with shape (767, 8)\
\
\
\
     I looked this up and the most prominent suggestion seemed to be upgrade keras and theno, which I did, but that didn’t resolve the problem.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401267)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 3, 2017 at 7:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401314 "Direct link to this comment")\
\
\
\
\
\
       Ensure you have copied the code exactly from the post.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401314)\
119. ![](https://secure.gravatar.com/avatar/549dc9183c8d7a7e46ce30089a42d3105f5f551894c9db19115cd78e40b7111c?s=40&d=mm&r=g)\
\
\
\
     Hemanth Kumar KJune 3, 2017 at 2:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401338 "Direct link to this comment")\
\
\
\
\
\
     hi Jason,\
\
\
     I am stuck with an error\
\
\
     TypeError: sigmoid\_cross\_entropy\_with\_logits() got an unexpected keyword argument ‘labels’\
\
\
     my tensor flow and keras virsions are\
\
\
     keras: 2.0.4\
\
\
     Tensorflow: 0.12\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401338)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 4, 2017 at 7:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401406 "Direct link to this comment")\
\
\
\
\
\
       I’m sorry to hear that, I have not seen that error before. Perhaps you could post a question to stackoverflow or the keras user group?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401406)\
120. ![](https://secure.gravatar.com/avatar/9fd1a27c28ef4ece45a56182585cd572a7a16ed01be01beab81d908a2fee907a?s=40&d=mm&r=g)\
\
\
\
     xenaJune 4, 2017 at 6:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401461 "Direct link to this comment")\
\
\
\
\
\
     can anyone tell me which neural network is being used here? Is it MLP??\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401461)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 5, 2017 at 7:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401518 "Direct link to this comment")\
\
\
\
\
\
       Yes, it is a multilayer perceptron (MLP) feedforward neural network.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-401518)\
121. ![](https://secure.gravatar.com/avatar/efab77f1147cb4effb865bb51d62b21a7d1b7ce4f24e3d97417e0473663fb7b7?s=40&d=mm&r=g)\
\
\
\
     Nirmesh ShahJune 9, 2017 at 11:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402024 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I have run this code successfully on PC with CPU.\
\
\
\
     If I have to run the same code n another PC which contains GPU, What line should I add to make it sure that it runs on the GPU\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402024)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 10, 2017 at 8:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402078 "Direct link to this comment")\
\
\
\
\
\
       The code would stay the same, your configuration of the Keras backend would change.\
\
\
\
       Please refer to TensorFlow or Theano documentation.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402078)\
122. ![](https://secure.gravatar.com/avatar/f73284bf7e1b5c74697c278ddc4d9fdffdb4ae1d38f16ad81e33bffa2d58a5ef?s=40&d=mm&r=g)\
\
\
\
     PrachiJune 12, 2017 at 7:30 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402284 "Direct link to this comment")\
\
\
\
\
\
     What if I want to train my neural which should detect whether the luggage is abandoned or not ? How do i proceed for it ?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402284)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 13, 2017 at 8:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402346 "Direct link to this comment")\
\
\
\
\
\
       This process will help you work through your predictive modeling problem end to end:\
\
       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402346)\
123. ![](https://secure.gravatar.com/avatar/0a3e6a0744018a0a62f365878497d8d382b46d499e221afe68dcbd59dfa8c416?s=40&d=mm&r=g)\
\
\
\
     EbtesamJune 14, 2017 at 11:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402529 "Direct link to this comment")\
\
\
\
\
\
     Hi\
\
\
     I was build neural machine translation model but the score i was get is 0 i am not sure why\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402529)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 15, 2017 at 8:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402577 "Direct link to this comment")\
\
\
\
\
\
       Here is a good list of things to try:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-402577)\
124. ![](https://secure.gravatar.com/avatar/fe694c7d34cc1983578bbacdce40e837e5aad31a540b6ba4699ab2bb45b12f21?s=40&d=mm&r=g)\
\
\
\
     Sarvottam PatelJune 20, 2017 at 7:31 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403198 "Direct link to this comment")\
\
\
\
\
\
     HHey Jason , first of all thank you very much from the core of my heart to make me understand this perfectly, I have an error after completing 150 iteration.\
\
\
\
     File “keras\_first\_network.py”, line 53, in\
\
\
     print(“\\n%s: %.2f” %(model.metrics\_names\[1\]\*100))\
\
\
     TypeError: not enough arguments for format string\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403198)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/fe694c7d34cc1983578bbacdce40e837e5aad31a540b6ba4699ab2bb45b12f21?s=40&d=mm&r=g)\
\
\
\
       Sarvottam PatelJune 20, 2017 at 8:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403203 "Direct link to this comment")\
\
\
\
\
\
       Sorry Sir my bad , actually I wrote it wrongly\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403203)\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)June 21, 2017 at 8:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403262 "Direct link to this comment")\
\
\
\
\
\
       Confirm that you have copied the line exactly:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
       | 1 | print("\\n%s: %.2f%%"%(model.metrics\_names\[1\],scores\[1\]\*100)) |\
\
\
\
\
\
\
\
\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-403262)\
125. ![](https://secure.gravatar.com/avatar/261463914c375ad8417bcbd9c9f778054372745fa57c4ee79ae370a8c6d47404?s=40&d=mm&r=g)\
\
\
\
     JoydeepJune 30, 2017 at 4:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404339 "Direct link to this comment")\
\
\
\
\
\
     Hi Dr Jason,\
\
\
\
     Thanks for the tutorial to get started using Keras.\
\
\
\
     I used the below snippet to directly load the dataset from the URL rather than downloading and saving as this makes the code more streamlined without having to navigate elsewhere.\
\
\
\
     \# load pima indians dataset\
\
\
     datasource = numpy.DataSource().open(“http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data”)\
\
\
     dataset = numpy.loadtxt(datasource, delimiter=”,”)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404339)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 1, 2017 at 6:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404408 "Direct link to this comment")\
\
\
\
\
\
       Thanks for the tip.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-404408)\
126. ![](https://secure.gravatar.com/avatar/b0b69bdd157210d40ee7b6537b1725256be5799d141d687264df7200d0a366b0?s=40&d=mm&r=g)\
\
\
\
     YvetteJuly 7, 2017 at 9:01 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405032 "Direct link to this comment")\
\
\
\
\
\
     Thanks for this helpful resource!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405032)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 9, 2017 at 10:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405248 "Direct link to this comment")\
\
\
\
\
\
       I’m glad it helped.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405248)\
127. ![](https://secure.gravatar.com/avatar/372a8293c4398b781308f2e4c1372cb06cb68a6a891e23a68988611f7243b21a?s=40&d=mm&r=g)\
\
\
\
     AndeepJuly 10, 2017 at 1:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405326 "Direct link to this comment")\
\
\
\
\
\
     Hi Dr Brownlee,\
\
\
\
     thank you very much for this great tutorial!\
\
\
     I would be grateful, if you could answer some questions:\
\
\
\
     1\. What does the 7 in “numpy.random.seed(7)” means?\
\
\
\
     2\. In my case I have 3 input neurons and 2 output neurons. Is the correct notation:\
\
\
     X = dataset\[:,0:3\]\
\
\
     Y = dataset\[:,3:4\] ?\
\
\
\
     3\. The batch size means how many training data are used in one epoch, am I right?\
\
\
     I have thought we have to use the whole training data set for the training. In this case I would determine the batch size as the number of training data pairs I have achieved through experiments etc.. In your example, does the batch (sized 10) means that the computer always uses the same 10 training data in every epoch or are the 10 training data randomly chosen among all training data before every epoch?\
\
\
\
     4\. When evaluating the model what does the loss means (e.g. in loss: 0.5105 – acc: 0.7396)?\
\
\
     Is it the sum of values of the error function (e.g. mean\_squared\_error) of the output neurons?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405326)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2017 at 10:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405465 "Direct link to this comment")\
\
\
\
\
\
       You can use any random seed you like, more here:\
\
       [https://machinelearningmastery.com/reproducible-results-neural-networks-keras/](https://machinelearningmastery.com/reproducible-results-neural-networks-keras/)\
\
\
\
       You are referring to the columns in your data. Your network will also need to be configured with the correct number of inputs and outputs (e.g. input and output layers).\
\
\
\
       Batch size is the number of samples in the dataset to work through before updating network weights. One epoch is comprised of one or more batches.\
\
\
\
       Loss is the term being optimized by the network. Here we use log loss:\
\
       [https://en.wikipedia.org/wiki/Cross\_entropy](https://en.wikipedia.org/wiki/Cross_entropy)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405465)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/372a8293c4398b781308f2e4c1372cb06cb68a6a891e23a68988611f7243b21a?s=40&d=mm&r=g)\
\
\
\
         AndeepJuly 16, 2017 at 7:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406077 "Direct link to this comment")\
\
\
\
\
\
         Thank you for your response, Dr Brownlee !!\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406077)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)July 16, 2017 at 8:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406086 "Direct link to this comment")\
\
\
\
\
\
           I hope it helps.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406086)\
128. ![](https://secure.gravatar.com/avatar/1735717d63c3ddada9381393b0c6fcb930957f2619f681740db884a1f47399cb?s=40&d=mm&r=g)\
\
\
\
     Patrick ZawadzkiJuly 11, 2017 at 5:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405445 "Direct link to this comment")\
\
\
\
\
\
     Is there anyway to see the relationship between these inputs? Essentially understand which inputs affect the output the most, or perhaps which pairs of inputs affect the output the most?\
\
\
\
     Maybe pairing this with unsupervised deep learning? I want to have less of a “black box” for the developed network if at all possible. Thank you for your great content!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405445)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2017 at 10:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405486 "Direct link to this comment")\
\
\
\
\
\
       Yes, try and RFE:\
\
       [https://machinelearningmastery.com/feature-selection-machine-learning-python/](https://machinelearningmastery.com/feature-selection-machine-learning-python/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405486)\
129. ![](https://secure.gravatar.com/avatar/6c9da196b48bdbd7741fed295927aafca32fe696c85029c02c9f8dfd665ca530?s=40&d=mm&r=g)\
\
\
\
     BerntJuly 13, 2017 at 10:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405783 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Thank you for sharing your skills and competence.\
\
\
\
     I want to study the change in weights and predictions between each epoch run.\
\
\
     Have tried to use the model.train\_on\_batch method and the model.fit method with epoch=1 and batch\_size equal all the samples.\
\
\
\
     But it seems like the model doesn’t save the new updated weights.\
\
\
     I print predictions before and after I dont see a change in the evaluation scores.\
\
\
\
     Parts of the code is printed below.\
\
\
\
     Any idea?\
\
\
     Thanks.\
\
\
\
     \# Compile model\
\
\
     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
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
     \# Run one update of the model trained run with X and compared with Y\
\
\
     model.train\_on\_batch(X, Y)\
\
\
\
     \# Fit the model\
\
\
     model.fit(X, Y, epochs=1, batch\_size=768)\
\
\
\
     scores = model.evaluate(X, Y)\
\
\
     print(“\\n%s: %.2f%%” % (model.metrics\_names\[1\], scores\[1\]\*100))\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405783)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 14, 2017 at 8:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405844 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I have not explored evaluating a Keras model this way.\
\
\
\
       Perhaps it is a fault, I would recommend preparing the smallest possible example that demonstrates the issue and post to the Keras GitHub issues.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-405844)\
130. ![](https://secure.gravatar.com/avatar/e15d449b59597b8a5875d2485a4a76d20ee4b37b8e3efbeae3be89a3159b9cde?s=40&d=mm&r=g)\
\
\
\
     imanJuly 18, 2017 at 11:18 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406391 "Direct link to this comment")\
\
\
\
\
\
     Hi, I tried to apply this to the titanic data set, however the predictions were all 0.4. What do you suggest for:\
\
\
     \# create model\
\
\
     model = Sequential()\
\
\
     model.add(Dense(12, input\_dim=4, activation=’relu’))\
\
\
     model.add(Dense(4, activation=’relu’))\
\
\
     model.add(Dense(1, activation=’sigmoid’))\
\
\
\
     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\]) #’sgd’\
\
\
\
     model.fit(X, Y, epochs=15, batch\_size=10)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406391)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 19, 2017 at 8:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406440 "Direct link to this comment")\
\
\
\
\
\
       This post will give you some ideas to list the skill of your model:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406440)\
131. ![](https://secure.gravatar.com/avatar/999bef3ec34f9bacfe5747c04f2882d12071e41ea156c5e6cf4769f507805471?s=40&d=mm&r=g)\
\
\
\
     CamusJuly 19, 2017 at 2:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406403 "Direct link to this comment")\
\
\
\
\
\
     Hi Dr Jason,\
\
\
     This is probably a stupid question but I cannot find out how to do it … and I am beginner on Neural Network.\
\
\
     I have relatively same number of inputs (7) and one output. This output can take numbers between -3000 and +3000.\
\
\
     I want to build a neural network model in python but I don’t know how to do it.\
\
\
     Do you have an example with outputs different from 0-1.\
\
\
     Tanks in advance\
\
\
\
     Camus\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406403)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 19, 2017 at 8:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406444 "Direct link to this comment")\
\
\
\
\
\
       Ensure you scale your data then use the above tutorial to get started.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406444)\
132. ![](https://secure.gravatar.com/avatar/91f0a2102c306b662c51fc5b4f733cab3fa9c88fc8d3c7c792d1ca39aec0e0b3?s=40&d=mm&r=g)\
\
\
\
     Khalid HussainJuly 21, 2017 at 11:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406782 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason Brownlee\
\
\
\
     I am using the same data “pima-indians-diabetes.csv” but all predicted values are less then 1 and are in fraction which could not distinguish any class.\
\
\
\
     If I round off then all become 0.\
\
\
\
     I am using model.predict(x) function\
\
\
\
     You are requested to kindly guide me what I am doing wrong are how can I achieve correct predicted value.\
\
\
\
     Thank you\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406782)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 22, 2017 at 8:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406834 "Direct link to this comment")\
\
\
\
\
\
       Consider you have copied all of the code exactly from the tutorial.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-406834)\
133. ![](https://secure.gravatar.com/avatar/a46f1c0053576cb878b4a55773c5e6251a9845a4af100aca036199aa992f50d7?s=40&d=mm&r=g)\
\
\
\
     LudoJuly 25, 2017 at 6:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407262 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     Thanks you for your great example. I have some comments.\
\
\
\
     – Why you have choice “12” inputs hidden layers ? and not 24 / 32 .. it’s arbitary ?\
\
\
     – Same question about epochs and batch\_size ?\
\
\
\
     This value are very sensible !! i have try with 32 inputs first layer , epchos=500 and batch\_size=1000 and the result is very differents… i’am at 65% accurancy.\
\
\
\
     Thx for you help.\
\
\
     Regards.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407262)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 26, 2017 at 7:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407316 "Direct link to this comment")\
\
\
\
\
\
       Yes, it is arbitrary. Tune the parameters of the model to your problem.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407316)\
134. ![](https://secure.gravatar.com/avatar/586c81bad8a6b6d72b040b53b8b05ce455deb39a0fe792266ed2096f11b9c352?s=40&d=mm&r=g)\
\
\
\
     Almoutasem Bellah RajabJuly 25, 2017 at 7:32 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407266 "Direct link to this comment")\
\
\
\
\
\
     Wow, you’re still replying to comments more than a year later!!!… you’re great,, thanks..\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407266)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 26, 2017 at 7:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407317 "Direct link to this comment")\
\
\
\
\
\
       Yep.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407317)\
135. ![](https://secure.gravatar.com/avatar/85334a979cb909cd2e897f0e1e7dc7269806c4ee0c5878501d3ed16261cf3893?s=40&d=mm&r=g)\
\
\
\
     JaneJuly 26, 2017 at 1:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407294 "Direct link to this comment")\
\
\
\
\
\
     Thanks for your tutorial, I found it very useful to get me started with Keras. I’ve previously tried TensorFlow, but found it very difficult to work with. I do have a question for you though. I have both Theano and TensorFlow installed, how do I know which back-end Keras is using? Thanks again\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407294)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 26, 2017 at 8:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407334 "Direct link to this comment")\
\
\
\
\
\
       Keras will print which backend it uses every time you run your code.\
\
\
\
       You can change the backend in the Keras configuration file (~/.keras/keras.json) which looks like:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
       | 1<br>2<br>3<br>4<br>5<br>6 | {<br>"image\_data\_format":"channels\_last",<br>"backend":"tensorflow",<br>"epsilon":1e-07,<br>"floatx":"float32"<br>} |\
\
\
\
\
\
\
\
\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407334)\
136. ![](https://secure.gravatar.com/avatar/3f83888d5f63dac0ae17a7a9a1561f9a2287918dc3fe611bd08f26e711a52c12?s=40&d=mm&r=g)\
\
\
\
     Masood ImranJuly 28, 2017 at 12:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407561 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
\
     My understanding of Machine Learning or evaluating deep learning models is almost 0. But, this article gives me lot of information. It is explained in a simple and easy to understand language.\
\
\
\
     Thank you very much for this article. Would you suggest any good read to further explore Machine Learning or deep learning models please?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407561)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)July 28, 2017 at 8:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407621 "Direct link to this comment")\
\
\
\
\
\
       Thanks.\
\
\
\
       Yes, start right here:\
\
       [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-407621)\
137. ![](https://secure.gravatar.com/avatar/ad1563157377ec3ddead4049d908d936c7b3c953907f89b07fbb06fa9943136d?s=40&d=mm&r=g)\
\
\
\
     PeggyAugust 3, 2017 at 7:14 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408414 "Direct link to this comment")\
\
\
\
\
\
     If I have trained prediction models or neural network function scripts. How can I use them to make predictions in an application that will be used by end users? I want to use python but it seems I will have to redo the training in Python again. Is there a way I can rewrite the scripts in Python without retraining and just call the function of predicting?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408414)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 4, 2017 at 6:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408470 "Direct link to this comment")\
\
\
\
\
\
       You need to train and save the final model then load it to make predictions.\
\
\
\
       This post will make it clear:\
\
       [https://machinelearningmastery.com/train-final-machine-learning-model/](https://machinelearningmastery.com/train-final-machine-learning-model/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408470)\
138. ![](https://secure.gravatar.com/avatar/1b1770ae111d7290de51f586c15d9d17899b89ecaafd324ca23e6b6ebeb9f47b?s=40&d=mm&r=g)\
\
\
\
     ShaneAugust 8, 2017 at 2:38 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408936 "Direct link to this comment")\
\
\
\
\
\
     Jason, I used your tutorial to install everything needed to run this tutorial. I followed your tutorial and ran the resulting program successfully. Can you please describe what the output means? I would like to thank you for your very informative tutorials.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408936)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/1b1770ae111d7290de51f586c15d9d17899b89ecaafd324ca23e6b6ebeb9f47b?s=40&d=mm&r=g)\
\
\
\
       ShaneAugust 8, 2017 at 2:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408937 "Direct link to this comment")\
\
\
\
\
\
       768/768 \[==============================\] – 0s – loss: 0.4807 – acc: 0.7826\
\
\
       Epoch 148/150\
\
\
       768/768 \[==============================\] – 0s – loss: 0.4686 – acc: 0.7812\
\
\
       Epoch 149/150\
\
\
       768/768 \[==============================\] – 0s – loss: 0.4718 – acc: 0.7617\
\
\
       Epoch 150/150\
\
\
       768/768 \[==============================\] – 0s – loss: 0.4772 – acc: 0.7812\
\
\
       32/768 \[>………………………..\] – ETA: 0s\
\
\
       acc: 77.99%\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408937)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)August 8, 2017 at 5:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408959 "Direct link to this comment")\
\
\
\
\
\
         It is summarizing the training of the model.\
\
\
\
         The final line evaluates the accuracy of the model’s predictions – really just to demonstrate how to make predictions.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408959)\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 8, 2017 at 5:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408958 "Direct link to this comment")\
\
\
\
\
\
       Well done Shane.\
\
\
\
       Which output?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-408958)\
139. ![](https://secure.gravatar.com/avatar/1fdd07c9f405e6aefaef4582889c879d793405c710b9ae0273645b40efe476ef?s=40&d=mm&r=g)\
\
\
\
     BeneAugust 9, 2017 at 1:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409000 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason, i really liked your Work and it helped me a lot with my first steps.\
\
\
\
     But i am not really familiar with the numpy stuff:\
\
\
\
     So here is my Question:\
\
\
\
     dataset = numpy.loadtxt(“pima-indians-diabetes.csv”, delimiter=”,”)\
\
\
     \# split into input (X) and output (Y) variables\
\
\
     X = dataset\[:,0:8\]\
\
\
     Y = dataset\[:,8\]\
\
\
\
     I get that the numpy.loadtxt is extracting the information from the cvs File\
\
\
\
     but what does the stuff in the Brackets mean like X = dataset\[:,0:8\]\
\
\
\
     why the “:” and why , 0:8\
\
\
\
     its probably pretty dumb but i can’t find a good explanation online 😀\
\
\
\
     thanks really much!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409000)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 9, 2017 at 6:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409042 "Direct link to this comment")\
\
\
\
\
\
       Good question Bene, it’s called array slicing:\
\
       [https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409042)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/1fdd07c9f405e6aefaef4582889c879d793405c710b9ae0273645b40efe476ef?s=40&d=mm&r=g)\
\
\
\
         BeneAugust 9, 2017 at 10:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409113 "Direct link to this comment")\
\
\
\
\
\
         That helped me out tank you Jason 🙂\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409113)\
140. ![](https://secure.gravatar.com/avatar/6c808020a97cd861b4d35190c705320d38a8a65e2c696c6d7c7cc8703f708ac0?s=40&d=mm&r=g)\
\
\
\
     ChenAugust 12, 2017 at 5:43 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409406 "Direct link to this comment")\
\
\
\
\
\
     Can I translate it to Chinese and put it to Internet in order to let other Chinese people can read your article?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409406)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 13, 2017 at 9:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409463 "Direct link to this comment")\
\
\
\
\
\
       No, please do not.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409463)\
141. ![](https://secure.gravatar.com/avatar/5230aa3cec9ee9c1f6332e5046c29f19311f7af156a32d0fc4ad7748384eec79?s=40&d=mm&r=g)\
\
\
\
     [Deep Learning](http://autonom.io/)August 12, 2017 at 7:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409412 "Direct link to this comment")\
\
\
\
\
\
     It seems that using this line:\
\
\
\
     np.random.seed(5)\
\
\
\
     …is redundant i.e. the Keras output in a loop running the same model with the same configuration will yield a similar variety of results regardless if it’s set at all, or which number it is set to. Or am I missing something?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409412)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 13, 2017 at 9:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409465 "Direct link to this comment")\
\
\
\
\
\
       Deep learning algorithms are stochastic (random within a range). That means that they will make different predictions/learn different things when the same model is trained on the same data. This is a feature:\
\
       [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)\
\
\
\
       You can fix the random seed to ensure you get the same result, and it is a good idea for tutorials to help beginners out:\
\
       [https://machinelearningmastery.com/reproducible-results-neural-networks-keras/](https://machinelearningmastery.com/reproducible-results-neural-networks-keras/)\
\
\
\
       When evaluating the skill of a model, I would recommend repeating the experiment n times and taking skill as the average of the runs. See here for the procedure:\
\
       [https://machinelearningmastery.com/evaluate-skill-deep-learning-models/](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)\
\
\
\
       Does that help?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409465)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5230aa3cec9ee9c1f6332e5046c29f19311f7af156a32d0fc4ad7748384eec79?s=40&d=mm&r=g)\
\
\
\
         [Deep Learning](http://autonom.io/)August 14, 2017 at 3:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409526 "Direct link to this comment")\
\
\
\
\
\
         Thanks Jason 🙂\
\
\
\
         I totally get what it should do, but as I had pointed out, it does not do it. If you run the codes you have provided above in a loop for say 10 times. First 10 with random seed set and the other 10 times without that line of code all together. Then compare the result. At least the result I’m getting, is suggesting the effect is not there i.e. both sets of 10 times will have similar variation in the result.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409526)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)August 14, 2017 at 6:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409536 "Direct link to this comment")\
\
\
\
\
\
           It may suggest that the model is overprescribed and easily addresses the training data.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409536)\
142. ![](https://secure.gravatar.com/avatar/5230aa3cec9ee9c1f6332e5046c29f19311f7af156a32d0fc4ad7748384eec79?s=40&d=mm&r=g)\
\
\
\
     [Deep Learning](http://autonom.io/)August 14, 2017 at 3:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409528 "Direct link to this comment")\
\
\
\
\
\
     Nice post by the way > [https://machinelearningmastery.com/evaluate-skill-deep-learning-models/](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)\
\
\
\
     Thanks for sharing it. Been lately thinking about the aspect of accuracy a lot, it seems that at the moment it’s a “hot mess” in terms of the way common tools do it out of the box. I think a lot of non PhD / non expert crowd (most people) will at least initially be easily confused and make the kinds of mistakes you point out in your post.\
\
\
\
     Thanks for all the amazing contributions you are making in this field!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409528)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 14, 2017 at 6:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409537 "Direct link to this comment")\
\
\
\
\
\
       I’m glad it helped.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409537)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/a76cd3194294ac9f677a5001d6f725bd76af06c7c5264eb273b8a084bf108b7f?s=40&d=mm&r=g)\
\
\
\
         HaneeshDecember 7, 2019 at 10:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514080 "Direct link to this comment")\
\
\
\
\
\
         Hi Jason,\
\
\
\
         i’m actually trying to find “spam filter for quora questions” where i have a dataset with label-0’s and 1’s and questions columns. please let me know the approach and path to build a model for this.\
\
\
\
         Thanks\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514080)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)December 8, 2019 at 6:10 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514151 "Direct link to this comment")\
\
\
\
\
\
           Sounds like a great project.\
\
\
\
           The tutorials here on text classification will help:\
\
           [https://machinelearningmastery.com/start-here/#nlp](https://machinelearningmastery.com/start-here/#nlp)\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-514151)\
143. ![](https://secure.gravatar.com/avatar/6879eb6d2a76b9add8ea245fc48e24cfed3bb3e9f93d06ce31337952df60d5b4?s=40&d=mm&r=g)\
\
\
\
     RATNA NITIN PATILAugust 14, 2017 at 8:16 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409584 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason, Thanks for a wonderful tutorial.\
\
\
     Can I use Genetic Algorithm for feature selection??\
\
\
     If yes, Could you please provide the link for it???\
\
\
     Thanks in advance.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409584)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2017 at 6:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409631 "Direct link to this comment")\
\
\
\
\
\
       Sure. Sorry, I don’t have any examples.\
\
\
\
       Generally, computers are so fast it might be easier to test all combinations in an exhaustive search.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409631)\
144. ![](https://secure.gravatar.com/avatar/e738a601cda74e9fd189dc8bc9112df07a144d24108e110e5149ae93f37cfef5?s=40&d=mm&r=g)\
\
\
\
     sunny1304August 15, 2017 at 3:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409677 "Direct link to this comment")\
\
\
\
\
\
     Hi Json,\
\
\
     Thank you for your awesome tutorial.\
\
\
     I have a question for you.\
\
\
\
     Is there any guideline on how to decide on neuron number for our network.\
\
\
     for example you used 12 for thr 1st layer and 8 for the second layer.\
\
\
     how do you decide on that ?\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409677)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2017 at 4:58 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409690 "Direct link to this comment")\
\
\
\
\
\
       No, there is no way to analytically determine the configuration of the network.\
\
\
\
       I use trial and error. You can grid search, random search, or copy configurations from tutorials or papers.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409690)\
145. ![](https://secure.gravatar.com/avatar/fd529be9156abd43c8fc0ad9e8519a4fa806530b68893f3f3afbd8dd9c0b44dc?s=40&d=mm&r=g)\
\
\
\
     yihadadAugust 16, 2017 at 6:53 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409831 "Direct link to this comment")\
\
\
\
\
\
     Hi Json,\
\
\
     Thanks for a wonderful tutorial.\
\
\
\
     Run a model generated by a CNN it takes how much ram, cpu ?\
\
\
\
     Thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409831)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)August 17, 2017 at 6:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409895 "Direct link to this comment")\
\
\
\
\
\
       It depends on the data you are using to fit the model and the size of the model.\
\
\
\
       Very large models could be 500MB of RAM or more.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-409895)\
146. ![](https://secure.gravatar.com/avatar/ff68defeab12fb62cccc663a420129c0ddbb28f41a04a7b5b53fc7855c1e2bb9?s=40&d=mm&r=g)\
\
\
\
     AnkurSeptember 1, 2017 at 3:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-411961 "Direct link to this comment")\
\
\
\
\
\
     Hi ,\
\
\
     Please let me know , how can i visualise the complete neural network in Keras……………….\
\
\
\
     I am looking for the complete architecture – like number of neurons in the Input Layer, hidden layer , output layer with weights.\
\
\
\
     Please have a look at the link present below, here someone has created a beutiful visualisation/architecture using neuralnet package in R.\
\
\
     Please let me know, can we create such type of model in KERAS\
\
\
\
     [https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/](https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-411961)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 1, 2017 at 6:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412000 "Direct link to this comment")\
\
\
\
\
\
       Use the Keras visualization API:\
\
       [https://keras.io/visualization/](https://keras.io/visualization/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412000)\
\
     - ![](https://secure.gravatar.com/avatar/cc7ecdd035e1f7aa0e7355d6c83fdb6b7cbd8d1a1b42c0c5cdc8411e2bc10233?s=40&d=mm&r=g)\
\
\
\
       ASADOctober 17, 2017 at 3:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416961 "Direct link to this comment")\
\
\
\
\
\
       Hello ANKUR,,,, how are you?\
\
\
\
       you have try visualization in keras which is suggested by Jason Brownlee?\
\
\
       if you have tried then please send me code i am also trying but didnot work..\
\
\
\
       please guide me\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416961)\
147. ![](https://secure.gravatar.com/avatar/372a8293c4398b781308f2e4c1372cb06cb68a6a891e23a68988611f7243b21a?s=40&d=mm&r=g)\
\
\
\
     AdamSeptember 3, 2017 at 1:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412324 "Direct link to this comment")\
\
\
\
\
\
     Thank you Dr. Brownlee for the great tutorial,\
\
\
\
     I have a question about your code:\
\
\
     is the argument metrics=\[‘accuracy’\] necessary in the code and does it change the results of the neural network or is it just for showing me the accuracy during compiling?\
\
\
\
     thank you!!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412324)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 3, 2017 at 5:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412353 "Direct link to this comment")\
\
\
\
\
\
       No, it just prints out the accuracy of the model at the end of each epoch. Learn more about Keras metrics here:\
\
       [https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/](https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412353)\
148. ![](https://secure.gravatar.com/avatar/eadef4eee5ebcf3dd2008d49d9a858ea58fff1bdfd3420baa2ea818cfef6c616?s=40&d=mm&r=g)\
\
\
\
     PottOfGoldSeptember 5, 2017 at 12:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412576 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     your work here is really great. It helped me a lot.\
\
\
     I recently stumbled upon one thing I cannot understand:\
\
\
\
     For the pimas dataset you state:\
\
\
     <>\
\
\
     When I look at the table of the pimas dataset, the examples are in rows and the features in columns, so your input dimension is the number of columns. As far as I can see, you don’t change the table.\
\
\
\
     For neural networks, isn’t the input normally: examples = columns, features=rows?\
\
\
     Is this different for Keras? Or can I use both shapes? An if yes, what’s the difference in the construction of the net?\
\
\
\
     Thank you!!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412576)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 7, 2017 at 12:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412938 "Direct link to this comment")\
\
\
\
\
\
       No, features are columns, rows are instances or examples.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412938)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/eadef4eee5ebcf3dd2008d49d9a858ea58fff1bdfd3420baa2ea818cfef6c616?s=40&d=mm&r=g)\
\
\
\
         PottOfGoldSeptember 7, 2017 at 3:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412995 "Direct link to this comment")\
\
\
\
\
\
         Thanks! 🙂\
\
\
         I had a lot of discussions because of that.\
\
\
         In Andrew Ng new Coursera course it’s explained as examples = columns, features=rows, but he doesn’t use Keras of course, but programms the neural networks from scratch.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-412995)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)September 9, 2017 at 11:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-413196 "Direct link to this comment")\
\
\
\
\
\
           I doubt that, I think you may have mixed it up. Columns are never examples.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-413196)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/eadef4eee5ebcf3dd2008d49d9a858ea58fff1bdfd3420baa2ea818cfef6c616?s=40&d=mm&r=g)\
\
\
\
             PottOfGoldOctober 6, 2017 at 6:26 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415846 "Direct link to this comment")\
\
\
\
\
\
             Thats what I thought, but I looked it up in the notation for the new coursera course (deeplearning.ai) and there it says: m is the numer of examples in the dataset and n is the input size, where X superscript n x m is the input matrix …\
\
\
             But either way, you helped me! Thank you. 🙂\
149. ![](https://secure.gravatar.com/avatar/62dceb9f3c19051d774921b29c5449f63684c1be38e9ebbd87f5a3148a95d8b1?s=40&d=mm&r=g)\
\
\
\
     Lin LiSeptember 16, 2017 at 1:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-413956 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, thank you so much for your tutorial, it helps me a lot. I need your help for the question below:\
\
\
     I copy the code and run it. Although I got the classification results, there were some warning messages in the process. As follows:\
\
\
\
     Warning (from warnings module):\
\
\
     File “C:\\Users\\llfor\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages\\keras\\callbacks.py”, line 120\
\
\
     % delta\_t\_median)\
\
\
     UserWarning: Method on\_batch\_end() is slow compared to the batch update (0.386946). Check your callbacks.\
\
\
\
     I don’t know why, and cannot find any answer to this question. I’m looking forward to your reply. Thanks again!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-413956)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)September 16, 2017 at 8:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-413998 "Direct link to this comment")\
\
\
\
\
\
       Sorry, I have not seen this message before. It looks like a warning, you might be able to ignore it.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-413998)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/62dceb9f3c19051d774921b29c5449f63684c1be38e9ebbd87f5a3148a95d8b1?s=40&d=mm&r=g)\
\
\
\
         Lin LiSeptember 16, 2017 at 12:24 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414011 "Direct link to this comment")\
\
\
\
\
\
         Thanks for your reply. I’m a start-learner on deep learning.I’d like to put it aside temporarily.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414011)\
150. ![](https://secure.gravatar.com/avatar/f9cfee091f21d760b61613186e6f1db30ff26791f1e0872b8c5ac2c0126886a9?s=40&d=mm&r=g)\
\
\
\
     SagarSeptember 22, 2017 at 2:51 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414537 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Great article, thumbs up for that. I am getting this error when I try to run the file on the command prompt. Any suggestions. Thanks for you response.\
\
\
\
     #######################################################################\
\
\
     C:\\Work\\ML>python keras\_first\_network.py\
\
\
     Using TensorFlow backend.\
\
\
     2017-09-22 10:11:11.189829: W C:\\tf\_jenkins\\home\\workspace\\rel-win\\M\\windows\\PY\\
\
\
     36\\tensorflow\\core\\platform\\cpu\_feature\_guard.cc:45\] The TensorFlow library wasn


     ‘t compiled to use AVX instructions, but these are available on your machine and


     could speed up CPU computations.


     2017-09-22 10:11:11.190829: W C:\\tf\_jenkins\\home\\workspace\\rel-win\\M\\windows\\PY\


     36\\tensorflow\\core\\platform\\cpu\_feature\_guard.cc:45\] The TensorFlow library wasn


     ‘t compiled to use AVX2 instructions, but these are available on your machine an


     d could speed up CPU computations.


     32/768 \[>………………………..\] – ETA: 0s


     acc: 78.52%


     #######################################################################



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414537)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 23, 2017 at 5:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414594 "Direct link to this comment")





       Looks like warning messages that you can ignore.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414594)




       - ![](https://secure.gravatar.com/avatar/f9cfee091f21d760b61613186e6f1db30ff26791f1e0872b8c5ac2c0126886a9?s=40&d=mm&r=g)



         SagarSeptember 24, 2017 at 3:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414673 "Direct link to this comment")





         Thanks I got to know what the problem was. According to section 6 I had set verbose argument to 0 while calling “model.fit()”. Now all the epochs are getting printed.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414673)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)September 24, 2017 at 5:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414689 "Direct link to this comment")





           Glad to hear it.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414689)
151. ![](https://secure.gravatar.com/avatar/fb3b18a7871403d2bee7813215d46a3d30336109cb16ff3d28f5f5cd3edf451d?s=40&d=mm&r=g)



     ValentinSeptember 26, 2017 at 6:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414856 "Direct link to this comment")





     Hi Jason,



     Thanks for the amazing article . Clear and straightforward.


     I had some problems installing Keras but was advised to prefix


     with tf.contrib.keras


     so I have code like



     model=tf.contrib.keras.models.Sequential()


     Dense=tf.contrib.keras.layers.Dense



     Now I try to train Keras on some small datafile to see how things work out:


     1,1,0,0,8


     1,2,1,0,4


     1,0,0,1,5


     1,0,1,0,7


     0,1,0,0,8


     1,4,1,0,4


     1,0,2,1,1


     1,0,1,0,7



     The first 4 columns are inputs and the 5-th column is output.


     I use the same code for training (adjust number of inputs) as in your article,


     but the network only gets to 12.5% accuracy.


     Any advise?



     Thanks,


     Valentin



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414856)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 27, 2017 at 5:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414891 "Direct link to this comment")





       Thanks Valentin.



       I have a good list of suggestions for improving model performance here:

       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-414891)
152. ![](https://secure.gravatar.com/avatar/75489acb1b81d5fa99d6aca2597238cb1920eeedbd3e0cf26af8df2fe2d6cde6?s=40&d=mm&r=g)



     PriyaOctober 3, 2017 at 2:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415424 "Direct link to this comment")





     Hi Jason,



     I tried replacing the pima data with random data as follows:



     X\_train = np.random.rand(18,61250)


     X\_test = np.random.rand(18,61250)


     Y\_train = np.array(\[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,\
\
\
     0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,\])


     Y\_test = np.array(\[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,\
\
\
     1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,\])



     \_, input\_size = X\_train.shape #put this in input\_dim in the first dense layer



     I took the round() off of the predictions so I could see the full value and then inserted my random test data in model.fit():



     predictions = model.predict(X\_test)


     preds = \[x\[0\] for x in predictions\]


     print(preds)



     model.fit(X\_train, Y\_train, epochs=100, batch\_size=10, verbose=2, validation\_data=(X\_test,Y\_test))



     I found something slightly odd; I expected the predicted values to be around 0.50, plus or minus some, but instead, I got this:



     \[0.49525392, 0.49652839, 0.49729034, 0.49670222, 0.49342978, 0.49490061, 0.49570397, 0.4962129, 0.49774086, 0.49475089, 0.4958384, 0.49506786, 0.49696651, 0.49869373, 0.49537542, 0.49613148, 0.49636957, 0.49723724\]



     which is near 0.50 but always less than 0.50. I ran this a few times with different random seeds, so it’s not coincidental. Would you have any explanation for why it does this?



     Thanks,


     Priya



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415424)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 3, 2017 at 3:46 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415433 "Direct link to this comment")





       Perhaps calculate the mean of your training data and compare it to the predicted value. It might be simple sampling error.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415433)

     - ![](https://secure.gravatar.com/avatar/75489acb1b81d5fa99d6aca2597238cb1920eeedbd3e0cf26af8df2fe2d6cde6?s=40&d=mm&r=g)



       PriyaOctober 4, 2017 at 1:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415486 "Direct link to this comment")





       I found out I was doing predictions before fitting the model. (I suppose that would mean the network hadn’t adjusted to the data’s distribution yet.)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415486)
153. ![](https://secure.gravatar.com/avatar/b71bbf45843fd6dc70f1ee0c122126576e1813cff836bfe4029140f56ae8d308?s=40&d=mm&r=g)



     SaurabhOctober 7, 2017 at 5:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415899 "Direct link to this comment")





     Hello Jason,



     I tried to train this model on my laptop, it is working fine. But I tried to train this model on google-cloud with the same instructions as in your example-5. But it is failing.


     Can you just let me know, which changes are to required for the model, so that I can train this on cloud.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415899)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 7, 2017 at 7:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415904 "Direct link to this comment")





       Sorry, I don’t know about google cloud.



       I have instructions here for running on AWS:

       [https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-415904)
154. ![](https://secure.gravatar.com/avatar/dbb6fbed3acd1645938036896c8c52df13290739177eec9684b1630949d44992?s=40&d=mm&r=g)



     tobegit3hubOctober 12, 2017 at 6:40 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416426 "Direct link to this comment")





     Great post. Thanks for sharing.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416426)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 13, 2017 at 5:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416483 "Direct link to this comment")





       You’re welcome.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416483)
155. ![](https://secure.gravatar.com/avatar/f940e782e1ccabb73a2e1b011d02e159b928d1f94ca526b4aa9acddaf895df41?s=40&d=mm&r=g)



     ManojOctober 12, 2017 at 11:43 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416454 "Direct link to this comment")





     Hi Jason,


     Is there a way to store the model, once it is created so that I can use it for different input data sets as and when needed.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416454)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 13, 2017 at 5:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416489 "Direct link to this comment")





       Yes, you can save it to file. See this tutorial:

       [https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-416489)
156. ![](https://secure.gravatar.com/avatar/adb071938e7485b3f568e965ff9910a96bb003b05dcc60e22f840e08c26540c2?s=40&d=mm&r=g)



     CamOctober 23, 2017 at 6:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417624 "Direct link to this comment")





     I get a syntax error for the



     model.fit() line in this example. Is it due to library conflicts with theano and tensorflow if i have both installed?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417624)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 24, 2017 at 5:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417679 "Direct link to this comment")





       Perhaps ensure your environment is up to date and that you copied the code exactly.



       This tutorial can help with setting up your environment:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417679)




       - ![](https://secure.gravatar.com/avatar/adb071938e7485b3f568e965ff9910a96bb003b05dcc60e22f840e08c26540c2?s=40&d=mm&r=g)



         CamOctober 24, 2017 at 2:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417721 "Direct link to this comment")





         Thanks, fixed!



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417721)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)October 24, 2017 at 4:01 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417741 "Direct link to this comment")





           Glad to hear it.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417741)
157. ![](https://secure.gravatar.com/avatar/1b3aa08e44856dcd58091f748cbee87facead16c7e164bae7f2fc9e6b7f97e9e?s=40&d=mm&r=g)



     [Diego Quintana](https://www.linkedin.com/in/diego-quintana-valenzuela/)October 25, 2017 at 7:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417842 "Direct link to this comment")





     Hi Jason, thanks for the example.



     How would you predict a single element from X? X\[0\] raises a ValueError



     ValueError: Error when checking : expected dense\_1\_input to have shape (None, 8) but got array with shape (8, 1)



     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417842)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 25, 2017 at 3:56 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417878 "Direct link to this comment")





       You can reshape it to have 1 row and 8 columns:











































































       |     |     |
       | --- | --- |
       | 1 | X=X.reshape((1,8)) |











       This post will give you further advice:

       [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-417878)




       - ![](https://secure.gravatar.com/avatar/32c7840c02a65f4b7a34218f488339fe3e3a7ce333545aff158a84268b204f90?s=40&d=mm&r=g)



         haraldApril 10, 2019 at 8:26 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-479500 "Direct link to this comment")





         Should it be: X\[0\].reshape((1,8)) ?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-479500)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)April 11, 2019 at 6:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-479596 "Direct link to this comment")





           Yep!



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-479596)
158. ![](https://secure.gravatar.com/avatar/63d006d5f2fc8c42f1b9f10989ae3b96267067b81bf9de50b141e8759a0b4b58?s=40&d=mm&r=g)



     Shahbaz WastiOctober 28, 2017 at 1:30 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418207 "Direct link to this comment")





     Dear Sir ,


     I have installed and configured the environment according to your directions but while running the program i have following error



     “from keras.utils import np\_utils”



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418207)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 29, 2017 at 5:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418284 "Direct link to this comment")





       What is the error exactly?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418284)
159. ![](https://secure.gravatar.com/avatar/51f9ea1c499debb5dc231a41705ed73ed1248130539b8dd0af7291c6415c7095?s=40&d=mm&r=g)



     ZhengpingOctober 30, 2017 at 12:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418366 "Direct link to this comment")





     Hi Jason, thanks for the great tutorials. I just learnt and repeated the program in your “Your First Machine Learning Project in Python Step-By-Step” without problem. Now trying this one, getting stuck at the line “model = Sequential()” when the Interactive window throws: NameError: name ‘Sequential’ is not defined. tried to google, can’t find a solution. I did import Sequential from keras.models as in ur example code. copy pasted as it is. Thanks in advance for your help.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418366)




     - ![](https://secure.gravatar.com/avatar/51f9ea1c499debb5dc231a41705ed73ed1248130539b8dd0af7291c6415c7095?s=40&d=mm&r=g)



       ZhengpingOctober 30, 2017 at 12:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418368 "Direct link to this comment")





       I’m running ur examples in Anaconda 4.4.0 environment in visual studio community version. relevant packages have been installed as in ur earlier tutorials instructed.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418368)




       - ![](https://secure.gravatar.com/avatar/51f9ea1c499debb5dc231a41705ed73ed1248130539b8dd0af7291c6415c7095?s=40&d=mm&r=g)



         ZhengpingOctober 30, 2017 at 12:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418369 "Direct link to this comment")





         >\> # create model


         … model = Sequential()


         …


         Traceback (most recent call last):


         File “”, line 2, in


         NameError: name ‘Sequential’ is not defined


         >>\> model.add(Dense(12, input\_dim=8, init=’uniform’, activation=’relu’))


         Traceback (most recent call last):


         File “”, line 1, in


         AttributeError: ‘SVC’ object has no attribute ‘add’



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418369)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)October 30, 2017 at 5:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418403 "Direct link to this comment")





           This does not look good. Perhaps post the error to stack exchange or other keras support. I have a list of keras support sites here:

           [https://machinelearningmastery.com/get-help-with-keras/](https://machinelearningmastery.com/get-help-with-keras/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418403)
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 30, 2017 at 5:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418402 "Direct link to this comment")





       Looks like you need to install Keras. I have a tutorial here on how to do that:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418402)
160. ![](https://secure.gravatar.com/avatar/6c9a809ce1eb6608ff8142f2523c0af5eda8bd6a225052fca565d5e945bbb02c?s=40&d=mm&r=g)



     AkhilOctober 30, 2017 at 5:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418462 "Direct link to this comment")





     Ho Jason,



     Thanks a lot for this wonderful tutorial.



     I have a question:



     I want to use your code to predict the classification (1 or 0) of unknown samples. Should I create one common csv file having the train (known) as well as the test (unknown) data. Whereas the ‘classification’ column for the known data will have a known value, 1 or 0, for the unknown data, should I leave the column empty (and let the code decide the outcome)?



     Thanks a lot



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418462)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 31, 2017 at 5:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418522 "Direct link to this comment")





       Great question.



       No, you only need the inputs and the model can predict the outputs, call model.predict(X).



       Also, this post will give a general idea on how to fit a final model:

       [https://machinelearningmastery.com/train-final-machine-learning-model/](https://machinelearningmastery.com/train-final-machine-learning-model/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418522)
161. ![](https://secure.gravatar.com/avatar/bd53fc83536d4c75dc13bccb5114a0c8b077525a8fcdf915943b969db8c7ef5e?s=40&d=mm&r=g)



     GuilhermeNovember 3, 2017 at 1:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418817 "Direct link to this comment")





     Hi Jason,



     This is really cool! I am blown away! Thanks so much for making it so simple for a beginner to have some hands on. I have a couple questions:



     1) where are the weights, can I save and/or retrieve them?



     2) if I want to train images with dogs and cats and later ask the neural network whether a new image has a cat or a dog, how do I get my input image to pass as an array and my output result to be “cat” or “dog”?



     Thanks again and great job!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418817)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 3, 2017 at 5:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418842 "Direct link to this comment")





       The weights are in the model, you can save them:

       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)



       Yes, you would save your model, then call model.predict() on the new data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-418842)
162. ![](https://secure.gravatar.com/avatar/b769cb00ad3823edf12ba4bed43c04fd1faae3b7b4a5cb02d2b678035fb3eaf7?s=40&d=mm&r=g)



     MichaelNovember 5, 2017 at 8:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419046 "Direct link to this comment")





     Hi Jason,



     Are you familiar with a python tool/package that can build neural network as in the tutorial, but suitable for data stream mining?



     Thanks,


     Michael



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419046)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 6, 2017 at 4:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419101 "Direct link to this comment")





       Not really, sorry.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419101)
163. ![](https://secure.gravatar.com/avatar/41e5af02f699ab0f7fa23c889bb2a65f244cf022561f0db8d9a20079b34eb3cd?s=40&d=mm&r=g)



     beaNovember 8, 2017 at 1:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419288 "Direct link to this comment")





     Hi, there. Could you please clarify why exactly you’ve built your network with 12 neurons in the first layer?



     “The first layer has 12 neurons and expects 8 input variables. The second hidden layer has 8 neurons and finally, the output layer has 1 neuron to predict the class (onset of diabetes or not)…”



     Should’nt it have 8 neurons at the start?



     Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419288)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 8, 2017 at 9:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419344 "Direct link to this comment")





       The input layer has 8, the first hidden layer has 12. I chose 12 through a little trial and error.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419344)
164. ![](https://secure.gravatar.com/avatar/bd53fc83536d4c75dc13bccb5114a0c8b077525a8fcdf915943b969db8c7ef5e?s=40&d=mm&r=g)



     GuilhermeNovember 9, 2017 at 12:54 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419442 "Direct link to this comment")





     Hi Jason,



     Do you have or else could you recommend a beginner’s level image segmentation approach that uses deep learning? For example, I want to train some neural net to automatically “find” a particular feature out of an image.



     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419442)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 9, 2017 at 10:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419493 "Direct link to this comment")





       Sorry, I don’t have image segmentation examples, perhaps in the future.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419493)
165. ![](https://secure.gravatar.com/avatar/5f3caf14b3bb71703967a48c0f4f35d9233dde3666b1dd830e752b6df17a87e9?s=40&d=mm&r=g)



     AndyNovember 12, 2017 at 6:56 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419823 "Direct link to this comment")





     Hi Jason,



     I just started my DL training a few weeks ago. According to what I learned in course, in order to train the parameters for the NN, we need to run the Forward and Backward propagation; however, looking at your Keras example, i don’t find any of these propagation processes. Does it mean that Keras has its own mechanism to find the parameters instead of using Forward and Backward propagation?



     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419823)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 13, 2017 at 10:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419870 "Direct link to this comment")





       It is performing those operations under the covers for you.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419870)
166. ![](https://secure.gravatar.com/avatar/ef514a767ad8c19d23428fe73d5fb851a36c81fae9a857b93ad81234c07e08c2?s=40&d=mm&r=g)



     BadrNovember 13, 2017 at 11:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419887 "Direct link to this comment")





     Hi Jason,



     Can you explain why I got the following output:



     ValueError Traceback (most recent call last)


     in ()


     —-\> 1 model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     2 model.fit(X, Y, epochs=150, batch\_size=10)


     3 scores = model.evaluate(X, Y)


     4 print(“\\n%s: %.2f%%” % (model.metrics\_names\[1\], scores\[1\]\*100))



     /Users/badrshomrani/anaconda/lib/python3.5/site-packages/keras/models.py in compile(self, optimizer, loss, metrics, sample\_weight\_mode, \*\*kwargs)


     545 metrics=metrics,


     546 sample\_weight\_mode=sample\_weight\_mode,


     –\> 547 \*\*kwargs)


     548 self.optimizer = self.model.optimizer


     549 self.loss = self.model.loss



     /Users/badrshomrani/anaconda/lib/python3.5/site-packages/keras/engine/training.py in compile(self, optimizer, loss, metrics, loss\_weights, sample\_weight\_mode, \*\*kwargs)


     620 loss\_weight = loss\_weights\_list\[i\]


     621 output\_loss = weighted\_loss(y\_true, y\_pred,


     –\> 622 sample\_weight, mask)


     623 if len(self.outputs) > 1:


     624 self.metrics\_tensors.append(output\_loss)



     /Users/badrshomrani/anaconda/lib/python3.5/site-packages/keras/engine/training.py in weighted(y\_true, y\_pred, weights, mask)


     322 def weighted(y\_true, y\_pred, weights, mask=None):


     323 # score\_array has ndim >= 2


     –\> 324 score\_array = fn(y\_true, y\_pred)


     325 if mask is not None:


     326 # Cast the mask to floatX to avoid float64 upcasting in theano



     /Users/badrshomrani/anaconda/lib/python3.5/site-packages/keras/objectives.py in binary\_crossentropy(y\_true, y\_pred)


     46


     47 def binary\_crossentropy(y\_true, y\_pred):


     —\> 48 return K.mean(K.binary\_crossentropy(y\_pred, y\_true), axis=-1)


     49


     50



     /Users/badrshomrani/anaconda/lib/python3.5/site-packages/keras/backend/tensorflow\_backend.py in binary\_crossentropy(output, target, from\_logits)


     1418 output = tf.clip\_by\_value(output, epsilon, 1 – epsilon)


     1419 output = tf.log(output / (1 – output))


     -\> 1420 return tf.nn.sigmoid\_cross\_entropy\_with\_logits(output, target)


     1421


     1422



     /Users/badrshomrani/anaconda/lib/python3.5/site-packages/tensorflow/python/ops/nn\_impl.py in sigmoid\_cross\_entropy\_with\_logits(\_sentinel, labels, logits, name)


     147 # pylint: disable=protected-access


     148 nn\_ops.\_ensure\_xent\_args(“sigmoid\_cross\_entropy\_with\_logits”, \_sentinel,


     –\> 149 labels, logits)


     150 # pylint: enable=protected-access


     151



     /Users/badrshomrani/anaconda/lib/python3.5/site-packages/tensorflow/python/ops/nn\_ops.py in \_ensure\_xent\_args(name, sentinel, labels, logits)


     1696 if sentinel is not None:


     1697 raise ValueError(“Only call `%s` with ”


     -\> 1698 “named arguments (labels=…, logits=…, …)” % name)


     1699 if labels is None or logits is None:


     1700 raise ValueError(“Both labels and logits must be provided.”)



     ValueError: Only call `sigmoid_cross_entropy_with_logits` with named arguments (labels=…, logits=…, …)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419887)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 14, 2017 at 10:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419938 "Direct link to this comment")





       Perhaps double check you have the latest versions of the keras and tensorflow libraries installed?!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419938)
167. ![](https://secure.gravatar.com/avatar/f199b20bbaa096c5e27e2f5574119c3dc7b4bd2647bd2bd2ca67691d99296a8a?s=40&d=mm&r=g)



     BadrNovember 14, 2017 at 10:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419958 "Direct link to this comment")





     keras was outdated



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-419958)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 15, 2017 at 9:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-420033 "Direct link to this comment")





       Glad to hear you fixed it.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-420033)
168. ![](https://secure.gravatar.com/avatar/2a7474b420f43fc4c860a4f8aa9df4ba6fdb0c4805ac2ee73a7f1444597a25fb?s=40&d=mm&r=g)



     MikaelNovember 22, 2017 at 8:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-420787 "Direct link to this comment")





     Hi Jason, thanks for your short tutorial, helps a lot to actually get your hands dirty with a simple example.


     I have tried 5 different parameters and got some interesting results to see what would happen. Unfortunately, I didnt record running time.



     Test 1 Test 2 Test 3 Test 4 Test 5 Test 6 Test 7


     number of layers 3 3 3 3 3 3 4


     Train set 768 768 768 768 768 768 768


     Iterations 150 100 1000 1000 1000 150 150


     Rate of update 10 10 10 5 1 1 5


     Errors 173 182 175 139 161 169 177


     Values 768 768 768 768 768 768 768


     % Error 23,0000% 23,6979% 22,7865% 18,0990% 20,9635% 22,0052% 23,0469%



     I can’t seem to see a trend here.. That could put me on the right track to adjust my hyperparameters.



     Do you have any advice on that?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-420787)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 22, 2017 at 11:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-420828 "Direct link to this comment")





       Something is wrong. Here is a good list of things to try:

       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-420828)
169. ![](https://secure.gravatar.com/avatar/d8ed323528abe945b78754220ead4c02c85116427ba80c8e588f298b4b5dffb3?s=40&d=mm&r=g)



     NikolaosNovember 28, 2017 at 10:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421444 "Direct link to this comment")





     Hi, I try to implement the above example with fer2013.csv but I receive an error, it is possible to help me to implement this correctly?











































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40 | keras.models import Sequential<br>from keras.layers import Dense<br>import numpy<br>import numpy asnp<br>\# fix Random seed for reproducibility<br>numpy.random.seed(7)<br>Y=\[\]<br>X=\[\]<br>#load dataset<br>forline inopen("fer2013.csv"):<br>row=line.split(',')<br>Y.append(int(row\[0\]))<br>X.append(\[int(p)forpinrow\[1\].split()\])<br>X,Y=np.array(X)/255.0,np.array(Y)<br>print(Y.shape)<br>print(X.shape)<br>#create model<br>model=Sequential()<br>model.add(Dense(12,input\_dim=(35887,2304),activation='tanh'))<br>model.add(Dense(8,activation='tanh'))<br>model.add(Dense(1,activation='sigmoid'))<br>#Compile Model<br>model.compile(loss='binary\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>#Fit Model<br>model.fit(X,Y,epochs=150,batch\_size=1)<br>\# evaluate the model<br>scores=model.evaluate(X,Y)<br>print("\\n%s: %.2f%%"%(model.metrics\_names\[1\],scores\[1\]\*100))<br>\# calculate predictions<br>predictions=model.predict(X)<br>\# round predictions<br>rounded=\[round(x\[0\])forxinpredictions\]<br>print(rounded) |











     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421444)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 29, 2017 at 8:10 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421532 "Direct link to this comment")





       Sorry, I cannot debug your code.



       What is the problem exactly?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421532)
170. ![](https://secure.gravatar.com/avatar/606771602e79455ae1926d2c20de6e1cb7829f460cf62ea9b8538d9d35e234af?s=40&d=mm&r=g)



     [Tanya](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)December 2, 2017 at 12:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421877 "Direct link to this comment")





     Hello,


     i have a a bit general question.


     I have to do a forecasting for restaurant sales (meaning that I have to predict 4 meals based on a historical daily sales data), weather condition (such as temperature, rain, etc), official holiday and in-off-season. I have to perform that forecasting using neuronal networks.


     I am unfortunately not a very skilled in python. On my computer I have Python 2.7 and I have install anaconda. I am trying to learn exercising with your codes, Mr. Brownlee. But somehow I can not run the code at all (in Spyder). Can you tell me what kind of version of python and anaconda I have to install on my computer and in which environment (jupiterlab,notebook,qtconsole, spyder, etc) I can run the code, so to work and not to give error from the very beginning?


     I will be very thankful for your response


     KG


     Tanya



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421877)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 2, 2017 at 9:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421918 "Direct link to this comment")





       Perhaps this tutorial will help you setup and confirm your environment:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       I would also recommend running code from the command like as IDEs and notebooks can introduce and hide errors.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-421918)
171. ![](https://secure.gravatar.com/avatar/85c16aa0afd59fa76194ecd03312659fcdc445c49327f7910c914cd3d008eba6?s=40&d=mm&r=g)



     EliahDecember 3, 2017 at 10:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422017 "Direct link to this comment")





     Hi Dr. Brownlee.



     I looked over the tutorial and I had a question regarding reading the data from a binary file? For instance I working on solving the sliding tiled n-puzzle using neural networks, but I seem to have trouble to getting my data which is in a binary file and it generates the number of move required for the n-puzzle to be solve in. Am not sure if you have dealt with this before, but any help would be appreciated.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422017)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 4, 2017 at 7:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422093 "Direct link to this comment")





       Sorry, I don’t know about your binary file.



       Perhaps after you load your data, you can convert it to a numpy array so that you can provide it to a neural net?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422093)




       - ![](https://secure.gravatar.com/avatar/85c16aa0afd59fa76194ecd03312659fcdc445c49327f7910c914cd3d008eba6?s=40&d=mm&r=g)



         EliahDecember 4, 2017 at 9:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422106 "Direct link to this comment")





         Thanks for the tip, I’ll try it.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422106)
172. ![](https://secure.gravatar.com/avatar/d4d1431b272c03b3d61de7ef99f480efd36eb51e3f753d30ef800d4229cdab51?s=40&d=mm&r=g)



     WafaaDecember 7, 2017 at 4:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422406 "Direct link to this comment")





     Thank you very very much for all your great tutorials.



     If I wanted to add batch layer after the input layer, how should I do it?



     Cuz I applied this tutorial on a different dataset and features and I think I need normalization or standardization and I want to do it the easiest way.



     Thank you,



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422406)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 8, 2017 at 5:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422452 "Direct link to this comment")





       I recommend preparing the data prior to fitting the model.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422452)
173. ![](https://secure.gravatar.com/avatar/9425845f96b964a1a02f1a09f90ac3af7c24edb91eba2eb6d35d2d555e4e1713?s=40&d=mm&r=g)



     zaheerDecember 9, 2017 at 3:03 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422523 "Direct link to this comment")





     thanks for sharing such nice tutorials, it helped me alot. i want to print the confusion matrix from the above example. and one more question.


     if i have


     20-input variable


1- class label (binary)

and 400 instances

how i would know , setting up the dense layer parameter in the first layer and hidden layer and output layer. like above example you have placed. 12,8,1

[Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422523)

     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 9, 2017 at 5:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422543 "Direct link to this comment")





       I recommend trial and error to configure the number of neurons in the hidden layer to see what works best for your specific problem.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422543)
174. ![](https://secure.gravatar.com/avatar/9425845f96b964a1a02f1a09f90ac3af7c24edb91eba2eb6d35d2d555e4e1713?s=40&d=mm&r=g)



     zaheerDecember 9, 2017 at 3:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422525 "Direct link to this comment")





     C:\\Users\\zaheer\\AppData\\Local\\Programs\\Python\\Python36\\python.exe C:/Users/zaheer/PycharmProjects/PythonBegin/Bin-CLNCL-Copy.py


     Using TensorFlow backend.


     Traceback (most recent call last):


     File “C:/Users/zaheer/PycharmProjects/PythonBegin/Bin-CLNCL-Copy.py”, line 28, in


     model.fit(x\_train , y\_train , epochs=100, batch\_size=100)


     File “C:\\Users\\zaheer\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\keras\\models.py”, line 960, in fit


     validation\_steps=validation\_steps)


     File “C:\\Users\\zaheer\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\keras\\engine\\training.py”, line 1574, in fit


     batch\_size=batch\_size)


     File “C:\\Users\\zaheer\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\keras\\engine\\training.py”, line 1407, in \_standardize\_user\_data


     exception\_prefix=’input’)


     File “C:\\Users\\zaheer\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\keras\\engine\\training.py”, line 153, in \_standardize\_input\_data


     str(array.shape))


     ValueError: Error when checking input: expected dense\_1\_input to have shape (None, 20) but got array with shape (362, 1)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422525)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 9, 2017 at 5:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422545 "Direct link to this comment")





       Ensure the input shape matches your data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422545)
175. ![](https://secure.gravatar.com/avatar/86c8586dfef8b488f4842394842ce8939f98234a4dc74c62e38b20cd0fc9e0c6?s=40&d=mm&r=g)



     [Anam Zahra](http://nill/)December 10, 2017 at 7:40 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422664 "Direct link to this comment")





     Dear Jason! Great job a very simple guide.


     I am trying to run the exact code but there is an eror


     str(array.shape))



     ValueError: Error when checking target: expected dense\_3 to have shape (None, 1) but got array with shape (768, 8)



     How can I resolve.



     I have windows 10 and spyder.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422664)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 11, 2017 at 5:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422711 "Direct link to this comment")





       Sorry to hear that, perhaps confirm that you have the latest version of Numpy and Keras installed?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422711)
176. ![](https://secure.gravatar.com/avatar/0d01cd09f239d1bb9acd175c8fa471a8e85de1734a2cebe6582fff374651a9be?s=40&d=mm&r=g)



     [nazek hassouneh](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)December 11, 2017 at 7:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422732 "Direct link to this comment")





     after run this code , i will calculate the accuracy , how i did , i


     i want to split the data set into test data , training data


     and evaluate the model and calculate the accuracy


     thank dr.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-422732)

177. ![](https://secure.gravatar.com/avatar/5bc1559b083c4f6360e94613acbf4dab9b430d38d4932736315e6d5618008c58?s=40&d=mm&r=g)



     SuchithDecember 21, 2017 at 2:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424287 "Direct link to this comment")





     In the model how many hidden layers are there ?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424287)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 21, 2017 at 3:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424298 "Direct link to this comment")





       There are 2 hidden layers, 1 input layer and 1 output layer.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424298)
178. ![](https://secure.gravatar.com/avatar/17c8f097fa1de06048490ef0899bb7b88b273ea0c226f2b1b0a14c5d95f28ab9?s=40&d=mm&r=g)



     Amare MahtesenuDecember 22, 2017 at 9:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424397 "Direct link to this comment")





     hi there. this blog is very awesome like the Adrian’s pyimagesearch blog. I have one question and that is do you have or will you have a tutorial on keras frame work with SSD or Yolo architechtures?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424397)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 22, 2017 at 4:16 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424422 "Direct link to this comment")





       Thanks for the suggestion, I hope to cover them in the future.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-424422)
179. ![](https://secure.gravatar.com/avatar/2d223e899190e8244d6f98aa2d7961068982d0f7725325df3b0e68592ea0ebc4?s=40&d=mm&r=g)



     [Kyujin Chae](http://intellibon.com/)January 8, 2018 at 2:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425868 "Direct link to this comment")





     Thanks for your awesome article.


     I am really enjoying


     ‘Machine Learning Mastery’!!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425868)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 8, 2018 at 3:54 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425874 "Direct link to this comment")





       Thanks!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425874)
180. ![](https://secure.gravatar.com/avatar/2b22b50c3ab4473d3463e129d0fa4cca1dfb870a9ff51d586537e8d162fe2a2a?s=40&d=mm&r=g)



     Luis GaldoJanuary 9, 2018 at 8:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425936 "Direct link to this comment")





     Hello Jason!



     This is an awesome article!


     I am writing a report for a subject in university and I have used your code during my implementation, would it be possible to cite this post in bibtex?



     Thank you!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425936)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 9, 2018 at 3:17 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425955 "Direct link to this comment")





       Sure, you can cite the webpage directly.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-425955)
181. ![](https://secure.gravatar.com/avatar/31fe90f063e74754801d0217b8278ca6b07be0e1ac22b432bf0a4db2eedeeaf3?s=40&d=mm&r=g)



     Nikhil GuptaJanuary 25, 2018 at 8:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427583 "Direct link to this comment")





     My question is regarding predict. I used to get decimals in the prediction array. Suddenly, I started seeing only Integers (0 or 1) in the run. Any idea what could be causing the change?



     predictions = model.predict(X2)



     predictions


     Out\[3\]:


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
     \[ 0.\],\
\
\
     \[ 0.\]\], dtype=float32)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427583)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 26, 2018 at 5:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427612 "Direct link to this comment")





       Perhaps check the activation function on the output layer?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427612)




       - ![](https://secure.gravatar.com/avatar/31fe90f063e74754801d0217b8278ca6b07be0e1ac22b432bf0a4db2eedeeaf3?s=40&d=mm&r=g)



         Nikhil GuptaJanuary 28, 2018 at 3:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427747 "Direct link to this comment")





         \# create model. Fully connected layers are defined using the Dense class


         model = Sequential()


         model.add(Dense(12, input\_dim=len(x\_columns), activation=’relu’)) #12 neurons, 8 inputs


         model.add(Dense(8, activation=’relu’)) #Hidden layer with 8 neurons


         model.add(Dense(1, activation=’sigmoid’)) #1 output layer. Sigmoid give 0/1



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427747)
182. ![](https://secure.gravatar.com/avatar/d558beb3482472444caa02b41e53b2a254341843ed63a261f9b45a71155e65f1?s=40&d=mm&r=g)



     joeJanuary 27, 2018 at 1:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427660 "Direct link to this comment")





     ================== RESTART: /Users/apple/Documents/deep1.py ==================


     Using TensorFlow backend.



     Traceback (most recent call last):


     File “/Users/apple/Documents/deep1.py”, line 20, in


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     File “/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/keras/models.py”, line 826, in compile


     \*\*kwargs)


     File “/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/keras/engine/training.py”, line 827, in compile


     sample\_weight, mask)


     File “/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/keras/engine/training.py”, line 426, in weighted


     score\_array = fn(y\_true, y\_pred)


     File “/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/keras/losses.py”, line 77, in binary\_crossentropy


     return K.mean(K.binary\_crossentropy(y\_true, y\_pred), axis=-1)


     File “/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/keras/backend/tensorflow\_backend.py”, line 3069, in binary\_crossentropy


     logits=output)


     TypeError: sigmoid\_cross\_entropy\_with\_logits() got an unexpected keyword argument ‘labels’


     >>>



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427660)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 27, 2018 at 5:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427705 "Direct link to this comment")





       I have not seem this error, sorry. Perhaps try posting to stack overflow?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427705)
183. ![](https://secure.gravatar.com/avatar/ffdced945176cb3e310dfb793effc3e1a818deab3ec9d562b40ea7efe096304a?s=40&d=mm&r=g)



     AtefehJanuary 27, 2018 at 4:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427730 "Direct link to this comment")





     Hello Mr.Janson


     After installing Anaconda and deep learning libraries, I read your Free mini-course and I tried to write the code about the handwritten digit recognition.


     I wrote the codes in jupyter notebook, am I right?


     if not where should I write the codes ?


     and if I want to use another dataset (my own data set) how can I use in the code?


     and how can I see the result, for example the accuracy percentage?


     I am really sorry for my simple questions! I have written a lot of code in “Matlab” but I am really a beginner in Python and Anaconda, my teacher force me to use Python and keras for my project.



     thank you very much for your help



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427730)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 28, 2018 at 8:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427769 "Direct link to this comment")





       A notebook is fine.



       You can write code in a Python script and then run the script directly.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427769)
184. ![](https://secure.gravatar.com/avatar/ffdced945176cb3e310dfb793effc3e1a818deab3ec9d562b40ea7efe096304a?s=40&d=mm&r=g)



     AtefehJanuary 28, 2018 at 12:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427742 "Direct link to this comment")





     Hello Mr.Janson again


     I wrote the code below from your Free mini course for hand written digit recognition, but after running I faced the syntaxerror:



     from keras.datasets import mnist


     …


     (X\_train, y\_train), (X\_test, y\_test) = mnist.load\_data()



     X\_train = X\_train.reshape(X\_train.shape\[0\], 1, 28, 28)


     X\_test = X\_test.reshape(X\_test.shape\[0\], 1, 28, 28)



     from keras.utils import np\_utils


     …


     y\_train = np\_utils.to\_categorical(y\_train)


     y\_test = np\_utils.to\_categorical(y\_test)



     model = Sequential()


     model.add(Conv2D(32, (3, 3), padding=’valid’, input\_shape=(1, 28, 28),


     activation=’relu’))


     model.add(MaxPooling2D(pool\_size=(2, 2)))


     model.add(Flatten())


     model.add(Dense(128, activation=’relu’))


     model.add(Dense(num\_classes, activation=’softmax’))


     model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])



     File “”, line 2


     2 model.add(Conv2D(32, (3, 3), padding=’valid’, input\_shape=(1, 28, 28),


     ^


     SyntaxError: invalid syntax



     would you please help me?!



     thanks a lot



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427742)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 28, 2018 at 8:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427777 "Direct link to this comment")





       This:











































































       |     |     |
       | --- | --- |
       | 1<br>2 | model.add(Conv2D(32,(3,3),padding=’valid’,input\_shape=(1,28,28),<br>activation=’relu’)) |











       should be:











































































       |     |     |
       | --- | --- |
       | 1 | model.add(Conv2D(32,(3,3),padding=’valid’,input\_shape=(1,28,28),activation=’relu’)) |











       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427777)
185. ![](https://secure.gravatar.com/avatar/d42385a19ca86fe371ad76865c1b2f552f45329f5eded2a0cfa3401d6474d91b?s=40&d=mm&r=g)



     LilaJanuary 29, 2018 at 8:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427822 "Direct link to this comment")





     Thank you for the awsome blog and explanations. I have just a question: How can we get predicted values by the model. . Many thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427822)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 29, 2018 at 8:21 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427834 "Direct link to this comment")





       As follows:











































































       |     |     |
       | --- | --- |
       | 1<br>2 | X=...<br>yhat=model.predict(X) |











       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427834)




       - ![](https://secure.gravatar.com/avatar/d42385a19ca86fe371ad76865c1b2f552f45329f5eded2a0cfa3401d6474d91b?s=40&d=mm&r=g)



         LilaJanuary 30, 2018 at 1:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427871 "Direct link to this comment")





         Thank you for your prompt answer. I am trying to learn how keras models work and I used. I trained the model like this:



         model.compile(loss=’mean\_squared\_error’, optimizer=’sgd’, metrics=\[‘MSE’\])



         As output I have those lines



         Epoch 10000/10000



         10/200 \[>………………………..\] – ETA: 0s – loss: 0.2489 – mean\_squared\_error: 0.2489


         200/200 \[==============================\] – 0s 56us/step – loss: 0.2652 – mean\_squared\_error: 0.2652



         and my question what the difference between the two lines (MSE values)



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427871)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)January 30, 2018 at 9:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427920 "Direct link to this comment")





           They should be the same thing. One may be calculated at the end of each batch, and one at the end of each epoch.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427920)
186. ![](https://secure.gravatar.com/avatar/ffdced945176cb3e310dfb793effc3e1a818deab3ec9d562b40ea7efe096304a?s=40&d=mm&r=g)



     AtefehJanuary 30, 2018 at 4:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427880 "Direct link to this comment")





     hello



     after running again it show an error:



     NameError Traceback (most recent call last)


     in ()


     —-\> 1 model = Sequential()


     2 model.add(Conv2D(32, (3, 3), padding=’valid’, input\_shape=(1, 28, 28), activation=’relu’))


     3 model.add(MaxPooling2D(pool\_size=(2, 2)))


     4 model.add(Flatten())


     5 model.add(Dense(128, activation=’relu’))



     NameError: name ‘Sequential’ is not defined



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427880)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 30, 2018 at 9:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427925 "Direct link to this comment")





       You are missing the imports. Ensure you copy all code from the complete example at the end.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427925)
187. ![](https://secure.gravatar.com/avatar/ffdced945176cb3e310dfb793effc3e1a818deab3ec9d562b40ea7efe096304a?s=40&d=mm&r=g)



     AtefehJanuary 31, 2018 at 1:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427976 "Direct link to this comment")





     from keras.datasets import mnist


     …


     (X\_train, y\_train), (X\_test, y\_test) = mnist.load\_data()


     X\_train = X\_train.reshape(X\_train.shape\[0\], 1, 28, 28)


     X\_test = X\_test.reshape(X\_test.shape\[0\], 1, 28, 28)


     from keras.utils import np\_utils


     …


     y\_train = np\_utils.to\_categorical(y\_train)


     y\_test = np\_utils.to\_categorical(y\_test)



     model = Sequential()


     2 model.add(Conv2D(32, (3, 3), padding=’valid’, input\_shape=(1, 28, 28), activation=’relu’))


     3 model.add(MaxPooling2D(pool\_size=(2, 2)))


     4 model.add(Flatten())


     5 model.add(Dense(128, activation=’relu’))


     6 model.add(Dense(num\_classes, activation=’softmax’))


     7 model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-427976)

188. ![](https://secure.gravatar.com/avatar/ffdced945176cb3e310dfb793effc3e1a818deab3ec9d562b40ea7efe096304a?s=40&d=mm&r=g)



     AtefehFebruary 2, 2018 at 5:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428159 "Direct link to this comment")





     hello


     please tell me how can I find out that tensorflow and keras are correctly installed on my system.


     maybe the problem is that, because no code runs in my jupyter. and no “import” acts well(for example import pandas)


     thank you



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428159)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 2, 2018 at 8:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428189 "Direct link to this comment")





       See this post:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428189)
189. ![](https://secure.gravatar.com/avatar/d804aebbeee18af8f83ea101335ecb414adea48d58576e5608f6346859be99e4?s=40&d=mm&r=g)



     DanFebruary 3, 2018 at 12:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428245 "Direct link to this comment")





     Hi. I’m totally new to machine learning and I’m trying to wrap my head around it.


     I have a problem I can’t quite solve yet. And don’t know where to start actually.


     I have a dictionary with a few key:value pairs. The key is a random 4 digit number from 0000 to 9999. And the value for each key is set as follows: if a digit in a number is either 0, 6 or 9 then its weight is 1, if a digit is 8 then it’s weight is 2, any other digit has a weight of 0. All the weights are summarised then and here you have the value for the key. (example: { ‘0000’: 4, ‘1234’: 0, ‘1692’: 2, ‘8800’: 6} – and so on).



     Now I’m trying to build a model that will predict the correct value of a given key. (i.e if I give it 2222 the answer is 0, if I give it 9011 – it’s 2). What I did first is created a CSV file with 5 columns, first four is a split (by a single digit) key from my dictionary, and the fifth column is the value for each key. Next I created a dataset and defined a model (like this tutorial but with input\_dim=4). Now when I train the model the accuracy won’t go higher then ~30%. Also your model is based on binary output, whereas mine should have an integer from 0 to 8. Where do I go from here?



     Thank you for all your effort in advance! 🙂



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428245)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 3, 2018 at 8:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428281 "Direct link to this comment")





       This post might help you nail down your problem as a predictive modeling problem:

       [https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428281)
190. ![](https://secure.gravatar.com/avatar/8478a960d8b387f98eedb441039dbf7731f7ac5737ac2714ea20504ebe5d3633?s=40&d=mm&r=g)



     AlexFebruary 5, 2018 at 5:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428391 "Direct link to this comment")





     There is one thing I just dont get.



     An example of row data is 6,148,72,35,0,33.6,0.627,50,1



     I guess the number at the end is if the person has diabetes (1) or does not (0) , but what I dont understand is how I know the ‘prediction’is about that 0 or 1, tehere are a lot of other variables in the data, and I dont see ‘diabetes’ being a label for any of that.



     So, how do I know or how do I set wich variable (number) I want to predict?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428391)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 5, 2018 at 7:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428412 "Direct link to this comment")





       You interpret the prediction in your application or usage.



       The model does not care what the inputs and outputs are, it does the best it can. It does not intrinsically care about diabetes.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428412)
191. ![](https://secure.gravatar.com/avatar/b50424dc47ab8ffd25077f057192a03a78e5a61fdef252d8d73b0a080f590071?s=40&d=mm&r=g)



     [blaisexen](https://github.com/cesarsouza/keras-sharp)February 6, 2018 at 9:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428546 "Direct link to this comment")





     hi,


     @Jason Brownlee, Master of Keras Python.



     I’m developing a face recognition testing, I successfully used Rprop, it was good for static images or face pictures, I also have test svm results.



     What do you think in your experienced that Keras is better or powerful than Rprop?



     because I was also thinking to used Keras(1:1) for final result of Rprop(1:many).



     or which do you think is better system?



     thanks in advance for the advices.



     I also heard one of the leader of commercial face recognizers uses PNN(uses libopenblas), so I really doubt which one to choose for my final thesis and application.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428546)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 6, 2018 at 9:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428571 "Direct link to this comment")





       What do you mean by rprop? I believe it is just an optimization algorithm, whereas Keras is a deep learning library.

       [https://en.wikipedia.org/wiki/Rprop](https://en.wikipedia.org/wiki/Rprop)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428571)




       - ![](https://secure.gravatar.com/avatar/b50424dc47ab8ffd25077f057192a03a78e5a61fdef252d8d73b0a080f590071?s=40&d=mm&r=g)



         blaisexenFebruary 17, 2018 at 10:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429768 "Direct link to this comment")





         Ok, I think I understand you.



         I used Accord.Net


         Rprop testing was good


         MLR testing was good


         SVM testing was good


         RBM testing was good



         I used classification for face images


         They are only good for static face pictures 100×100



         but if I used another picture from them,


         these 4 testing I have failed.



         Do you think if I used Keras in image face recognition will have a good result or good prediction?



         because if Keras will have a good result then I’ll have to used cesarsouza keras c#

         [https://github.com/cesarsouza/keras-sharp](https://github.com/cesarsouza/keras-sharp)



         thanks for the reply.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429768)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)February 18, 2018 at 6:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429807 "Direct link to this comment")





           Try it and see.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429807)
192. ![](https://secure.gravatar.com/avatar/55030963f6bc34dc1a00b26bc9427b363259d3e28255d0ebfcbe6e06b7175e38?s=40&d=mm&r=g)



     CHIRANJEEVIFebruary 8, 2018 at 8:52 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428803 "Direct link to this comment")





     What is the difference between the accuracy we get when we fit the model and the accuracy\_score() of sklearn.metrics , what they mean exactly ?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428803)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 9, 2018 at 9:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428841 "Direct link to this comment")





       Accuracy is a summary of the number of predictions that were made correctly out of all predictions that were made.



       It is used as an estimate of model skill on new out of sample data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428841)
193. ![](https://secure.gravatar.com/avatar/a19b61a15b7c39703245de079ad9b13a45e8c190ae376a90bdc56bab50eee9ce?s=40&d=mm&r=g)



     ShinanFebruary 8, 2018 at 9:09 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428805 "Direct link to this comment")





     is weather forecasting can done using RNN?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428805)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 9, 2018 at 9:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428843 "Direct link to this comment")





       No. Weather forecasting is done with ensembles of physics simulations on very large computers.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428843)
194. ![](https://secure.gravatar.com/avatar/55030963f6bc34dc1a00b26bc9427b363259d3e28255d0ebfcbe6e06b7175e38?s=40&d=mm&r=g)



     CHIRANJEEVIFebruary 9, 2018 at 3:56 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428876 "Direct link to this comment")





     we haven’t predicting anyting during the fit (its just a training , like mapping F(x)=Y)


     but still getting acc , what is this acc?



     Epoch 1/150


     768/768 \[==============================\] – 1s 1ms/step – loss: 0.6771 – acc: 0.6510



     Thank you in advance



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428876)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 10, 2018 at 8:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428958 "Direct link to this comment")





       Predictions are made as part of back propagating error.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-428958)
195. ![](https://secure.gravatar.com/avatar/a28db95aa277f23476a45d15df185c0c1aaaeb487304da8c0ad60d952fed5565?s=40&d=mm&r=g)



     lcy1031February 12, 2018 at 1:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429159 "Direct link to this comment")





     Hi Jason,



     Many thanks to you for a great tutorial. I have couple questions to you as followings.


     1). How can I get the score of Prediction?


     2). How can I output the result of predict run to a file in which the output is listed by vertical?



     I see you everywhere to answer questions and help people. Your time and patience were greatly appreciated!



     Charles



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429159)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 12, 2018 at 2:50 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429179 "Direct link to this comment")





       You can make predictions with a model as follows:



       yhat = model.predict(X)



       You can then save the numpy array result to file.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-429179)
196. ![](https://secure.gravatar.com/avatar/9071c47ebeaecb1236802a24d4db029d91027eb14d860167d329d25b7e0f4543?s=40&d=mm&r=g)



     CallumFebruary 21, 2018 at 10:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430118 "Direct link to this comment")





     Hi I’ve just finished this tutorial but the only problem is what are we actually finding in the results as in what do accuracy and loss mean and what we are actually finding out.



     I’m really new to the whole neural networks thing and don’t really understand them yet, I’d be very grateful if you’re able to reply



     Many Thanks



     Callum



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430118)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 22, 2018 at 11:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430192 "Direct link to this comment")





       Accuracy is the model skill in terms of the number of correct predictions divided by the total number of predictions.



       Loss the function that the network is optimising, something differentiable and relatable to the metric of interest for the model, in this case logarithmic loss used for classification.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430192)
197. ![](https://secure.gravatar.com/avatar/90cd22b852f664a6155b0ee9d6410ca3262252729aaea8c5a53e1a370c3b84df?s=40&d=mm&r=g)



     Pedro WennerFebruary 23, 2018 at 1:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430259 "Direct link to this comment")





     Hi Jason,



     First of all congratulations for your awesome work, I finally got the hang of ML (hopefully, haha).


     So, testing some changes in the number of neurons and batch size/epochs, I achieved 99.87% of accuracy.



     The parameters I used were:



     \# create model


     model = Sequential()


     model.add(Dense(240, input\_dim=8, init=’uniform’, activation=’relu’))


     model.add(Dense(160, init=’uniform’, activation=’relu’))


     model.add(Dense(1, init=’uniform’, activation=’sigmoid’))


     \# Compile model


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     \# Fit the model


     model.fit(X, Y, epochs=1500, batch\_size=100, verbose=2)



     And when I run it, I always get 99,87% of accuracy, which I think it’s a good thing, right? Please tell me if I did something wrong or if this is a false positive.



     Thank you in advance and sorry for the bad english 😉



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430259)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 23, 2018 at 12:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430311 "Direct link to this comment")





       that accuracy is great, there will always be some error.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430311)
198. ![](https://secure.gravatar.com/avatar/0f3f769d50c01fa8aece909e42d640fe22f0ae7b30b3234ecc6a4acf872cbf36?s=40&d=mm&r=g)



     ShinyMarch 2, 2018 at 12:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430836 "Direct link to this comment")





     The above example is very good sir, I want to do price change prediction of electronics in online shopping project. Can you give any suggestions about my project. You had any example of price prediction using neural network please send a link sir.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430836)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2018 at 5:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430856 "Direct link to this comment")





       I would recommend following this process:

       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-430856)
199. ![](https://secure.gravatar.com/avatar/76bad145180c5180fa34f13c8dd23e705c1cc0af343dfbcf1df3835fc13a0ba0?s=40&d=mm&r=g)



     [awaludin](http://www.polban.ac.id/)March 6, 2018 at 12:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431238 "Direct link to this comment")





     Hi, very helpful example. But I still don’t understand why you load


     X = dataset\[:,0:8\]


     Y = dataset\[:,8\]


     If I do


     X = dataset\[:,0:7\] it won’t work



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431238)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 6, 2018 at 6:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431284 "Direct link to this comment")





       You can learn more about indexing and slicing numpy arrays here:

       [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431284)
200. ![](https://secure.gravatar.com/avatar/17dacd2a4f260107504d64179a8b232d68e0219ec3f34669fe23a93bbf2ce34e?s=40&d=mm&r=g)



     Jeong KimMarch 8, 2018 at 1:48 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431579 "Direct link to this comment")





     Thank you for the tutorial.


     Perhaps, someone already told you this. The data set is no longer available.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431579)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 8, 2018 at 2:55 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431587 "Direct link to this comment")





       Thanks for the note, I’ll fix that up ASAP.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431587)
201. ![](https://secure.gravatar.com/avatar/f0d8d03da0c25a391cdf5fde8f892fd3473bb5d49250186a1de46b8b6687f69a?s=40&d=mm&r=g)



     Wesley CampbellMarch 9, 2018 at 1:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431614 "Direct link to this comment")





     Thanks very much for the concise example! As an “interested amateur” with more experience coding for scientific data manipulation than for software development, a simple, high-level explanation like this one is much appreciated. I find sometimes that documentation pages can be a bit low-level for my liking, even with coding experience multiple languages. This article was all I needed to get started, and was much more helpful than other “official tutorials.”



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431614)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 9, 2018 at 6:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431638 "Direct link to this comment")





       Thanks, I’m glad to hear that Wesley.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431638)
202. ![](https://secure.gravatar.com/avatar/6b8164287c12d0460a7f044cce624fd136d1b18c85fed6df982f099b1c138798?s=40&d=mm&r=g)



     TrungMarch 10, 2018 at 12:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431708 "Direct link to this comment")





     Thank you for your tutorial, but the data set is not accessible. Could you please fix it.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431708)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 10, 2018 at 6:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431747 "Direct link to this comment")





       Thanks, I’ll fix it.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-431747)
203. ![](https://secure.gravatar.com/avatar/ffdced945176cb3e310dfb793effc3e1a818deab3ec9d562b40ea7efe096304a?s=40&d=mm&r=g)



     atefehMarch 16, 2018 at 10:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432365 "Direct link to this comment")





     hello



     I have found a code to converting my image data to mnist format . but I face to an error below.


     would you please help me?



     import os


     from PIL import Image


     from array import \*


     from random import shuffle



     \# Load from and save to


     Names = \[\[‘./training-images’,’train’\], \[‘./test-images’,’test’\]\]



     for name in Names:



     data\_image = array(‘B’)


     data\_label = array(‘B’)



     FileList = \[\]


     for dirname in os.listdir(name\[0\])\[1:\]: # \[1:\] Excludes .DS\_Store from Mac OS


     path = os.path.join(name\[0\],dirname)


     for filename in os.listdir(path):


     if filename.endswith(“.png”):


     FileList.append(os.path.join(name\[0\],dirname,filename))



     shuffle(FileList) # Usefull for further segmenting the validation set



     for filename in FileList:



     label = int(filename.split(‘/’)\[2\])



     Im = Image.open(filename)



     pixel = Im.load()



     width, height = Im.size



     for x in range(0,width):


     for y in range(0,height):


     data\_image.append(pixel\[y,x\])



     data\_label.append(label) # labels start (one unsigned byte each)



     hexval = “{0:#0{1}x}”.format(len(FileList),6) # number of files in HEX



     # header for label array



     header = array(‘B’)


     header.extend(\[0,0,8,1,0,0\])


     header.append(int(‘0x’+hexval\[2:\]\[:2\],16))


     header.append(int(‘0x’+hexval\[2:\]\[2:\],16))



     data\_label = header + data\_label



     # additional header for images array



     if max(\[width,height\]) <= 256:


     header.extend(\[0,0,0,width,0,0,0,height\])


     else:


     raise ValueError('Image exceeds maximum size: 256×256 pixels');



     header\[3\] = 3 # Changing MSB for image data (0x00000803)



     data\_image = header + data\_image



     output\_file = open(name\[1\]+'-images-idx3-ubyte', 'wb')


     data\_image.tofile(output\_file)


     output\_file.close()



     output\_file = open(name\[1\]+'-labels-idx1-ubyte', 'wb')


     data\_label.tofile(output\_file)


     output\_file.close()



     \# gzip resulting files



     for name in Names:


     os.system('gzip '+name\[1\]+'-images-idx3-ubyte')


     os.system('gzip '+name\[1\]+'-labels-idx1-ubyte')



     FileNotFoundError Traceback (most recent call last)


     in ()


     13


     14 FileList = \[\]


     —\> 15 for dirname in os.listdir(name\[0\])\[1:\]: # \[1:\] Excludes .DS\_Store from Mac OS


     16 path = os.path.join(name\[0\],dirname)


     17 for filename in os.listdir(path):



     FileNotFoundError: \[WinError 3\] The system cannot find the path specified: ‘./training-images’



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432365)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 17, 2018 at 8:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432408 "Direct link to this comment")





       Looks like the code cannot find your images. Perhaps change the path in the code?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432408)
204. ![](https://secure.gravatar.com/avatar/e594af2b6bcaa6cd7c16a5b564513f51eacd41fe2f8ac52c873705e6b18643e4?s=40&d=mm&r=g)



     SayanMarch 17, 2018 at 4:57 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432435 "Direct link to this comment")





     Thanks a lot sir, this was a very good and intuitive tutorial



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432435)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 18, 2018 at 6:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432485 "Direct link to this comment")





       Thanks, I’m glad it helped.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432485)
205. ![](https://secure.gravatar.com/avatar/31fe90f063e74754801d0217b8278ca6b07be0e1ac22b432bf0a4db2eedeeaf3?s=40&d=mm&r=g)



     Nikhil GuptaMarch 19, 2018 at 11:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432618 "Direct link to this comment")





     I got a prediction model running successfully for fraud detection. My dataset is over 50 million and growing. I am seeing a peculiar issue.


     When the loaded data is 10million or less, My prediction is OK.


     As soon as I load 11 million data, My prediction saturates to a particular (say 0.48) and keeps on repeating. That is all predictions will be 0.48, irrespective of the input.



     I have tried will multiple combinations of the dense model.


     \# create model


     model = Sequential()


     model.add(Dense(32, input\_dim=4, activation=’tanh’))


     model.add(Dense(28, activation=’tanh’))


     model.add(Dense(24, activation=’tanh’))


     model.add(Dense(20, activation=’tanh’))


     model.add(Dense(16, activation=’tanh’))


     model.add(Dense(12, activation=’tanh’))


     model.add(Dense(8, activation=’tanh’))


     model.add(Dense(1, activation=’sigmoid’))



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432618)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 20, 2018 at 6:21 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432665 "Direct link to this comment")





       Perhaps check whether you need to train on all data, often a small sample is sufficient.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432665)




       - ![](https://secure.gravatar.com/avatar/31fe90f063e74754801d0217b8278ca6b07be0e1ac22b432bf0a4db2eedeeaf3?s=40&d=mm&r=g)



         Nikhil GuptaMarch 22, 2018 at 2:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432845 "Direct link to this comment")





         Oh. I believe that the machine learning accuracy will improve as we get more data over time.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-432845)
206. ![](https://secure.gravatar.com/avatar/3b370df996e5a9f5be3b16e5c69f67104b427614c0030492408a319b80bd3577?s=40&d=mm&r=g)



     Chandra Sutrisno TjhongMarch 28, 2018 at 4:43 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433467 "Direct link to this comment")





     HI,



     How do you define number of hidden layers and neurons per layer?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433467)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 29, 2018 at 6:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433528 "Direct link to this comment")





       There are no good heuristics, trial and error is a good approach. Discover what works best for your specific data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433528)
207. ![](https://secure.gravatar.com/avatar/c07218a28e64fa760313a692a264bd0f23eea5a622175cce8d77d25c4d6da7c6?s=40&d=mm&r=g)



     AravindMarch 30, 2018 at 12:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433619 "Direct link to this comment")





     I executed the code and got the output, but how to use this prediction in the application.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433619)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 30, 2018 at 6:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433659 "Direct link to this comment")





       Depends on the application.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433659)
208. ![](https://secure.gravatar.com/avatar/c07218a28e64fa760313a692a264bd0f23eea5a622175cce8d77d25c4d6da7c6?s=40&d=mm&r=g)



     SabarishMarch 30, 2018 at 12:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433620 "Direct link to this comment")





     What does the value 1.0 and 0..0 signifies??



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433620)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 30, 2018 at 6:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433660 "Direct link to this comment")





       In what context?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433660)
209. ![](https://secure.gravatar.com/avatar/093de7e9414b39104d590c8730b78e6864a768225e95329caf0c5e6e1c485a44?s=40&d=mm&r=g)



     AnandApril 1, 2018 at 3:51 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433828 "Direct link to this comment")





     If number of inputs are 8 then why did you use 12 neurons in input layer ? Moreover why is activation function used in input layer ?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433828)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 2, 2018 at 5:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433870 "Direct link to this comment")





       The number of neurons in the first hidden layer can be different to the number of neurons in the input layer (e.g. number of input features). They are only loosely related.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433870)
210. ![](https://secure.gravatar.com/avatar/d42385a19ca86fe371ad76865c1b2f552f45329f5eded2a0cfa3401d6474d91b?s=40&d=mm&r=g)



     LiaApril 1, 2018 at 11:49 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433849 "Direct link to this comment")





     Hello Sir,


     Does the neural network use a standardized independent variable values, or should we feed it with standardized ones in the fitting and predicting stages. Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433849)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 2, 2018 at 5:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433879 "Direct link to this comment")





       Try both and see what works best for your specific predictive modeling problem.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-433879)




       - ![](https://secure.gravatar.com/avatar/3ddf66f0d012c56505c80581e218eb82d68a0b36135433d3c5c1efadacf38dba?s=40&d=mm&r=g)



         Mark LittlewoodOctober 27, 2021 at 9:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632647 "Direct link to this comment")





         Hi I was playing with a 2 input data set and when I had the first layer set at Dense(4 it only output NaN for the loss. However when I reduced this to 3 I got meaningful loss output. Is there something about the maximum Dens value in relation to the inputs that causes this ?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632647)




         - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



           Adrian TamOctober 27, 2021 at 12:56 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632703 "Direct link to this comment")





           There should not be. It is more likely due to how the layers are initialized than number of neurons in the Dense layer.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632703)
211. ![](https://secure.gravatar.com/avatar/05f58d9cde6cd56fbc0cd7879949ada0ab15281814aae0e312aca9e17f939ff7?s=40&d=mm&r=g)



     tareknahoolApril 4, 2018 at 5:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434047 "Direct link to this comment")





     you always fantastic, it’s a great lesson. But, frankly I don’t know what is the meaning of


     “\\n%s: %.2f%%” % and why you used the number(1)in that code(model.metrics\_names\[1\], scores\[1\]\*100))



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434047)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 4, 2018 at 6:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434077 "Direct link to this comment")





       This is Python string formatting:

       [https://pyformat.info/](https://pyformat.info/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434077)
212. ![](https://secure.gravatar.com/avatar/85fd5b84eb0c649a4a171c300bd4fd46d24b2e89dfa0f774ee504090122ebe70?s=40&d=mm&r=g)



     Abhilash MenonApril 5, 2018 at 6:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434190 "Direct link to this comment")





     Dr. Brownlee,



     When we predict, is it possible to have the predictions for each row in the test data set right next to it in the same row. I thought of printing predictions and then copying it in excel but I am not sure if Keras preserves order. Could you please help me out with this issue? Thanks so much for all your help!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434190)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 5, 2018 at 3:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434209 "Direct link to this comment")





       Yes, the order of predictions matches the order of input values.



       Does that help?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434209)
213. ![](https://secure.gravatar.com/avatar/f7b879ba9f1474b1212ad17308400fccaf02b25bd802942b8f800ea413fb7837?s=40&d=mm&r=g)



     [Andrea Grandi](https://www.andreagrandi.it/)April 9, 2018 at 6:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434507 "Direct link to this comment")





     Is Deep Learning some kind of “black magic” 🙂 ?



     I had previously used scikit-learn and Machine Learning for the same dataset, trying to apply all the techniques I did learn both here and on books, to get a 76% accuracy.



     I tried this Keras tutorial, using TensorFlow as backend and I’m getting 80% accuracy at first try O\_o



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434507)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 10, 2018 at 6:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434581 "Direct link to this comment")





       No, not magic, just different.



       Well done though!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434581)
214. ![](https://secure.gravatar.com/avatar/28689088b687eb42f1fb71168e9d36b405e912a597d48998ba88241534cddbec?s=40&d=mm&r=g)



     Manny CorraoApril 11, 2018 at 8:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434701 "Direct link to this comment")





     Can you tell us the column names? I think that is important because it helps us understand what the network is evaluating and learning about.



     Thanks,



     Manny



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434701)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 11, 2018 at 4:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434722 "Direct link to this comment")





       Yes, they are listed here:

       [https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.names](https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.names)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434722)
215. ![](https://secure.gravatar.com/avatar/acb32a74323337656a3325d2dccebbd250adef19897474f388784582934d11ff?s=40&d=mm&r=g)



     rachitApril 11, 2018 at 7:13 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434743 "Direct link to this comment")





     While Executing versions.py



     i am getting this error



     Traceback (most recent call last):


     File “versions.py”, line 2, in


     import scipy


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\scipy\\\_\_init\_\_.py”, line 61, in


     from numpy import show\_config as show\_numpy\_config


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\\_\_init\_\_.py”, line 142, in


     from . import add\_newdocs


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\add\_newdocs.py”, line 13, in


     from numpy.lib import add\_newdoc


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\lib\\\_\_init\_\_.py”, line 8, in


     from .type\_check import \*


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\lib\\type\_check.py”, line 11, in


     import numpy.core.numeric as \_nx


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\core\\\_\_init\_\_.py”, line 74, in


     from numpy.testing import \_numpy\_tester


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\testing\\\_\_init\_\_.py”, line 12, in


     from . import decorators as dec


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\testing\\decorators.py”, line 6, in


     from .nose\_tools.decorators import \*


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\testing\\nose\_tools\\decorators.py”, line 20, in


     from .utils import SkipTest, assert\_warns


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\numpy\\testing\\nose\_tools\\utils.py”, line 15, in


     from tempfile import mkdtemp, mkstemp


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\tempfile.py”, line 45, in


     from random import Random as \_Random


     File “C:\\Users\\ATIT GARG\\random.py”, line 7, in


     from keras.models import Sequential


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\keras\\\_\_init\_\_.py”, line 3, in


     from . import utils


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\keras\\utils\\\_\_init\_\_.py”, line 4, in


     from . import data\_utils


     File “C:\\Users\\ATIT GARG\\Anaconda3\\lib\\site-packages\\keras\\utils\\data\_utils.py”, line 23, in


     from six.moves.urllib.error import HTTPError


     ImportError: cannot import name ‘HTTPError’



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434743)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 12, 2018 at 8:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434793 "Direct link to this comment")





       Perhaps you need to update your environment?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434793)
216. ![](https://secure.gravatar.com/avatar/dbe1c210715043a2fb3cabef143793ac293feeb071ba21ab52e1543e6f14a831?s=40&d=mm&r=g)



     GrayApril 14, 2018 at 4:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434948 "Direct link to this comment")





     Jason – very impressive work! Even more impressive is your detailed answer to every question. I went through them all and got a lot of useful information. Great job!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434948)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 14, 2018 at 6:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434974 "Direct link to this comment")





       Thanks Gray!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434974)
217. ![](https://secure.gravatar.com/avatar/0ac97caa3bac7d2b364c401566c1127feec31a32b3c7513278551cea72f612a3?s=40&d=mm&r=g)



     octdesApril 14, 2018 at 2:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434998 "Direct link to this comment")





     Hello Jason,


     Thank’s for the good tuto !


     How would you name/describe the structure of this neuronal network ?


     The point is that i find strange that you can have a different nmber of input and of neurones in the input layer. Most of the neuronal network diagramm i have seen, each input is directly connected with one neurone of the input layer. I have never seen a neuronal network diagramm where the number of input is different with the number of neurones in the input layer.


     Do you have counterexample or do there is something i understand wrong ?


     Thank you for your work and sharing your knowledge 🙂



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-434998)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 15, 2018 at 6:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435041 "Direct link to this comment")





       The type of neural network in this post is a multi-layer perceptron or an MLP for short.



       The first “layer” in the code actually defines both the input layer and the first hidden layer at the same time.



       The number of inputs must match the number of columns in the input data. The number of neurons in the first hidden layer can be anything you want.



       Does that help?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435041)
218. ![](https://secure.gravatar.com/avatar/6ddd433bf26d1454c867be0049e0fb3842a7b1ffca0bef8b0632dff46d5e76cc?s=40&d=mm&r=g)



     AshleyApril 16, 2018 at 7:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435117 "Direct link to this comment")





     Thank you VERY much for this tutorial, Jason! It is the best I have found on the internet. As a political scientist pursuing complex outcomes like this one, I was looking for models that allow for more complicated relationships. Your code and post are so clearly articulated; I was able to adapt it for my purposes more easily than I thought would be possible. One possible extension of your work, and possibly this tutorial, would be to map the layers and nodes onto a theory of the data generating process.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435117)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 16, 2018 at 2:54 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435132 "Direct link to this comment")





       Thanks Ashley, I’m glad it helped.



       Thanks for the suggestion.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435132)
219. ![](https://secure.gravatar.com/avatar/dbdfe9a2c5edb624583218fb9dac0502cdd96bf34c3b74e39aff8d5b9ffb6778?s=40&d=mm&r=g)



     Eric MilesApril 20, 2018 at 1:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435460 "Direct link to this comment")





     I’m just starting out working through your site – thanks for the great resource! I wanted to point out what I think is a typo: in the code block just before Section 2 “Define Model” I believe we just want X = dataset\[:,0:7\] so that we don’t include the output variables in our inputs.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435460)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 20, 2018 at 6:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435487 "Direct link to this comment")





       No, it is correct Eric.



       X will have 8 columns (0-7), the original dataset has 9.



       You can learn more about array slicing and ranges in Python here:

       [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-435487)
220. ![](https://secure.gravatar.com/avatar/9108e8cabfa759c2a0cd9e38d211bd6304371ea0ba23cd4dcc0dfa437c7d3f85?s=40&d=mm&r=g)



     [Rafa](https://www.aprenderpython.net/)April 28, 2018 at 12:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436132 "Direct link to this comment")





     Great tutorial, finally I have found a good web about deep learning (Y)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436132)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 28, 2018 at 5:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436158 "Direct link to this comment")





       Thanks.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436158)
221. ![](https://secure.gravatar.com/avatar/9b7d39e111b2a715e275456624b11a3a3abf016284d17429acd4ad5853b311bd?s=40&d=mm&r=g)



     VivekMay 7, 2018 at 8:31 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436827 "Direct link to this comment")





     Great tutorial thank for help. I have one project in which i have to do CAD images(basically 3-d mechanical image classification). can you please give road map how can i proceed?


     I am new and i dont have any idea



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436827)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 8, 2018 at 6:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436871 "Direct link to this comment")





       This is my general roadmap for a predictive modeling problem:

       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436871)




       - ![](https://secure.gravatar.com/avatar/9b7d39e111b2a715e275456624b11a3a3abf016284d17429acd4ad5853b311bd?s=40&d=mm&r=g)



         VivekMay 9, 2018 at 10:03 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437052 "Direct link to this comment")





         Thanks a lot sir. This will help me to proceed



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437052)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)May 10, 2018 at 6:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437084 "Direct link to this comment")





           I’m glad to hear that.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437084)
222. ![](https://secure.gravatar.com/avatar/f49c77790015c40c6024af622c39463e6c97f9a343eb4791e08d9be059531efe?s=40&d=mm&r=g)



     Rahmad arsMay 8, 2018 at 1:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436842 "Direct link to this comment")





     Thanks sir for the tutorial.


     Actually i still have some question:


     1\. Is this backpropagation neural network?


     2\. How to initialize nguyen-widrow random weights


     3\. I have my own dataset, each consist of 1×64 matrix, which is the correct one? I normalize each column of it, or each row of it?



     Thanks.


     Im the one who asked u in backpropagation from scratch page



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436842)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 8, 2018 at 6:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436877 "Direct link to this comment")





       Yes, it uses backpropgation to update the weights.



       Sorry, I don’t know about that initialization method, you can see the supported methods here:

       [https://keras.io/initializers/](https://keras.io/initializers/)



       Try a suite of data preparation schemes to see what works best for your specific dataset and chosen model.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-436877)
223. ![](https://secure.gravatar.com/avatar/748f3c4fcc0df5f1dccae3b23806ab0b709f889995f7d1048f62982834f32ea3?s=40&d=mm&r=g)



     HusseinMay 9, 2018 at 10:33 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437056 "Direct link to this comment")





     Hi Jason,



     This is a very nice intro to a daunting but intriguing technology! I wanted to play around with your code and see if I could come up with some simple dataset and see how the predictions will work out – one idea that occurred to me is, can I make a model that predicts what country a telephone number belongs to. So the training dataset looks like a 2 column CSV, phone number and country…that’s basically one feature. Do you think this would be effective at all? What other features could be added here? I’ll still give this a shot, but would appreciate any thoughts/ideas!



     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437056)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 10, 2018 at 6:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437088 "Direct link to this comment")





       The country code would make it too simple a problem – e.g. it can be solved with a look-up table.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437088)




       - ![](https://secure.gravatar.com/avatar/748f3c4fcc0df5f1dccae3b23806ab0b709f889995f7d1048f62982834f32ea3?s=40&d=mm&r=g)



         HusseinMay 10, 2018 at 4:24 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437125 "Direct link to this comment")





         True, I just wanted to see if machine learning could be used to “figure out” the lookup table as opposed to be provided with one by the user, given enough data..not a practical use-case, but as a learning exercise. As it turns out, my data-set of about 700 phone numbers wasn’t effective for this. But again, is this because the problem had too few features, i.e in my case, just one? What if I increased the number of features, say phone number, country code, city the phone number belongs to, maybe even the cellphone company the number is registered to, do you think that would make the training more effective?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437125)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)May 11, 2018 at 6:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437175 "Direct link to this comment")





           If you can write an if statement or use a look-up table to solve the problem, then it might be a bad fit for machine learning.



           This post will help you frame your problem:

           [https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437175)




           - ![](https://secure.gravatar.com/avatar/748f3c4fcc0df5f1dccae3b23806ab0b709f889995f7d1048f62982834f32ea3?s=40&d=mm&r=g)



             HusseinMay 11, 2018 at 5:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437230 "Direct link to this comment")





             Thanks Jason for that resource. I’ll check it out. I also came across this ( [https://elitedatascience.com/machine-learning-projects-for-beginners](https://elitedatascience.com/machine-learning-projects-for-beginners)) that I’m reading through, for anyone else that’s looking for a small ML problem to solve as a learning experience.

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)May 12, 2018 at 6:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437286 "Direct link to this comment")





             Great.
224. ![](https://secure.gravatar.com/avatar/1d60457d1ba39b758a40d9e5dd4d142465d0f85420a9224460034d02f15416a2?s=40&d=mm&r=g)



     Frank LuMay 14, 2018 at 7:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437499 "Direct link to this comment")





     Great tutorial very helpful ,then I have a question .Which accounted for the largest proportion in 8 inputs? We have 8 factors in the dataset like pregnancies, glucose, bloodpressure and the others. So , Which factor is most related to diabetes used? How do we know this proportion through MLP?


     Thanks！



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437499)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 15, 2018 at 7:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437529 "Direct link to this comment")





       We might not know. This is the difference between descriptive and predictive models.



       This is really the issue of model interpretability, I write more about it here:

       [https://machinelearningmastery.com/faq/single-faq/how-do-i-interpret-the-predictions-from-my-model](https://machinelearningmastery.com/faq/single-faq/how-do-i-interpret-the-predictions-from-my-model)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437529)
225. ![](https://secure.gravatar.com/avatar/31fba75c63d289492984a810afdef09853dfe127bd9ca8823d788d770568fe19?s=40&d=mm&r=g)



     PaoloMay 16, 2018 at 7:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437683 "Direct link to this comment")





     Hi Jason,


     thanks for your tutorials.



     I have a question, do you use keras with pandas too? In this case, it is better to import data wih numpy anyway? What do you suggest?



     Thank you again,


     Paolo



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437683)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 17, 2018 at 6:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437724 "Direct link to this comment")





       Yes, and yes.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-437724)




       - ![](https://secure.gravatar.com/avatar/bb6864c5a7862a40a2a410820eec1dd768318020391ad69d90c4b115b22130d4?s=40&d=mm&r=g)



         StefanNovember 10, 2018 at 1:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454222 "Direct link to this comment")





         How so? I usually see pandas.readcsv() to read files. Does keras only accept numpy arrays?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454222)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)November 10, 2018 at 6:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454264 "Direct link to this comment")





           Correct.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454264)
226. ![](https://secure.gravatar.com/avatar/8d0beab566eb2cdebf51e2e7ba08980668bd7a6c61be9c0907e975b04ebe12a8?s=40&d=mm&r=g)



     zohrehMay 20, 2018 at 9:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438027 "Direct link to this comment")





     Thanks for your great tutorial. I have a credit card dataset and I want to do fraud detection on it. it has 312 columns, So before doing DNN, I should do dimension reduction, then using DNN? and another question is that Is it possible to do CNN on my dataset as well?



     Thank you



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438027)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 21, 2018 at 6:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438085 "Direct link to this comment")





       Yes, choose the features that best map to the output variable.



       A CNN can be used if there is a spatial relationship in the data, such as a sequence of transactions over space or time.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438085)




       - ![](https://secure.gravatar.com/avatar/8d0beab566eb2cdebf51e2e7ba08980668bd7a6c61be9c0907e975b04ebe12a8?s=40&d=mm&r=g)



         zohrehMay 23, 2018 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438297 "Direct link to this comment")





         Thanks for your answer, So I think CNN doesn’t make sense for my dataset,


         Do you have any tutorial for active learning?


         thanks for your time.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438297)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)May 23, 2018 at 2:37 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438325 "Direct link to this comment")





           I don’t know if it is appropriate, I was trying to provide enough information for you to make that call.



           I hope to cover active learning in the future.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438325)




           - ![](https://secure.gravatar.com/avatar/8d0beab566eb2cdebf51e2e7ba08980668bd7a6c61be9c0907e975b04ebe12a8?s=40&d=mm&r=g)



             zohrehMay 24, 2018 at 3:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438390 "Direct link to this comment")





             yes I understand, I said according to your provided information, thank you so much for your answers and great tutorials.
227. ![](https://secure.gravatar.com/avatar/727cbfdcf3421bbe5a6eaac5f37205906105239fd835b3f7c6dbea917056ac8f?s=40&d=mm&r=g)



     Miguel GarcíaMay 24, 2018 at 11:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438429 "Direct link to this comment")





     Can you share a tutorial for first neural netowrk with multilabel support?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438429)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 24, 2018 at 1:51 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438441 "Direct link to this comment")





       Thanks for the suggestion.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438441)
228. ![](https://secure.gravatar.com/avatar/2cad328e8eb4cf750e668d07af5704d86e68f408afd47748c64e14e752ada31a?s=40&d=mm&r=g)



     SathishMay 24, 2018 at 12:57 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438435 "Direct link to this comment")





     how to create convolutional layers and visualize features in keras



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438435)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 24, 2018 at 1:51 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438442 "Direct link to this comment")





       Good question, sorry, I don’t have a worked example.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438442)
229. ![](https://secure.gravatar.com/avatar/ca2fa1a681f0b9196a8b3aebee012924d0f78fff69468fe3ca7a7f3e11c9e162?s=40&d=mm&r=g)



     AnamMay 28, 2018 at 3:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438862 "Direct link to this comment")





     Dear Jason,


     I get an error”ValueError: could not convert string to float: “Kindly help to solve the issue.And I am using my own dataset which consist of text not numbers(like the dataset you have used).


     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438862)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 28, 2018 at 6:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438880 "Direct link to this comment")





       This might give you some ideas:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438880)
230. ![](https://secure.gravatar.com/avatar/ca2fa1a681f0b9196a8b3aebee012924d0f78fff69468fe3ca7a7f3e11c9e162?s=40&d=mm&r=g)



     AnamMay 29, 2018 at 7:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438975 "Direct link to this comment")





     Dear Jason,


     I am running your code example from section 6.But I get an error in the following code snippet:



     Code Snippet:


     dataset = numpy.loadtxt(“pima\_indians.csv”, delimiter=”,”)


     \# split into input (X) and output (Y) variables


     X = dataset\[:,0:8\]


     Y = dataset\[:,8\]



     Error:


     ValueError: could not convert string to float: “6



     Kindly guide me to solve the issue. Thanks for your precious time.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438975)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 29, 2018 at 2:49 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438989 "Direct link to this comment")





       I’m sorry to hear that, I have some suggestions here:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-438989)

     - ![](https://secure.gravatar.com/avatar/d45862b1b0fcfe8824a55eb79f4da36dfc0c0089648328a1234bc6623fdfbad3?s=40&d=mm&r=g)



       Gautam SharmaJune 19, 2018 at 1:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441194 "Direct link to this comment")





       Did you find any solution as I am getting the same error?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441194)
231. ![](https://secure.gravatar.com/avatar/b526dcae4330feca556f914a5f152628f86dd2574751108af9fd6dfe3626b714?s=40&d=mm&r=g)



     [moti](http://no/)June 4, 2018 at 3:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439669 "Direct link to this comment")





     Hi Doctor, in this python code where shall I get the “keras” package?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439669)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 4, 2018 at 6:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439698 "Direct link to this comment")





       This tutorial shows you how to install Keras:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439698)
232. ![](https://secure.gravatar.com/avatar/dd38791572bd609484955e7388606db0858118be05070d4d744ca109d49c26b5?s=40&d=mm&r=g)



     Ammara HabibJune 5, 2018 at 5:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439814 "Direct link to this comment")





     Hy jason, Thanks for an amazing post. I have a question here that can we use dense layer as input for text classification(e.g : sentiment classification of movie reviews).If yes than how can we convert the text dataset into numeric for dense layer.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439814)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 5, 2018 at 6:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439838 "Direct link to this comment")





       You can, although it is common to one hot encode the text or use an embedding layer.



       I have examples of both on the blog.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439838)
233. ![](https://secure.gravatar.com/avatar/dd38791572bd609484955e7388606db0858118be05070d4d744ca109d49c26b5?s=40&d=mm&r=g)



     Ammara HabibJune 5, 2018 at 9:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439848 "Direct link to this comment")





     Thanks for your precious time.Sir, you mean that first i use embedding layer as input layer and then i use dense layer as the hidden layer?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439848)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 5, 2018 at 3:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439867 "Direct link to this comment")





       Yes.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-439867)
234. ![](https://secure.gravatar.com/avatar/f261e62d7613f4235f3feb05ebfce2fefe4dff289151e1bb3ebf71cb62370e4e?s=40&d=mm&r=g)



     Lisa XieJune 15, 2018 at 1:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-440886 "Direct link to this comment")





     Hi,thanks for your tutorial. I am wondering how you set the number neurons and activation functions for each layer, eg. 12 neurons for the 1st layer and 8 for the second.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-440886)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 15, 2018 at 2:50 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-440894 "Direct link to this comment")





       I used a little trial and error.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-440894)
235. ![](https://secure.gravatar.com/avatar/e3ce5f4659dd50925d9880bf467fb3959bc10323fda3acf007085671ee74ce46?s=40&d=mm&r=g)



     MarwaJune 18, 2018 at 1:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441091 "Direct link to this comment")





     Hi jason,



     I developped two neural networks using keras but I have this error:



     line 1336, in \_do\_call


     raise type(e)(node\_def, op, message)



     ResourceExhaustedError: OOM when allocating tensor with shape\[7082368,50\]


     \[\[Node: training\_1/Adam/Variable\_14/Assign = Assign\[T=DT\_FLOAT, \_class=\[“loc:@training\_1/Adam/Variable\_14″\], use\_locking=true, validate\_shape=true, \_device=”/job:localhost/replica:0/task:0/device:GPU:0”\](training\_1/Adam/Variable\_14, training\_1/Adam/zeros\_14)\]\]



     Have you an idea?


     Thanks.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441091)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 18, 2018 at 6:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441124 "Direct link to this comment")





       Sorry, I have not seen this error before. Perhaps try posting/searching on stackoverflow?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441124)
236. ![](https://secure.gravatar.com/avatar/3223e77927db6d5abaf9cfa1c981e460301fd2767bf34f37e351b4adf811deaa?s=40&d=mm&r=g)



     prateek bhadauriaJune 23, 2018 at 11:38 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441796 "Direct link to this comment")





     sir i have a regression related dataset which contains an array of 49999 rows and 20 coloumns , i want to implement CNN on this dataset ,



     i put my code as per my perception kindly give me suggestion , to correct it i was stuck mainly by putting my dense dimension specially



     from keras.models import Sequential


     from keras.layers import Dense


     import numpy as np


     import tensorflow as tf


     from matplotlib import pyplot


     from sklearn.datasets import make\_regression


     from sklearn.preprocessing import MinMaxScaler


     from sklearn.metrics import mean\_squared\_error


     from keras.wrappers.scikit\_learn import KerasRegressor


     from sklearn.preprocessing import StandardScaler


     from keras.layers import Dense, Dropout, Flatten


     from keras.layers import Conv2D, MaxPooling2D


     from keras.optimizers import SGD



     seed = 7


     np.random.seed(seed)


     from scipy.io import loadmat


     dataset = loadmat(‘matlab2.mat’)


     Bx=basantix\[:, 50001:99999\]


     Bx=np.transpose(Bx)


     Fx=fx\[:, 50001:99999\]


     Fx=np.transpose(Fx)



     from sklearn.cross\_validation import train\_test\_split


     Bx\_train, Bx\_test, Fx\_train, Fx\_test = train\_test\_split(Bx, Fx, test\_size=0.2, random\_state=0)



     scaler = StandardScaler() # Class is create as Scaler


     scaler.fit(Bx\_train) # Then object is created or to fit the data into it


     Bx\_train = scaler.transform(Bx\_train)


     Bx\_test = scaler.transform(Bx\_test)



     model = Sequential()


     def base\_model():



     keras.layers.Dense(Dense(49999, input\_shape=(20,), activation=’relu’))


     model.add(Dense(20))


     model.add(Dense(49998, init=’normal’, activation=’relu’))


     model.add(Dense(49998, init=’normal’))


     model.compile(loss=’mean\_squared\_error’, optimizer = ‘adam’)


     return model



     scale = StandardScaler()


     Bx = scale.fit\_transform(Bx)


     Bx = scale.fit\_transform(Bx)



     clf = KerasRegressor(build\_fn=base\_model, nb\_epoch=100, batch\_size=5,verbose=0)



     clf.fit(Bx,Fx)


     res = clf.predict(Bx)



     \## line below throws an error


     clf.score(Fx,res)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441796)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 24, 2018 at 7:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441839 "Direct link to this comment")





       Sorry, I cannot debug your code for you. Perhaps post your code and error to stackoverflow?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441839)
237. ![](https://secure.gravatar.com/avatar/edd12f7f19c321d99952207a6a8a5a0aa3bb758336ddb851650bfbe7bf7b1725?s=40&d=mm&r=g)



     Madhav PrakashJune 24, 2018 at 3:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441809 "Direct link to this comment")





     Hi Jason,


     Looking at the dataset, I could find that there were many attributes with each of them differing in terms of units. Why haven’t you rescaled/normalised the data? but still managed to get an accuracy of 75%?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441809)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 24, 2018 at 7:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441842 "Direct link to this comment")





       Ideally, we should rescale the data.



       The relu activation function is more flexible with unscaled data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441842)




       - ![](https://secure.gravatar.com/avatar/edd12f7f19c321d99952207a6a8a5a0aa3bb758336ddb851650bfbe7bf7b1725?s=40&d=mm&r=g)



         Madhav PrakashJune 24, 2018 at 4:23 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441869 "Direct link to this comment")





         Ohkay, thanks.


         Also, I’ve implemented a NN on a database similar to this, where the accuracy varies b/w 70-75%. I’ve tried to increase the accuracy by tuning various parameters and functions (learning rate, no. of layers, neurons per level, earlystopping, activation fn, initialization, optimizer etc…) but it was not a success. My question is when do i come to know that i’ve reached the maximum accuracy possible for my implementation? Do i stay content with the current accuracy?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441869)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)June 25, 2018 at 6:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441934 "Direct link to this comment")





           When we run out of time or ideas.



           I list some more ideas here:

           [https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)



           And here:

           [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-441934)
238. ![](https://secure.gravatar.com/avatar/d4e463703d0ebe18d633415ed721f92d84fe0557c835eb632c761c212c7ee3e7?s=40&d=mm&r=g)



     Aarron WilsonJuly 8, 2018 at 8:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442947 "Direct link to this comment")





     First of all thanks for the tutorial. Also I acknowledge that this network is more for educational purposes. Yet this network can be improved to 83-84% accuracy with standard normalization alone. Also it can hit 93-95% accuracy by using a deeper model.



     #Standard normalization


     X= StandardScaler().fit\_transform(X)



     #and a deeper model


     model = Sequential()


     model.add(Dense(12, input\_dim=8, activation=’relu’))


     model.add(Dense(12, activation=’relu’))


     model.add(Dense(12, activation=’relu’))


     model.add(Dense(12, activation=’relu’))


     model.add(Dense(12, activation=’relu’))


     model.add(Dense(8, activation=’relu’))


     model.add(Dense(1, activation=’sigmoid’))



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442947)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 9, 2018 at 6:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442988 "Direct link to this comment")





       Thanks, yes, normalization is a good idea in general when working with neural nets.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-442988)
239. ![](https://secure.gravatar.com/avatar/8478a960d8b387f98eedb441039dbf7731f7ac5737ac2714ea20504ebe5d3633?s=40&d=mm&r=g)



     AlexJuly 10, 2018 at 3:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443055 "Direct link to this comment")





     Hi, thank you for this great article



     Imagine that in my dataset instead of diabetes being a 0 or 1 I have 3 results, I mean, the data rows are like this



     data1, data2, sickness


     123, 124, 0


     142, 541, 0


     156, 418, 1


     142, 541, 1


     156, 418, 2



     So, I need to categorize for 3 values, If I use this same example you gave us how can I determine the output?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443055)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 10, 2018 at 6:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443089 "Direct link to this comment")





       The output will be sickness Alex. Perhaps I don’t understand your question?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443089)




       - ![](https://secure.gravatar.com/avatar/8478a960d8b387f98eedb441039dbf7731f7ac5737ac2714ea20504ebe5d3633?s=40&d=mm&r=g)



         AlexJuly 10, 2018 at 7:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443097 "Direct link to this comment")





         The output will be sickness yes



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443097)
240. ![](https://secure.gravatar.com/avatar/8478a960d8b387f98eedb441039dbf7731f7ac5737ac2714ea20504ebe5d3633?s=40&d=mm&r=g)



     AlexJuly 10, 2018 at 10:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443106 "Direct link to this comment")





     Sorry for my English, it is not my natal tongue, I will re do my quesyion. What I mean is this, I will be having a label with more than 2 results, 0 is one sickness, 1 will be other and 2 will be other.



     How can I use the model you showed us to fit the 3 results?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443106)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 10, 2018 at 2:26 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443119 "Direct link to this comment")





       I see, this is called a multi-class classification problem.



       This tutorial will help:

       [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443119)
241. ![](https://secure.gravatar.com/avatar/f0b7ae172b6f3ad46db5562f3f429502d6cd77e9ac7cfb5159c3e8adb483a883?s=40&d=mm&r=g)



     adsadJuly 11, 2018 at 1:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443157 "Direct link to this comment")





     is it possible to predict the lottery outcome. if so how?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443157)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2018 at 5:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443180 "Direct link to this comment")





       No. I explain more here:

       [https://machinelearningmastery.com/faq/single-faq/can-i-use-machine-learning-to-predict-the-lottery](https://machinelearningmastery.com/faq/single-faq/can-i-use-machine-learning-to-predict-the-lottery)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443180)
242. ![](https://secure.gravatar.com/avatar/bd3aa78232a3d06400628548d31f15c15c524313f5f9be5e780dcc07aaf4f458?s=40&d=mm&r=g)



     TomJuly 14, 2018 at 2:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443400 "Direct link to this comment")





     Hi Jason, I run your first example code in this tutorial. but what makes me confused is:



     Why the final training accuracy (0.7656) is different from the evaluated scores (78.26%) in the same datasets (training set) ? I can’t figure it out. Can you tell me please? Thanks a lot!



     Epoch 150/150


     768/768 \[==============================\] – 0s – loss: 0.4827 – acc: 0.7656


     32/768 \[>………………………..\] – ETA: 0s


     acc: 78.26%



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443400)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 14, 2018 at 6:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443434 "Direct link to this comment")





       One is the performance on the training set, the other on the validation set.



       You can learn more about the difference here:

       [https://machinelearningmastery.com/difference-test-validation-datasets/](https://machinelearningmastery.com/difference-test-validation-datasets/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443434)
243. ![](https://secure.gravatar.com/avatar/bd3aa78232a3d06400628548d31f15c15c524313f5f9be5e780dcc07aaf4f458?s=40&d=mm&r=g)



     TomJuly 14, 2018 at 9:09 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443482 "Direct link to this comment")





     Thanks for the rapid reply. But I noticed that in your code the training set and validation set are exactly the same dataset. Please check it for confirmation. The code is in the part “6. Tie It All Together”.



     \# Fit the model


     model.fit(X, Y, epochs=150, batch\_size=10)


     \# evaluate the model


     scores = model.evaluate(X, Y)



     So, my problem is still the same: Why the final training accuracy (0.7656) is different from the evaluated scores (78.26%) in the same datasets?


     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443482)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 15, 2018 at 6:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443517 "Direct link to this comment")





       Perhaps verbose output might be accumulated over each batch rather than summarizing skill at the end of the training epoch.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443517)
244. ![](https://secure.gravatar.com/avatar/ce9b8f1c9cd693f045fe5a54c0b310ebed6330199464c10874106941c5ec83c9?s=40&d=mm&r=g)



     amiJuly 16, 2018 at 2:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443564 "Direct link to this comment")





     Hello Jason,


     Do you have some tutorial on signal processing using CNN ? I have csv files of some biomedical signals like ECG and i want to classify normal and abnormal signals using deep learning.



     With Regards



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443564)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 16, 2018 at 6:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443575 "Direct link to this comment")





       Yes, I have a suite of tutorials scheduled on this topic. They should be out soon.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443575)
245. ![](https://secure.gravatar.com/avatar/a61b90d68e1ba4c0b1b4a16395ed5c9ddbd5b3d1ce95d1a9af4c33fedf794f4e?s=40&d=mm&r=g)



     ELJuly 16, 2018 at 7:19 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443615 "Direct link to this comment")





     Hi, thank you so much for your tutorial. I am trying to make a neural network that will take a dataset and return if it is suitable to be analyzed by another program i have. Is it possible to feed this with acceptable datasets and unacceptable datasets and then call it on a new dataset and then return whether this dataset is acceptable? Thank you for your help, I am very new to machine learning.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443615)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 17, 2018 at 6:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443658 "Direct link to this comment")





       Try it and see how you go.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443658)
246. ![](https://secure.gravatar.com/avatar/ce9b8f1c9cd693f045fe5a54c0b310ebed6330199464c10874106941c5ec83c9?s=40&d=mm&r=g)



     amiJuly 18, 2018 at 2:37 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443784 "Direct link to this comment")





     Oh really ! Thank you so much. Can you please notify me when the tutorials will be out because i am doing a project and i am stuck right now.



     With Regards



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-443784)

247. ![](https://secure.gravatar.com/avatar/763de47e829990e8686732539d10e4bddb26498c20a8626a7f93eb84c2b62ce9?s=40&d=mm&r=g)



     DiagramsJuly 30, 2018 at 2:45 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444792 "Direct link to this comment")





     It would be very very helpful for newcomers if you had a diagram of the network, showing individual nodes and graph edges (and bias nodes and activation functions), and indicating on it which parts were generated by which model.add commands/parameters. Similar to [https://zhu45.org/posts/2017/May/25/draw-a-neural-network-through-graphviz/](https://zhu45.org/posts/2017/May/25/draw-a-neural-network-through-graphviz/)



     I’ve tried visualizing it with from keras.utils.plot\_model and tensorboard, but neither produce a node-level diagram.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444792)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 31, 2018 at 5:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444827 "Direct link to this comment")





       Thanks for the suggestion.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444827)
248. ![](https://secure.gravatar.com/avatar/341fa85ff6b8c74c1f737db46cd6da2a3c1c95aa6803576a40cd02e92c83b648?s=40&d=mm&r=g)



     AravindJuly 30, 2018 at 7:57 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444801 "Direct link to this comment")





     can anyone tell a simple way to run my ann keras tensorflow backend in GPU. Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444801)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 31, 2018 at 6:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444830 "Direct link to this comment")





       The simplest way I know how:

       [https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-444830)
249. ![](https://secure.gravatar.com/avatar/2b483c23c4b0153356501f56ec9248fa056f2cc9de85cc0c64d1f7c7e2779dfc?s=40&d=mm&r=g)



     farliAugust 6, 2018 at 1:08 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445332 "Direct link to this comment")





     Did you use back propagation here?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445332)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 6, 2018 at 2:54 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445337 "Direct link to this comment")





       Yes.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445337)




       - ![](https://secure.gravatar.com/avatar/2b483c23c4b0153356501f56ec9248fa056f2cc9de85cc0c64d1f7c7e2779dfc?s=40&d=mm&r=g)



         farliAugust 13, 2018 at 9:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445975 "Direct link to this comment")





         Can you please make a tutorial on convolutional neural net? That would be really helpful ..:)



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445975)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)August 13, 2018 at 2:27 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445982 "Direct link to this comment")





           Yes, i have many on the blog already. Try the blog search.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445982)
250. ![](https://secure.gravatar.com/avatar/4ed3d04bca6467a839f7a4f878bc15737c3c4afa9cb3a5184e0062c73429cff2?s=40&d=mm&r=g)



     Karim GamalAugust 7, 2018 at 8:52 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445467 "Direct link to this comment")





     I have a problem where I get the result as shown below



     Epoch 146/150 – 0s – loss: -1.2037e+03 – acc: 0.0000e +00


     Epoch 147/150 – 0s – loss: -1.2037e+03 – acc: 0.0000e +00


     Epoch 148/150 – 0s – loss: -1.2037e+03 – acc: 0.0000e +00


     Epoch 149/150 – 0s – loss: -1.2037e+03 – acc: 0.0000e +00


     Epoch 150/150 – 0s – loss: -1.2037e+03 – acc: 0.0000e +00



     where in my data set the output is a value between 0 to 500 not only 0 and 1


     so how can I fix this in my code



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445467)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 8, 2018 at 6:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445516 "Direct link to this comment")





       Sounds like a regression problem. Change the activation function in the output layer to linear and the loss function to ‘mse’.



       See this tutorial:

       [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-445516)
251. ![](https://secure.gravatar.com/avatar/3b2a730d77959614321cccd79a5f2e6ae99fddbcbc1a0a1140f2c8239b6258ef?s=40&d=mm&r=g)



     [Tim](http://none/)August 15, 2018 at 5:54 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446117 "Direct link to this comment")





     AWESOME!!! Thanks so much for this.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446117)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 15, 2018 at 6:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446143 "Direct link to this comment")





       You’re welcome, I’m happy it helped.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-446143)
252. ![](https://secure.gravatar.com/avatar/11dbe708b663a5204d81a0f580475943bccbef72f75eff73818be545a4d0e54f?s=40&d=mm&r=g)



     taniaAugust 27, 2018 at 8:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-447069 "Direct link to this comment")





     Hi Jason,



     Thank you for the tutorial. I am relatively new to ML and I am currently working on a classification problem that is non binary.



     My dataset consists of a number of labeled samples – all measuring the same quantity/unit. The amount typically ranges from 10 to 20 labeled samples/inputs. However, the feed forward or testing sample will only contain 7 of those inputs (at random).



     I’m struggling to find a solution to designing a system that accepts fewer inputs than what is typically found in the training set.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-447069)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 28, 2018 at 5:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-447096 "Direct link to this comment")





       Perhaps try following this process:

       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-447096)
253. ![](https://secure.gravatar.com/avatar/8bc6583719bc529b474acb0efe993fb4393299c869db2cb3468f6b89fef4bdaf?s=40&d=mm&r=g)



     Vaibhav JaiswalSeptember 10, 2018 at 6:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-448318 "Direct link to this comment")





     Great tutorial there! But the main aspect of the model is to predict on a sample. If i print the first predicted value,it shows me some values for all the columns of categorical features. How to get the predicted number from the sample?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-448318)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 11, 2018 at 6:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-448355 "Direct link to this comment")





       The order of the predictions matches the order of the inputs.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-448355)
254. ![](https://secure.gravatar.com/avatar/67134c81db5d01faba8aefffa04de934139ee4d90a92155e86d3e196b225290f?s=40&d=mm&r=g)



     GlenSeptember 19, 2018 at 10:45 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449337 "Direct link to this comment")





     I think I must be doing something wrong, I keep getting the error:


     File “C:\\Users\\glens\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\errors\_impl.py”, line 519, in \_\_exit\_\_


     c\_api.TF\_GetCode(self.status.status))



     InvalidArgumentError: Input to reshape is a tensor with 10 values, but the requested shape has 1


     \[\[Node: training\_19/Adam/gradients/loss\_21/dense\_64\_loss/Mean\_1\_grad/Reshape = Reshape\[T=DT\_FLOAT, Tshape=DT\_INT32, \_class=\[“loc:@training\_19/Adam/gradients/loss\_21/dense\_64\_loss/Mean\_1\_grad/truediv”\], \_device=”/job:localhost/replica:0/task:0/device:GPU:0″\](training\_19/Adam/gradients/loss\_21/dense\_64\_loss/mul\_grad/Sum, training\_19/Adam/gradients/loss\_21/dense\_64\_loss/Mean\_1\_grad/DynamicStitch/\_1703)\]\]



     Are you able to shed any light on why I would get this error?



     Thankyou



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449337)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 20, 2018 at 7:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449367 "Direct link to this comment")





       I have not seen this error, I have some suggestions here:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449367)
255. ![](https://secure.gravatar.com/avatar/d894cac9cf13bac29c719267d9887dfe5ce89d4aee574c27d65e243274e971ac?s=40&d=mm&r=g)



     SnehasishSeptember 19, 2018 at 11:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449339 "Direct link to this comment")





     Hi Jason, thanks for this awesome tutorial. I have one doubt – why did the evaluation not produce 100% accuracy? After all, we used the same dataset for evaluation as the one used for training itself.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449339)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 20, 2018 at 8:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449368 "Direct link to this comment")





       Good question!



       We are approximating a challenging mapping function, not memorizing examples. As such, there will always be error.



       I explain more here:

       [https://machinelearningmastery.com/faq/single-faq/why-cant-i-get-100-accuracy-or-zero-error-with-my-model](https://machinelearningmastery.com/faq/single-faq/why-cant-i-get-100-accuracy-or-zero-error-with-my-model)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449368)
256. ![](https://secure.gravatar.com/avatar/0d0d001ced670edd125cb85a23c8f208804e305f16ff149ddcf2cb13df3ef486?s=40&d=mm&r=g)



     Mark CSeptember 27, 2018 at 12:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449873 "Direct link to this comment")





     How do you predict something you want to predict such as new data. for example I did a spam detection but dont know how to predict whether a sentence i write is spam or not .



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449873)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 27, 2018 at 6:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449895 "Direct link to this comment")





       You can call model.predict() with a finalized model. More here:

       [https://machinelearningmastery.com/faq/single-faq/how-do-i-make-predictions](https://machinelearningmastery.com/faq/single-faq/how-do-i-make-predictions)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-449895)
257. ![](https://secure.gravatar.com/avatar/ea84e0212b31a49f0592c0f20c89047bdda3a9278c6479ee5b66ebc71e73f053?s=40&d=mm&r=g)



     VivekOctober 1, 2018 at 3:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450335 "Direct link to this comment")





     Hello Sir,



     I am new and understood some part of your code. I have question in prediction model basically we divide our data into training and test set. In the example above the entire dataset is used as training dataset. How can we train the model on training set use it for the prediction on test set?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450335)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 1, 2018 at 6:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450355 "Direct link to this comment")





       Great question, yes, train the model on all available data and then use it to start making predictions.



       More here:

       [https://machinelearningmastery.com/train-final-machine-learning-model/](https://machinelearningmastery.com/train-final-machine-learning-model/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450355)
258. ![](https://secure.gravatar.com/avatar/ea84e0212b31a49f0592c0f20c89047bdda3a9278c6479ee5b66ebc71e73f053?s=40&d=mm&r=g)



     Vivek35October 1, 2018 at 7:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450365 "Direct link to this comment")





     Hello Sir,


     It’s great tutorial to understand. However, I am new and want to understand something out of it. In the above code we have treated entire dataset as training set. Can we divide this into training set and test set, apply model to training set and use it for test set prediction.How can we achieve with the above code?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450365)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 1, 2018 at 2:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450386 "Direct link to this comment")





       Thanks.



       Yes, you can split the dataset manually or use scikit-learn to make the split for you. I explain more here:

       [https://machinelearningmastery.com/faq/single-faq/how-do-i-evaluate-a-machine-learning-algorithm](https://machinelearningmastery.com/faq/single-faq/how-do-i-evaluate-a-machine-learning-algorithm)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450386)
259. ![](https://secure.gravatar.com/avatar/674bd5ad3a7585306b5b1ba9e9f91cb698b951cb28452dca8eb3f3f26a790b5e?s=40&d=mm&r=g)



     LipiOctober 5, 2018 at 6:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450742 "Direct link to this comment")





     Hi Jason,



     I am trying to predict using my neural network. I have used MinMaxScaler in the features while training the data. I don’t get a good prediction if I don’t use the same transform function on the prediction data set which I used on the features while training the data. Could you suggest me the correct approach in this situation?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450742)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 5, 2018 at 2:29 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450762 "Direct link to this comment")





       You must use the same transform to both prepare training data and to make predictions on new data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450762)




       - ![](https://secure.gravatar.com/avatar/674bd5ad3a7585306b5b1ba9e9f91cb698b951cb28452dca8eb3f3f26a790b5e?s=40&d=mm&r=g)



         LipiOctober 5, 2018 at 10:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450783 "Direct link to this comment")





         Thank you!



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450783)
260. ![](https://secure.gravatar.com/avatar/d0ce2e5f49ab105dfbcf56d88b75d8621e751881cc4a48bd2b416d3b0a127fbd?s=40&d=mm&r=g)



     neenuOctober 6, 2018 at 3:57 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450849 "Direct link to this comment")





     hi i am new to this i writew following code in spyder


     from keras.models import Sequential


     from keras.layers import Dense


     import numpy


     \# fix random seed for reproducibility


     numpy.random.seed(7)


     \# load pima indians dataset


     dataset = numpy.loadtxt(“pima-indians-diabetes.txt”,encoding=”UTF8″, delimiter=”,”)


     \# split into input (X) and output (Y) variables


     X = dataset\[:,0:8\]


     Y = dataset\[:,8\]



     \# create model


     model = Sequential()


     model.add(Dense(12, input\_dim=8, activation=’relu’))


     model.add(Dense(8, activation=’relu’))


     model.add(Dense(1, activation=’sigmoid’))


     \# Compile model


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     \# Fit the model


     model.fit(X, Y, epochs=150, batch\_size=10)


     \# evaluate the model


     scores = model.evaluate(X, Y)


     print(“\\n%s: %.2f%%” % (model.metrics\_names\[1\], scores\[1\]\*100))



     And i got this as output



     runfile(‘C:/Users/DELL/Anaconda3/Scripts/temp.py’, wdir=’C:/Users/DELL/Anaconda3/Scripts’)


     Using TensorFlow backend.


     Traceback (most recent call last):



     File “”, line 1, in


     runfile(‘C:/Users/DELL/Anaconda3/Scripts/temp.py’, wdir=’C:/Users/DELL/Anaconda3/Scripts’)



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\spyder\_kernels\\customize\\spydercustomize.py”, line 668, in runfile


     execfile(filename, namespace)



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\spyder\_kernels\\customize\\spydercustomize.py”, line 108, in execfile


     exec(compile(f.read(), filename, ‘exec’), namespace)



     File “C:/Users/DELL/Anaconda3/Scripts/temp.py”, line 1, in


     from keras.models import Sequential



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\keras\\\_\_init\_\_.py”, line 3, in


     from . import utils



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\keras\\utils\\\_\_init\_\_.py”, line 6, in


     from . import conv\_utils



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\keras\\utils\\conv\_utils.py”, line 9, in


     from .. import backend as K



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\keras\\backend\\\_\_init\_\_.py”, line 89, in


     from .tensorflow\_backend import \*



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow\_backend.py”, line 5, in


     import tensorflow as tf



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\tensorflow\\\_\_init\_\_.py”, line 22, in


     from tensorflow.python import pywrap\_tensorflow # pylint: disable=unused-import



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\\_\_init\_\_.py”, line 49, in


     from tensorflow.python import pywrap\_tensorflow



     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\pywrap\_tensorflow.py”, line 74, in


     raise ImportError(msg)



     ImportError: Traceback (most recent call last):


     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\pywrap\_tensorflow\_internal.py”, line 14, in swig\_import\_helper


     return importlib.import\_module(mname)


     File “C:\\Users\\DELL\\Anaconda3\\lib\\importlib\\\_\_init\_\_.py”, line 126, in import\_module


     return \_bootstrap.\_gcd\_import(name\[level:\], package, level)


     File “”, line 994, in \_gcd\_import


     File “”, line 971, in \_find\_and\_load


     File “”, line 955, in \_find\_and\_load\_unlocked


     File “”, line 658, in \_load\_unlocked


     File “”, line 571, in module\_from\_spec


     File “”, line 922, in create\_module


     File “”, line 219, in \_call\_with\_frames\_removed


     ImportError: DLL load failed with error code -1073741795



     During handling of the above exception, another exception occurred:



     Traceback (most recent call last):


     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\pywrap\_tensorflow.py”, line 58, in


     from tensorflow.python.pywrap\_tensorflow\_internal import \*


     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\pywrap\_tensorflow\_internal.py”, line 17, in


     \_pywrap\_tensorflow\_internal = swig\_import\_helper()


     File “C:\\Users\\DELL\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\pywrap\_tensorflow\_internal.py”, line 16, in swig\_import\_helper


     return importlib.import\_module(‘\_pywrap\_tensorflow\_internal’)


     File “C:\\Users\\DELL\\Anaconda3\\lib\\importlib\\\_\_init\_\_.py”, line 126, in import\_module


     return \_bootstrap.\_gcd\_import(name\[level:\], package, level)


     ModuleNotFoundError: No module named ‘\_pywrap\_tensorflow\_internal’



     Failed to load the native TensorFlow runtime.



     See [https://www.tensorflow.org/install/install\_sources#common\_installation\_problems](https://www.tensorflow.org/install/install_sources#common_installation_problems)



     for some common reasons and solutions. Include the entire stack trace


     above this error message when asking for help.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450849)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 7, 2018 at 7:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450894 "Direct link to this comment")





       I recommend this tutorial to help you setup your environment:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       I recommend that you don’t use am IDE or notebook:

       [https://machinelearningmastery.com/faq/single-faq/why-dont-use-or-recommend-notebooks](https://machinelearningmastery.com/faq/single-faq/why-dont-use-or-recommend-notebooks)



       Instead, I recommend you save code to a .py file and run from the command line:

       [https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line](https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-450894)
261. ![](https://secure.gravatar.com/avatar/c7ce40186e6030c6cd631991460a52f31a1510365b727a0d0d055fe6d7aa24d3?s=40&d=mm&r=g)



     kamalOctober 15, 2018 at 1:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-451623 "Direct link to this comment")





     sir please provide the python code for adaptive neuro fuzzy classifier



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-451623)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 15, 2018 at 7:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-451654 "Direct link to this comment")





       Thanks for the suggestion.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-451654)




       - ![](https://secure.gravatar.com/avatar/cb54697cd91d76f40ed6d3119efb4e5c72118204ac6bc3e179dd6154faa074b8?s=40&d=mm&r=g)



         Rajan KumarJune 29, 2021 at 3:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-614958 "Direct link to this comment")





         I am waiting too for it.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-614958)
262. ![](https://secure.gravatar.com/avatar/5250e263fd2bb271c362c06d8a2f9d2b444705e0d063420e340b5b77f24ec9ea?s=40&d=mm&r=g)



     ShahbazOctober 24, 2018 at 4:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-452504 "Direct link to this comment")





     blessed on u sir,


     can u give me idea about OCR system, for my final year project, plz give me back-end stratigy for OCR , r u have any code on OCR



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-452504)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 24, 2018 at 6:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-452526 "Direct link to this comment")





       Perhaps start here:

       [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-452526)
263. ![](https://secure.gravatar.com/avatar/b017c74c77bb0133ad889b5a47ba3356e7508c2480842c4b67b71de3bc827649?s=40&d=mm&r=g)



     Andrew AgibOctober 29, 2018 at 10:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453064 "Direct link to this comment")





     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])



     show a syntax error on that sentence what could be the reason



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453064)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 30, 2018 at 6:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453096 "Direct link to this comment")





       I have some suggestions here:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453096)
264. ![](https://secure.gravatar.com/avatar/00533085a9a28ad2e817e4bae72c36698b432806f821333e00f965f9d0932748?s=40&d=mm&r=g)



     VASUDEV K PNovember 3, 2018 at 10:13 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453573 "Direct link to this comment")





     Hello Jason,



     I have the theano back end installed. I am using Windows OS and during execution I am getting an error “No module named TensorFlow”. Please help



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453573)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 4, 2018 at 6:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453603 "Direct link to this comment")





       You may have to change the configuration of Keras to use Theano instead.



       More details here:

       [https://keras.io/backend/](https://keras.io/backend/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453603)
265. ![](https://secure.gravatar.com/avatar/2f3979b65e819d822fbbbd91442383b3f5393f28e1b311b36c25f5727199c12e?s=40&d=mm&r=g)



     Imen DrsNovember 4, 2018 at 7:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453614 "Direct link to this comment")





     Hi Jason,


     Please,how can we calculate the precision and recall of this example?


     And thanks.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453614)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 5, 2018 at 6:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453678 "Direct link to this comment")





       You can use scikit-learn metrics:

       [http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-453678)
266. ![](https://secure.gravatar.com/avatar/bb6864c5a7862a40a2a410820eec1dd768318020391ad69d90c4b115b22130d4?s=40&d=mm&r=g)



     StefanNovember 10, 2018 at 2:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454229 "Direct link to this comment")





     I thought sigmoid and softmax were quite similar activation functions. But when trying the same model with softmax as activation for the last layer instead of sigmoid, my accuracy is much much worse.



     Does that make sense to you? If so why? I feel like I see softmax more often in other code than sigmoid.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454229)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 10, 2018 at 6:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454267 "Direct link to this comment")





       Nope.



       Sigmoid for 2 classes.


       Softmax for >2 classes



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454267)
267. ![](https://secure.gravatar.com/avatar/ed7105a3a225e1c29b8ebecd70879ab929e4bd7041b43f68df59d6514bafdb08?s=40&d=mm&r=g)



     Amuda KamorudeenNovember 10, 2018 at 4:46 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454305 "Direct link to this comment")





     I’m working on model that will predict propensity of customer that are likely to terminate their service with company. I have dataset of 70000 rows and 500 columns, Please how can I pass numeric data as an input to a convolutional neural network (CNN) .



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454305)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 11, 2018 at 5:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454365 "Direct link to this comment")





       CNNs are only appropriate for data with a spatial relationship, such as images, time series and text.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-454365)
268. ![](https://secure.gravatar.com/avatar/1ea1acce714ee8eddf2e92800b5d6c10150b5db8c35a3a4ca2f5d3e60b649f5d?s=40&d=mm&r=g)



     irfanNovember 18, 2018 at 3:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-455292 "Direct link to this comment")





     hi jason,



     i am using tensor flow as backend.


     from keras.models import Sequential


     from keras.layers import Dense


     import sys


     from keras import layers


     from keras.utils import plot\_model



     print (model.layer())



     erro.



     —————————————————————————


     AttributeError Traceback (most recent call last)


     in


     9 model.add(Dense(512, activation=’relu’))


     10 model.add(Dense(10, activation=’sigmoid’))


     —\> 11 print (model.layer())


     12 # Compile model


     13 model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])



     AttributeError: ‘Sequential’ object has no attribute ‘layer’



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-455292)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 19, 2018 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-455431 "Direct link to this comment")





       Why are you trying to print model.layer()?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-455431)
269. ![](https://secure.gravatar.com/avatar/8ba65e396a654fa000dbfe2fa951a3dba45140db0f325599ae4965ac9910bac0?s=40&d=mm&r=g)



     MarioDecember 2, 2018 at 5:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-456881 "Direct link to this comment")





     Hi Jason


     First thanks for amazing tutorial , since your scripts are using list of values while my inputs are list of 24×20 matrices which are filled out by values in especial order how they measured for 3 parameters in 3000 cycles , how can I feed this type matrice-data or let’s say how can I feed stream of images for 3 different parameters I already extracted from raw dataset and after preprocessing I convert them to 24\*20 matrices or .png images ? How should I change this script so that I can use my dataset?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-456881)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 2, 2018 at 6:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-456907 "Direct link to this comment")





       When using an MLP with images, you must flatten each matrix of pixel data to a single row vector.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-456907)
270. ![](https://secure.gravatar.com/avatar/fd99f5c4efbad3ff552612a9839fa8104a135015d3e1674fea52d104c437c505?s=40&d=mm&r=g)



     Evangelos ArgyropoulosDecember 18, 2018 at 6:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-458970 "Direct link to this comment")





     Hi Jason,


     Thank for tutorial. 1 questions.


     I use the algorithm for time series prediction 0=buy 1=sell. Does this model overfit?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-458970)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 18, 2018 at 6:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-458972 "Direct link to this comment")





       You can only know if you try fitting it and evaluating learning curves on train and validation datasets.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-458972)
271. ![](https://secure.gravatar.com/avatar/3c49eb174705de97d9f1d376e8d934e0176653c8ccb59a427013794967451205?s=40&d=mm&r=g)



     SOURAV MONDALDecember 28, 2018 at 7:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460394 "Direct link to this comment")





     Great tutorial Sir.


     Is there a way to visualize different layers with their nodes and interconnections among them, of a model created in keras (i mean the basic structure of a neural network with layers of nodes and interconnections among them).



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460394)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 29, 2018 at 5:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460500 "Direct link to this comment")





       Yes, check out this tutorial:

       [https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/](https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460500)
272. ![](https://secure.gravatar.com/avatar/2f3979b65e819d822fbbbd91442383b3f5393f28e1b311b36c25f5727199c12e?s=40&d=mm&r=g)



     Imen DrsDecember 28, 2018 at 11:29 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460473 "Direct link to this comment")





     Thanks for this tutorial.



     I have a problem when i try to compile and fit my model. It return value error : ValueError: could not convert string to float: ’24, 26, 99, 31, 623, 863, 77, 32, 362, 998, 1315, 33, 291, 14123, 39, 8, 335, 2308, 349, 403, 409, 1250, 417, 47, 1945, 50, 188, 51, 4493, 3343, 13419, 6107, 84, 18292, 339, 9655, 22498, 1871, 782, 1276, 2328, 56, 17633, 24004, 24236, 1901, 6112, 22506, 26397, 816, 502, 352, 24238, 18330, 7285, 2160, 220, 511, 17680, 68, 5137, 26398, 875, 542, 354, 2045, 555, 2145, 93, 327, 26399, 3158, 7501, 26400, 8215′ .



     Can you help me please.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460473)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 29, 2018 at 5:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460509 "Direct link to this comment")





       Perhaps your data contains a string?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460509)




       - ![](https://secure.gravatar.com/avatar/2f3979b65e819d822fbbbd91442383b3f5393f28e1b311b36c25f5727199c12e?s=40&d=mm&r=g)



         Imen DrsDecember 29, 2018 at 7:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460522 "Direct link to this comment")





         The data contains ” user, number\_of\_followers, list\_of\_followers, number\_of\_followee, list\_of\_followee, number\_of\_mentions, list\_of\_user\_mentioned…”


         the values in the list are separated by commas.


         For example: “36 ; 3 ; 52,3,87 ; 5 ; 63,785,22,11,6 ; 0 ; “



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-460522)
273. ![](https://secure.gravatar.com/avatar/4c112b742acb9523a3bd1d99bba9f168a29b39dc1243063b1d7306af33e6895c?s=40&d=mm&r=g)



     SomashekharJanuary 2, 2019 at 4:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461081 "Direct link to this comment")





     Hi, Is there a solution posted for solving pima-indians-diabetes.csv for prediction using LSTM?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461081)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 2, 2019 at 6:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461103 "Direct link to this comment")





       No. LSTMs are for sequential data only, and the pima indians dataset is not a sequence prediction problem.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461103)
274. ![](https://secure.gravatar.com/avatar/2f3979b65e819d822fbbbd91442383b3f5393f28e1b311b36c25f5727199c12e?s=40&d=mm&r=g)



     Imen DrsJanuary 4, 2019 at 9:56 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461441 "Direct link to this comment")





     Is there a way to use specific fields in the dataset instead of the entire uploaded dataset.


     And thanks.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461441)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 5, 2019 at 6:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461494 "Direct link to this comment")





       Yes, fields are columns in the dataset matrix and you can remove those columns that you do not want to use as inputs to your model.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461494)
275. ![](https://secure.gravatar.com/avatar/4a62e1c812db9f8ad802e2807271c822317585c1950f2391a91531938b6db5db?s=40&d=mm&r=g)



     KahinaJanuary 5, 2019 at 12:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461452 "Direct link to this comment")





     Thank you so much ! It’s helpful



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461452)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 5, 2019 at 6:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461498 "Direct link to this comment")





       I’m happy to hear that it was helpful.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-461498)
276. ![](https://secure.gravatar.com/avatar/f613313144b217bc22ce4ddb7f4fec467abe885663d2d1b7facc00405593c245?s=40&d=mm&r=g)



     KhemmarutJanuary 12, 2019 at 11:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-463190 "Direct link to this comment")





     Traceback (most recent call last):


     File “C:/Users/Admin/PycharmProjects/NN/nnt.py”, line 119, in


     rounded = \[round(X\[:1\]) for x in predictions\]


     File “C:/Users/Admin/PycharmProjects/NN/nnt.py”, line 119, in


     rounded = \[round(X\[:1\]) for x in predictions\]


     TypeError: type numpy.ndarray doesn’t define \_\_round\_\_ method



     Help me please



     Thank you.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-463190)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 13, 2019 at 5:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-463237 "Direct link to this comment")





       Perhaps ensure that your libraries are up to date?



       This might help:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-463237)
277. ![](https://secure.gravatar.com/avatar/ebed8cbebe02f9928bc8ee75f905f2bf07aefa71b1a0efb474bcf534a803e6c2?s=40&d=mm&r=g)



     Priti PachpandeJanuary 31, 2019 at 2:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465582 "Direct link to this comment")





     Hi Jason,


     Thank you for the amazing tutorial. I am trying to build an autoencoder model in keras using backend tensorflow.


     I need to use tensorflow(like tf.ifft,tf.fft) functions in the model. Can you guide me towards how can I do it? I tried using lambda layer but the accuracy decreases when I use it.



     Also, I m using model.predict() function to check the values between the intermediate layers. Am I doing it right?



     Also, can you guide me towards how to use reshape function in keras?



     Thanks for your help



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465582)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 31, 2019 at 5:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465608 "Direct link to this comment")





       Sorry, I don’t know about the functions you are using. Perhaps post on stackoverflow?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465608)
278. ![](https://secure.gravatar.com/avatar/5cfe2440c512933d971be7878479b58d4266cfcbdbad1cca5e4c03957b99abd3?s=40&d=mm&r=g)



     CrawfordJanuary 31, 2019 at 9:34 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465775 "Direct link to this comment")





     Hi Jason,


     Your tutorials are brilliant, thanks for putting all this together.


     In this tutorial the result is either a 1 or 0, but what if you have data with more than two possible results, e.g. 0, 1, 2, or similar?


     Can I do something with the code you have presented here, or is a whole other approach required?


     I have somewhat achieved what I’m trying to do using your “first machine learning project” using a knn model, but I had to simplify my data by stripping out some variables. I believe there is value in these extra variables, so thought the neural network might be useful, but like I said I have three classifications not two.


     Thanks.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465775)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 1, 2019 at 5:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465860 "Direct link to this comment")





       Yes, here is an example of a multi-class classification with a neural net:

       [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465860)




       - ![](https://secure.gravatar.com/avatar/5cfe2440c512933d971be7878479b58d4266cfcbdbad1cca5e4c03957b99abd3?s=40&d=mm&r=g)



         CrawfordFebruary 1, 2019 at 10:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466073 "Direct link to this comment")





         Brilliant, thanks.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466073)
279. ![](https://secure.gravatar.com/avatar/ace21dcfd87299842ac36ca75c63b00177e5a606871fc995767160189278a27a?s=40&d=mm&r=g)



     SergioFebruary 1, 2019 at 10:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465925 "Direct link to this comment")





     Hi, Im trying to construct a neural network using complex number as inputs, I followed your recommendatins but i get the following warning:


     \`


     ComplexWarning: Casting complex values to real discards the imaginary part return array(a, dtype, copy=False, order=order)



     The code run without problems, but the predictions is 25 % exact.



     Is possible to use complex number in neural networks..?



     Do u have some advices?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465925)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 1, 2019 at 11:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465944 "Direct link to this comment")





       I don’t think the Keras API supports complex numbers as input.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465944)




       - ![](https://secure.gravatar.com/avatar/ace21dcfd87299842ac36ca75c63b00177e5a606871fc995767160189278a27a?s=40&d=mm&r=g)



         SergioFebruary 1, 2019 at 2:17 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465988 "Direct link to this comment")





         Do u have any suggestion to deal with complex numbers?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-465988)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)February 2, 2019 at 6:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466148 "Direct link to this comment")





           Not off hand, sorry.



           Perhaps post to the Keras users group to see if anyone has tried this before:

           [https://machinelearningmastery.com/get-help-with-keras/](https://machinelearningmastery.com/get-help-with-keras/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466148)
280. ![](https://secure.gravatar.com/avatar/9a53276f433c16f32dc194a0a559a4a6c894c832c20208f4fa55f08b4db20fb2?s=40&d=mm&r=g)



     Arnab Kumar MishraFebruary 1, 2019 at 9:47 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466068 "Direct link to this comment")





     Hi Jason,



     I am trying to run the code in the tutorial with some minor modifications, but I am facing a problem with the training.



     The training loss and accuracy both are staying the same across epochs (Please take a look at the code snippet and the output below). This is for a different dataset, not the diabetes dataset.



     I have tried to solve this problem using the suggestions given in [https://stackoverflow.com/questions/37213388/keras-accuracy-does-not-change](https://stackoverflow.com/questions/37213388/keras-accuracy-does-not-change)



     But the problem is still there.



     Can you please take a look at this and help me solve this problem? Thanks.



     CODE and OUTPUT Snippets:



     \# create model


     model = Sequential()


     model.add(Dense(15, input\_dim=9, activation=’relu’))


     model.add(Dense(10, activation=’relu’))


     model.add(Dense(5, activation=’relu’))


     model.add(Dense(1, activation=’sigmoid’))



     \# compile model


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])



     \# Fit the model


     model.fit(xTrain, yTrain, epochs=500, batch\_size=10)



     Epoch 1/200


     81/81 \[==============================\] – 0s 177us/step – loss: -8.4632 – acc: 0.4691


     Epoch 2/200


     81/81 \[==============================\] – 0s 148us/step – loss: -8.4632 – acc: 0.4691


     Epoch 3/200


     81/81 \[==============================\] – 0s 95us/step – loss: -8.4632 – acc: 0.4691


     Epoch 4/200


     81/81 \[==============================\] – 0s 116us/step – loss: -8.4632 – acc: 0.4691


     Epoch 5/200


     81/81 \[==============================\] – 0s 106us/step – loss: -8.4632 – acc: 0.4691


     Epoch 6/200


     81/81 \[==============================\] – 0s 98us/step – loss: -8.4632 – acc: 0.4691


     Epoch 7/200


     81/81 \[==============================\] – 0s 145us/step – loss: -8.4632 – acc: 0.4691


     Epoch 8/200


     81/81 \[==============================\] – 0s 138us/step – loss: -8.4632 – acc: 0.4691


     Epoch 9/200


     81/81 \[==============================\] – 0s 105us/step – loss: -8.4632 – acc: 0.4691


     Epoch 10/200


     81/81 \[==============================\] – 0s 128us/step – loss: -8.4632 – acc: 0.4691


     Epoch 11/200


     81/81 \[==============================\] – 0s 129us/step – loss: -8.4632 – acc: 0.4691


     Epoch 12/200


     81/81 \[==============================\] – 0s 111us/step – loss: -8.4632 – acc: 0.4691


     Epoch 13/200


     81/81 \[==============================\] – 0s 106us/step – loss: -8.4632 – acc: 0.4691


     Epoch 14/200


     81/81 \[==============================\] – 0s 144us/step – loss: -8.4632 – acc: 0.4691


     Epoch 15/200


     81/81 \[==============================\] – 0s 106us/step – loss: -8.4632 – acc: 0.4691


     Epoch 16/200


     81/81 \[==============================\] – 0s 180us/step – loss: -8.4632 – acc: 0.4691


     Epoch 17/200


     81/81 \[==============================\] – 0s 125us/step – loss: -8.4632 – acc: 0.4691


     Epoch 18/200


     81/81 \[==============================\] – 0s 183us/step – loss: -8.4632 – acc: 0.4691


     Epoch 19/200


     81/81 \[==============================\] – 0s 149us/step – loss: -8.4632 – acc: 0.4691


     Epoch 20/200


     81/81 \[==============================\] – 0s 146us/step – loss: -8.4632 – acc: 0.4691


     Epoch 21/200


     81/81 \[==============================\] – 0s 206us/step – loss: -8.4632 – acc: 0.4691


     Epoch 22/200


     81/81 \[==============================\] – 0s 135us/step – loss: -8.4632 – acc: 0.4691


     Epoch 23/200


     81/81 \[==============================\] – 0s 116us/step – loss: -8.4632 – acc: 0.4691


     Epoch 24/200


     81/81 \[==============================\] – 0s 135us/step – loss: -8.4632 – acc: 0.4691


     Epoch 25/200


     81/81 \[==============================\] – 0s 121us/step – loss: -8.4632 – acc: 0.4691


     Epoch 26/200


     81/81 \[==============================\] – 0s 110us/step – loss: -8.4632 – acc: 0.4691


     Epoch 27/200


     81/81 \[==============================\] – 0s 104us/step – loss: -8.4632 – acc: 0.4691


     Epoch 28/200


     81/81 \[==============================\] – 0s 122us/step – loss: -8.4632 – acc: 0.4691


     Epoch 29/200


     81/81 \[==============================\] – 0s 117us/step – loss: -8.4632 – acc: 0.4691


     Epoch 30/200


     81/81 \[==============================\] – 0s 111us/step – loss: -8.4632 – acc: 0.4691


     Epoch 31/200


     81/81 \[==============================\] – 0s 123us/step – loss: -8.4632 – acc: 0.4691


     Epoch 32/200


     81/81 \[==============================\] – 0s 116us/step – loss: -8.4632 – acc: 0.4691


     Epoch 33/200


     81/81 \[==============================\] – 0s 120us/step – loss: -8.4632 – acc: 0.4691


     Epoch 34/200


     81/81 \[==============================\] – 0s 156us/step – loss: -8.4632 – acc: 0.4691


     Epoch 35/200


     81/81 \[==============================\] – 0s 131us/step – loss: -8.4632 – acc: 0.4691


     Epoch 36/200


     81/81 \[==============================\] – 0s 122us/step – loss: -8.4632 – acc: 0.4691


     Epoch 37/200


     81/81 \[==============================\] – 0s 110us/step – loss: -8.4632 – acc: 0.4691


     Epoch 38/200


     81/81 \[==============================\] – 0s 121us/step – loss: -8.4632 – acc: 0.4691


     Epoch 39/200


     81/81 \[==============================\] – 0s 123us/step – loss: -8.4632 – acc: 0.4691


     Epoch 40/200


     81/81 \[==============================\] – 0s 111us/step – loss: -8.4632 – acc: 0.4691


     Epoch 41/200


     81/81 \[==============================\] – 0s 115us/step – loss: -8.4632 – acc: 0.4691


     Epoch 42/200


     81/81 \[==============================\] – 0s 119us/step – loss: -8.4632 – acc: 0.4691


     Epoch 43/200


     81/81 \[==============================\] – 0s 115us/step – loss: -8.4632 – acc: 0.4691


     Epoch 44/200


     81/81 \[==============================\] – 0s 133us/step – loss: -8.4632 – acc: 0.4691


     Epoch 45/200


     81/81 \[==============================\] – 0s 114us/step – loss: -8.4632 – acc: 0.4691


     Epoch 46/200


     81/81 \[==============================\] – 0s 112us/step – loss: -8.4632 – acc: 0.4691


     Epoch 47/200


     81/81 \[==============================\] – 0s 143us/step – loss: -8.4632 – acc: 0.4691


     Epoch 48/200


     81/81 \[==============================\] – 0s 124us/step – loss: -8.4632 – acc: 0.4691


     Epoch 49/200


     81/81 \[==============================\] – 0s 129us/step – loss: -8.4632 – acc: 0.4691


     Epoch 50/200



     The same goes on for the rest of the epochs as well.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466068)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 2, 2019 at 6:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466163 "Direct link to this comment")





       I have some suggestions here that might help:

       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466163)
281. ![](https://secure.gravatar.com/avatar/6d295b05abfafe2a4db18785c9a71dd078ed088b86c65a3c8aa3a07207c6066a?s=40&d=mm&r=g)



     NageshFebruary 4, 2019 at 1:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466523 "Direct link to this comment")





     Hi Jason,



     Can you please update me, whether we can plot a graph(epoch vs acc)?


     If yes then how.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466523)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 4, 2019 at 5:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466543 "Direct link to this comment")





       I show how here:

       [https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466543)
282. ![](https://secure.gravatar.com/avatar/0c88ff0563d659015063e31a311bd33f75c84820cb5101c70f4dc6d83dda24ab?s=40&d=mm&r=g)



     NilsFebruary 5, 2019 at 1:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466743 "Direct link to this comment")





     Great stuff, thanks!



     I just wondered that in chapter 2 there is a description of the “init” parameter, but in all sources it was missing.


     I added it like:



     model.add(Dense(12, input\_dim=8, init=’uniform’ ,activation=’relu’))



     Then I got this warning:


     pima\_diabetes.py:25: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(12, input_dim=8, activation="relu"

     , kernel_initializer="uniform")`


     model.add(Dense(12, input\_dim=8, init=’uniform’ ,activation=’relu’))



     Solution for me was to use the “kernel\_initializer” instead:


     model.add(Dense(12, input\_dim=8, activation=”relu”, kernel\_initializer=”uniform”))



     Regarding the same line I got one question: Is it correct, that it adds one input layer with 8 neurons AND another hidden layer with 12 neurons?


     So, would it result in the same ANN to do this?


     model.add(Dense(8, input\_dim=8, kernel\_initializer=’uniform’))


     model.add(Dense(8, activation=”relu”, kernel\_initializer=’uniform’))



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466743)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 5, 2019 at 8:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466830 "Direct link to this comment")





       Yes, perhaps your version of the book is out of date, email me to get the latest version?



       Yes, the definition of the first hidden layer also defines the input layer via an argument.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-466830)
283. ![](https://secure.gravatar.com/avatar/c8ce6a38309d00054150339ae8f570ca5efc3b41969a1822471f0ba3bd54532a?s=40&d=mm&r=g)



     ShujaFebruary 8, 2019 at 12:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467404 "Direct link to this comment")





     Hi Jason


     I am getting the following error


     (env) shuja@latitude:~$ python keras\_test.py


     Using TensorFlow backend.


     Traceback (most recent call last):


     File “keras\_test.py”, line 8, in


     dataset = numpy.loadtxt(“pima-indians-diabetes.csv”, delimiter=”,”)


     File “/home/shuja/env/lib/python3.6/site-packages/numpy/lib/npyio.py”, line 955, in loadtxt


     fh = np.lib.\_datasource.open(fname, ‘rt’, encoding=encoding)


     File “/home/shuja/env/lib/python3.6/site-packages/numpy/lib/\_datasource.py”, line 266, in open


     return ds.open(path, mode, encoding=encoding, newline=newline)


     File “/home/shuja/env/lib/python3.6/site-packages/numpy/lib/\_datasource.py”, line 624, in open


     raise IOError(“%s not found.” % path)


     OSError: pima-indians-diabetes.csv not found.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467404)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 8, 2019 at 7:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467452 "Direct link to this comment")





       Looks like the dataset was not downloaded and place in the same directory as your script.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467452)
284. ![](https://secure.gravatar.com/avatar/e51031b6673599163ae9a11f0b84d8e223f0e232fbbb997899a3c1a695e59972?s=40&d=mm&r=g)



     ShubhamFebruary 12, 2019 at 4:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467850 "Direct link to this comment")





     Hi, Jason



     Thanks for the tutorial.


     Do you have some good reference or an example where I can learn about setting up “Adversarial Neural Networks”.



     Shubham



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467850)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 12, 2019 at 8:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467926 "Direct link to this comment")





       Not at this stage, I hope to cover the topic in the future.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-467926)
285. ![](https://secure.gravatar.com/avatar/432024d3456c9c705a988507430d0040e623a1fea6c1cc8d51ae63a92d3b4f6c?s=40&d=mm&r=g)



     [Daniel](https://github.com/Danielslee51)March 13, 2019 at 8:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-473964 "Direct link to this comment")





     Hey Jason,



     I’ve been reading your tutorials for a while now on a variety of ML topics, and I think that you write very cleanly and concisely. Thank you for making almost every topic I’ve encountered understandable.



     However, one thing I have noticed is that the comment sections on your pages sometimes cover the bulk of the webpage. The first couple times I saw this site, I saw how tiny my scroll bar was and I assumed that the tutorial would be 15 pages long, only to find that your introductions were in fact “gentle” as promised and everything but the first sliver of the page were people’s responses and your responses back. I think it would be very useful if you could somehow condense the responses (maybe a “show responses” button?) to only show the actual content. Not only would everything look better, but I think it would also prevent people from initially thinking your blog was exceptionally long, like I did a few times.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-473964)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 13, 2019 at 8:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-473966 "Direct link to this comment")





       Great feedback, thanks Daniel. I’ll see if there are some good wordpress plugins for this.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-473966)
286. ![](https://secure.gravatar.com/avatar/e7e312e2d44603a2adc15a46fa3a37c1a1d301aed8d1957596bee25ff1eff318?s=40&d=mm&r=g)



     ismaelMarch 22, 2019 at 5:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-475828 "Direct link to this comment")





     do not work why



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-475828)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 22, 2019 at 8:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-475873 "Direct link to this comment")





       Sorry to hear that you’re having trouble, what is the problem exactly?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-475873)
287. ![](https://secure.gravatar.com/avatar/e6e9b577878dec1aa00ad05384c5ebb7eea633c2ea53db81c490311a83f70baf?s=40&d=mm&r=g)



     Felix DanielMarch 30, 2019 at 7:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-477348 "Direct link to this comment")





     Awesome work on machine learning… I was just thinking on how to start my journey into Machine Learning, I randomly searched for people in Machine Learning on LinkedIn that’s how I find myself here… I’m delighted to see this… Here is my final bus stop to start building up in ML. Thanks for accepting my connection on LinkedIn.



     I have a project that am about to start but I don’t know how and the road Map. Please I need your detailed guideline.



     Here is the topic



     Human Activity Recognition System that Controls overweight in Children and Adults.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-477348)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 31, 2019 at 9:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-477542 "Direct link to this comment")





       Sounds like a great project, you can get started here:

       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-477542)
288. ![](https://secure.gravatar.com/avatar/70962919af7b8b88feb8baf2f282aabbebfa7ab0c4e1f9be114e91c04c2afadd?s=40&d=mm&r=g)



     Akshaya EApril 13, 2019 at 11:38 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480256 "Direct link to this comment")





     can you please explain me why we use 12 neurons in the first layer ? 8 are inputs and are the rest 4 biases ?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480256)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 14, 2019 at 5:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480294 "Direct link to this comment")





       No, the 12 refers to the 12 nodes in the first hidden layer, not the input layer.



       The input layer is defined by a input\_dim argument on the first hidden layer.



       I explain more here:

       [https://machinelearningmastery.com/faq/single-faq/how-do-you-define-the-input-layer-in-keras](https://machinelearningmastery.com/faq/single-faq/how-do-you-define-the-input-layer-in-keras)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480294)




       - ![](https://secure.gravatar.com/avatar/70962919af7b8b88feb8baf2f282aabbebfa7ab0c4e1f9be114e91c04c2afadd?s=40&d=mm&r=g)



         Akshaya EApril 14, 2019 at 8:09 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480361 "Direct link to this comment")





         thank you for the immediate response. my doubt has been cleared.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480361)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)April 15, 2019 at 7:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480434 "Direct link to this comment")





           Happy to hear that.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-480434)
289. ![](https://secure.gravatar.com/avatar/467167eae47a06814c99d1692f8b0898cad3ee561bb9afa0dedbd3709e76bea9?s=40&d=mm&r=g)



     AbhiramApril 19, 2019 at 11:50 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481337 "Direct link to this comment")





     hii Jason, above predictions are between 0 to 1,My labels are 1,1,1,2,2,2,3,3,3……..36,36,36.


     Now i want to predict class 36 then what should i do??



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481337)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 20, 2019 at 7:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481402 "Direct link to this comment")





       What problem are you having exactly?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481402)
290. ![](https://secure.gravatar.com/avatar/b07cecf01cd8972afbeac74f1e465e8224f2e1fa259d6d325280a127103cee8a?s=40&d=mm&r=g)



     AkashApril 22, 2019 at 12:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481745 "Direct link to this comment")





     Hi Jason,



     I am learning NLP and facing difficulties with understanding NLP with Deep Learning.


     Please, can you help with converting the following N:N to N:1 model?


     I want to change my vec\_y from max\_input\_words\_amount length to 1.


     How should I define the layers and use LSTM or RNN or …?


     Thank You.



     x=df1\[‘Question’\].tolist()


     y=df1\[‘Answer’\].tolist()



     max\_input\_words\_amount = 0


     tok\_x = \[\]


     for i in range(len(x)) :


     tokenized\_q = nltk.word\_tokenize(re.sub(r”\[^a-z0-9\]+”, ” “, x\[i\].lower()))


     max\_input\_words\_amount = max(len(tokenized\_q), max\_input\_words\_amount)


     tok\_x.append(tokenized\_q)



     vec\_x=\[\]


     for sent in tok\_x:


     sentvec = \[ft\_cbow\_model\[w\] for w in sent\]


     vec\_x.append(sentvec)



     vec\_y=\[\]


     for sent in y:


     sentvec = \[ft\_cbow\_model\[sent\]\]


     vec\_y.append(sentvec)



     for tok\_sent in vec\_x:


     tok\_sent\[max\_input\_words\_amount-1:\]=\[\]


     tok\_sent.append(ft\_cbow\_model\[‘\_E\_’\])



     for tok\_sent in vec\_x:


     if len(tok\_sent)<max\_input\_words\_amount:


     for i in range(max\_input\_words\_amount-len(tok\_sent)):


     tok\_sent.append(ft\_cbow\_model\['\_E\_'\])



     for tok\_sent in vec\_y:


     tok\_sent\[max\_input\_words\_amount-1:\]=\[\]


     tok\_sent.append(ft\_cbow\_model\['\_E\_'\])



     for tok\_sent in vec\_y:


     if len(tok\_sent)<max\_input\_words\_amount:


     for i in range(max\_input\_words\_amount-len(tok\_sent)):


     tok\_sent.append(ft\_cbow\_model\['\_E\_'\])



     vec\_x=np.array(vec\_x,dtype=np.float64)


     vec\_y=np.array(vec\_y,dtype=np.float64)



     x\_train,x\_test, y\_train,y\_test = train\_test\_split(vec\_x, vec\_y, test\_size=0.2, random\_state=1)



     model=Sequential()


     model.add(LSTM(output\_dim=100,input\_shape=x\_train.shape\[1:\],return\_sequences=True, init='glorot\_normal', inner\_init='glorot\_normal', activation='sigmoid'))


     model.add(LSTM(output\_dim=100,input\_shape=x\_train.shape\[1:\],return\_sequences=True, init='glorot\_normal', inner\_init='glorot\_normal', activation='sigmoid'))


     model.add(LSTM(output\_dim=100,input\_shape=x\_train.shape\[1:\],return\_sequences=True, init='glorot\_normal', inner\_init='glorot\_normal', activation='sigmoid'))


     model.add(LSTM(output\_dim=100,input\_shape=x\_train.shape\[1:\],return\_sequences=False, init='glorot\_normal', inner\_init='glorot\_normal', activation='sigmoid'))


     model.compile(loss='cosine\_proximity', optimizer='adam', metrics=\['accuracy'\])



     model.fit(x\_train, y\_train, nb\_epoch=100,validation\_data=(x\_test, y\_test),verbose=0)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481745)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 22, 2019 at 6:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481809 "Direct link to this comment")





       I’m happy to answer questions, but I don’t have the capacity to review your code, sorry.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481809)
291. ![](https://secure.gravatar.com/avatar/3f8dd19a875d29455b7e0976963a889559805c18dda7649827c409ffd01cc7e7?s=40&d=mm&r=g)



     CharlieApril 22, 2019 at 8:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481833 "Direct link to this comment")





     Jason – I think you are honestly the best teacher of these concepts on the web. Would you do a graph convolutions post? Maybe working through the concepts in Kipf and Welling 2016 GCN ( [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)) paper, and/or (ideally) a worked example applying to a graph network problem in Keras, maybe using Spektral, the recent graph convolutions Keras library ( [https://github.com/danielegrattarola/spektral](https://github.com/danielegrattarola/spektral) ) – would HUGELY appreciate it, and with the rise of graph ML eg per this DeepMind paper ( [https://arxiv.org/abs/1806.01261](https://arxiv.org/abs/1806.01261)) I’m sure there will be lots of great applications and interest for people but there’s not much online that’s easy to follow. Thanks so much in hope.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481833)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 22, 2019 at 2:26 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481887 "Direct link to this comment")





       Thanks.



       Thanks for the suggestion.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-481887)
292. ![](https://secure.gravatar.com/avatar/8dfd1fc362b3099b5d1b9a282b1fd60e18fd4b01c55306a293c52cd97f6e6065?s=40&d=mm&r=g)



     KudaApril 23, 2019 at 10:01 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482207 "Direct link to this comment")





     Hi Jason



     Thank you so much for your examples they are crystal clear. Do you have the implementation of RBF neural network in python?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482207)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 24, 2019 at 8:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482281 "Direct link to this comment")





       Not at this stage, sorry.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482281)
293. ![](https://secure.gravatar.com/avatar/7ee49a3da45224c936dee988fc0727426473e9aea645e59282349eb536877d0b?s=40&d=mm&r=g)



     Tom ColeApril 25, 2019 at 5:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482437 "Direct link to this comment")





     Do you have updated python code for this model on github? I’m enjoying working through the model but having some difference in the library loads required to do the data splitting and the model fitting steps.


     Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482437)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 25, 2019 at 8:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482481 "Direct link to this comment")





       What problem are you having exactly?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482481)
294. ![](https://secure.gravatar.com/avatar/2273ae3a077c5cab3a0c611e4c9b5bd3418f05e893502ceba05e09e4545471d6?s=40&d=mm&r=g)



     MridulApril 26, 2019 at 3:20 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482719 "Direct link to this comment")





     Hi! Jeson Brownlee,


     I try to implement the model in Jupyter notebook.


     But when i try to run,an error message show me that “module ‘tensorflow’ has no attribute ‘get\_default\_graph'” for compiling model = Sequential().I have try lot to overcome it.But couldn’t solve it.


     well you please help on this.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482719)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 27, 2019 at 6:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482825 "Direct link to this comment")





       I recommend running code from the command line and not from a notebook, here’s how:

       [https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line](https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-482825)
295. ![](https://secure.gravatar.com/avatar/fb2e9ce8b39a376a2e2a37a22013b99b313f7587f907565099bc37ca1ed1c162?s=40&d=mm&r=g)



     RoyalMay 5, 2019 at 10:18 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484151 "Direct link to this comment")





     Hi Jason,


     Super tutorials!



     If I run Your First Neural Network once and then repeat several times (without resetting the seed, during the same python session) using only this code:



     model.fit(X, Y, epochs=150, batch\_size=10, verbose=0)


     scores = model.evaluate(X, Y)


     print(“\\n%s: %.2f%%” % (model.metrics\_names\[1\], scores\[1\]\*100))



     then I get on average a ca. 3% improvement in accuracy (range 77.85% – 83.07%). Apparently the initialization values are benefitting from the previous runs.


     Does it make sense to use a model based on the best fit found after running several times? That would provide an almost 5% greater accuracy!


     Or are we overfitting?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484151)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 6, 2019 at 6:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484183 "Direct link to this comment")





       Yes, see this post:

       [https://machinelearningmastery.com/faq/single-faq/why-do-i-get-different-results-each-time-i-run-the-code](https://machinelearningmastery.com/faq/single-faq/why-do-i-get-different-results-each-time-i-run-the-code)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484183)
296. ![](https://secure.gravatar.com/avatar/ca27fd46ca99029928655b5e4ed1e946c59be295d2adefb4c333978cbe4bf627?s=40&d=mm&r=g)



     RogerMay 12, 2019 at 1:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484935 "Direct link to this comment")





     (base) C:\\Users\\Roger\\Documents\\Python Scripts>python firstnn.py


     Using Theano backend.


     Traceback (most recent call last):


     File “firstnn.py”, line 14, in


     model.add(Dense(12, input\_dim=8, activation=’relu’))


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\keras\\engine\\sequential.py”, line 165, in add


     layer(x)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\keras\\engine\\base\_layer.py”, line 431, in \_\_call\_\_


     self.build(unpack\_singleton(input\_shapes))


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\keras\\layers\\core.py”, line 866, in build


     constraint=self.kernel\_constraint)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\keras\\legacy\\interfaces.py”, line 91, in wrapper


     return func(\*args, \*\*kwargs)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\keras\\engine\\base\_layer.py”, line 249, in add\_weight


     weight = K.variable(initializer(shape),


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\keras\\initializers.py”, line 218, in \_\_call\_\_


     dtype=dtype, seed=self.seed)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\keras\\backend\\theano\_backend.py”, line 2600, in random\_uniform


     return rng.uniform(shape, low=minval, high=maxval, dtype=dtype)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\sandbox\\rng\_mrg.py”, line 872, in uniform


     rstates = self.get\_substream\_rstates(nstreams, dtype)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\configparser.py”, line 117, in res


     return f(\*args, \*\*kwargs)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\sandbox\\rng\_mrg.py”, line 779, in get\_substream\_rstates


     multMatVect(rval\[0\], A1p72, M1, A2p72, M2)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\sandbox\\rng\_mrg.py”, line 62, in multMatVect


     \[A\_sym, s\_sym, m\_sym, A2\_sym, s2\_sym, m2\_sym\], o, profile=False)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\compile\\function.py”, line 317, in function


     output\_keys=output\_keys)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\compile\\pfunc.py”, line 486, in pfunc


     output\_keys=output\_keys)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\compile\\function\_module.py”, line 1841, in orig\_function


     fn = m.create(defaults)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\compile\\function\_module.py”, line 1715, in create


     input\_storage=input\_storage\_lists, storage\_map=storage\_map)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\link.py”, line 699, in make\_thunk


     storage\_map=storage\_map)\[:3\]


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\vm.py”, line 1091, in make\_all


     impl=impl))


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\op.py”, line 955, in make\_thunk


     no\_recycling)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\op.py”, line 858, in make\_c\_thunk


     output\_storage=node\_output\_storage)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\cc.py”, line 1217, in make\_thunk


     keep\_lock=keep\_lock)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\cc.py”, line 1157, in \_\_compile\_\_


     keep\_lock=keep\_lock)


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\cc.py”, line 1609, in cthunk\_factory


     key = self.cmodule\_key()


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\cc.py”, line 1300, in cmodule\_key


     c\_compiler=self.c\_compiler(),


     File “C:\\Users\\Roger\\Anaconda3\\lib\\site-packages\\theano\\gof\\cc.py”, line 1379, in cmodule\_key\_


     np.core.multiarray.\_get\_ndarray\_c\_version())


     AttributeError: (‘The following error happened while compiling the node’, DotModulo(A, s, m, A2, s2, m2), ‘\\n’, “module ‘numpy.core.multiarray’ has no attribute ‘\_get\_ndarray\_c\_version'”)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484935)




     - ![](https://secure.gravatar.com/avatar/ca27fd46ca99029928655b5e4ed1e946c59be295d2adefb4c333978cbe4bf627?s=40&d=mm&r=g)



       RogerMay 12, 2019 at 1:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484936 "Direct link to this comment")





       I followed all the steps to set up the environment but when I ran the code I got an attribute error ‘module ‘numpy.core.multiarray’ has no attribute ‘\_get\_ndarray\_c\_version”



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484936)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)May 12, 2019 at 6:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484978 "Direct link to this comment")





         Perhaps try searching/posting on stackoverflow?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484978)
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 12, 2019 at 6:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484977 "Direct link to this comment")





       Ouch, perhaps numpy is not installed correctly?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-484977)
297. ![](https://secure.gravatar.com/avatar/ca27fd46ca99029928655b5e4ed1e946c59be295d2adefb4c333978cbe4bf627?s=40&d=mm&r=g)



     RogerMay 12, 2019 at 8:34 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-485062 "Direct link to this comment")





     No numpy 1.16.2 does not work with theano 1.0.3 as served up currently by Anaconda. I downgraded to numpy 1.13.0.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-485062)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 13, 2019 at 6:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-485122 "Direct link to this comment")





       Thanks Roger.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-485122)
298. ![](https://secure.gravatar.com/avatar/83068472ab696893940c19b94af9d27543a19b71bb89dc343e09dc7fb3fdabb1?s=40&d=mm&r=g)



     AdityaMay 21, 2019 at 5:02 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-486120 "Direct link to this comment")





     Hi Jason,


     Thanks for this amazing example!


     What I observe in the example is the database used is purely numeric.


     My doubt is:


     How can the example be modified to handle categorical input?


     Will it work if the inputs are One Hot Encoded?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-486120)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 22, 2019 at 7:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-486200 "Direct link to this comment")





       Yes, you can use a one hot encoding for our input categorical variables.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-486200)




       - ![](https://secure.gravatar.com/avatar/83068472ab696893940c19b94af9d27543a19b71bb89dc343e09dc7fb3fdabb1?s=40&d=mm&r=g)



         AdityaMay 31, 2019 at 3:41 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-487446 "Direct link to this comment")





         Can you please provide a good reference point for OHE in python?


         Thanks in advance! 🙂



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-487446)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)June 1, 2019 at 6:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-487546 "Direct link to this comment")





           Sure:

           [https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-487546)




           - ![](https://secure.gravatar.com/avatar/83068472ab696893940c19b94af9d27543a19b71bb89dc343e09dc7fb3fdabb1?s=40&d=mm&r=g)



             AdityaJune 2, 2019 at 3:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-487609 "Direct link to this comment")





             I read the link and it was helpful. Now, I have a doubt specific to my network.


             I have 3 categorical input which have different sizes. One has around 15 ‘categories’ while the other two have 5. So after I One Hot encode each of them, do I have to make their sizes same by padding? Or it’ll work as it it?

           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



             [Jason Brownlee](https://machinelearningmastery.com/)June 2, 2019 at 6:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-487633 "Direct link to this comment")





             You can encode each variable and concatenate them together into one vector.



             Or you can have a model with one input for each variable and let the model concatenate them.
299. ![](https://secure.gravatar.com/avatar/a52e520611a1dcf13f35676d490ffb2ced3b29b13530974ff5ba59687b3c0fc5?s=40&d=mm&r=g)



     SriJune 17, 2019 at 7:29 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489229 "Direct link to this comment")





     Hi,



     If there is one independent variable (say country) with more than 100 labels, how to resolve it.


     I think only one hot encoding will not work including scaling.



     Is there any alternative for it



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489229)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 18, 2019 at 6:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489280 "Direct link to this comment")





       You can try:



       – integer encoding


       – one hot encoding


       – embedding



       Test each and see what works best for your specific dataset.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489280)
300. ![](https://secure.gravatar.com/avatar/043fa4af78fc84f92211317fdc24955806bb2f78c0c2345a7de63cc9a54a65ac?s=40&d=mm&r=g)



     MKJune 21, 2019 at 7:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489744 "Direct link to this comment")





     Hi jason,



     thanks a lot for your posts, helped me a lot.



     1\. How can I add confusion matrix?



     2\. How can I change learning rate?



     Cheers Martin



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489744)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 22, 2019 at 6:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489814 "Direct link to this comment")





       Add a confusion matrix:

       [https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/](https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/)



       Tune learning rate:

       [https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-489814)
301. ![](https://secure.gravatar.com/avatar/e4af15f99fe433bc3c838e2ba5638dee9c4482fe454a0128da3d715c30b8b540?s=40&d=mm&r=g)



     Guhan palanivelJuly 1, 2019 at 10:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491139 "Direct link to this comment")





     hi jason,


     I have trained a neural network model with 6 months data and deployed at a remote site ,


     when receiving the new data for upcoming months ,


     is there any way to automatically update the model with addition of new training data ?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491139)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 2, 2019 at 7:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491202 "Direct link to this comment")





       Yes, perhaps the easiest way is to refit the model on the new data or on all available data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491202)
302. ![](https://secure.gravatar.com/avatar/e51031b6673599163ae9a11f0b84d8e223f0e232fbbb997899a3c1a695e59972?s=40&d=mm&r=g)



     ShubhamJuly 5, 2019 at 8:46 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491754 "Direct link to this comment")





     Hi jason,



     I want to print the neural network score as a function of one of the variable., how do i do that?



     Regards


     Shubham



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491754)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 6, 2019 at 8:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491826 "Direct link to this comment")





       Perhaps try a linear activation unit and a mse loss function?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-491826)
303. ![](https://secure.gravatar.com/avatar/3c1ca3b657791f6463d6a877ed4399450f687447bb9837427c112ac07ea203e1?s=40&d=mm&r=g)



     Maha LakshmiJuly 17, 2019 at 7:37 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-493215 "Direct link to this comment")





     Sir, I am working with sklearn.neural\_network.MLPClassifier in Python. now I want to give my own Initial Weights to Classifier.how to do that? please help me. Thanks in Advance



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-493215)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 18, 2019 at 8:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-493280 "Direct link to this comment")





       Sorry, I don’t have an example of this.



       Perhaps try posting on stackoverflow?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-493280)
304. ![](https://secure.gravatar.com/avatar/3c1ca3b657791f6463d6a877ed4399450f687447bb9837427c112ac07ea203e1?s=40&d=mm&r=g)



     Maha LakshmiJuly 18, 2019 at 4:09 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-493321 "Direct link to this comment")





     Thank you for your response



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-493321)

305. ![](https://secure.gravatar.com/avatar/ffddd7ef271f0c524fbb0b27b1d88be5f91471b71c410f39e72733711058dd5a?s=40&d=mm&r=g)



     RonJuly 24, 2019 at 8:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494056 "Direct link to this comment")





     Normalization of the data increases the accuracy in the 90’s.

     [https://stackoverflow.com/questions/39525358/neural-network-accuracy-optimization](https://stackoverflow.com/questions/39525358/neural-network-accuracy-optimization)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494056)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 24, 2019 at 2:19 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494076 "Direct link to this comment")





       Thanks for sharing.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494076)
306. ![](https://secure.gravatar.com/avatar/b15de902ef31df97448f66909032ac689638c48486554001d9c896cfdb2cf3de?s=40&d=mm&r=g)



     HammadJuly 29, 2019 at 6:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494635 "Direct link to this comment")





     Dear sir,



     I would like to apply above shared example on arrays produced by “train\_test\_split” but it does not work, as these arrays are not in the form of numpy.



     Let me give you the details, I have “XYZ” dataset. The dataset has the following specifications:



     Total Images = 630


     2500 features has been extracted from each image. Each feature has float type.


     Total Classes = 7



     Now, after processing the feature file, I have got results in the following variables:



     XData: contains features data in two dimensional array form (rows: 630, columns: 2500)


     YData: contain original labels of classes in one dimensional array form (rows: 630, column: 1)



     So, by using the following code, I split the data set into train and testing data:



     from sklearn.model\_selection import train\_test\_split


     x\_train, x\_test, y\_train, y\_test = train\_test\_split(XData, YData, stratify=YData, test\_size=0.25)



     Now, I would like to apply the deep-learning examples shared on this blog on my dataset which is now in the form arrays, and generate output as prediction of testing data and accuracy.



     Can you please let me know about it, which can work on the above arrays?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494635)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 30, 2019 at 6:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494704 "Direct link to this comment")





       Yes, the Keras model can operate on numpy arrays directly.



       Perhaps I don’t follow the problem that you’re having exactly?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494704)




       - ![](https://secure.gravatar.com/avatar/b15de902ef31df97448f66909032ac689638c48486554001d9c896cfdb2cf3de?s=40&d=mm&r=g)



         HammadJuly 30, 2019 at 6:01 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494797 "Direct link to this comment")





         Dear sir,



         Thanks, I converted my arrays into numpy format.



         Now, I have followed your tutorial on multi-classification problem ( [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)) and use the following code:



         ############################################################


         import pandas


         from keras.models import Sequential


         from keras.layers import Dense


         from keras.wrappers.scikit\_learn import KerasClassifier


         from keras.utils import np\_utils


         from sklearn.model\_selection import cross\_val\_score


         from sklearn.model\_selection import KFold


         from sklearn.preprocessing import LabelEncoder


         from sklearn.pipeline import Pipeline


         from sklearn.metrics import accuracy\_score



         seed=5


         totalclasses=7 # Class Labels are: ‘p1’, ‘p2’, ‘p3’, ‘p4’, ‘p5’, ‘p6’, ‘p7′


         totalimages=630


         totalfeatures=2500 #features generated from images



         \# Data has been imported from feature file, which results two arrays XData and YData


         \# XData contains features dataset without numpy array form


         \# YData contains labels without numpy array form



         \# encode class values as integers


         encoder = LabelEncoder()


         encoder.fit(YData)


         encoded\_Y = encoder.transform(YData)


         \# convert integers to dummy variables (i.e. one hot encoded)


         dummy\_y = np\_utils.to\_categorical(encoded\_Y)



         \# define baseline model


         def baseline\_model():


         # create model


         model = Sequential()


         model.add(Dense(8, input\_dim=totalfeatures+1, activation=’relu’))


         model.add(Dense(totalclasses, activation=’softmax’))


         # Compile model


         model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’, metrics=


         ‘accuracy’\])


         return model



         estimator = KerasClassifier(build\_fn=baseline\_model, nb\_epoch=200, batch\_size=5, verbose=0)



         x\_train, x\_test, y\_train, y\_test = train\_test\_split(XData, dummy\_y, test\_size=0.25, random\_state=seed)



         x\_train = np.array(x\_train)


         x\_test = np.array(x\_test)


         y\_train = np.array(y\_train)


         y\_test = np.array(y\_test)



         estimator.fit(x\_train, y\_train)


         predictions = estimator.predict(x\_test)



         print(predictions)


         print(encoder.inverse\_transform(predictions))



         ########################################################



         The code generates no syntax error.



         Now, I would like to ask:



         1\. Does I have applied the deep learning (Neural Network Model) in a right way?


         2\. How could I calculate the accuracy, confusion matrix, and classification\_report?


         3\. Can you please suggest what other type of deep learning algorithms could I apply on this type of problem?



         After applying different deep learning algorithm, I would like to compare their accuracies such as, you did in tutorial [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/), by plotting graphs.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494797)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)July 31, 2019 at 6:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494858 "Direct link to this comment")





           Sorry, I don’t have the capacity to review your code.



           This post shows how to calculate metrics:

           [https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)



           I recommend testing a suite of methods in order to discover what works best for your specific dataset:

           [https://machinelearningmastery.com/faq/single-faq/what-algorithm-config-should-i-use](https://machinelearningmastery.com/faq/single-faq/what-algorithm-config-should-i-use)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-494858)
307. ![](https://secure.gravatar.com/avatar/5012f99b8c1609c54855ae5ce36055bb2dbc28adf1baede67f0a4f26b60d26ec?s=40&d=mm&r=g)



     TysonSeptember 3, 2019 at 10:07 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-499542 "Direct link to this comment")





     Hi Jason,


     Great tutorial. I am now trying new data sets from the UCI archive. However I am running into problems when the data is incomplete. Rather than a number there is a ‘?’ indicating that the data is missing or unknown. So I am getting


     ValueError: could not convert string to float: ‘?’



     Is there a way to ignore that data? I am sure many data sets have this issue where pieces are missing.



     Thanks in advance!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-499542)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 4, 2019 at 5:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-499598 "Direct link to this comment")





       Yes, you can replace missing data with the mean or median of the variable – at least as a starting point.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-499598)
308. ![](https://secure.gravatar.com/avatar/6464e0fdd65013e12f57eca56b84beed86344f1454d2815d8d5c65bf8d248c79?s=40&d=mm&r=g)



     SrinuSeptember 10, 2019 at 9:07 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-500884 "Direct link to this comment")





     Can you provide GUI code for the same data like calling the ANN model from a website or from android application.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-500884)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 11, 2019 at 5:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-500928 "Direct link to this comment")





       I don’t see why not.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-500928)
309. ![](https://secure.gravatar.com/avatar/091f09e85db889bb91b86af8ed431858e53c7b6adedb0d3ba6fbf1e3a8490f55?s=40&d=mm&r=g)



     Hemanth KumarSeptember 20, 2019 at 12:58 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502289 "Direct link to this comment")





     dear sir


     ValueError: Error when checking input: expected conv2d\_5\_input to have 4 dimensions, but got array with shape (250, 250, 3)


     I am getting this error



     what steps I did


     original\_image->resized to same resolution->converted to numpy array ->saved and loaded to x\_train -> fed into network model ->modal.fit(x\_train .. getting this error



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502289)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 20, 2019 at 1:42 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502300 "Direct link to this comment")





       Perhaps start with this tutorial for image classification:

       [https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502300)
310. ![](https://secure.gravatar.com/avatar/091f09e85db889bb91b86af8ed431858e53c7b6adedb0d3ba6fbf1e3a8490f55?s=40&d=mm&r=g)



     Hemanth KumarSeptember 20, 2019 at 3:14 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502309 "Direct link to this comment")





     thanks for response sir 🙂


     after that I am getting list index out of range error at model.fit



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502309)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 21, 2019 at 6:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502376 "Direct link to this comment")





       I’m sorry to hear that, I have some suggestions here that may help:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502376)
311. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaSeptember 26, 2019 at 2:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502988 "Direct link to this comment")





     Dear Dr Jason,


     Thank you for this tutorial.


     I have been playing around with the number of layers and the number of neurons.


     In the current code







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4 | model=Sequential()<br>model.add(Dense(12,input\_dim=8,activation='relu'))<br>model.add(Dense(8,activation='relu'))<br>model.add(Dense(1,activation='sigmoid')) |











     I have played around with increasing the numbers in the first layer:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4 | model=Sequential()<br>model.add(Dense(100,input\_dim=8,activation='relu'))<br>model.add(Dense(8,activation='relu'))<br>model.add(Dense(1,activation='sigmoid')) |







     The result is that the accuracy didn’t improve much.


     There was an improvement in the addition of layers.


     When each layer had say a large number of neurons, the accuracy improved.


     This is not the only example, but playing around with the following code:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7 | model=Sequential()<br>model.add(Dense(200,input\_dim=8,activation='relu'))<br>model.add(Dense(800,activation='relu'))<br>model.add(Dense(200,activation='relu'))<br>model.add(Dense(400,activation='relu'))<br>model.add(Dense(200,activation='relu'))<br>model.add(Dense(1,activation='sigmoid')) |







     The accuracy achieved was 91.1%



     I added two more layers







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9 | model=Sequential()<br>model.add(Dense(200,input\_dim=8,activation='relu'))<br>model.add(Dense(800,activation='relu'))<br>model.add(Dense(200,activation='relu'))<br>model.add(Dense(400,activation='relu'))<br>model.add(Dense(200,activation='relu'))<br>model.add(Dense(400,activation='relu'))<br>model.add(Dense(800,activation='relu'))<br>model.add(Dense(1,activation='sigmoid')) |







     The accuracy dropped slightly to 88%



     From these brief experiments, increasing the number of neurons as in your first example did not increase accuracy.


     However adding more layers especially with a large number of neurons did increase the accuracy to about 91%


     BUT if there are too many layers there is a slight drop in accuracy to 88%.



     My question is there a way to increase the accuracy any further than 91%?



     Thank you,


     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502988)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 26, 2019 at 6:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503015 "Direct link to this comment")





       If this is the pima indians dataset, then the best accuracy is about 78% via 10-fold cross validation, anything more is probably overfitting.



       Yes, I have tons of tutorials on diagnosing issues with models and lifting performance, you can start here:

       [https://machinelearningmastery.com/start-here/#better](https://machinelearningmastery.com/start-here/#better)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503015)
312. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaSeptember 26, 2019 at 6:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502997 "Direct link to this comment")





     Dear Dr Jason,


     Further experimentation, I played with the following code







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7 | model=Sequential()<br>model.add(Dense(25,input\_dim=8,activation='relu'))<br>model.add(Dense(89,activation='relu'))<br>model.add(Dense(377,activation='relu'))<br>model.add(Dense(233,activation='relu'))<br>model.add(Dense(55,activation='relu'))<br>model.add(Dense(1,activation='sigmoid')) |







     I obtained an accuracy of 95% by playing around with the number of neurons increasing then decreasing.


     I cannot work out a systematic way of improving the accuracy.



     Thank you,


     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-502997)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 26, 2019 at 6:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503018 "Direct link to this comment")





       Haha, yes. That is the great open problem with neural nets (no good theories for how to configure them) and why we must use empirical methods.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503018)
313. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaSeptember 26, 2019 at 1:57 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503044 "Direct link to this comment")





     Dear Dr Jason,


     thank you for those replies.



     Yes, it was the Pima Indian dataset that is covered in this tutorial.



     Before I indulge in further readings on 10-fold cross validation, please briefly answer:


     \\* what is the meaning of overfit.


     \\* why is an accuracy of 96% regarded as overfit.



     To do:


     Play around with simple functions and play around with this tutorial and then look at overfitting:


     For example suppose we have x = 0, 1, 2, 3, 4, 5 and f(x) = x^2







































































     |     |     |
     | --- | --- |
     | 1<br>2 | x:0,1,2,3,4,5<br>f(x):0,1,4,9,16,25 |







     The aim:


     \\* to see if there is an accurate mapping of the function of x and f(x) for x = 0..5


     \\* to see what happens when we predict for x = 6, 7, 8. Will it be 36, 49, 64?


     \\* we ask if there is such a thing as overfitting the model exists.



     Thank you,


     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503044)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 27, 2019 at 7:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503116 "Direct link to this comment")





       Overfit means better performance on the training set at the cost of performing worse on the test set.



       It can also mean better performance on a test/validation set at the cost of worse performance on new data.



       I know from experience that the limit on that dataset is 77-78% after having worked with it in tutorials for about 20 years.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503116)
314. ![](https://secure.gravatar.com/avatar/a2f7601dffdab66d53fa77b5b67b0e1f5d4dcb93d226c554f6ff947f899c514a?s=40&d=mm&r=g)



     AndreySeptember 29, 2019 at 8:32 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503375 "Direct link to this comment")





     Hi Jason,



     I see the data is not divided for that of training and for the test. Why is that? What does prediction mean in this case?



     Andrey



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503375)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 30, 2019 at 6:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503418 "Direct link to this comment")





       It might mean that the result is a little optimistic.



       I did that to keep this example very simple and easy to follow.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503418)
315. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaSeptember 29, 2019 at 9:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503377 "Direct link to this comment")





     Dear Dr Jason,


     I tried to do the same for a deterministic model of x and fx where x = \[0,1,2,3,4,5\] and fx = x\*\*2


     I want to see how machine learning operates with a deterministic function.


     However I am only getting 16.67% accuracy.


     Here is the code based on the this tutorial







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27 | from keras.models import Sequential<br>from keras.layers import Dense<br>import numpy asnp<br>#Aim is to see how a deterministic function will operate using machine learning<br>#In year 7 algebra we have x and y. y is known as f(x). <br>#So here we aim to have a structore of \[indep var, dep var\]<br>#that is \[x, fx\]<br>#Making a 2D (like) list<br>x=\[iforiinrange(6)\];\# have a list of x = \[0,1,2,3,4,5\]<br>#x = np.array(x) <br>fx=\[x\*\*2forxinx\];\# have a list of fx = \[0,1,4,9,16,25\]<br>#fx = np.array(fx) <br>model=Sequential()<br>model.add(Dense(100,input\_dim=1,activation='relu'))<br>model.add(Dense(200,activation='relu'))<br>model.add(Dense(1,activation='relu'))<br>model.compile(loss='binary\_crossentropy',optimizer='adam',metrics=\['accuracy'\])<br>model.fit(x,fx,epochs=150,batch\_size=2,verbose=0)<br>\_,accuracy=model.evaluate(x,fx)<br>print('Accuracy: %.2f'%(accuracy\*100)) |











     We know that fx = x\*\*2 is predictable. What do I need to do.



     Thank you,


     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503377)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 30, 2019 at 6:10 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503420 "Direct link to this comment")





       Perhaps you need hundreds of thousands of examples?



       And perhaps the model will need to be tuned for your problem, e.g. perhaps using mse loss and a linear activation function in the output layer because it is a regression problem.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503420)
316. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 1, 2019 at 5:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503516 "Direct link to this comment")





     Dear Dr Jason,


     I tried with mse-loss and linear activation function and still only obtained 1% accuracy.











































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26 | from keras.models import Sequential<br>from keras.layers import Dense<br>import numpy asnp<br>#Aim is to see how a deterministic function will operate using machine learning<br>#In year 7 algebra we have x and y. y is known as f(x). <br>x=\[iforiinrange(100)\];\# have a list of x = \[0,1,2,3,4,5\]<br>x=np.array(x)<br>fx=\[x\*\*2forxinx\];\# have a list of fx = \[0,1,4,9,16,25\]<br>fx=np.array(fx)<br>model=Sequential()<br>model.add(Dense(12,input\_dim=1,activation='linear'))<br>model.add(Dense(33,activation='linear'))<br>model.add(Dense(1,activation='linear'))<br>#model.compile(loss='mean\_squared\_error', optimizer='softmax', metrics=\['accuracy'\])<br>model.compile(loss='mean\_squared\_error',optimizer='sgd')<br>#model.compile(loss='mean\_squared\_error')<br>model.fit(x,fx,epochs=10,batch\_size=1,verbose=0)<br>\_,accuracy=model.evaluate(x,fx)<br>print('Accuracy: %.2f'%(accuracy\*100)) |







     However I get this:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6 | 32/100\[========>.....................\]-ETA:0s<br>100/100\[==============================\]-0s312us/step<br>Traceback(most recent call last):<br>File"C:\\Python36\\deterministicII.py",line25,in<br>\_,accuracy=model.evaluate(x,fx)<br>TypeError:'float'objectisnotiterable |











     I want to map a deterministic function to see if machine learning will work out f(x) without the formula.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503516)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 1, 2019 at 7:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503537 "Direct link to this comment")





       Accuracy is not a valid metric for regression problems:

       [https://machinelearningmastery.com/faq/single-faq/how-do-i-calculate-accuracy-for-regression](https://machinelearningmastery.com/faq/single-faq/how-do-i-calculate-accuracy-for-regression)



       You are very close!



       Also, try a much larger dataset of examples. Hundreds or thousands.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503537)
317. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 1, 2019 at 10:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503549 "Direct link to this comment")





     Dear Dr Jason,


     I removed the model.evaluate from the program. BUT still I have not got a satisfactory match of the expected and actual values.











































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26 | from keras.models import Sequential<br>from keras.layers import Dense<br>import numpy asnp<br>#Aim is to see how a deterministic function will operate using machine learning<br>#In year 7 algebra we have x and y. y is known as f(x). <br>x=\[iforiinrange(100)\];\# have a list of x = \[0,1,2,3,4,5\]<br>x=np.array(x)<br>fx=\[x\*\*2forxinx\];\# have a list of fx = \[0,1,4,9,16,25\]<br>fx=np.array(fx)<br>model=Sequential()<br>model.add(Dense(100,input\_dim=1,activation='linear'))<br>model.add(Dense(100,activation='linear'))<br>model.add(Dense(1,activation='linear'))<br>model.compile(loss='mean\_squared\_error',optimizer='adam')<br>model.fit(x,fx,epochs=1000,batch\_size=1000,verbose=0)<br>#Removing the model.evaluate code<br>predictions=model.predict\_classes(x)<br>foriinrange(6):<br>print('%s => %d (expected %d)'%(x\[i\],predictions\[i\],fx\[i\])) |











     Output







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6 | 0=>0(expected0)<br>1=>0(expected1)<br>2=>0(expected4)<br>3=>0(expected9)<br>4=>1(expected16)<br>5=>1(expected25) |











     Not yet getting a match of the expected and the actual values



     Thank you,


     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503549)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 1, 2019 at 2:17 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503567 "Direct link to this comment")





       Perhaps the model architecture (layers and nodes) needs tuning?


       Perhaps the learning rate needs tuning?


       Perhaps you need more training examples?


       Perhaps you need more or fewer epochs?


       …



       More ideas here:

       [https://machinelearningmastery.com/start-here/#better](https://machinelearningmastery.com/start-here/#better)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503567)
318. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 2, 2019 at 7:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503650 "Direct link to this comment")





     Dear Dr Jason,


     I cannot find a systematic way to find a way for a machine learning algorithm to use it to compute a deterministic equation such as y = f(x) where f(x) = x\*\*2.



     I am still having trouble. I will be posting this on the page. Essentially is (i) adding/dropping layers, (ii) adjusting the number of epochs, (iii) adjusting the batch\_size. But I haven’t come close yet.



     Also using the function model.predict rather than model.predict\_classes.



     Here is the program with most of the commented out lines deleted.











































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29 | from keras.models import Sequential<br>from keras.layers import Dense<br>import numpy asnp<br>#Aim is to see how a deterministic function will operate using machine learning<br>#In year 7 algebra we have x and y. y is known as f(x). Here y = f(x) = x\*\*2 <br>x=\[iforiinrange(100)\];\# have a list of x = \[0,1,2,3,4,5,.....,99\]<br>x=np.array(x)<br>fx=\[x\*\*2forxinx\];\# have a list of fx = x\*\*2 = \[0,1,4,9,16,25,...,9801\] <br>fx=np.array(fx)<br>model=Sequential()<br>model.add(Dense(55,input\_dim=1,activation='linear'))<br>model.add(Dense(34,activation='linear'))<br>model.add(Dense(21,activation='linear'))<br>model.add(Dense(13,activation='linear'))<br>model.add(Dense(1,activation='linear'))<br>model.compile(loss='mean\_squared\_error',optimizer='adam')<br>model.fit(x,fx,epochs=89,batch\_size=144,verbose=0)<br>predictions=model.predict(x);#This seems to work instead of model.predict\_classes<br>print("x, predicted, expected")<br>foriinrange(6):<br>print('%s => %d (expected %d)'%(x\[i\],predictions\[i\],fx\[i\])) |











     The output is:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7 | x,predicted,expected<br>0=>29(expected0)<br>1=>110(expected1)<br>2=>191(expected4)<br>3=>272(expected9)<br>4=>353(expected16)<br>5=>434(expected25) |







     No matter how much I adjust the number of neurons per layer, the number of layers, the no of epochs and the batch size, the “predicted” appears like an arithmetic progression, not a geometric progression.



     Note the terms tn+1 – tn is 81 for all the predicted values in the machine learning model.



     BUT we know that the difference between successive terms in y = f(x) is not the same.



     For example, in non linear relation such as f(x) = x\*\*2, f(x) = 0, 1, 2, 4, 9, 16, 25, 36, the difference between the terms is: 1, 1, 2, 5, 7, 9, 11, that is tn+1 – tn != tn+2 – tn+1.



     So still having trouble working out how to get a machine learning algorithm evaluate f(x) without the formula.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503650)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 2, 2019 at 8:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503675 "Direct link to this comment")





       Here is the solution, hope it helps











































































       |     |     |
       | --- | --- |
       | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32 | \# fit an mlp on x vs x^2<br>from sklearn.preprocessing import MinMaxScaler<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from numpy import asarray<br>from matplotlib import pyplot<br>\# define data<br>x=asarray(\[iforiinrange(1000)\])<br>y=asarray(\[a\*\*2forainx\])<br>\# reshape into rows and cols<br>x=x.reshape((len(x),1))<br>y=y.reshape((len(y),1))<br>\# scale data<br>x\_s=MinMaxScaler()<br>x=x\_s.fit\_transform(x)<br>y\_s=MinMaxScaler()<br>y=y\_s.fit\_transform(y)<br>\# fit a model<br>model=Sequential()<br>model.add(Dense(10,input\_dim=1,activation='relu'))<br>model.add(Dense(1))<br>model.compile(loss='mse',optimizer='adam')<br>model.fit(x,y,epochs=150,batch\_size=10,verbose=0)<br>mse=model.evaluate(x,y,verbose=0)<br>print(mse)<br>\# predict<br>yhat=model.predict(x)<br>\# plot real vs predicted<br>pyplot.plot(x,y,label='y')<br>pyplot.plot(x,yhat,label='yhat')<br>pyplot.legend()<br>pyplot.show() |











       I guess you could also do an inverse\_transform() on the predicted values to get back to original units.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503675)
319. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 2, 2019 at 9:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503680 "Direct link to this comment")





     Dear Dr Jason,


     Thank you very much for your reply. I got an mse in the order of 3 x 10\*\*-6.



     Despite this, I will be studying the program and learn myself about (i) the MinMaxScaler and why we use it, (ii) fit\_transform(y) and (iii) one hidden layer of 10 neurons, and (iii) I will still have to learn about the choice of activation function and loss functions. The keras website has a section on loss functions at [https://keras.io/losses/](https://keras.io/losses/) but having a look at the Python “IDLE” program, a look at from keras import losses, there are many more loss functions which are necessary to compile a model.



     In addition, the predicted values will have to be re-computed to its unscaled values. So I will also look up ‘rescaling’.



     Thank you again,


     Anthony, Sydney NSW



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503680)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 2, 2019 at 10:10 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503685 "Direct link to this comment")





       Yes, you can use inverse\_transform to unscale the predictions, as I mentioned.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503685)
320. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 3, 2019 at 6:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503779 "Direct link to this comment")





     Dear Dr Jason,


     I know how to use the inverse\_transform function:


     First apply the MinMaxScaler to scale to 0 to 1







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4 | x\_s=MinMaxScaler()<br>x=x\_s.fit\_transform(x)<br>y\_s=MinMaxScaler()<br>y=y\_s.fit\_transform(y) |







     If we want to reconstitute x and y, it is simple to:







































































     |     |     |
     | --- | --- |
     | 1<br>2 | x\_original=x\_s.inverse\_transform(x);\# where x was transformed/scaled<br>y\_original=y\_s.inverse\_transform(y);\# where y was transformed/scaled |







     x\_s and y\_s has the min and max values stored of the original pre-transformed data.



     BUT how do you transform yhat to its original scale when it was not subject to the inverse\_transform function.



     If I relied on the y\_s.inverse\_transform(yhat), where you get this:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12 | yhat\_restored=y\_s.inverse\_transform(yhat);#using the values of ymin and ymax of original data<br>yhat\_restored\[0:10\]<br>array(\[\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\]\],dtype=float32) |







     I was ‘hoping’ for something close to the original:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12 | >>>y\_restored=y\_s.inverse\_transform(y)<br>>>>y\_restored\[0:10\]<br>array(\[\[0.\],<br>\[1.\],<br>\[4.\],<br>\[9.\],<br>\[16.\],<br>\[25.\],<br>\[36.\],<br>\[49.\],<br>\[64.\],<br>\[81.\]\]) |











     BUT yhat does not use the MinMaxScaler at the start.



     Do I have to rewrite my own function?



     Thanks,


     Anthony of Sydney NSW



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503779)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 3, 2019 at 6:54 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503809 "Direct link to this comment")





       The model predicts scaled values, apply the inverse transform on yhat directly.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503809)
321. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 3, 2019 at 2:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503867 "Direct link to this comment")





     Dear Dr Jason,


     I did that apply the inverse transform of yhat directly, BUT GOT these


     Cut down version of code







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27 | y\_s=MinMaxScaler()<br>y=y\_s.fit\_transform(y);#y\_s stores the min and max values according to the sklearn doc<br>#Note the above is for y. WE DON'T KNOW yhat(min) & yhat(max<br>yhat=model.predict(x);#we have the scaled estimate.<br>x\_original=x\_s.inverse\_transform(x);#this printed okay<br>#Printout of yhat transformed<br>#Calculate yhat scaled using the min and max values of f(x) = y<br>yhat\_restored=y\_s.inverse\_transform(yhat)<br>#Print yhat<br>print(yhat\_restored\[0:10\])<br>array(\[\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\],<br>\[6838.43\]\],dtype=float32) |











     Don’t understand how to get an inverse transform of yhat when I don’t know the ‘untransformed’ value because I have not estimated it.



     Thank you,


     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503867)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 4, 2019 at 5:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503973 "Direct link to this comment")





       You can inverse transform y and yhat and plot both.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503973)
322. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 4, 2019 at 3:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503949 "Direct link to this comment")





     Dear Dr Jason,


     I tried it again to illustrate that despite the predicted fitting a parabola for scaled predicted and expected values of f(x) the resulting values when ‘unscaled’ back to the original does seems quite absurd.


     Code – relevant







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21 | \# plot real vs predicted<br>pyplot.plot(x,y,label='y')<br>pyplot.plot(x,yhat,label='yhat')<br>pyplot.legend()<br>print("The graph of the (x, predicted f(x) and (x, f(x) is on a separate window")<br>pyplot.show()<br>y\_predicted=y\_s.inverse\_transform(yhat)<br>y\_expected=y\_s.inverse\_transform(y)<br>x\_original=x\_s.inverse\_transform(x)<br>#print(y\_predicted\[0:10,\].tolist(), x\_original\[0:10,\].tolist())<br>print("Printing the first 10, predicted, expected, and x")<br>foriinrange(10):<br>print(y\_predicted\[i\],y\_expected\[i\],x\_original\[i\])<br>print("let's try some other arbitrary section, say 10:20")<br>#print(y\_predicted\[9:21,\].tolist(),y\_predicted\[9:21,\].tolist(), x\_original\[9:21,\].tolist())<br>print("printing 10th to 20th, predicted, expected, and x")<br>foriinrange(10):<br>print(y\_predicted\[i+10\],y\_expected\[i+10\],x\_original\[i+10\]) |











     The resulting output:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24 | 5.472537487406726e-06<br>Printing the first10,predicted,expected,andx<br>\[1030.0833\]\[0.\]\[0.\]<br>\[1030.0833\]\[1.\]\[1.\]<br>\[1030.0833\]\[4.\]\[2.\]<br>\[1030.0833\]\[9.\]\[3.\]<br>\[1030.0833\]\[16.\]\[4.\]<br>\[1030.0833\]\[25.\]\[5.\]<br>\[1030.0833\]\[36.\]\[6.\]<br>\[1030.0833\]\[49.\]\[7.\]<br>\[1030.0833\]\[64.\]\[8.\]<br>\[1030.0833\]\[81.\]\[9.\]<br>let'strysome other arbitrary section,say10:20<br>printing10thto20th,predicted,expected,andx<br>\[1030.0833\]\[100.\]\[10.\]<br>\[1030.0833\]\[121.\]\[11.\]<br>\[1030.0833\]\[144.\]\[12.\]<br>\[1030.0833\]\[169.\]\[13.\]<br>\[1030.0833\]\[196.\]\[14.\]<br>\[1030.0833\]\[225.\]\[15.\]<br>\[1030.0833\]\[256.\]\[16.\]<br>\[1030.0833\]\[289.\]\[17.\]<br>\[1030.0833\]\[324.\]\[18.\]<br>\[1030.0833\]\[361.\]\[19.\] |







     When I plotted (x, yhat) and (x,f(x)), the plot was as expected. BUT when I rescaled the yhat back, all the values of unscaled yhat were 1030.0833 which is quite odd.



     Why?



     Thank you,


     Anthony of Sydney NSW



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503949)

323. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 4, 2019 at 3:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503950 "Direct link to this comment")





     Dear Dr Jason,


     I printed the yhat, and they were all the same.



     This is despite that the plot of the scaled values (x, yhat) looked like a parabola


     Note: this is prior to scaling.







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14 | \# plot real vs predicted<br>pyplot.plot(x,y,label='y')<br>pyplot.plot(x,yhat,label='yhat')<br>pyplot.legend()<br>print("the graph is printed on another window")<br>pyplot.show()<br>print("Printing the output of the scaled values of yhat, f(x) and x")<br>print("printing the first 10")<br>foriinrange(10):<br>print(yhat\[i\],y\[i\],x\[i\])<br>print("printing the 10th to 20th")<br>foriinrange(10):<br>print(yhat\[i+10\],y\[i+10\],x\[i+10\]) |











     Yet despite the expected plots of scaled values (x,yhat), and (x, y), yhat’s values are the same







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23 | Printing the output of the scaled values<br>printing the first10<br>\[0.00117336\]\[0.\]\[0.\]<br>\[0.00117336\]\[1.002003e-06\]\[0.001001\]<br>\[0.00117336\]\[4.00801202e-06\]\[0.002002\]<br>\[0.00117336\]\[9.01802704e-06\]\[0.003003\]<br>\[0.00117336\]\[1.60320481e-05\]\[0.004004\]<br>\[0.00117336\]\[2.50500751e-05\]\[0.00500501\]<br>\[0.00117336\]\[3.60721081e-05\]\[0.00600601\]<br>\[0.00117336\]\[4.90981472e-05\]\[0.00700701\]<br>\[0.00117336\]\[6.41281923e-05\]\[0.00800801\]<br>\[0.00117336\]\[8.11622433e-05\]\[0.00900901\]<br>printing the10thto20th<br>\[0.00117336\]\[0.0001002\]\[0.01001001\]<br>\[0.00117336\]\[0.00012124\]\[0.01101101\]<br>\[0.00117336\]\[0.00014429\]\[0.01201201\]<br>\[0.00117336\]\[0.00016934\]\[0.01301301\]<br>\[0.00117336\]\[0.00019639\]\[0.01401401\]<br>\[0.00117336\]\[0.00022545\]\[0.01501502\]<br>\[0.00117336\]\[0.00025651\]\[0.01601602\]<br>\[0.00117336\]\[0.00028958\]\[0.01701702\]<br>\[0.00117336\]\[0.00032465\]\[0.01801802\]<br>\[0.00117336\]\[0.00036172\]\[0.01901902\] |







     I don’t get it.You would expect a similarity of yhat and f(x).



     I would appreciate a response



     Thank you,


     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503950)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 4, 2019 at 5:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503987 "Direct link to this comment")





       Sorry, I don’t have the capacity to debug your examples further. I hope that you can understand.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503987)
324. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 4, 2019 at 6:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503993 "Direct link to this comment")





     Dear Dr Jason,


     I asked the question at [https://datascience.stackexchange.com/questions/61223/reconstituting-estimated-predicted-values-to-original-scale-from-minmaxscaler](https://datascience.stackexchange.com/questions/61223/reconstituting-estimated-predicted-values-to-original-scale-from-minmaxscaler) and hope that there is an answer.


     Thanks


     Anthony Of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-503993)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 4, 2019 at 8:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504003 "Direct link to this comment")





       Here is the solution











































































       |     |     |
       | --- | --- |
       | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36 | \# fit an mlp on x vs x^2<br>from sklearn.preprocessing import MinMaxScaler<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from numpy import asarray<br>from matplotlib import pyplot<br>\# define data<br>x=asarray(\[iforiinrange(1000)\])<br>y=asarray(\[a\*\*2forainx\])<br>\# reshape into rows and cols<br>x=x.reshape((len(x),1))<br>y=y.reshape((len(y),1))<br>\# scale data<br>x\_s=MinMaxScaler()<br>x=x\_s.fit\_transform(x)<br>y\_s=MinMaxScaler()<br>y=y\_s.fit\_transform(y)<br>\# fit a model<br>model=Sequential()<br>model.add(Dense(10,input\_dim=1,activation='relu'))<br>model.add(Dense(1))<br>model.compile(loss='mse',optimizer='adam')<br>model.fit(x,y,epochs=150,batch\_size=10,verbose=0)<br>mse=model.evaluate(x,y,verbose=0)<br>print(mse)<br>\# predict<br>yhat=model.predict(x)<br>\# inverse transforms<br>x=x\_s.inverse\_transform(x)<br>y=y\_s.inverse\_transform(y)<br>yhat=y\_s.inverse\_transform(yhat)<br>\# plot real vs predicted<br>pyplot.plot(x,y,label='y')<br>pyplot.plot(x,yhat,label='yhat')<br>pyplot.legend()<br>pyplot.show() |











       The three missing lines were:











































































       |     |     |
       | --- | --- |
       | 1<br>2<br>3<br>4 | \# inverse transforms<br>x=x\_s.inverse\_transform(x)<br>y=y\_s.inverse\_transform(y)<br>yhat=y\_s.inverse\_transform(yhat) |











       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504003)
325. ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



     Anthony The KoalaOctober 4, 2019 at 9:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504015 "Direct link to this comment")





     Dear Dr Jason,


     I am coming to the conclusion that there must be a bug NOT in your solution and neither in my solution. I think it is coming from a bug in the lower implementation of the language.



     I printed the scaled version of yhat, f(x) actual and x and got this.


     NOTE the values are the same for the scaled version of yhat.


     That is:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6 | model.fit(x,y,epochs=277,batch\_size=200,verbose=0)<br>mse=model.evaluate(x,y,verbose=0)<br>print("the value of the mse")<br>print(mse)<br>\# predict<br>yhat=model.predict(x) |











     DESPITE the successful plot of (x, yhat) and (x, f(x),


     the resulting output of the first 10 of the scaled output of yhat is the same,



     That is we would get a FLAT LINE if we plotted (x, yhat), BUT THE PLOT WAS A PARABOLA.











































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10 | \[0.00161531\]\[0.\]\[0.\]<br>\[0.00161531\]\[1.002003e-06\]\[0.001001\]<br>\[0.00161531\]\[4.00801202e-06\]\[0.002002\]<br>\[0.00161531\]\[9.01802704e-06\]\[0.003003\]<br>\[0.00161531\]\[1.60320481e-05\]\[0.004004\]<br>\[0.00161531\]\[2.50500751e-05\]\[0.00500501\]<br>\[0.00161531\]\[3.60721081e-05\]\[0.00600601\]<br>\[0.00161531\]\[4.90981472e-05\]\[0.00700701\]<br>\[0.00161531\]\[6.41281923e-05\]\[0.00800801\]<br>\[0.00161531\]\[8.11622433e-05\]\[0.00900901\] |







     When we did the following transforms:







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3 | x=x\_s.inverse\_transform(x)<br>y=y\_s.inverse\_transform(y)<br>yhat=y\_s.inverse\_transform(yhat) |







     WE STILL GOT THE SAME FAULT FOR THE UNSCALED VALUES of yhat. The 2nd column is f(x) and third column is x.







































































     |     |     |
     | --- | --- |
     | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10 | \[1612.0857\]\[0.\]\[0.\]<br>\[1612.0857\]\[1.\]\[1.\]<br>\[1612.0857\]\[4.\]\[2.\]<br>\[1612.0857\]\[9.\]\[3.\]<br>\[1612.0857\]\[16.\]\[4.\]<br>\[1612.0857\]\[25.\]\[5.\]<br>\[1612.0857\]\[36.\]\[6.\]<br>\[1612.0857\]\[49.\]\[7.\]<br>\[1612.0857\]\[64.\]\[8.\]<br>\[1612.0857\]\[81.\]\[9.\] |











     Conclusion: It is not a programmatical bug in either your solution or my solution. I believe it may be a lower implementation problem.



     Why am I ‘persistent’ in this matter: because in case I have more complex models I want to see the predicted/yhat values that are re-scaled.



     I don’t know if there are people at stackexchange who may have an insight.



     I appreciate your time, many blessings to you,



     Anthony of Sydney



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504015)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 6, 2019 at 8:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504307 "Direct link to this comment")





       I believe is correct, given that it is an exponential, the model has decided that it can give up correctness at the low end for correctness at the high end – given the reduction in MSE.



       Consider changing the number of examples from 1K to 100, then review all 100 values manually – you’ll see what I mean.



       All of this is a good exercise, well done.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504307)




       - ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



         Anthony The KoalaOctober 13, 2019 at 10:57 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505515 "Direct link to this comment")





         Dear Dr Jason,


         I did this problem again and got very good results!


         I cannot explain why I got accurate results, when I expected to get accurate results, BUT they are certainly an improvement.



         The rescaled original and fitted values produced an RMS of 0.0.



         Here is the code with variable names changed slightly.











































































         |     |     |
         | --- | --- |
         | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52<br>53<br>54<br>55<br>56 | from sklearn.preprocessing import MinMaxScaler<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from numpy import sqrt<br>from matplotlib import pyplot<br>x=asarray(\[iforiinrange(100)\])<br>y=asarray(\[i\*\*2foriinx\])<br>x=x.reshape((len(x),1))<br>y=y.reshape((len(y),1))<br>x\_s=MinMaxScaler()<br>xscaled=x\_s.fit\_transform(x)<br>y\_s=MinMaxScaler()<br>yscaled=y\_s.fit\_transform(y)<br>model=Sequential()<br>model.add(Dense(100,input\_dim=1,activation='relu')<br>model.add(Dense(1))<br>model.compile(loss='mse',optimizer='adam')<br>model.fit(xscaled,yscaled,epochs=150,batch\_size=10,verbose=0)<br>mse=model.evaluate(xscaled,yscaled,verbose=0)<br>mse<br>2.9744908551947447e-05<br>yhat=model.predict(x)<br>yhat\_original=y\_s.inverse\_transform(yscaled)<br>#First five elements of predicted values<br>yhat\_original\[:5\].T<br>array(\[\[0.,1.,4.,9.,16.\]\])<br>#First five elements of original y<br>y\[:5\].T<br>array(\[\[0,1,4,9,16\]\])<br>#Last five elements of the original series.<br>y\[-5:\].T<br>array(\[\[9025,9216,9409,9604,9801\]\])<br>#Last five elements of predicted values<br>yhat\_original\[-5:\].T<br>array(\[\[9025.,9216.,9409.,9604.,9801.\]\])<br>#Now determining the RMS of the predicted and original values of y<br>cum\_sum=0<br>foriinrange(len(yhat\_original)):<br>cum\_sum+=(yoriginal\[i\]-yhat\_original\[i\])\*\*2/len(yhat\_original)<br>rms=sqrt(cum\_sum)<br>rms\[0\]<br>0.0<br>#Plotting the rescaled original and rescaled yhat<br>pyplot.plot(xoriginal,yoriginal,label='y')<br>pyplot.plot(xoriginal,yhat\_original,label='fitted')<br>pyplot.legend()<br>pyplot.show() |











         It works, the rescaled yhat is as expected but cannot explain why it was “cuckoo”, in the previous. More experimentation on this.



         Nevertheless, my next project is k-folds sampling on a deterministic function to see if the gaps in the resampled data fold will give us an accurate prediction despite the random sampling in each fold.



         Thank you,


         Anthony of Sydney



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505515)




         - ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)



           Anthony The KoalaOctober 13, 2019 at 11:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505518 "Direct link to this comment")





           Dear Dr Jason,


           Apologies, I thought the RMS was ‘unrealistic’. I had a programming error.


           Nevertheless, I did it again, and still produced results which looked pleasing.











































































           |     |     |
           | --- | --- |
           | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41 | x=x.reshape((len(x),1))<br>y=y.reshape((len(y),1))<br>x\_s=MinMaxScaler()<br>y\_s=MinMaxScaler()<br>x\_scaled=x\_s.fit\_transform(x)<br>y\_scaled=y\_s.fit\_transform(y)<br>model=Sequential()<br>model.add(Dense(100,input\_dim=1,activation='relu'))<br>model.add(Dense(1))<br>model.compile(loss='mse',optimizer='adam')<br>model.fit(x\_scaled,y\_scaled,epochs=100,batch\_size=10,verbose=0)<br>mse=model.evaluate(x\_scaled,y\_scaled,verbose=0)<br>mse<br>1.0475558547113905e-05<br>yhat=model.predict(x\_scaled)<br>yhat\_original=y\_s.inverse\_transform(yhat)<br>#First five of yhat\_original (yhat rescaled)<br>yhat\_original\[:5\].T<br>array(\[\[11.835742,11.835742,11.835742,11.835742,11.835742\]\]<br>#compared to first original 5 elements of y = 0,1,4,9,16<br>#Last five of yhat\_original (yhat rescaled)<br>yhat\_original\[-5:\].T<br>array(\[\[8985.839,9154.454,9323.067,9491.684,9660.3\]<br>#compared to last original 5 elements of y = 9025, 9216, 9409, 9604, 9801<br>#Now determine the RMS of the predicted and original values<br>cum\_sum=0<br>foriinrange(len(yhat\_original)):<br>cum\_sum+=(y\[i\]-yhat\_original\[i\])\*\*2/len(yhat\_original)<br>mse=sqrt(cum\_sum)<br>mse<br>array(\[31.72189417\])<br>pyplot.plot(x,y,label='y')<br>pyplot.plot(x,yhat\_original,label='estimated')<br>pyplot.legend()<br>pyplot.show() |\
\
\
\
\
\
\
\
\
\
\
\
           In sum, the rescaled yhat produced results closer to the original values. The lower values of yhat rescaled appear to be odd.\
\
\
\
           Despite that the values need to be more realistic at the bottom end even though the plot of the rescaled x & rescaled y, and rescaled x and rescaled yhat look close.\
\
\
\
           More investigations needed on the batch size, epochs and optimizers.\
\
\
\
           Next, to do k-folds sampling on a deterministic function to see if the gaps in the resampled data fold will give us an accurate prediction despite the random sampling in each fold.\
\
\
\
           Again apologies for the mistake in the previous post.\
\
\
\
           Anthony of Sydney\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505518)\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)October 14, 2019 at 8:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505575 "Direct link to this comment")\
\
\
\
\
\
           Well done.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505575)\
\
\
\
\
           - ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)\
\
\
\
             Anthony The KoalaNovember 21, 2019 at 5:03 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-511730 "Direct link to this comment")\
\
\
\
\
\
             Dear Dr Jason,\
\
\
             A person ‘Serali’ a particle physicist relied to me at “StackExchange” replied and suggested that I shuffle the original data. The shuffling of data in this context has nothing to do with the shuffling in k-folds. According to the contributor, the results should improve. Source [https://datascience.stackexchange.com/questions/61223/reconstituting-estimated-predicted-values-to-original-scale-from-minmaxscaler](https://datascience.stackexchange.com/questions/61223/reconstituting-estimated-predicted-values-to-original-scale-from-minmaxscaler)\
\
\
\
             The code is exactly the same as what I was experimenting with. So I will show the necessary code to shuffe at the start and de-shuffle at the end.\
\
\
\
             Shuffling code at the beginning:\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
             | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39 | from sklearn.preprocessing import MinMaxScaler<br>from keras.models import Sequential<br>from keras.layers import Dense<br>from numpy import asarray<br>from numpy import sqrt<br>from matplotlib import pyplot<br>from numpy.random import seed<br>from numpy.random import shuffle<br>from numpy.random import sample<br>import numpy asnp<br>#We will want to reshuffle the data<br>x=\[iforiinrange(100)\]<br>y=\[i\*\*2foriinx\]<br>xfx=np.vstack((x,y)).T<br>xy=xfx<br>shuffle(xy)<br>#x = asarray(\[i for i in range(100)\])<br>#y = asarray(\[i\*\*2 for i in x\])<br>#x = asarray(xy\[:,0\]);#x.reshape((len(x),1))<br>x=np.reshape(xy\[:,0\],(100,1))<br>#print('debug, size x = %d ' + str(np.shape(x)))<br>y=np.reshape(xy\[:,1\],(100,1))<br>#y = asarray(xy\[:,1\]);#y.reshape((len(y),1))<br>#print('debug, size y = %d ' + str(np.shape(y)))<br>x\_s=MinMaxScaler()<br>y\_s=MinMaxScaler()<br>x\_scaled=x\_s.fit\_transform(x)<br>y\_scaled=y\_s.fit\_transform(y)<br>#The rest is fed into model<br>......<br>.......<br>yhat=model.predict(x\_scaled)<br>yhat\_original=y\_s.inverse\_transform(yhat) |\
\
\
\
\
\
\
\
\
\
\
\
             The end code was ‘unshuffled’/sorted in order to display the difference between the actual and predicted.\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
             | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19 | #Plotting dots instead of lineplot otherwise we get a zig-zag plot <br>pyplot.plot(x,y,'r.',label='y')<br>pyplot.plot(x,yhat\_original,'b.',label='estimated')<br>#printing the first values - we have to sort the values in order to see them in <br>#their proper context.<br>xy=np.vstack((x\[:,0\],y\[:,0\])).T<br>xyhat=np.vstack((x\[:,0\],yhat\_original\[:,0\])).T<br>xyy=np.sort(xy,axis=0)<br>xyhatt=np.sort(xyhat,axis=0)<br>print("printing x, y, yhat")<br>forloop inrange(10):<br>print(xyy\[loop,0\],xyy\[loop,1\],xyhatt\[loop,1\])<br>pyplot.legend()<br>pyplot.show() |\
\
\
\
\
\
\
\
\
\
\
\
             Here is a listing of x, f(x) and yhat\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
             | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11 | printingx,y,yhat<br>001.4915295839309692<br>112.66086745262146<br>244.75526237487793<br>399.125076293945312<br>41615.723174095153809<br>52524.287418365478516<br>63635.04938507080078<br>74947.73912811279297<br>86462.95930480957031<br>98180.16889190673828 |\
\
\
\
\
\
\
\
\
\
\
\
             Things to improve:\
\
\
             \\* adjusting the number of layers.\
\
\
             \\* adjusting how many neurons in each layer\
\
\
             \\* adjusting the batch size\
\
\
             \\* adjusting the epoch size\
\
\
             In addition\
\
\
             \\* look at k-folds for further model refinement.\
\
\
\
             Thank you\
\
\
             Anthony of Sydney\
\
           - ![](https://secure.gravatar.com/avatar/4b47ee02646f260b78df6a9a9972da4f21650097571c2771ece53142d4f9a6a0?s=40&d=mm&r=g)\
\
\
\
             Anthony The KoalaNovember 24, 2019 at 4:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512172 "Direct link to this comment")\
\
\
\
\
\
             Dear Dr Jason,\
\
\
             Here is an even improved version with very close results.\
\
\
             Instead of MinMaxScaler, I took the logs (to the base e) of the inputs x and f(x) applied my model, then retransformed my model to its original values.\
\
\
\
             Snippets of code transforming the data\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
             | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24 | #We will want to reshuffle the data<br>x=\[iforiinrange(100)\]<br>y=\[i\*\*2foriinx\]<br>xfx=np.vstack((x,y)).T<br>xy=xfx<br>seed(1)<br>shuffle(xy)<br>x=np.reshape(xy\[:,0\],(100,1))<br>print('debug, size x = %d '+str(np.shape(x)))#shape is (100,1)<br>y=np.reshape(xy\[:,1\],(100,1))<br>print('debug, size y = %d '+str(np.shape(y)))<br>#x\_s = MinMaxScaler()<br>#x\_s = MinMaxScaler(feature\_range =(0,200))<br>#y\_s = MinMaxScaler()<br>#x\_scaled = x\_s.fit\_transform(x)<br>#y\_scaled = y\_s.fit\_transform(y)<br>x\_scaled=np.log(x+1);\# we add 1 so as not to have an error as log(0) produces an error<br>y\_scaled=np.log(y+1);\# we add 1 so as not to have an error as log(0) produces an error<br>model=Sequential()<br>...\# the model is applied on the transformed data |\
\
\
\
\
\
\
\
\
\
\
\
             The\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
             | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26 | #We need to resort the numbers<br>#in order to print the first 10 values<br>xy=np.vstack((x\[:,0\],y\[:,0\])).T<br>xyhat=np.vstack((x\[:,0\],yhat\_original\[:,0\])).T<br>xyy=np.sort(xy,axis=0)<br>xyhatt=np.sort(xyhat,axis=0)<br>print("printing x, y, yhat")<br>forloop inrange(10):<br>print(xyy\[loop,0\],xyy\[loop,1\],xyhatt\[loop,1\])<br>#want to predict for the values 100 and 200<br>Xnew=np.reshape(\[100,200\],(2,1))<br>print("let's predict for values 100 and 200")<br>print("the values of x = Xnew before transform %s, %s "%(Xnew\[0\],Xnew\[1\]))<br>Xnew=np.log(Xnew+1)<br>print("values of scaled xnew to put into the model %s, %s "%(Xnew\[0\],Xnew\[1\]))<br>ynew=model.predict(Xnew)<br>#Re-transform  the original values<br>ynew=np.exp(ynew)-1<br>print("The values of Xnew and its predicted yhat")<br>forloop inrange(len(Xnew)):<br>print("Xnew\[%s\] = %s, ynew\[%s\] = %s "%(loop,Xnew\[loop\],loop,ynew\[loop\])) |\
\
\
\
\
\
\
\
\
\
\
\
             The resulting output: Note how close the actual f(x) is to the predicted f(x)\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
             | 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19 | printingx,y,yhat<br>000.00208890438079834<br>110.9818048477172852<br>244.111057281494141<br>399.025933265686035<br>41615.918327331542969<br>52524.944564819335938<br>63636.00426483154297<br>74949.05435562133789<br>86463.969764709472656<br>98180.93276977539062<br>let'spredict forvalues100and200<br>the values ofx=Xnew before transform\[100\],\[200\]<br>values of scaled xnew toput into the model\[4.61512052\],\[5.30330491\]<br>The values of Xnew andits predicted yhat<br>Xnew\[0\]=\[100.\],ynew\[0\]=\[10008.037\]<br>Xnew\[1\]=\[200.\],ynew\[1\]=\[40082.062\] |\
\
           - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
             [Jason Brownlee](https://machinelearningmastery.com/)November 25, 2019 at 6:21 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512250 "Direct link to this comment")\
\
\
\
\
\
             Nice work.\
326. ![](https://secure.gravatar.com/avatar/f7dca5999b5879bf967f9d9fbe56b76f58859b5451b9903097236d9e64b7230d?s=40&d=mm&r=g)\
\
\
\
     kamuOctober 6, 2019 at 7:51 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504402 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
     Thank you very much for “Your First Deep Learning Project in Python with Keras Step-By-Step” tutorial. It is very useful for me. I want to ask you:\
\
\
     Can I code:\
\
\
\
     model.add(Dense(8)) # input layer\
\
\
     model.add(Dense(12, activation=’relu’)) # first hidden layer\
\
\
\
     Instead of:\
\
\
\
     model.add(Dense(12, input\_dim=8, activation=’relu’)) # input layer and first hidden layer\
\
\
\
     Sincerely.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504402)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 7, 2019 at 8:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504472 "Direct link to this comment")\
\
\
\
\
\
       No.\
\
\
\
       The input\_dim argument defines the input layer.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-504472)\
327. ![](https://secure.gravatar.com/avatar/dd089ff54180bce35eb02b444aa3b257ace873e55c651013cd22156651854ee4?s=40&d=mm&r=g)\
\
\
\
     keryumsOctober 17, 2019 at 1:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505953 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason, is it not necessary to use the keras utilility ‘to\_categorical’ to convert your y vector into a matrix before fitting the model?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505953)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 17, 2019 at 6:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505988 "Direct link to this comment")\
\
\
\
\
\
       You can, or you can use the sklearn tools to do the same thing.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505988)\
328. ![](https://secure.gravatar.com/avatar/af48175eac546aa890f6e4e518acbe03a85d7c0a0166a237e650bc1d78de976c?s=40&d=mm&r=g)\
\
\
\
     Aquilla Setiawan KanadiOctober 17, 2019 at 6:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505986 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks a lot for your tutorial about deep learning project, it really help me a lot in my journey to learn machine learning.\
\
\
\
     I have a question about the data splitting in code above, how is the splitting work between data for training and the data for validate the training data? I’ve tried to read your tutorial about the data splitting but i have no ideas about the data splitting work above.\
\
\
\
     Thankyou,\
\
\
\
     Aquilla\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505986)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 17, 2019 at 6:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505998 "Direct link to this comment")\
\
\
\
\
\
       We did not split the data, we fit and evaluated on one set. We did this for brevity.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-505998)\
329. ![](https://secure.gravatar.com/avatar/9124899785937d3f43c4c27a59aef41b40e63173dcbd45bdffa3f0f7440955c3?s=40&d=mm&r=g)\
\
\
\
     Love your work!October 17, 2019 at 11:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506013 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I just wanted to thank you. This tutorial is incredibly clear and well presented. Unlike many other online tutorials you explain very eloquently the intuition behind the lines of code and what is being accomplished which is very useful. As someone just starting out with Keras I had been finding some of the coding, as well as how Keras and Tensorflow interact, confusing. After your explanations Keras seems incredibly basic. I’ve been looking over some of my recent code from other Keras tutorials and I now understand how everything works.\
\
\
\
     Thanks again!\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506013)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 17, 2019 at 1:50 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506030 "Direct link to this comment")\
\
\
\
\
\
       Well done on your progress and thanks for your support!\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506030)\
330. ![](https://secure.gravatar.com/avatar/6769442fc165f5a2e597755363088d64900dbf5e59ef79f364619341b5119c10?s=40&d=mm&r=g)\
\
\
\
     AhmedOctober 19, 2019 at 6:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506209 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason. I am deeply grateful to this amazing work. Everything works well so far. King Regards\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506209)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 19, 2019 at 6:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506241 "Direct link to this comment")\
\
\
\
\
\
       Thanks, well done on your progress!\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-506241)\
331. ![](https://secure.gravatar.com/avatar/6c93bb8bc185240045d575e980e6cdc94e72bb51b2c5a009136f042aa59daafb?s=40&d=mm&r=g)\
\
\
\
     JAMES JONAHOctober 28, 2019 at 10:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507719 "Direct link to this comment")\
\
\
\
\
\
     Please i need help, which algorithms is the best in cyber threat detection and how to implement it. thanks\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507719)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 28, 2019 at 1:18 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507735 "Direct link to this comment")\
\
\
\
\
\
       This is a common question that I answer here:\
\
       [https://machinelearningmastery.com/faq/single-faq/what-algorithm-config-should-i-use](https://machinelearningmastery.com/faq/single-faq/what-algorithm-config-should-i-use)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507735)\
332. ![](https://secure.gravatar.com/avatar/067bd056a7a78c0c9f0ba8904021c61af888da1232b29ae3d0dafa75eb968870?s=40&d=mm&r=g)\
\
\
\
     shivanOctober 29, 2019 at 7:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507820 "Direct link to this comment")\
\
\
\
\
\
     hello sir\
\
\
     do you have an implementation about (medical image analysis with deep learning).\
\
\
     i need to start with medical image NOT real world image\
\
\
     thanks for your help.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507820)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 29, 2019 at 1:47 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507847 "Direct link to this comment")\
\
\
\
\
\
       Not really, sorry.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507847)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/067bd056a7a78c0c9f0ba8904021c61af888da1232b29ae3d0dafa75eb968870?s=40&d=mm&r=g)\
\
\
\
         shivanOctober 31, 2019 at 9:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508037 "Direct link to this comment")\
\
\
\
\
\
         so, what do you recommend me about it\
\
\
         thanks.\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508037)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)October 31, 2019 at 1:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508050 "Direct link to this comment")\
\
\
\
\
\
           Perhaps start by collecting a dataset.\
\
\
\
           Then consider reviewing the literature to see what types of data prep and models other have used for similar data.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508050)\
333. ![](https://secure.gravatar.com/avatar/1a62d3330e52fcc8eb11b104ae20ff2d8893187b90eeb21c9b73ad597d2d165e?s=40&d=mm&r=g)\
\
\
\
     Nasir ShahOctober 30, 2019 at 7:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507917 "Direct link to this comment")\
\
\
\
\
\
     Sir. i am new to neural network. so from where i start it. or which tutorial i watch . i didn’t have any idea about it.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507917)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)October 30, 2019 at 1:55 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507938 "Direct link to this comment")\
\
\
\
\
\
       Yes, you can start here:\
\
       [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-507938)\
334. ![](https://secure.gravatar.com/avatar/612dfe11dbacf581fa93d2ffad3fe0433beb2a8b80003497e1dee925da0bdf0d?s=40&d=mm&r=g)\
\
\
\
     hima hansiNovember 3, 2019 at 1:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508490 "Direct link to this comment")\
\
\
\
\
\
     hello sir, I’m new to this field. I’m going to develop monophonic musical instrument classification system using python and Keras. sir,I want to find monophonic data set, how can I find it.\
\
\
     I try to get piano music from you tube and convert it to .waw file and splitting it. Is it a good or bad ? or an other methods available to get free data set on the web.. give your suggestions please ??\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508490)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 4, 2019 at 6:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508552 "Direct link to this comment")\
\
\
\
\
\
       Perhaps this will help:\
\
       [https://machinelearningmastery.com/faq/single-faq/where-can-i-get-a-dataset-on-\_\_\_](https://machinelearningmastery.com/faq/single-faq/where-can-i-get-a-dataset-on-___)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-508552)\
335. ![](https://secure.gravatar.com/avatar/dda0ffecd59f823d4431a1d7c5f1c7cec59527bf7925605a4f1910c96907298f?s=40&d=mm&r=g)\
\
\
\
     Mona AhmedNovember 20, 2019 at 3:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-511590 "Direct link to this comment")\
\
\
\
\
\
     i got score 76.69\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-511590)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 20, 2019 at 6:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-511624 "Direct link to this comment")\
\
\
\
\
\
       Well done!\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-511624)\
336. ![](https://secure.gravatar.com/avatar/0a94d56f514ce03d1b11b72f62a61877735745fbf3909e8d2fff2191929c966e?s=40&d=mm&r=g)\
\
\
\
     Niall XieNovember 26, 2019 at 8:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512423 "Direct link to this comment")\
\
\
\
\
\
     Hello, I just want to say that I am elated to use your tutorial. So, I am working on a group project with my team and I used datasets representing heart disease, diabetes and breast cancer for this tutorial. However, this code example will give an error when the cell contains a string value, in this case… title names like clump\_thickess and ? will produce an error. how do I fix this?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512423)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 26, 2019 at 1:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512454 "Direct link to this comment")\
\
\
\
\
\
       Thanks.\
\
\
\
       Perhaps try encoding your categories using a one hot encoding first:\
\
       [https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/](https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512454)\
337. ![](https://secure.gravatar.com/avatar/228eecebae85a837186bf94d26e21cafc9c8ef2fc5d8a6405b5be0ce4260a274?s=40&d=mm&r=g)\
\
\
\
     MohamedNovember 28, 2019 at 10:46 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512847 "Direct link to this comment")\
\
\
\
\
\
     thank you sir for this article, would you please suggest an example with testing data ?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512847)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)November 29, 2019 at 6:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512894 "Direct link to this comment")\
\
\
\
\
\
       Sorry I don’t understand your question, can you elaborate?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-512894)\
338. ![](https://secure.gravatar.com/avatar/cc2b5fd35821c78ad0cda7701243a71e99189b13b7a7cd3708ba97a1927a921b?s=40&d=mm&r=g)\
\
\
\
     ChrisDecember 3, 2019 at 10:49 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-513499 "Direct link to this comment")\
\
\
\
\
\
     I believe there is something wrong with the (150/10) 15 updates to the model weights. The internal coefficients are updated after every single batch. Our data is comprised of 768 samples. Since batch\_size=10, we obtain 77 batches (76 with 10 samples and one with 8). Therefore, at each epoch we should see 77 updates of weights and coefficients and not 15. Moreover, the total number of updates must be: 150\*77=11550. Am I missing something important?\
\
\
\
     Really good job and very well-written article (all your articles). Keep up the good job. Cheers\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-513499)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 4, 2019 at 5:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-513555 "Direct link to this comment")\
\
\
\
\
\
       You’re right. Not sure what I was thinking there. Simplified.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-513555)\
339. ![](https://secure.gravatar.com/avatar/a3318725b7b90a6401420e58cc6df38fd6af1e1dba05387de423a102b2927f3b?s=40&d=mm&r=g)\
\
\
\
     JustineDecember 14, 2019 at 9:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515062 "Direct link to this comment")\
\
\
\
\
\
     Thanks! This is my first foray into keras, and the tutorial went swimmingly. Am now training on my own data. It is not performing worse than on my other machine learning models (that’s a win :).\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515062)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 15, 2019 at 6:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515181 "Direct link to this comment")\
\
\
\
\
\
       Well done!\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515181)\
340. ![](https://secure.gravatar.com/avatar/52ad9a366f90566fc043b14d98a2dd88c6902b9ccbcb4125967936b66ae41d1a?s=40&d=mm&r=g)\
\
\
\
     xDecember 17, 2019 at 8:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515522 "Direct link to this comment")\
\
\
\
\
\
     Hi，Jason. Thanks so much for your answer. Now my question is why I can’t found my directory in Jupyter and put the ‘pima-indians-diabetes.csv’ in it.\
\
\
     OSError Traceback (most recent call last)\
\
\
     in\
\
\
     4 from keras.layers import Dense\
\
\
     5 # load the dataset\
\
\
     —-\> 6 dataset = loadtxt(‘pima-indians-diabetes.csv’, delimiter=’,’)\
\
\
     7 # split into input (X) and output (y) variables\
\
\
     8 X = dataset\[:,0:8\]\
\
\
\
     D:\\anaconda\\lib\\site-packages\\numpy\\lib\\npyio.py in loadtxt(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, max\_rows)\
\
\
     966 fname = os\_fspath(fname)\
\
\
     967 if \_is\_string\_like(fname):\
\
\
     –\> 968 fh = np.lib.\_datasource.open(fname, ‘rt’, encoding=encoding)\
\
\
     969 fencoding = getattr(fh, ‘encoding’, ‘latin1’)\
\
\
     970 fh = iter(fh)\
\
\
\
     D:\\anaconda\\lib\\site-packages\\numpy\\lib\\\_datasource.py in open(path, mode, destpath, encoding, newline)\
\
\
     267\
\
\
     268 ds = DataSource(destpath)\
\
\
     –\> 269 return ds.open(path, mode, encoding=encoding, newline=newline)\
\
\
     270\
\
\
     271\
\
\
\
     D:\\anaconda\\lib\\site-packages\\numpy\\lib\\\_datasource.py in open(self, path, mode, encoding, newline)\
\
\
     621 encoding=encoding, newline=newline)\
\
\
     622 else:\
\
\
     –\> 623 raise IOError(“%s not found.” % path)\
\
\
     624\
\
\
     625\
\
\
\
     OSError: pima-indians-diabetes.csv not found.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515522)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 17, 2019 at 1:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515534 "Direct link to this comment")\
\
\
\
\
\
       Perhaps try running the code file from the command line, as follows:\
\
       [https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line](https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-515534)\
341. ![](https://secure.gravatar.com/avatar/f187af3840e70f26d13edb66f265304e4445b19ecd7eaeba4fd179065dab36a0?s=40&d=mm&r=g)\
\
\
\
     Manohar NookalaDecember 22, 2019 at 9:32 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-516052 "Direct link to this comment")\
\
\
\
\
\
     Hi sir,\
\
\
     My name is manohar. i trained a deep learning model on car price prediction. i got\
\
\
\
     loss: nan – acc: 0.0000e+00. if you give me your email ID then i will send you. you can tell me the problem. please do this help because i am a beginner.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-516052)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)December 23, 2019 at 6:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-516076 "Direct link to this comment")\
\
\
\
\
\
       Perhaps you need to scale the data prior to fitting?\
\
\
       Perhaps you need to use relu activation?\
\
\
       Perhaps you need some type of regularization?\
\
\
       Perhaps you need a larger or smaller model?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-516076)\
342. ![](https://secure.gravatar.com/avatar/44a0e62ce14cca112949ba7d50428fa6c6ae595fd632362dcc3fd25bafa5acb0?s=40&d=mm&r=g)\
\
\
\
     Shone XuJanuary 5, 2020 at 1:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517112 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     thanks and it is a great tutorial. just 1 question. do we have to train the model by “model.fit(x, y, epochs=150, batch\_size=10)” every time before making the prediction because it takes a very long time to train the model. I am just wondering whether it is possible to save the trained model and go straight to the prediction skipping the model.fit (eg: pickle)?\
\
\
\
     many thanks for your advice in advance\
\
\
\
     cheers\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517112)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 5, 2020 at 7:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517147 "Direct link to this comment")\
\
\
\
\
\
       No, you can fit the model once, then save it:\
\
       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)\
\
\
\
       Then later load it and make predictions.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517147)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/44a0e62ce14cca112949ba7d50428fa6c6ae595fd632362dcc3fd25bafa5acb0?s=40&d=mm&r=g)\
\
\
\
         Shone XuJanuary 7, 2020 at 2:16 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517309 "Direct link to this comment")\
\
\
\
\
\
         Thanks and will check it out\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517309)\
343. ![](https://secure.gravatar.com/avatar/3a4f5aa48bc7ab6b4e6c8ae0ee3fd48af3263b49ec02f4e5d18d2d5582a57c6e?s=40&d=mm&r=g)\
\
\
\
     ustenggJanuary 8, 2020 at 7:40 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517414 "Direct link to this comment")\
\
\
\
\
\
     Thank you so much for this tutorial sir but How can I use the model to predict using data outside the dataset?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517414)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 9, 2020 at 7:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517456 "Direct link to this comment")\
\
\
\
\
\
       Call model.predict() with the new inputs.\
\
\
\
       See the “Make Predictions” section.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517456)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/3a4f5aa48bc7ab6b4e6c8ae0ee3fd48af3263b49ec02f4e5d18d2d5582a57c6e?s=40&d=mm&r=g)\
\
\
\
         ustenggJanuary 9, 2020 at 4:07 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517506 "Direct link to this comment")\
\
\
\
\
\
         Nice! Thank you so much, Sir. I figured it out using the link on the “Make predictions” section. I’ve learned a lot from your tutorials. You’re the best!\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517506)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)January 10, 2020 at 7:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517540 "Direct link to this comment")\
\
\
\
\
\
           Nice work!\
\
\
\
           Thanks.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-517540)\
344. ![](https://secure.gravatar.com/avatar/8423ac069a7c39ab8b1c86bf580ce66674fa4bf0689777fcc5b7191ee004bf90?s=40&d=mm&r=g)\
\
\
\
     monicaJanuary 23, 2020 at 4:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518782 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thanks for sharing this post.\
\
\
\
     I have a question, when I tried to split the dataset\
\
\
     (X = dataset\[:,0:8\]\
\
\
     y = dataset\[:,8\])\
\
\
\
     it gives me an error: TypeError: ‘(slice(None, None, None), slice(0, 8, None))’ is an invalid key\
\
\
\
     how can I fix it?\
\
\
\
     Thanks,\
\
\
\
     monica\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518782)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 23, 2020 at 6:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518813 "Direct link to this comment")\
\
\
\
\
\
       Sorry to hear that, this might help:\
\
       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518813)\
345. ![](https://secure.gravatar.com/avatar/a663512ea38e7cc00ec4cef207573e8cc6c0e3245e9ee766a2bee194e5aedb20?s=40&d=mm&r=g)\
\
\
\
     Sam SarjantJanuary 23, 2020 at 9:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518914 "Direct link to this comment")\
\
\
\
\
\
     Thanks for the tutorial! This is a wonderful ‘Hello World’ to Deep Learning\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518914)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 24, 2020 at 7:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518986 "Direct link to this comment")\
\
\
\
\
\
       Thanks, I’m happy it was helpful.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-518986)\
346. ![](https://secure.gravatar.com/avatar/59bf97356442eb3e998aa4f68c1ccf1d913fdaea41540a61ad2771e2d6909544?s=40&d=mm&r=g)\
\
\
\
     KeerthanJanuary 24, 2020 at 4:01 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519037 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason! hope you are doing good.\
\
\
     I am actually doing a project on classification of thyroid disease using back propagation with stocastic gradient descent method,can you help me out with the code a little bit?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519037)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 25, 2020 at 8:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519146 "Direct link to this comment")\
\
\
\
\
\
       Perhaps start by adapting the code in the above tutorial?\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519146)\
347. ![](https://secure.gravatar.com/avatar/462966c7bc78ebfd226b5118fcaeafb123ce925df13f462e96ec3f7964af3210?s=40&d=mm&r=g)\
\
\
\
     ShakirJanuary 25, 2020 at 1:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519097 "Direct link to this comment")\
\
\
\
\
\
     Dear Sir\
\
\
     I want to predict air pollution using deep learning techniques please suggest how to go about with my data sets\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519097)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)January 25, 2020 at 8:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519163 "Direct link to this comment")\
\
\
\
\
\
       Start here:\
\
       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-519163)\
348. ![](https://secure.gravatar.com/avatar/0e6057357818e46b099abed2de54471718cabdee23194b748a0529739b6f7518?s=40&d=mm&r=g)\
\
\
\
     YaredFebruary 7, 2020 at 4:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520743 "Direct link to this comment")\
\
\
\
\
\
     AttributeError: module ‘tensorflow’ has no attribute ‘get\_default\_graph’AttributeError: module ‘tensorflow’ has no attribute ‘get\_default\_graph’\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520743)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 8, 2020 at 7:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520822 "Direct link to this comment")\
\
\
\
\
\
       Perhaps confirm you are using TF 2 and Keras 2.3.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520822)\
349. ![](https://secure.gravatar.com/avatar/0e6057357818e46b099abed2de54471718cabdee23194b748a0529739b6f7518?s=40&d=mm&r=g)\
\
\
\
     YaredFebruary 7, 2020 at 4:41 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520744 "Direct link to this comment")\
\
\
\
\
\
     I went to detect agreement errors in a sentence using LSTM techniques please suggest how to go about with my data sets\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520744)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)February 8, 2020 at 7:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520823 "Direct link to this comment")\
\
\
\
\
\
       You can get started with NLP problems here:\
\
       [https://machinelearningmastery.com/start-here/#nlp](https://machinelearningmastery.com/start-here/#nlp)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-520823)\
350. ![](https://secure.gravatar.com/avatar/441388b14a98e66af8b502d686e867a559db9b30d80e5cadc258961a34d5827a?s=40&d=mm&r=g)\
\
\
\
     Pavitra NayakFebruary 29, 2020 at 2:55 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-523817 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason\
\
\
     I am using this code for my project. It works perfectly for your dataset. But I have a dataset which has too many 0’s and 1’s. So I am getting the wrong prediction. What can I do to solve this problem?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-523817)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 1, 2020 at 5:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-523906 "Direct link to this comment")\
\
\
\
\
\
       Here are some suggestions:\
\
       [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-523906)\
351. ![](https://secure.gravatar.com/avatar/0031eacc444798cfdaac7124b52ff787b48595447f3da5b3a75086ad23f4360a?s=40&d=mm&r=g)\
\
\
\
     nurulMarch 6, 2020 at 5:50 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524612 "Direct link to this comment")\
\
\
\
\
\
     hi. I wanna ask. i had follow all the steps but i’m stuck at the fit the model. This error occured. How can I solve this problem?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524612)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/0031eacc444798cfdaac7124b52ff787b48595447f3da5b3a75086ad23f4360a?s=40&d=mm&r=g)\
\
\
\
       kikiMarch 6, 2020 at 6:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524615 "Direct link to this comment")\
\
\
\
\
\
       I have already tried this step and stuck at the fit phase and got this error. Do you have any solution for my problem?\
\
\
\
       —————————————————————————\
\
\
       ValueError Traceback (most recent call last)\
\
\
       in\
\
\
       1 # fit the keras model on the dataset\
\
\
       —-\> 2 model.fit(x, y, batch\_size=10,epochs=150)\
\
\
\
       ~\\Anaconda4\\lib\\site-packages\\keras\\engine\\training.py in fit(self, x, y, batch\_size, epochs, verbose, callbacks, validation\_split, validation\_data, shuffle, class\_weight, sample\_weight, initial\_epoch, steps\_per\_epoch, validation\_steps, validation\_freq, max\_queue\_size, workers, use\_multiprocessing, \*\*kwargs)\
\
\
       1152 sample\_weight=sample\_weight,\
\
\
       1153 class\_weight=class\_weight,\
\
\
       -\> 1154 batch\_size=batch\_size)\
\
\
       1155\
\
\
       1156 # Prepare validation data.\
\
\
\
       ~\\Anaconda4\\lib\\site-packages\\keras\\engine\\training.py in \_standardize\_user\_data(self, x, y, sample\_weight, class\_weight, check\_array\_lengths, batch\_size)\
\
\
       577 feed\_input\_shapes,\
\
\
       578 check\_batch\_axis=False, # Don’t enforce the batch size.\
\
\
       –\> 579 exception\_prefix=’input’)\
\
\
       580\
\
\
       581 if y is not None:\
\
\
\
       ~\\Anaconda4\\lib\\site-packages\\keras\\engine\\training\_utils.py in standardize\_input\_data(data, names, shapes, check\_batch\_axis, exception\_prefix)\
\
\
       143 ‘: expected ‘ + names\[i\] + ‘ to have shape ‘ +\
\
\
       144 str(shape) + ‘ but got array with shape ‘ +\
\
\
       –\> 145 str(data\_shape))\
\
\
       146 return data\
\
\
       147\
\
\
\
       ValueError: Error when checking input: expected dense\_133\_input to have shape (16,) but got array with shape (17,)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524615)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
         [Jason Brownlee](https://machinelearningmastery.com/)March 7, 2020 at 7:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524665 "Direct link to this comment")\
\
\
\
\
\
         Perhaps this will help you copy the code from the tutorial:\
\
         [https://machinelearningmastery.com/faq/single-faq/how-do-i-copy-code-from-a-tutorial](https://machinelearningmastery.com/faq/single-faq/how-do-i-copy-code-from-a-tutorial)\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524665)\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 7, 2020 at 7:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524663 "Direct link to this comment")\
\
\
\
\
\
       I’m sorry to hear that, perhaps this will help:\
\
       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524663)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/0031eacc444798cfdaac7124b52ff787b48595447f3da5b3a75086ad23f4360a?s=40&d=mm&r=g)\
\
\
\
         kikiMarch 9, 2020 at 12:18 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524941 "Direct link to this comment")\
\
\
\
\
\
         Thanks for the answer jason\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524941)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 10, 2020 at 5:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525007 "Direct link to this comment")\
\
\
\
\
\
           You’re welcome.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525007)\
352. ![](https://secure.gravatar.com/avatar/af806aaf82baeb21391123aacc7f6aeb3e5a0d6009c3eb3b6b2c7c4b41279c54?s=40&d=mm&r=g)\
\
\
\
     lazMarch 7, 2020 at 2:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524718 "Direct link to this comment")\
\
\
\
\
\
     Hey, Jason!\
\
\
\
     Again… Thanks for your awesome tutorials and for giving your knowledge to the public! >800 comments and nearly all answered, you’re great. I can’t understand how you manage all that, writing great content, do ml stuff, teach, learn, great respect!\
\
\
\
     2 general questions:\
\
\
\
     Question(1):\
\
\
\
     Why and when do we need to flatten() inputs and in which cases not?\
\
\
\
     For example 4 numeric inputs, a lag of 2 of every input means 4\*2=8 values per batch:\
\
\
\
     I always do this, no matter how many inputs or lags, i give that as flat array to the input:\
\
\
\
     1 set/batch: \[\[1.0,1.1, 2.0,2.1, 3.0,3.1, 4.0,4.1\]\]\
\
\
\
     Input(shape=(8,)) # keras func api\
\
\
\
     Does it make sense to input a structure like this, if so – why/when?\
\
\
\
     Better? \[\[\[1.0,1.1\], \[2.0,2.1\], \[3.0,3.1\], \[4.0,4.1\]\]\]\
\
\
\
     Question(2):\
\
\
\
     Are you still using Theano? As they do not update it, it becomes older, but not worse ;). I tried Tensorflow a lot – but always with lower performance in terms of speed. Theano is much faster (factor 3-10) for me. But using more than 1 core is always slower for me, in both theano and tf. Did you experienced similar things? I also tried torch, nice but it was also slower as the good old theano. Any ideas or alternatives (i can’t use gpu/external/aws)?\
\
\
\
     I would be happy to see you doing some deep reinforcement learning (DRL) stuff, what do you think? Are you?\
\
\
\
     Regards, keep it up 😉\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524718)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 8, 2020 at 6:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524809 "Direct link to this comment")\
\
\
\
\
\
       You need to flatten when the output shape of one layer does not match the input shape of another, e.g. CNN output to a Dense.\
\
\
\
       No. I use and recommend tensorflow and have for years. Tensorflow used to not work for windows users, so I recommend theano for them – and still do if they have trouble. Theano works fine and will continue to work fine for most applications.\
\
\
\
       No, RL is not practical/useful:\
\
       [https://machinelearningmastery.com/faq/single-faq/do-you-have-tutorials-on-deep-reinforcement-learning](https://machinelearningmastery.com/faq/single-faq/do-you-have-tutorials-on-deep-reinforcement-learning)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524809)\
353. ![](https://secure.gravatar.com/avatar/af806aaf82baeb21391123aacc7f6aeb3e5a0d6009c3eb3b6b2c7c4b41279c54?s=40&d=mm&r=g)\
\
\
\
     lazMarch 8, 2020 at 11:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524848 "Direct link to this comment")\
\
\
\
\
\
     Dear Jason, thanks for your answer ;)…\
\
\
\
     “flatten when the output shape of one layer does not match the input shape of another, e.g. CNN output to a Dense.”\
\
\
\
     Thanks. The question about the “flatten” operation was not about the flatten() between layers, it was about how to present inputs to the input layer. Sorry for being vague. Maybe I misunderstood something, are there use cases where the FEATURES/INPUTS/LAGS are not flattened?\
\
\
\
     “RL is not practical/useful”\
\
\
     Is this statement based on your experience or do you take the opinion of others without checking it yourself here ;)? Please do not misunderstand, you are the expert here. However, i can refute some arguments against RL.\
\
\
\
     Rewards are hard to create: depends on your environment\
\
\
     Unstable: depends on your environment, code, setup\
\
\
\
     I started experimenting with a simple DQN, I expanded it step by step and now I have a “Dueling Double DQN”. It learns well and quick. I admit – on simple data. But it does it repeatable and reproducible! So i would say: In general, it works.\
\
\
\
     I have to see how it works with more complicated data. That is why I emphasized that the performance of this method strongly depends on the area of application.\
\
\
\
     But there is a huge problem, most public sources contain incorrect code or incorrect implementations. I have never reported or found so many bugs on any subject. These errors are copied again and again and in the end many think that they are correct. I have collected tons of links and pdf files to understand and debug this beast.\
\
\
\
     No matter, you have to decide for yourself. If you want to take a look at it, take a simple example, even the DQN (without dueling or double) is able to learn – if the code is correct. And although I’m not a mathematician: to understand how it works and what possibilities it offers – made me smile 😉 …\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524848)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 9, 2020 at 7:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524916 "Direct link to this comment")\
\
\
\
\
\
       For more on the input shape of LSTMs/1d CNNs, see this:\
\
       [https://machinelearningmastery.com/faq/single-faq/what-is-the-difference-between-samples-timesteps-and-features-for-lstm-input](https://machinelearningmastery.com/faq/single-faq/what-is-the-difference-between-samples-timesteps-and-features-for-lstm-input)\
\
\
\
       I don’t yet see an ROI for “developers at work” in covering RL as described in the link.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524916)\
354. ![](https://secure.gravatar.com/avatar/af806aaf82baeb21391123aacc7f6aeb3e5a0d6009c3eb3b6b2c7c4b41279c54?s=40&d=mm&r=g)\
\
\
\
     lazMarch 8, 2020 at 10:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524880 "Direct link to this comment")\
\
\
\
\
\
     Interesting read:\
\
\
\
     “We use a double deep Q-learning network (DDQN) to find the right material type and the optimal geometrical design for metasurface holograms to reach high efficiency. The DDQN acts like an intelligent sweep and could identify the optimal results in ~5.7 billion states after only 2169 steps. The optimal results were found between 23 different material types and various geometrical properties for a three-layer structure. The computed transmission efficiency was 32% for high-quality metasurface holograms; this is two times bigger than the previously reported results under the same conditions.”\
\
\
\
     [https://www.nature.com/articles/s41598-019-47154-z](https://www.nature.com/articles/s41598-019-47154-z)\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524880)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 9, 2020 at 7:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524920 "Direct link to this comment")\
\
\
\
\
\
       Thanks for sharing.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-524920)\
355. ![](https://secure.gravatar.com/avatar/c78fc043d4d80fb513a240a1afe0301314efbd942f7ad9f01ebb959b79d599f8?s=40&d=mm&r=g)\
\
\
\
     YzNMarch 11, 2020 at 4:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525131 "Direct link to this comment")\
\
\
\
\
\
     Literally the best “first neural network tutorial”\
\
\
     Got 85.68 acc by adding layers and decreasing batch size\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525131)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 11, 2020 at 5:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525166 "Direct link to this comment")\
\
\
\
\
\
       Thanks.\
\
\
\
       Well done!\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525166)\
356. ![](https://secure.gravatar.com/avatar/f6f796295d5552f2dca151e785c2cf4d78d2861b1039b8832e36cf06be891d16?s=40&d=mm&r=g)\
\
\
\
     NehaMarch 14, 2020 at 12:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525517 "Direct link to this comment")\
\
\
\
\
\
     Hello Jason,\
\
\
     I have a quick question.\
\
\
     I am trying to build just 1 sigmoid neuron for a binary classification task, basically I am implying this is how 1 sigmoid model is:\
\
\
\
     model = Sequential()\
\
\
     model.add(Dense(1, activation=’sigmoid’))\
\
\
\
     My inputs are images of size = (39\*39\*3)\
\
\
\
     I am unsure as to how to input these images to my Dense layer (which is the only layer I am using)\
\
\
\
     I am currently using below for inputting my images:\
\
\
\
     train\_generator = train\_datagen.flow\_from\_directory(train\_data\_dir,\
\
\
     target\_size=(39, 39),\
\
\
     batch\_size=batch\_size)\
\
\
     class\_mode=’binary’)\
\
\
\
     But somehow Dense layer cannot accept input shape (39, 39, 3).\
\
\
\
     So my question is, how do I input my images data to the Dense layer?\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525517)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 14, 2020 at 8:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525552 "Direct link to this comment")\
\
\
\
\
\
       You can flatten the input or use a CNN as the input instead that is designed for 3d input samples.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-525552)\
357. ![](https://secure.gravatar.com/avatar/8b01ac3d7a6d246cea0feadb0be4bb41e3f91525c30583789474d631888e0972?s=40&d=mm&r=g)\
\
\
\
     Bertrand BruMarch 29, 2020 at 12:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527335 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     Thank you very much for your tutorial.\
\
\
\
     I am new in the world of deep leraning. I have been able to modify your code and make it work for a set of data I recorded with a 3 axis accelerometer. My goal was to detect if I was walking or running. I recorded around 50 trials of each activities. From the signal, I calculated specific parameters that enable the code to differenciate the two activities. Amongst the parameters, I calculated for all axis, the mean, min and max values, and some parameters in the domain frequencies (the 3 first peak of the power spectrum and their respective position).\
\
\
\
     It works very well and I am able to easily detect if I am running or walking.\
\
\
\
     I then decided to add a thrid activities: standing. I also recorded 50 trials of this activity. If I train my model with standing and running, I can identify the two activity. Same if I train it with standing and walking or with walking and running.\
\
\
\
     It is more complicated if I train my model with the three activities. In fact, it can’t do it. It can only recgonise the first two activities. So for example if standing, walking and running have the following ID: 0, 1 and 2, then it can only detect 0 and 1 (standing and walking). It thinks that all running trials are walking trials. If standing, running and walinking have the following ID: 0, 1 and 2, then it can only detect 0 and 1 (standing and running). It thinks that all walking trials are running trials.\
\
\
\
     So here is my question: Assuming you have the dataset, if you needed to adapt your code so it can detect if people are 0: not diabetic, 1: people are diabetic type 1, and 2: people are diabetic type 2, how would you modify your script?\
\
\
\
     Thank you very much for your help.\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527335)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)March 29, 2020 at 6:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527398 "Direct link to this comment")\
\
\
\
\
\
       You’re welcome.\
\
\
\
       Well done.\
\
\
\
       This is called multi-class classification, this tutorial will help:\
\
       [https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527398)\
\
\
\
\
       - ![](https://secure.gravatar.com/avatar/8b01ac3d7a6d246cea0feadb0be4bb41e3f91525c30583789474d631888e0972?s=40&d=mm&r=g)\
\
\
\
         Bertrand BruMarch 29, 2020 at 7:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527414 "Direct link to this comment")\
\
\
\
\
\
         Thank you so much for coming to me so quickly.\
\
\
         This is exactly what I was looking for.\
\
\
         Cheers,\
\
\
\
         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527414)\
\
\
\
\
         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
           [Jason Brownlee](https://machinelearningmastery.com/)March 30, 2020 at 5:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527533 "Direct link to this comment")\
\
\
\
\
\
           You’re welcome.\
\
\
\
           I’m happy to hear that.\
\
\
\
           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527533)\
358. ![](https://secure.gravatar.com/avatar/df1bb3cc2d77568c1a4c427f40a429e7c764fe393015b0284ce96598b74a0132?s=40&d=mm&r=g)\
\
\
\
     Dipak KambaleMarch 31, 2020 at 10:16 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527875 "Direct link to this comment")\
\
\
\
\
\
     Hi Jason,\
\
\
\
     I got accuracy 75.52 . Is it ok?? please let me know\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527875)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 1, 2020 at 5:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527928 "Direct link to this comment")\
\
\
\
\
\
       Well done. Try running the example a few times.\
\
\
\
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527928)\
359. ![](https://secure.gravatar.com/avatar/671ef85f77c7beec33b62f30b4a7443c117ea79a065a08322a7092b844537d58?s=40&d=mm&r=g)\
\
\
\
     islamuddinApril 1, 2020 at 6:20 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527996 "Direct link to this comment")\
\
\
\
\
\
     hello sir jason.\
\
\
     sir how to satiable accuracy run the cod one given out for example 86% next time 82% how to solve this!\
\
\
\
     #import\
\
\
     from numpy import loadtxt\
\
\
     from keras.models import Sequential\
\
\
     from keras.layers import Dense\
\
\
\
     \# load the dataset\
\
\
     dataset = loadtxt(‘E:/ms/impotnt/iwp1.csv’, delimiter=’,’)\
\
\
     \# split into input (X) and output (y) variables\
\
\
     X = dataset\[:,0:8\]\
\
\
     y = dataset\[:,8\]\
\
\
     \# define the keras model\
\
\
\
     #model = Sequential()\
\
\
     model = Sequential()\
\
\
     #model.add(Dense(25, input\_dim=8, init=’uniform’, activation=’relu’))\
\
\
     model.add(Dense(30, input\_dim=8, activation=’relu’))\
\
\
     model.add(Dense(95, activation=’relu’))\
\
\
     model.add(Dense(377, activation=’relu’))\
\
\
     model.add(Dense(233, activation=’relu’))\
\
\
     model.add(Dense(55, activation=’relu’))\
\
\
     model.add(Dense(1, activation=’sigmoid’))\
\
\
\
     \# compile the keras model\
\
\
     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])\
\
\
     \# fit the keras model on the dataset\
\
\
     model.fit(X, y, epochs=150, batch\_size=10)\
\
\
     \# evaluate the keras model\
\
\
     \_, accuracy = model.evaluate(X, y)\
\
\
     print(‘Accuracy: %.2f’ % (accuracy\*100))\
\
\
\
     output\
\
\
\
     0.1153 – accuracy: 0.9531\
\
\
     Epoch 149/150\
\
\
     768/768 \[==============================\] – 0s 278us/step – loss: 0.1330 – accuracy: 0.9401\
\
\
     Epoch 150/150\
\
\
     768/768 \[==============================\] – 0s 277us/step – loss: 0.1468 – accuracy: 0.9375\
\
\
     768/768 \[==============================\] – 0s 41us/step\
\
\
     Accuracy: 94.01\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-527996)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 2, 2020 at 5:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-528056 "Direct link to this comment")\
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
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-528056)\
360. ![](https://secure.gravatar.com/avatar/fed950043073e196d61e42de4b573e8bf455ed59b2d69dfe1c1c82da56354aec?s=40&d=mm&r=g)\
\
\
\
     M Husnain Ali NasirApril 3, 2020 at 2:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-528166 "Direct link to this comment")\
\
\
\
\
\
     Traceback (most recent call last):\
\
\
     File “keras\_first\_network.py”, line 7, in\
\
\
     dataset = loadtxt(‘pima-indians-diabetes.csv’, delimiter=’,’)\
\
\
     File “C:\\Users\\Hussnain\\anaconda3\\lib\\site-packages\\numpy\\lib\\npyio.py”, line 1159, in loadtxt\
\
\
     for x in read\_data(\_loadtxt\_chunksize):\
\
\
     File “C:\\Users\\Hussnain\\anaconda3\\lib\\site-packages\\numpy\\lib\\npyio.py”, line 1087, in read\_data\
\
\
     items = \[conv(val) for (conv, val) in zip(converters, vals)\]\
\
\
     File “C:\\Users\\Hussnain\\anaconda3\\lib\\site-packages\\numpy\\lib\\npyio.py”, line 1087, in\
\
\
     items = \[conv(val) for (conv, val) in zip(converters, vals)\]\
\
\
     File “C:\\Users\\Hussnain\\anaconda3\\lib\\site-packages\\numpy\\lib\\npyio.py”, line 794, in floatconv\
\
\
     return float(x)\
\
\
     ValueError: could not convert string to float: ‘”6’\
\
\
\
     I AM HAVIN THE ABOVE ERROR WHILE RUNNING IT PLEaSE HELP. I am using Anaconda 3 , Python 3.7 , tensorflow ,keras\
\
\
\
     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-528166)\
\
\
\
\
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)\
\
\
\
       [Jason Brownlee](https://machinelearningmastery.com/)April 3, 2020 at 6:57 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-528213 "Direct link to this comment")\
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
       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-528213)\
361. ![](https://secure.gravatar.com/avatar/4806a292782c8834a3feeb3e7552111a274e010a4d81abee2f0eb6f9369529f3?s=40&d=mm&r=g)\
\
\
\
     Madhawa AkalankaApril 9, 2020 at 6:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529036 "Direct link to this comment")\
\
\
\
\
\
     (base) C:\\Users\\Madhawa Akalanka\\python codes>python keras\_first\_network.py\
\
\
     Using TensorFlow backend.\
\
\
     2020-04-09 13:42:28.003791: I tensorflow/core/platform/cpu\_feature\_guard.cc:142\]


     Your CPU supports instructions that this TensorFlow binary was not compiled to


     use: AVX AVX2


     2020-04-09 13:42:28.014066: I tensorflow/core/common\_runtime/process\_util.cc:147


     \] Creating new thread pool with default inter op setting: 2. Tune using inter\_op


     \_parallelism\_threads for best performance.


     Traceback (most recent call last):


     File “keras\_first\_network.py”, line 12, in


     model.fix(X,Y,epochs=150,batch\_size=10)


     AttributeError: ‘Sequential’ object has no attribute ‘fix’



     I had this error while it’s being run. please help.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529036)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 10, 2020 at 8:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529127 "Direct link to this comment")





       Sorry to hear that, see this:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529127)
362. ![](https://secure.gravatar.com/avatar/06b3cc9442130e53e7bc081762d14c0c39363a3edfdf90170b8e85ae3171b328?s=40&d=mm&r=g)



     [Rahim Dehkharghani](https://rtims.ubonab.ac.ir/~rdehkharghani/en/)April 14, 2020 at 2:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529621 "Direct link to this comment")





     Dear Jason


     Thanks for your wonderful website and books. I am a PhD holder and one of your fans in Deep Learning. Sometimes I get disappointed because I cannot achieve my goal in this area. My goal is to discover something new and publish it. Although I understand your codes mostly but having contribution in this field is difficult and requires understanding the whole theory which I have not been able to do so far. Can you please give me some tips to continue? Thanks a lot



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529621)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 14, 2020 at 6:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529664 "Direct link to this comment")





       You’re welcome.



       Keep working on it every day. That’s my best advice.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-529664)
363. ![](https://secure.gravatar.com/avatar/f67b14138923c88a0bdc052037d94e50baaea3eaf04b49cca3fe2aa790af8c02?s=40&d=mm&r=g)



     MattGurneyApril 16, 2020 at 10:52 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530019 "Direct link to this comment")





     There is a typo “input to the model lis defined”



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530019)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 17, 2020 at 6:21 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530077 "Direct link to this comment")





       Thanks! Fixed.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530077)
364. ![](https://secure.gravatar.com/avatar/f67b14138923c88a0bdc052037d94e50baaea3eaf04b49cca3fe2aa790af8c02?s=40&d=mm&r=g)



     MattGurneyApril 16, 2020 at 11:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530026 "Direct link to this comment")





     Using the latest libraries today I get a number of warnings due to latest numpy: 1.18.1 not being compatible with latest TensorFlow: 1.13.1.



     i.e:


     FutureWarning: Passing (type, 1) or ‘1type’ … (6 times)


     to\_int32 (from tensorflow.python.ops.math\_ops) is deprecated



     Options are to revert to an older numpy or suppress the warnings, I took the suppress route with this code:



     \# first neural network with keras tutorial



     \# Suppress warnings due to TF / numpy version incompatibility: [https://github.com/tensorflow/tensorflow/issues/30427#issuecomment-527891497](https://github.com/tensorflow/tensorflow/issues/30427#issuecomment-527891497)


     import warnings


     warnings.filterwarnings(‘ignore’, category=FutureWarning)



     import tensorflow



     \# Suppress warning from TF: to\_int32 (from tensorflow.python.ops.math\_ops) is deprecated: [https://github.com/aamini/introtodeeplearning/issues/25#issuecomment-578404772](https://github.com/aamini/introtodeeplearning/issues/25#issuecomment-578404772)


     import logging


     logging.getLogger(‘tensorflow’).setLevel(logging.ERROR)



     import keras


     from numpy import loadtxt


     from keras.models import Sequential


     from keras.layers import Dense



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530026)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 17, 2020 at 6:21 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530078 "Direct link to this comment")





       I recommend using Keras 2.3 and TensorFlow 2.1.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530078)




       - ![](https://secure.gravatar.com/avatar/f67b14138923c88a0bdc052037d94e50baaea3eaf04b49cca3fe2aa790af8c02?s=40&d=mm&r=g)



         MattGurneyApril 17, 2020 at 12:18 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530128 "Direct link to this comment")





         Yes, upgrading to tensorFlow 2.1 fixed it, I have now removed my warnings suppression and I don’t see the warnings in the output



         I upgraded TF like this:


         pip install –upgrade tensorflow



         I did follow your installation instructions from [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) and ended up with TF version 1.13.1. The command I ran was:


         conda install -c conda-forge tensorflow



         I am on Mac, I see possible relevant discussion here on TF2.1 not on conda: [https://github.com/tensorflow/tensorflow/issues/35754](https://github.com/tensorflow/tensorflow/issues/35754)



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530128)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)April 17, 2020 at 1:31 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530144 "Direct link to this comment")





           Well done!



           I use macports myself:

           [https://machinelearningmastery.com/install-python-3-environment-mac-os-x-machine-learning-deep-learning/](https://machinelearningmastery.com/install-python-3-environment-mac-os-x-machine-learning-deep-learning/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530144)
365. ![](https://secure.gravatar.com/avatar/e91b8deafa48d7e693d26e338090bec391ad7fffecfe5a431e996107d0c7d56b?s=40&d=mm&r=g)



     meryemApril 17, 2020 at 1:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530039 "Direct link to this comment")





     Thank you Jason for the tutoriel.I applied your example to mine by adding dropout and standarisation of X



     X = dataset\[:, 0:7\]


     y = dataset\[:, 7\]



     scaler = MinMaxScaler(feature\_range=(0, 1))


     X = scaler.fit\_transform(X)


     \# define the keras model


     model = Sequential()


     model.add(Dense(6, input\_dim=7, activation=’relu’))


     model.add(Dropout(rate=0.3))


     model.add(Dense(6, activation=’relu’))


     model.add(Dropout(rate=0.3))


     model.add(Dense(1, activation=’sigmoid’))


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\]


     history=model.fit(X, y, epochs=30, batch\_size=30, validation\_split=0.1)


     \_, accuracy = model.evaluate(X, y)


     print(‘Accuracy: %.2f’ % (accuracy\*100))



     shows me an accuracy of 100 which is not normal. to adjust my model, what should I do?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530039)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 17, 2020 at 6:22 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530080 "Direct link to this comment")





       Well done!



       Perhaps evaluate your model using k-fold cross validation.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530080)
366. ![](https://secure.gravatar.com/avatar/e91b8deafa48d7e693d26e338090bec391ad7fffecfe5a431e996107d0c7d56b?s=40&d=mm&r=g)



     meryemApril 17, 2020 at 7:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530100 "Direct link to this comment")





     yes i followed your example using k-flod cross validation it gives me always 100%



     if i move standarisation he gives 83% ,can you guide me please



     seed = 4


     numpy.random.seed(seed)


     dataset = loadtxt(‘data.csv’, delimiter=’,’)


     X = dataset\[:, 0:7\]


     Y = dataset\[:, 7\]


     from sklearn.preprocessing import StandardScaler


     sc = StandardScaler()


     X = sc.fit\_transform(X)


     kfold = StratifiedKFold(n\_splits=5, shuffle=True, random\_state=seed)


     cvscores = \[\]


     for train, test in kfold.split(X,Y):


     model = Sequential()


     model.add(Dense(12, input\_dim=7, activation=”relu”))


     model.add(Dropout(rate=0.2))


     model.add(Dense(6, activation=”relu”))


     model.add(Dropout(rate=0.2))


     model.add(Dense(1, activation=”sigmoid”))


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     model.fit(X\[train\], Y\[train\], epochs=20, batch\_size=10, verbose=1)


     scores = model.evaluate(X\[test\], Y\[test\], verbose=0)


     print(“%s: %.2f%%” % (model.metrics\_names\[1\], scores\[1\]\*100))


     cvscores.append(scores\[1\] \* 100)


     print(“%.2f%% (+/- %.2f%%)” % (numpy.mean(cvscores), numpy.std(cvscores)))



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530100)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 17, 2020 at 7:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530109 "Direct link to this comment")





       Nice work! Perhaps your prediction task is trivial?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530109)
367. ![](https://secure.gravatar.com/avatar/e91b8deafa48d7e693d26e338090bec391ad7fffecfe5a431e996107d0c7d56b?s=40&d=mm&r=g)



     meryemApril 17, 2020 at 8:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530113 "Direct link to this comment")





     you are very helpful .


     or because I don’t have enough data.So there is nothing else I can use?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530113)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 17, 2020 at 1:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530139 "Direct link to this comment")





       Perhaps.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530139)
368. ![](https://secure.gravatar.com/avatar/353da2833516ce284f3a9b40f14d75d262f6571fe5415f2e7071d61e2af3c5bf?s=40&d=mm&r=g)



     Farjad HaiderApril 17, 2020 at 11:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530203 "Direct link to this comment")





     Sir Jason you are awesome! Such a nice and easy to comprehend the tutorial. Great Work!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530203)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 18, 2020 at 5:57 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530277 "Direct link to this comment")





       Thanks!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530277)
369. ![](https://secure.gravatar.com/avatar/c871ee450e740093824115d870734853b107504af656262f184d670607dfe67f?s=40&d=mm&r=g)



     Joan EstradaApril 19, 2020 at 3:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530386 "Direct link to this comment")





     “Note, the most confusing thing here is that the shape of the input to the model is defined as an argument on the first hidden layer. This means that the line of code that adds the first Dense layer is doing 2 things, defining the input or visible layer and the first hidden layer.”



     Could you better explain this? Thanks, nice work!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530386)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 19, 2020 at 6:02 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530412 "Direct link to this comment")





       Yes, see this:

       [https://machinelearningmastery.com/faq/single-faq/how-do-you-define-the-input-layer-in-keras](https://machinelearningmastery.com/faq/single-faq/how-do-you-define-the-input-layer-in-keras)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530412)
370. ![](https://secure.gravatar.com/avatar/3954dc257277948fecc1c96de39845397d883d9e15b03f89ac11236d5d7ffea7?s=40&d=mm&r=g)



     HanyApril 19, 2020 at 9:57 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530436 "Direct link to this comment")





     Actually, I cannot thank you enough Dr. Brownlee.



     God Bless you.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530436)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 19, 2020 at 1:14 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530449 "Direct link to this comment")





       Thanks. You’re very welcome!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530449)
371. ![](https://secure.gravatar.com/avatar/06b3cc9442130e53e7bc081762d14c0c39363a3edfdf90170b8e85ae3171b328?s=40&d=mm&r=g)



     [Rahim](https://rtims.ubonab.ac.ir/~rdehkharghani/en/)April 22, 2020 at 5:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530893 "Direct link to this comment")





     Dear Jason


     Thanks for this interesting code. I tested this code on pima-indians-diabetes in my computer with keras 2.3.1 but strangely I got the accuracy of 52%. I wonder why there is this much difference between your accuracy (76%) and mine (52%).



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530893)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 22, 2020 at 6:10 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530925 "Direct link to this comment")





       You’re welcome.



       Perhaps try running the example a few times?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-530925)
372. ![](https://secure.gravatar.com/avatar/68a9afbfe9bea326f5464ac3fc2252e8e159c5be3ae46b68713dced28d0836d0?s=40&d=mm&r=g)



     SarmadApril 24, 2020 at 8:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531525 "Direct link to this comment")





     want to ask: in the first layer(a hidden layer) as we defined input\_dim=8 w.r.t features we have right. and we specify neurons = 12. but concerned is that a thing i studied is that we specify neurons w.r.t to inputs(features) . Means if we have 8 inputs so neurons will also be 8. but you specified as 12. Why?


     2) In any of problem we have to specified a neural network right. it can be any eg: convolutional, recurrent etc. so which neural network we have choose here. and where?


     3) we have to assign weights. so where we have assigned?


     please let me know. Thanks sir.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531525)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 25, 2020 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531581 "Direct link to this comment")





       The first line of the model defines 2 things, the input or visible layer (8) and the first hidden layer (12). More here:

       [https://machinelearningmastery.com/faq/single-faq/how-do-you-define-the-input-layer-in-keras](https://machinelearningmastery.com/faq/single-faq/how-do-you-define-the-input-layer-in-keras)



       These two things can have different values, they are not directly related.



       Yes, this will help you choose models:

       [https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/](https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/)



       Weights are assigned small random numbers automatically when you call compile():

       [https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/](https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531581)




       - ![](https://secure.gravatar.com/avatar/68a9afbfe9bea326f5464ac3fc2252e8e159c5be3ae46b68713dced28d0836d0?s=40&d=mm&r=g)



         SarmadApril 26, 2020 at 7:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531819 "Direct link to this comment")





         sir still confuse that as in ML algorithm we specify which algorithm to implement wrt to scenario like for regression we can choose linear regression , logistic regression etc.


         now at this time what neural net we have chosen? convoltiona, rntn etc?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531819)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)April 27, 2020 at 5:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531906 "Direct link to this comment")





           Linear regression is for regression, logistic regression is for classification.



           Here are some regression algorithms to try on a regression task:

           [https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/](https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531906)
373. ![](https://secure.gravatar.com/avatar/68a9afbfe9bea326f5464ac3fc2252e8e159c5be3ae46b68713dced28d0836d0?s=40&d=mm&r=g)



     SarmadApril 24, 2020 at 8:31 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531527 "Direct link to this comment")





     where are the weights, bias and input values?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531527)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 25, 2020 at 6:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531583 "Direct link to this comment")





       Weights are initialized to small random values when we call compile().



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531583)
374. ![](https://secure.gravatar.com/avatar/da6742a3920498dae116c6c520da632bd028379da7765b9a966e0c6fc05b16b4?s=40&d=mm&r=g)



     mounaApril 26, 2020 at 8:51 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531826 "Direct link to this comment")





     Hello Jason,



     Congratulations fro all the good job, i want to ask you:


     How we can know of all epochs the average of training time and validation time for a model?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531826)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 27, 2020 at 5:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531908 "Direct link to this comment")





       You could extrapolate the time of one epoch to the number of epochs you want to train.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-531908)
375. ![](https://secure.gravatar.com/avatar/f4b33f5deea1d8174590d2f1bb7da7c147245a105ff3e84713d57455c7893818?s=40&d=mm&r=g)



     Jason ChiaApril 28, 2020 at 2:41 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532162 "Direct link to this comment")





     Hi Jason,


     I am very new to deep learning. I understand that you do model.fit to fit the data and model.predict to predict the values of the class variable y. However, is it also possible to extract the parameter estimate and derive f(X) = y (similar to regression)?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532162)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 29, 2020 at 6:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532248 "Direct link to this comment")





       Perhaps for small models, but it would be a mess with thousands of coefficients. The model is complex circuit.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532248)
376. ![](https://secure.gravatar.com/avatar/0031eacc444798cfdaac7124b52ff787b48595447f3da5b3a75086ad23f4360a?s=40&d=mm&r=g)



     DinaApril 28, 2020 at 4:34 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532176 "Direct link to this comment")





     Hi JAson, do you have an idea on how to predict price or range of value?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532176)




     - ![](https://secure.gravatar.com/avatar/0031eacc444798cfdaac7124b52ff787b48595447f3da5b3a75086ad23f4360a?s=40&d=mm&r=g)



       DinaApril 28, 2020 at 4:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532178 "Direct link to this comment")





       If I use keras model to predict price/range of value, it is possible for me to find the accuracy of keras model?because in your article only to predict the binary output



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532178)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)April 29, 2020 at 6:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532255 "Direct link to this comment")





         You are describing a regression problem, I recommend starting here:

         [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532255)
     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 29, 2020 at 6:17 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532253 "Direct link to this comment")





       A prediction range is called a prediction interval, learn more here:

       [https://machinelearningmastery.com/prediction-intervals-for-machine-learning/](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-532253)
377. ![](https://secure.gravatar.com/avatar/3103113313c60888018b01db5dca574e3bf9a4fb027bec78c2f31c975cc413f7?s=40&d=mm&r=g)



     HumeMay 5, 2020 at 10:54 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-533326 "Direct link to this comment")





     thank you for your explanation, i am a beginner for machine learning as well as python.woluld you please help me in getting the exact CSV data file for predicting the Hepatitis B virus.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-533326)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 5, 2020 at 1:37 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-533346 "Direct link to this comment")





       This will help you locate a dataset:

       [https://machinelearningmastery.com/faq/single-faq/where-can-i-get-a-dataset-on-\_\_\_](https://machinelearningmastery.com/faq/single-faq/where-can-i-get-a-dataset-on-___)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-533346)
378. ![](https://secure.gravatar.com/avatar/e3b3fc026790c92154a498376994d1c3f3ebc46ca3699e12ac2c5ea7f013b988?s=40&d=mm&r=g)



     [Ababou Nabil](http://aababou.com/)May 12, 2020 at 2:01 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-534410 "Direct link to this comment")





     768/768 \[==============================\] – 2s 3ms/step


     Accuracy: 76.56



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-534410)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 13, 2020 at 6:21 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-534481 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-534481)
379. ![](https://secure.gravatar.com/avatar/ae6c8ee098be9b17498c2f3e5a0418e9bea817849fc848ca2abeb977c70d5c10?s=40&d=mm&r=g)



     MAHESH MADHUSHANMay 24, 2020 at 11:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536163 "Direct link to this comment")





     Why didn’t you normalize data? Is not that necessary ? I have seen on some tutorials, they normalize data for common scale using as –>from sklearn.preprocessing import StandardScaler . What is the difference that method and your method?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536163)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 25, 2020 at 5:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536238 "Direct link to this comment")





       It can help for some algorithms to normalize or standardize the data input data. Perhaps try it and see.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536238)
380. ![](https://secure.gravatar.com/avatar/a5a960d2c41fee633d1177cffcb9072d45446da9f69eeb55a320e8da0b58bd63?s=40&d=mm&r=g)



     Henry LevkineMay 26, 2020 at 7:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536434 "Direct link to this comment")





     Jason,



     You are the best!



     My name for your program here is “helloDL.py”



     I am sure your future book “Hello Deep Learning” will be the most popular on the market.



     People need in programs



     helloClassification.py


     helloRegression.py


     helloHelloPrediction.py


     helloDogsCats.py


     helloFaces.py



     and so on!



     Thank you for your hard work!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536434)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 26, 2020 at 1:19 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536463 "Direct link to this comment")





       Thanks.



       You can find all of these on the blog, use the search.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-536463)
381. ![](https://secure.gravatar.com/avatar/bb859e5fc69b446008e0044800d6de3dd2701a65423d4bf81dcde21921ccdbe0?s=40&d=mm&r=g)



     ThijsJune 12, 2020 at 12:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539143 "Direct link to this comment")





     Hello,



     is there a possibility to access the accuracy of the last epoch? If yes, how can i access this and save it?



     Kind regards



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539143)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 12, 2020 at 6:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539176 "Direct link to this comment")





       Yes, the history object contains the scores calculated on each epoch:

       [https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539176)
382. ![](https://secure.gravatar.com/avatar/6c3799e9ce18facd7b10f5127f49f894b61aa03737d6648243f24fa97f12654d?s=40&d=mm&r=g)



     KrishanJune 16, 2020 at 11:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539673 "Direct link to this comment")





     Accuracy: 82.42


     epochs=1500


     batch\_size=1



     I don’t know if what I did was appropriate. Any advise is appreciated.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539673)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 16, 2020 at 1:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539689 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-539689)
383. ![](https://secure.gravatar.com/avatar/f19ad8efd9bb9ef7d5b004d1601aeb100e052d92a6c8073f5ad9de5219dc6865?s=40&d=mm&r=g)



     SaadJune 19, 2020 at 9:20 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-540181 "Direct link to this comment")





     Hi Jason,



     Thanks a lot for this wonderful learning platform.



     Why were 12 neurons used in the first hidden layer, what is the criteria behind it? Is it random or there is an underlying reason/calculation?



     (I presumed that the number of neurons in a hidden layer would always be between the number of inputs and the number of outputs)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-540181)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 20, 2020 at 6:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-540229 "Direct link to this comment")





       I chose the configuration after a little trial and error.



       There is no good theory for configuring neural nets:

       [https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network](https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-540229)
384. ![](https://secure.gravatar.com/avatar/9d3a12b1c7272a050a5421f2df12e7e57dd0a469df7cb8e1f5a213425366add5?s=40&d=mm&r=g)



     Paras MemonJuly 30, 2020 at 9:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-546512 "Direct link to this comment")





     Hello Jason,



     I have this shape of training and testing data sets:


     xTrain\_CN.shape, yTrain\_CN.shape, xTest\_CN.shape


     ((320, 56, 6251), (320,), (80, 56, 6251))



     I am getting this error: ValueError: Error when checking input: expected dense\_20\_input to have 2 dimensions, but got array with shape (320, 56, 6251)



     Below is the code:



     def nn\_keras(xTrain\_CN, yTrain\_CN, xTest\_CN):



     model = Sequential()


     model.add(Dense(12, input\_dim=6251, activation=’relu’))


     model.add(Dense(8, activation=’relu’))


     model.add(Dense(1, activation=’sigmoid’))


     # compile the keras model


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     # fit the keras model on the dataset


     model.fit(xTrain\_CN, yTrain\_CN, epochs=150, batch\_size=10)


     # evaluate the keras model


     \_, accuracy = model.evaluate(xTrain\_CN, yTrain\_CN)


     print(‘Training Accuracy: %.2f’ % (accuracy\*100))



     \_, accuracy = model.evaluate(xTrain\_CN, yTrain\_CN)


     print(‘Testing Accuracy: %.2f’ % (accuracy\*100))



     nn\_keras(xTrain\_CN, yTrain\_CN, xTest\_CN)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-546512)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 30, 2020 at 1:44 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-546532 "Direct link to this comment")





       A MLP must take 2d data as input (rows and columns) and 1d data as output during training.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-546532)
385. ![](https://secure.gravatar.com/avatar/c6d902d71d6ed2ef0b2910a9f938cd6a63b2e857b135656239b1062d73c2216c?s=40&d=mm&r=g)



     JoanneAugust 12, 2020 at 1:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-548749 "Direct link to this comment")





     Hi Jason,



     This is a great tutorial, very easy to understand!! Is there a tutorial for how to add weight and bias into our model?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-548749)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 12, 2020 at 6:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-548770 "Direct link to this comment")





       Thanks!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-548770)
386. ![](https://secure.gravatar.com/avatar/71077e3d39d7ebc8e19ebd2fdd0422120225f28ab819c5a95371cbed18404e22?s=40&d=mm&r=g)



     Luis CorderoAugust 20, 2020 at 12:05 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550245 "Direct link to this comment")





     Hello, if I have a prediction problem, it is absolutely necessary to scale the input variables to use the sigmoid or relu activation functions or the one you decide to use?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550245)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 20, 2020 at 1:37 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550265 "Direct link to this comment")





       No, but try it and compare results.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550265)
387. ![](https://secure.gravatar.com/avatar/ecdaf000f4508dbb947e673c409a3fdb8e3c0c19af6b1f25f63e7e46bec14982?s=40&d=mm&r=g)



     Luis CorderoAugust 20, 2020 at 1:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550255 "Direct link to this comment")





     how I can create a configuration that has more than one output, i.e. the output layer has 2 or more values



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550255)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 20, 2020 at 1:39 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550267 "Direct link to this comment")





       Yes, just specify the number of targets in the output layer and prepare your training data accordingly.



       I have a tutorial on exactly this written and scheduled – for next week I think.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-550267)




       - ![](https://secure.gravatar.com/avatar/ecdaf000f4508dbb947e673c409a3fdb8e3c0c19af6b1f25f63e7e46bec14982?s=40&d=mm&r=g)



         Luis CorderoSeptember 1, 2020 at 4:29 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-556647 "Direct link to this comment")





         what will been name of tutorial to find it



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-556647)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)September 2, 2020 at 6:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-557154 "Direct link to this comment")





           Right here:

           [https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/](https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/)



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-557154)
388. ![](https://secure.gravatar.com/avatar/d18f21083e333e398789355809c74c3d6ccf29c69f4bb15dcf889f88dbddbd7e?s=40&d=mm&r=g)



     Simon SuarezAugust 30, 2020 at 8:27 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-554556 "Direct link to this comment")





     Hi Jason.



     I thank you for the great quality of this article. I am experienced with Machine Learning using Scikit-Learn, and reading this post (and some of your previous on the topic) helped me a lot to get into making Multilayer Perceptrons.


     I tested the knowledge I learned here with the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. I got around 92.965% Accuracy for train and 96.491% for test, only using 3 features (radius, texture, smoothness) and the following topology:


     • Epochs = 250


     • Batch\_size = 60


     • Función de activación = ReLu


     • Optimizador = ‘Nadam’



     Layer; Number of neurons; Activation function


     Input; 3; None


     Hidden 1; 4; ReLu


     Hidden 2; 4; ReLu


     Hidden 3; 2; ReLu


     Output; 1; Sigmoid



     Train and test were splitted using: train\_test\_split(X, y, test\_size=0.33, random\_state=42)


     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-554556)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 31, 2020 at 5:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-555337 "Direct link to this comment")





       Thanks.



       Well done on your results Simon!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-555337)
389. ![](https://secure.gravatar.com/avatar/047054daf90ec4b61a75082be9cf89560d56bf3b298135e3e505ca0b104cd085?s=40&d=mm&r=g)



     Berns BuenaobraSeptember 7, 2020 at 7:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-560332 "Direct link to this comment")





     0s 833us/step – loss: 0.4607 – accuracy: 0.7773



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-560332)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 7, 2020 at 8:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-560372 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-560372)
390. ![](https://secure.gravatar.com/avatar/8f8c850ac3b3c7cee44a8f2023a9a6ebb287ef77cc19a0939cb4027ccb529d70?s=40&d=mm&r=g)



     Berns BuenaobraSeptember 7, 2020 at 7:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-560333 "Direct link to this comment")





     Second iteration with laptop GPU gives:


     0s 958us/step – loss: 0.4119 – accuracy: 0.8216


     Accuracy: 82.16



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-560333)

391. ![](https://secure.gravatar.com/avatar/224af12aba4823a4ac48b4610ec498c8803645885ece5f3aff9277df0d95b684?s=40&d=mm&r=g)



     Ahmed NuruSeptember 8, 2020 at 5:01 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-561060 "Direct link to this comment")





     Hi janson how can predict image forgery and genuine using pretrained deep-learning model



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-561060)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 9, 2020 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-561370 "Direct link to this comment")





       Perhaps prepare a dataset of real and fake images and train a binary classification model to differentiate the two.



       Perhaps this tutorial will help you to get started:

       [https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-561370)
392. ![](https://secure.gravatar.com/avatar/16da183e22a961b3d340832c5748cc6b0a5d2565bd237770373f796f1265f559?s=40&d=mm&r=g)



     Fatma ZohraSeptember 11, 2020 at 2:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562060 "Direct link to this comment")





     Hello Jason ,



     Can you please guide me how to make a query and a document as an input in our NN (knowing that they both are represented by frequency vectors ) ?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562060)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 11, 2020 at 6:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562130 "Direct link to this comment")





       Perhaps start here:

       [https://machinelearningmastery.com/start-here/#nlp](https://machinelearningmastery.com/start-here/#nlp)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562130)
393. ![](https://secure.gravatar.com/avatar/16da183e22a961b3d340832c5748cc6b0a5d2565bd237770373f796f1265f559?s=40&d=mm&r=g)



     fatma zohraSeptember 13, 2020 at 2:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562699 "Direct link to this comment")





     Hi Dr Jason,



     Thanks a lot for the reply , the link was useful for me ,


     yet i’am still lost a bit since i’am new dealing with NN, actualy i want to calculate the similarity between the query and the doc using the NN , the inputs are (the TF vector of the doc and TF vector of the query , and the output is the similarity (0 if no , 1 if yes ) , i have the idea of my NN but i don’t know from where to start…


     i would be gratful if you could help me (a similar code that i can take as exemple maybe ),



     Waiting for your reply..thanks in advance



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562699)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 13, 2020 at 6:10 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562743 "Direct link to this comment")





       I think you’re asking about calculating text similarity. If so, sorry I don’t have tutorials on that topic.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562743)




       - ![](https://secure.gravatar.com/avatar/16da183e22a961b3d340832c5748cc6b0a5d2565bd237770373f796f1265f559?s=40&d=mm&r=g)



         fatma zohraSeptember 13, 2020 at 6:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562748 "Direct link to this comment")





         yeah , this is what i was asking for , anyways thanks a lot for your tutorials they are very clear and fruitful..



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562748)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)September 13, 2020 at 8:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562753 "Direct link to this comment")





           You’re welcome.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-562753)
394. ![](https://secure.gravatar.com/avatar/64867aec5a12b678fbaedcf87286ef3965c53831a9e99753be37c1993483e2d5?s=40&d=mm&r=g)



     yibrah fissehaSeptember 22, 2020 at 11:41 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564367 "Direct link to this comment")





     I would like to thank you a lot for your tutorials. can you please guide me on how to evaluate the model using confusion matrix parameters such as recall, precision, f1 score?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564367)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 23, 2020 at 6:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564402 "Direct link to this comment")





       Yes, here are examples:

       [https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564402)
395. ![](https://secure.gravatar.com/avatar/c005de9a8df07a54875b1c594d07927511f9244408422e33ed21de46eb18bf5f?s=40&d=mm&r=g)



     deryaSeptember 23, 2020 at 5:03 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564382 "Direct link to this comment")





     great tutorial helped a lot !



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564382)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 23, 2020 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564408 "Direct link to this comment")





       Thanks!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564408)
396. ![](https://secure.gravatar.com/avatar/d1e2434bc23ed5b2f61b25e5bf543456ef2e80eafd5b648662eb8aff58884618?s=40&d=mm&r=g)



     Sean H. KelleySeptember 23, 2020 at 6:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564399 "Direct link to this comment")





     Hi Jason, thank you very much for this.



     I appreciate the extra in depth explanations in the links to other pages.



     I am wondering how to keep the state of mind. Like you train it while it runs and get a level of accuracy. If you finally get the level of accuracy from training a certain configuration, how do you keep that configuration/state of mind/level of accuracy of the artificial neural net without having to train it all over again?



     Can you store a snapshot of that “state of mind” somewhere so that when you have a good working model, you just use that to run new data against or am I still missing some key elements in my attempting to grasp this?



     Thank you!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564399)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)September 23, 2020 at 6:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564410 "Direct link to this comment")





       You can save your model and load it later to make predictions, see this tutorial:

       [https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564410)




       - ![](https://secure.gravatar.com/avatar/d1e2434bc23ed5b2f61b25e5bf543456ef2e80eafd5b648662eb8aff58884618?s=40&d=mm&r=g)



         Sean H. KelleySeptember 24, 2020 at 12:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564517 "Direct link to this comment")





         Thank you very much!



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564517)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)September 24, 2020 at 6:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564554 "Direct link to this comment")





           You’re welcome.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-564554)
397. ![](https://secure.gravatar.com/avatar/2835f467f71ed753167990b9005a6528f6d1c0b5bcac3c8d302c0722cd07119f?s=40&d=mm&r=g)



     Muhammad Asad ArshedOctober 10, 2020 at 12:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566357 "Direct link to this comment")





     Awesome blog and technical skill would you like to refer me to some other blogs.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566357)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 10, 2020 at 7:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566417 "Direct link to this comment")





       Thanks!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566417)
398. ![](https://secure.gravatar.com/avatar/84bbbead6a11095f4e251f51cd2518bf4a805c1941eafbac70c5b438f4dc83e8?s=40&d=mm&r=g)



     BrijeshOctober 10, 2020 at 5:57 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566473 "Direct link to this comment")





     Hi



     Can we use only CSV file format?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566473)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 11, 2020 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566538 "Direct link to this comment")





       No, deep learning can use images, text data, audio data, almost anything that can be represented with numbers.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-566538)
399. ![](https://secure.gravatar.com/avatar/66daf32f154b58182e358b134297af879f937610c76ee0170e3724b856da8ff4?s=40&d=mm&r=g)



     imeneOctober 18, 2020 at 4:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568371 "Direct link to this comment")





     with epoch =10000 and batch-size = 20 a got accuracy = 84% and loss =loss: 0.3434



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568371)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 18, 2020 at 6:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568399 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568399)

     - ![](https://secure.gravatar.com/avatar/ed527df0a07b57821c80e903aa5f33d56c233d6bdcf9d5bc88e87cafdda71ae8?s=40&d=mm&r=g)



       YAŞAR SAİD DERDİMANDecember 27, 2020 at 4:12 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589156 "Direct link to this comment")





       this is good but probably, your model’s generalization error is higher. Because more epoch means more overfitting, Therefore you should use less epoch for any deep learning training.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589156)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)December 28, 2020 at 5:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589265 "Direct link to this comment")





         Good advice.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589265)
400. ![](https://secure.gravatar.com/avatar/66daf32f154b58182e358b134297af879f937610c76ee0170e3724b856da8ff4?s=40&d=mm&r=g)



     imeneOctober 18, 2020 at 4:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568374 "Direct link to this comment")





     first thanks for your good explanation,


     how can i save the trained model to be used for test becaus the trainnig repeat each time i try to execute the program


     tanks.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568374)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 18, 2020 at 6:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568400 "Direct link to this comment")





       Good question, this will show you how:

       [https://machinelearningmastery.com/save-load-keras-deep-learning-models/](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-568400)
401. ![](https://secure.gravatar.com/avatar/97ab6baca9fb573e359a2a4501b9cb0790cfd73c76fe9f4c4b1c2d8608089207?s=40&d=mm&r=g)



     FatimaOctober 24, 2020 at 5:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-570325 "Direct link to this comment")





     Hi Jason, I applied the Deep Neural Network algorithm(DNN) to do the prediction, It works and it is perfect, I have a problem in evaluating the predicted results I used (metrics.confusion\_matrix), It gave me this error:


     ValueError: Classification metrics can’t handle a mix of binary and continuous targets



     any suggestions to solve the error?


     note: my class label (outcome variable) is binary (0,1)



     Thanks in advanced



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-570325)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 24, 2020 at 7:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-570373 "Direct link to this comment")





       See this tutorial:

       [https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-570373)
402. ![](https://secure.gravatar.com/avatar/7e1cf165a2ba42a1c5f20eaaa8cddb920c21d1b8f44c146331d53a12e71afdbc?s=40&d=mm&r=g)



     K AlOctober 27, 2020 at 2:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-570964 "Direct link to this comment")





     First of all, please allow me to thank you for this great tutorial and for your valuable time.


     I wonder: you trained and evaluated the network on the same data set. Why did not it generate a 100% accuracy then?



     Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-570964)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)October 27, 2020 at 6:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-571000 "Direct link to this comment")





       All models have error.



       If we get perfect skill/100% accuracy then the problem is likely too simple and machine learning is not required:

       [https://machinelearningmastery.com/faq/single-faq/what-does-it-mean-if-i-have-0-error-or-100-accuracy](https://machinelearningmastery.com/faq/single-faq/what-does-it-mean-if-i-have-0-error-or-100-accuracy)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-571000)
403. ![](https://secure.gravatar.com/avatar/d900846d22d2ed735752de27f0fbbdd0b8a986ac6e66949ae1a274e4c8b0802d?s=40&d=mm&r=g)



     ZuzanaNovember 1, 2020 at 11:15 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572278 "Direct link to this comment")





     Hi, great tutorial, everything works, except when trying to add predictions, I get the following error message. Could you please, help? Thanks a lot.



     WARNING:tensorflow:From C:/Users/ZuzanaŠútová/Desktop/RTP new/3\_training\_deep\_learning/data\_PDS/keras\_first\_network\_including\_predictions.py:27: Sequential.predict\_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.


     Instructions for updating:


     Please use instead:\* `np.argmax(model.predict(x), axis=-1)`, if your model does multi-class classification (e.g. if it uses a `softmax` last-layer activation).\* `(model.predict(x) > 0.5).astype("int32")`, if your model does binary classification (e.g. if it uses a `sigmoid` last-layer activation).



     Warning (from warnings module):


     File “C:\\Users\\ZuzanaŠútová\\AppData\\Roaming\\Python\\Python38\\site-packages\\tensorflow\\python\\keras\\engine\\sequential.py”, line 457


     return (proba > 0.5).astype(‘int32’)


     RuntimeWarning: invalid value encountered in greater


     Traceback (most recent call last):


     File “C:\\Users\\ZuzanaŠútová\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\pandas\\core\\indexes\\base.py”, line 2895, in get\_loc


     return self.\_engine.get\_loc(casted\_key)


     File “pandas\\\_libs\\index.pyx”, line 70, in pandas.\_libs.index.IndexEngine.get\_loc


     File “pandas\\\_libs\\index.pyx”, line 101, in pandas.\_libs.index.IndexEngine.get\_loc


     File “pandas\\\_libs\\hashtable\_class\_helper.pxi”, line 1032, in pandas.\_libs.hashtable.Int64HashTable.get\_item


     File “pandas\\\_libs\\hashtable\_class\_helper.pxi”, line 1039, in pandas.\_libs.hashtable.Int64HashTable.get\_item


     KeyError: 0



     The above exception was the direct cause of the following exception:



     Traceback (most recent call last):


     File “C:/Users/ZuzanaŠútová/Desktop/RTP new/3\_training\_deep\_learning/data\_PDS/keras\_first\_network\_including\_predictions.py”, line 30, in


     print(‘%s => %d (expected %d)’ % (X\[i\].tolist(), predictions\[i\], y\[i\]))


     File “C:\\Users\\ZuzanaŠútová\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\pandas\\core\\frame.py”, line 2902, in \_\_getitem\_\_


     indexer = self.columns.get\_loc(key)


     File “C:\\Users\\ZuzanaŠútová\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\pandas\\core\\indexes\\base.py”, line 2897, in get\_loc


     raise KeyError(key) from err


     KeyError: 0



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572278)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 2, 2020 at 6:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572331 "Direct link to this comment")





       Sorry to hear that, this may help:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572331)
404. ![](https://secure.gravatar.com/avatar/d900846d22d2ed735752de27f0fbbdd0b8a986ac6e66949ae1a274e4c8b0802d?s=40&d=mm&r=g)



     ZuzanaNovember 2, 2020 at 6:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572339 "Direct link to this comment")





     I am sorry but none of that helped :/



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572339)

405. ![](https://secure.gravatar.com/avatar/b8001adc1d30b6ef7380a9e2b53a47e2a08d4058bcce58f4a956951f5eb233a9?s=40&d=mm&r=g)



     Julian A EppsNovember 3, 2020 at 7:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572565 "Direct link to this comment")





     Where can I find documentation on these keras functions that you are using. I don’t know how any of these functions work.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572565)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 3, 2020 at 10:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572574 "Direct link to this comment")





       Good question, here:

       [https://keras.io/api/](https://keras.io/api/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-572574)
406. ![](https://secure.gravatar.com/avatar/ef2db15ba9b59f13ac4ebde2275dc27290e3f5e9e6a76123ea427d71045f83c2?s=40&d=mm&r=g)



     Umair RasoolNovember 8, 2020 at 4:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-573561 "Direct link to this comment")





     Hello Sir, i am not actually familiar with ML so someone doing my task for prediction using raster dataset with python. He just giving final results and CSV file rather than final prediction map as raster, Could you please guide me ML works like this or he is missing something to generate final map. Please Response. Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-573561)




     - ![](https://secure.gravatar.com/avatar/ef2db15ba9b59f13ac4ebde2275dc27290e3f5e9e6a76123ea427d71045f83c2?s=40&d=mm&r=g)



       Umair RasoolNovember 8, 2020 at 4:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-573562 "Direct link to this comment")





       sorry i have a little mistake “final result as CSV file”



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-573562)

     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 8, 2020 at 6:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-573589 "Direct link to this comment")





       Perhaps this framework will help:

       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-573589)
407. ![](https://secure.gravatar.com/avatar/3904433020d62d281e071b020f8cb2d1b4bfd8a8d96e8be731cdb6b5219e5bf5?s=40&d=mm&r=g)



     HalilNovember 27, 2020 at 6:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-578930 "Direct link to this comment")





     Thank you for this brilliantly explained tutorial ! Actually, I am bored of watching videos which have lots of boring talks and superficial explanations. I discovered my main resource now



     By the way, I guess there is an error here. No?


     rounded = \[round(x\[0\]) for x in predictions\] —> should be “round(X…..”



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-578930)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 27, 2020 at 6:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-578949 "Direct link to this comment")





       You’re welcome.



       There are many ways to round an array.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-578949)




       - ![](https://secure.gravatar.com/avatar/3904433020d62d281e071b020f8cb2d1b4bfd8a8d96e8be731cdb6b5219e5bf5?s=40&d=mm&r=g)



         HalilNovember 30, 2020 at 5:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579596 "Direct link to this comment")





         I mean, that “x” should be “X”. No?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579596)
408. ![](https://secure.gravatar.com/avatar/797b84670d2b63f3f5e9c18f2ad0c3ae8381d62bed4360c96319b105ef0533fe?s=40&d=mm&r=g)



     RAJSHREE SRIVASTAVANovember 28, 2020 at 4:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579161 "Direct link to this comment")





     Hi jason,



     Hope you are doing well. I am working on ANN for image classification in google colab. I am getting this error , can you help me to find solution for this?



     InvalidArgumentError: Incompatible shapes: \[100,240,240,1\] vs. \[100,1\]


     \[\[node gradient\_tape/mean\_squared\_error/BroadcastGradientArgs (defined at :14) \]\] \[Op:\_\_inference\_train\_function\_11972\]



     Function call stack:


     train\_function



     Waitting for your reply.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579161)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 28, 2020 at 6:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579185 "Direct link to this comment")





       Sorry, I don’t know about colab:

       [https://machinelearningmastery.com/faq/single-faq/do-code-examples-run-on-google-colab](https://machinelearningmastery.com/faq/single-faq/do-code-examples-run-on-google-colab)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579185)
409. ![](https://secure.gravatar.com/avatar/797b84670d2b63f3f5e9c18f2ad0c3ae8381d62bed4360c96319b105ef0533fe?s=40&d=mm&r=g)



     RAJSHREE SRIVASTAVANovember 28, 2020 at 8:14 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579328 "Direct link to this comment")





     Hi jason thanks for your reply.



     ok in python I am working on ANN for image classification . I am getting this error , can you help me to find solution for this?



     InvalidArgumentError: Incompatible shapes: \[100,240,240,1\] vs. \[100,1\]


     \[\[node gradient\_tape/mean\_squared\_error/BroadcastGradientArgs (defined at :14) \]\] \[Op:\_\_inference\_train\_function\_11972\]



     Function call stack:


     train\_function



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579328)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)November 29, 2020 at 8:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579418 "Direct link to this comment")





       Sorry, the cause of the error is not clear, you may need to debug your model.



       Here are some suggestions:

       [https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code](https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-579418)
410. ![](https://secure.gravatar.com/avatar/408a0d98ec3a3a99380c006be560c3addb1123f26f363e619b92e6e09c4aff07?s=40&d=mm&r=g)



     HanemDecember 17, 2020 at 11:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-585691 "Direct link to this comment")





     Thanks a million, it helped me a lot. Actually, all of your articles are informative and goog guide for me.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-585691)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)December 17, 2020 at 12:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-585721 "Direct link to this comment")





       You’re welcome, I’m happy to hear that!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-585721)
411. ![](https://secure.gravatar.com/avatar/4e42cd26ed7ccceeac7d890fc2d8c325b7c153d5bf07ebbbf8a6644886453521?s=40&d=mm&r=g)



     John SmithDecember 28, 2020 at 7:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589294 "Direct link to this comment")





     This was a brilliant tutorial I think what could be done to improve this is adding an example of actual predictions.



     The prediction bit is quite brief I don’t quite have an understanding how to use that array of “predictions” to actually predict something.



     Like if I wanted to feed it some test data and get a prediction how could I do that?



     I will consult some of your other helpful guides but would be great to have it all in this 1 tutorial.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589294)




     - ![](https://secure.gravatar.com/avatar/4e42cd26ed7ccceeac7d890fc2d8c325b7c153d5bf07ebbbf8a6644886453521?s=40&d=mm&r=g)



       John SmithDecember 28, 2020 at 8:07 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589295 "Direct link to this comment")





       I did not have my coffee when I wrote this.



       I see now we are passing the original variables back into the model and predicting and printing out the predication vs actual.



       🙂



       Thanks – you made a great tutorial!



       Have a good christmas and new year.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589295)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)December 28, 2020 at 8:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589301 "Direct link to this comment")





         No problem at all!



         I’m happy it helped you kick start your journey with deep learning.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-589301)
412. ![](https://secure.gravatar.com/avatar/75918557e4bca8085ba668ac96b0d4137b8276d672163d1398f92906807fa3ae?s=40&d=mm&r=g)



     JoeJanuary 3, 2021 at 5:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-590889 "Direct link to this comment")





     Hi Jason,



     Happy new year!



     You are predicting on the same data set, X, that you used to train the model.



     I would have thought that the model would’ve produced close to 100% accuracy in this case since the model is so well trained specifically with respect to X (maybe even overfitted).



     Why are we only getting 76.9% accuracy, not close to 100%?



     Thanks


     Joe



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-590889)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 3, 2021 at 6:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-590916 "Direct link to this comment")





       Yes, I that to keep the example simple, I explain more here:

       [https://machinelearningmastery.com/faq/single-faq/why-do-you-use-the-test-dataset-as-the-validation-dataset](https://machinelearningmastery.com/faq/single-faq/why-do-you-use-the-test-dataset-as-the-validation-dataset)



       No model is perfect, they are all trying to generalize from the training data.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-590916)
413. ![](https://secure.gravatar.com/avatar/b9cf4c2005040c1126c28c80a70059c2b73ca869b3d742b99378092638b34f4c?s=40&d=mm&r=g)



     [Roberto Aguirre Maturana](https://poppercorner.wordpress.com/)January 7, 2021 at 12:19 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592112 "Direct link to this comment")





     Excelent tutorial, well explained and very easy to follow. It seems you have to update one line that was deprecated in 2021:



     #instead of


     #predictions = model.predict(X)



     #now you have to use


     predictions = (model.predict(X) > 0.5).astype(“int32”)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592112)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 7, 2021 at 2:04 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592124 "Direct link to this comment")





       Thanks.



       I don’t think so:

       [https://keras.io/api/models/model\_training\_apis/#predict-method](https://keras.io/api/models/model_training_apis/#predict-method)



       And:

       [https://www.tensorflow.org/api\_docs/python/tf/keras/Sequential#predict](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#predict)



       If you want labels you can use model.predict\_classes(), this will help:

       [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592124)
414. ![](https://secure.gravatar.com/avatar/7247941aa3a56d380ae83bd4a06de2d95123a2e3e05021182073251cec519368?s=40&d=mm&r=g)



     Girish AhireJanuary 8, 2021 at 8:27 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592328 "Direct link to this comment")





     I got 65%



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592328)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 9, 2021 at 6:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592384 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-592384)
415. ![](https://secure.gravatar.com/avatar/e256946bc59cbc593cf1af59d3d9dbeb7321e8e4bc5f6057aef2582639c3a6a2?s=40&d=mm&r=g)



     Tom RauchJanuary 15, 2021 at 6:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593189 "Direct link to this comment")





     Hi, I have these installed in my VirtualEnv (along with other libraries)



     Keras==2.4.3


     Keras-Preprocessing==1.1.2



     But when I run this:



     \# first neural network with keras tutorial


     from numpy import loadtxt


     from keras.models import Sequential


     from keras.layers import Dense



     I get a ‘Dead Kernel’ error message in jupyter; the first line runs fine but the ‘dead kernel’ message appears when it gets to keras.



     Any idea on how to fix?



     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593189)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 15, 2021 at 8:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593205 "Direct link to this comment")





       I recommend not using a notebook as they cause problems for almost everyone:

       [https://machinelearningmastery.com/faq/single-faq/why-dont-use-or-recommend-notebooks](https://machinelearningmastery.com/faq/single-faq/why-dont-use-or-recommend-notebooks)



       Instead save the code using a simple text editor like sublime or atom and run the script from the command line:

       [https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line](https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593205)
416. ![](https://secure.gravatar.com/avatar/e256946bc59cbc593cf1af59d3d9dbeb7321e8e4bc5f6057aef2582639c3a6a2?s=40&d=mm&r=g)



     Tom RauchJanuary 15, 2021 at 9:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593208 "Direct link to this comment")





     Thank you Jason! I will give the command line a try.



     Tom



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593208)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 15, 2021 at 11:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593214 "Direct link to this comment")





       You’re welcome.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593214)
417. ![](https://secure.gravatar.com/avatar/e256946bc59cbc593cf1af59d3d9dbeb7321e8e4bc5f6057aef2582639c3a6a2?s=40&d=mm&r=g)



     Tom RauchJanuary 15, 2021 at 12:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593216 "Direct link to this comment")





     Hi Jason, I followed your instructions but still running into issues with Keras, maybe I did not install it correctly?



     (rec\_engine) tom@machine:~/code$ python keras.py


     Traceback (most recent call last):


     File “keras.py”, line 3, in


     from keras.models import Sequential


     File “/home/tom/code/keras.py”, line 3, in


     from keras.models import Sequential


     ModuleNotFoundError: No module named ‘keras.models’; ‘keras’ is not a package



     but when I run this, I do see it installed



     (rec\_engine) tom@machine:~/code$ pip list \| grep Keras


     Keras 2.4.3


     Keras-Preprocessing 1.1.2



     I followed the pip install found in this guide:



     [https://www.liquidweb.com/kb/how-to-install-keras/](https://www.liquidweb.com/kb/how-to-install-keras/)



     I think my next step may be to create a new VirtualEnv for just Keras and TensorFlow.



     Thanks, Tom



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593216)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 15, 2021 at 1:26 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593222 "Direct link to this comment")





       I think there may be an issue with your environment, perhaps this tutorial will help:

       [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593222)
418. ![](https://secure.gravatar.com/avatar/308027ea680ccf82462124599f59473ff83718ed26cc6319865f045f36229444?s=40&d=mm&r=g)



     Govind KelkarJanuary 15, 2021 at 10:58 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593260 "Direct link to this comment")





     Hi Dr. Jason,



     I executed your code in google colab and got it executing only change I found is while predicting the new data


     you had listed the sequence as 10101 and I got it as 01010


     Also did the few changes to the code.


     Nonetheless I got the code working at least. Now I will try and play with it to get more accuracies.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593260)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 16, 2021 at 6:55 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593290 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593290)
419. ![](https://secure.gravatar.com/avatar/e256946bc59cbc593cf1af59d3d9dbeb7321e8e4bc5f6057aef2582639c3a6a2?s=40&d=mm&r=g)



     Tom RauchJanuary 16, 2021 at 9:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593306 "Direct link to this comment")





     Hi Jason, I created a new virtual env and loaded Keras, TensorFlow etc and created a .py with all of your code, then ran it at the command line in the directory that contains both the csv and py.



     But, I got this error:



     (ML) tom@machine:~/code$ python mykerasloader.py


     Illegal instruction (core dumped)



     Is there a logger I should be using to see more detail?



     Thanks, Tom



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593306)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 16, 2021 at 1:20 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593320 "Direct link to this comment")





       That does not look good, I suspect there is something up with your environment.



       Perhaps you can try posting/searching on stackoverflow.com



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593320)
420. ![](https://secure.gravatar.com/avatar/14d3d0b45a45654c70ac49177499f8fad4115c5d71148dc0bfbbec8f4ca354ab?s=40&d=mm&r=g)



     Francisco SantiagoJanuary 17, 2021 at 9:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593381 "Direct link to this comment")





     Creating neural network


     24/24 \[==============================\] – 0s 756us/step – loss: 0.3391 – accuracy: 0.8503


     Accuracy: 85.03



     Wo hooo!!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593381)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 17, 2021 at 1:27 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593386 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593386)
421. ![](https://secure.gravatar.com/avatar/e2aa9db4db34b5426f0fddff7d5d7f0e2e3568207bcb6365af74016faeff0b6d?s=40&d=mm&r=g)



     JeremyJanuary 17, 2021 at 4:38 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593399 "Direct link to this comment")





     Dr. Brownlee,



     Good morning, sir! Curious for your thoughts on something: is there value in running the algorithm, say, fifty times and averaging the accuracy? I’ve used that technique before to good effect, but since this is relatively new to me, having an experienced teacher of machines set me straight would be helpful.



     If this is something you think is useful, I have one more question that comes from my still limited understanding of things: where would I start the ‘for’ loop? My first thought was starting it before ‘model = Sequential()’, but that would mean redefining the NN structure each time, which doesn’t make much sense. Second thought was starting it before ‘model.fit()’, in which case the model stays the same, and loss/optimization functions stay the same.



     Thank you very much for your time!



     V/r,


     Jeremy



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593399)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 18, 2021 at 6:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593468 "Direct link to this comment")





       Yes, it reduces the variance in the method and can be used for both evaluating model performance and making predictions.



       More details are here:

       [https://machinelearningmastery.com/faq/single-faq/why-do-i-get-different-results-each-time-i-run-the-code](https://machinelearningmastery.com/faq/single-faq/why-do-i-get-different-results-each-time-i-run-the-code)



       The loop is around the definition, training and evaluation of the model.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593468)
422. ![](https://secure.gravatar.com/avatar/e256946bc59cbc593cf1af59d3d9dbeb7321e8e4bc5f6057aef2582639c3a6a2?s=40&d=mm&r=g)



     Tom RauchJanuary 18, 2021 at 6:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593486 "Direct link to this comment")





     Hi Jason, any tuts on using your code in this posting in Google colabs? Not sure how to point to the csv using colabs.



     Thanks, Tom



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593486)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 18, 2021 at 8:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593493 "Direct link to this comment")





       This is a common question that I answer here:

       [https://machinelearningmastery.com/faq/single-faq/do-code-examples-run-on-google-colab](https://machinelearningmastery.com/faq/single-faq/do-code-examples-run-on-google-colab)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593493)
423. ![](https://secure.gravatar.com/avatar/870b31c384d626c1070880b9b0f5862f8ab9e2c5539f63e035c3205ec0c10e41?s=40&d=mm&r=g)



     AnnaJanuary 21, 2021 at 8:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593863 "Direct link to this comment")





     Hello Jason I have a question.



     I want to create a model to predict the urban development. I started with your model above.


     I use the information about the urban and the non-urban points for 4 years (2000,2006,2012,2018). I also use information about the slope and some distances for every point.


     I have create a dataset witch contains information in the columns like this.


     2000-2006


     2006-2012



     After the train I have accuracy 94%


     But when I give to the model the year 2006 it doesn’t predict the 2012 very well. There many problems.


     I thought that with this accuracy the model would have predict the 2012 very well.



     I don’t where it might be the problem… At the train section, at the predict or somewhere else??


     Please tell your opinion because I am stuck in this for weeks and I have to find the solution quickly!!!!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593863)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 22, 2021 at 7:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593937 "Direct link to this comment")





       It sounds like your working with a time series dataset.



       If so, it would not be valid to train the model on the future and predict the past.



       I recommend starting here:

       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-593937)
424. ![](https://secure.gravatar.com/avatar/c48bd4c3c116175db9fafe38428787a609e3ab9ac6983c6e5ba45c4c9bbb5e54?s=40&d=mm&r=g)



     James ParkerJanuary 22, 2021 at 8:43 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594018 "Direct link to this comment")





     Thank you for this great article but I have a question what does \_, before accuracy stands for


     I searched it on the internet but couldn’t find it



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594018)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 23, 2021 at 7:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594073 "Direct link to this comment")





       We use underscore (\_) in python to eat up return values or variables we don’t care about. In this case the loss, as we only care about accuracy.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594073)
425. ![](https://secure.gravatar.com/avatar/be68a31f7522e0fc221ad4855194a433d4ee4b68e69a5795706f2398c38293fa?s=40&d=mm&r=g)



     FOGANG FOKOAJanuary 24, 2021 at 12:43 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594203 "Direct link to this comment")





     Hello,



     Input an array of (50385, ) where each is an array of (x, 127) into MLP)



     I want to input a numpy 2d array into MLP but I have an array of 50395 rows that contains many 2d array of shape `(x, 129)`. ``x`` because some matrices have different row numbers. Here is an example :



     train\[‘spec’\].shape


     >>(50395,)


     train\[‘spec’\]\[0\].shape


     >>(41, 129)


     train\[‘spec’\]\[5\].shape


     >>(71, 129)



     Here an snippet of my code :



     X\_train = train\[‘spec’\].values; X\_valid = valid\[‘spec’\].values


     y\_train = train\[‘label’\].values; y\_valid = valid\[‘label’\].values


     model.add(Dense(12, input\_shape=(50395, ), activation=’relu’));


     model.fit(X\_train, y\_train, validation\_data=(X\_valid, y\_valid), epochs=500, batch\_size=1);



     I get this error on last line (`model.fit`) :

     ``ValueError: Error when checking input: expected dense\_54\_input to have shape (50395,) but got array with shape (1,)``



     How to fix this problem so that the network can take as input all `50395` matrices of shape ``(x, 129)``?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594203)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 24, 2021 at 12:52 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594205 "Direct link to this comment")





       Perhaps they are “time steps” and if so this may help:

       [https://machinelearningmastery.com/faq/single-faq/what-is-the-difference-between-samples-timesteps-and-features-for-lstm-input](https://machinelearningmastery.com/faq/single-faq/what-is-the-difference-between-samples-timesteps-and-features-for-lstm-input)



       And then pad all sequences to the same length:

       [https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/](https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594205)




       - ![](https://secure.gravatar.com/avatar/be68a31f7522e0fc221ad4855194a433d4ee4b68e69a5795706f2398c38293fa?s=40&d=mm&r=g)



         FOGANG FOKOAJanuary 24, 2021 at 1:40 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594209 "Direct link to this comment")





         In fact I absolutely must use an MLP. I had sounds of 1s of frequency16000hz. As a result, all of my audio gave me an array of 16000. After removing the silence in those audios, I ended up with arrays of different sizes.



         Then I transformed these audio into a numpy matrix of numbers using the spectrograme algorithm to input them to the neural network.



         I ended up with matrices of 2 dimensions of the same columns but of different rows.



         is it possible to pass them in knowing that the matrix have different sizes?



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594209)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)January 25, 2021 at 5:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594272 "Direct link to this comment")





           As a first step, perhaps try padding all inputs to the same size and use a masking input layer followed by dense/mlp architecture.



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594272)
426. ![](https://secure.gravatar.com/avatar/be68a31f7522e0fc221ad4855194a433d4ee4b68e69a5795706f2398c38293fa?s=40&d=mm&r=g)



     FOGANG FOKOAJanuary 28, 2021 at 12:56 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594785 "Direct link to this comment")





     I did as you advised me. And I passed this difficulty there! Now my code looks like this



     `` model = Sequential();



     model.add(Dense(units =8, input\_shape=(71, 129), activation=’relu’));


     model.add(Dense(units=8, activation=’relu’));


     model.add(Dense(units=11, activation=’sigmoid’));



     \# Compile model


     model.compile(loss=’categorical\_crossentropy’, optimizer=’sgd’, metrics=\[‘accuracy’\]);


     #model = mpl\_model();


     X\_train = list(train\_df\[‘spec’\]); X\_valid = list(valid\_df\[‘spec’\]);


     y\_train = train\_df\[‘label’\]; y\_valid = valid\_df\[‘label’\];



     #labels = \[‘yes’, ‘no’, ‘up’, ‘down’, ‘left’,’right’, ‘on’, ‘off’, ‘stop’, ‘go’\];


     encoder = LabelEncoder();


     encoder.fit(y\_train);


     encoded\_y\_train = encoder.transform(y\_train);



     dummy\_y\_train = to\_categorical(encoded\_y\_train);



     \# Fit model , validation\_data=(np.array(X\_valid), y\_valid)


     model.fit(np.array(X\_train), np.array(list(dummy\_y\_train)), epochs=50, batch\_size=50); ``



     and I get this error :



     `` ValueError: A target array with shape (50395, 11) was passed for an output of shape (None, 71, 11) while using as loss `categorical_crossentropy`. This loss expects targets to have the same shape as the output. ``



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594785)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 28, 2021 at 6:01 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594835 "Direct link to this comment")





       Ouch, looks like the shape of the data does not match the expectations of the model.



       Perhaps focus on the prepared data and inspect it after each change – get that right, then focus on the modeling part.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-594835)




       - ![](https://secure.gravatar.com/avatar/be68a31f7522e0fc221ad4855194a433d4ee4b68e69a5795706f2398c38293fa?s=40&d=mm&r=g)



         FOGANG FOKOAJanuary 29, 2021 at 7:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595126 "Direct link to this comment")





         Okay. It’s done and It works well.. thank you



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595126)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)January 29, 2021 at 7:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595130 "Direct link to this comment")





           Nice work!



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595130)
427. ![](https://secure.gravatar.com/avatar/dfbc89b35616d05d75a9c4f69b28e57bd34d335ffef40c011821285ebe0c4e34?s=40&d=mm&r=g)



     Kinson VERNETJanuary 29, 2021 at 1:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595034 "Direct link to this comment")





     Hello, thank you for this tutorial.



     For 100 times I got score = 76.82 for the accuracy.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595034)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 29, 2021 at 6:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595106 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595106)
428. ![](https://secure.gravatar.com/avatar/0ff5a23c63b1119a5b281bc5b343ee79979b27788d6acf3374ff2656df45d7cf?s=40&d=mm&r=g)



     KamalJanuary 30, 2021 at 12:14 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595498 "Direct link to this comment")





     It’s a superb tutorial to implement your first deep neural network in Python. Thank you, dear Jason Brownlee.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595498)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)January 30, 2021 at 12:35 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595504 "Direct link to this comment")





       Thanks, well done on your progress!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-595504)
429. ![](https://secure.gravatar.com/avatar/c0cc633926a5a5eff5fc6e3f4968b84da32de4241bbd9d34f2d96badfd04d966?s=40&d=mm&r=g)



     RobFebruary 18, 2021 at 1:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-597912 "Direct link to this comment")





     Hi there,


     I’m currently stuck on fitting the model. Only thing I have done differently is use read\_csv so I didn’t have to put anything locally. But I’ve validated the X/y outputs to be the same.



     My error is:



     ValueError: logits and labels must have the same shape ((None, 11) vs (None, 1))



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-597912)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 19, 2021 at 5:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-597989 "Direct link to this comment")





       It suggests your data was not loaded correctly, perhaps this will help:

       [https://machinelearningmastery.com/load-machine-learning-data-python/](https://machinelearningmastery.com/load-machine-learning-data-python/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-597989)




       - ![](https://secure.gravatar.com/avatar/c0cc633926a5a5eff5fc6e3f4968b84da32de4241bbd9d34f2d96badfd04d966?s=40&d=mm&r=g)



         RobMarch 1, 2021 at 11:59 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599410 "Direct link to this comment")





         Ah thanks, it turns out it was an issue with the wrong number of nodes on the sigmoid layer.



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599410)




         - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



           [Jason Brownlee](https://machinelearningmastery.com/)March 2, 2021 at 5:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599460 "Direct link to this comment")





           Happy to hear you solved your problem!



           [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599460)
430. ![](https://secure.gravatar.com/avatar/64546aaf70ab9fb6e7df391d95c987ec7cce81306ed8bc3b138142174e6c1cd6?s=40&d=mm&r=g)



     SofiaFebruary 24, 2021 at 3:53 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-598580 "Direct link to this comment")





     Another great tutorial!!



     When I run the program it crashes with an error as seen below:



     2021-02-23 18:50:50.497125: W tensorflow/stream\_executor/platform/default/dso\_loader.cc:59\] Could not load dynamic library ‘cudart64\_101.dll’; dlerror: cudart64\_101.dll not found


     2021-02-23 18:50:50.498601: I tensorflow/stream\_executor/cuda/cudart\_stub.cc:29\] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


     Traceback (most recent call last):


     File “C:/Users/USER/PycharmProjects/Sofia/main.py”, line 26, in


     X = dataset\[:,0:8\]


     File “C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\pandas\\core\\frame.py”, line 3024, in \_\_getitem\_\_


     indexer = self.columns.get\_loc(key)


     File “C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\pandas\\core\\indexes\\base.py”, line 3080, in get\_loc


     return self.\_engine.get\_loc(casted\_key)


     File “pandas\\\_libs\\index.pyx”, line 70, in pandas.\_libs.index.IndexEngine.get\_loc


     File “pandas\\\_libs\\index.pyx”, line 75, in pandas.\_libs.index.IndexEngine.get\_loc


     TypeError: ‘(slice(None, None, None), slice(0, 8, None))’ is an invalid key



     How would I go about fixing this error? Thank you in advance!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-598580)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 24, 2021 at 5:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-598606 "Direct link to this comment")





       Thanks!



       Sorry to hear that, perhaps these tips will help:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-598606)
431. ![](https://secure.gravatar.com/avatar/1e4c7d052ade946d9ceb7b0370a1f8b3afe2e389a24032ee2795b7d88b5bdff7?s=40&d=mm&r=g)



     SlavaFebruary 27, 2021 at 3:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599027 "Direct link to this comment")





     It looks like the `model.predict_classes()` was deprecated on 2021-01-01.


     Cheers,


     Slava



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599027)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)February 27, 2021 at 6:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599072 "Direct link to this comment")





       Thanks.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599072)

     - ![](https://secure.gravatar.com/avatar/014f743b89f5064038bd40c34260657355ec0730dd96bc801746a65373191244?s=40&d=mm&r=g)



       Atsushi IsobeMarch 3, 2021 at 11:26 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599795 "Direct link to this comment")





       What is the new method to use? I can not run the predict method after finishing the training.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599795)




       - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



         [Jason Brownlee](https://machinelearningmastery.com/)March 4, 2021 at 5:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599830 "Direct link to this comment")





         This will help:

         [https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-599830)
432. ![](https://secure.gravatar.com/avatar/b7fdfade1b14ed96050ed392a1a029e50f7c522d6a00cd287e2ea28d825a55a4?s=40&d=mm&r=g)



     MitchellMarch 11, 2021 at 8:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600494 "Direct link to this comment")





     Jason, I have a couple of questions regarding the layers and how they choose filters.



     model = Sequential()


     model.add(Dense(12, input\_dim=8, activation=’relu’))


     model.add(Dense(8, activation=’relu’))


     model.add(Dense(1, activation=’sigmoid’)



     1)What is the filter size for each layer above ? 3×3 or 7×7.


     2) Are there any pre-defined 3×3 filters, 7×7 filers,?


     3) In hidden layers, filters are used to produce next layer usually. How does the model choose filters? For example, if a layer has 16 nodes, and how would I choose 32 filters so that the next layer will have 32 nodes (neurons) ?



     When you create a model, do you need to specify filters for each layer needed? like size of a filter and how many filters. .



     Thanks!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600494)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 11, 2021 at 1:25 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600537 "Direct link to this comment")





       There are no filters in a Dense layer, filters is something to do with convolutional layers:

       [https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600537)
433. ![](https://secure.gravatar.com/avatar/ab18ae4e7054bf9af2837d89703c9781414505cc0d056d4f166de27545692a9b?s=40&d=mm&r=g)



     marineboyMarch 12, 2021 at 8:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600670 "Direct link to this comment")





     hello Jason


     i have a problem ! can u have me :



     when I predict\_classes(Z) #Z=\[100,100,100,100,100,100,100,100\] as you see this data so difference but output still 0 or 1. i want output = don’t know label :((((( how can i make it pls have me



     thanks you so much, sir



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600670)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 13, 2021 at 5:29 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600728 "Direct link to this comment")





       Sorry, I don’t understand.



       Perhaps you can rephrase the problem you’re having?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-600728)
434. ![](https://secure.gravatar.com/avatar/e17ae0954dd5fbc0c60bf3fd07b04cdc78e6d2e2d3b795935d5260e6c5334191?s=40&d=mm&r=g)



     FranklinMarch 17, 2021 at 3:00 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601188 "Direct link to this comment")





     It’s an awesome blog. Keep the good work.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601188)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 18, 2021 at 5:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601250 "Direct link to this comment")





       Thanks!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601250)
435. ![](https://secure.gravatar.com/avatar/9d64c583a97e7b1eae866122b730d29be4da6daec342433c178901eaa96b262e?s=40&d=mm&r=g)



     HamzaMarch 19, 2021 at 12:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601331 "Direct link to this comment")





     79.53 accuracy



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601331)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 19, 2021 at 6:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601372 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601372)
436. ![](https://secure.gravatar.com/avatar/ba8a7694540287270650c9e8c5e6a4020a522fc48e2f77ce7847869e7564b16b?s=40&d=mm&r=g)



     Oriyomi RaheemMarch 20, 2021 at 6:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601509 "Direct link to this comment")





     I am trying to train a permeability data in las file and predict them afterwards. Please help



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601509)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)March 21, 2021 at 6:00 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601569 "Direct link to this comment")





       Perhaps this process will help you to work through your project:

       [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-601569)
437. ![](https://secure.gravatar.com/avatar/36b2b85f3023378e2e718ebb2f0303f458fab45e0d74ccfb18fe5056a0dd0b99?s=40&d=mm&r=g)



     Bangash 李忠勇March 31, 2021 at 6:41 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-602835 "Direct link to this comment")





     accuracy: 0.7865


     Accuracy: 78.65



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-602835)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 1, 2021 at 8:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-602933 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-602933)
438. ![](https://secure.gravatar.com/avatar/1e1e1bd909aee9cfe54666b9cd952f03ed425229628dc822891c662294c659b8?s=40&d=mm&r=g)



     PankajApril 23, 2021 at 7:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-606642 "Direct link to this comment")





     With categorical features, how would I prevent a Keras model from making a prediction on test samples that it has not seen in the training set, and instead either use another model or throw an exception?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-606642)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 24, 2021 at 5:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-606806 "Direct link to this comment")





       Sorry, I don’t understand. Perhaps you can elaborate?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-606806)
439. ![](https://secure.gravatar.com/avatar/0579c916b7ca8c26d6b8d79d5ec40c8998a15f592dc7085e61dd60b09453cde7?s=40&d=mm&r=g)



     LucaApril 26, 2021 at 8:31 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-607081 "Direct link to this comment")





     All the content you create and offer is absolutely amazing.


     Very informative, very up-to-date and cristal-clear.



     THANK YOU!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-607081)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)April 27, 2021 at 5:16 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-607119 "Direct link to this comment")





       You’re welcome.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-607119)
440. ![](https://secure.gravatar.com/avatar/7c670abfc606d6fdd22dc38e0afb5c90099a8c8ea2cc0c326ac4103f7713bcdc?s=40&d=mm&r=g)



     Ronald SsebaddukaMay 5, 2021 at 4:53 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-608109 "Direct link to this comment")





     File “/Users/ronaldssebadduka/PycharmProjects/pythonProject1/venv/lib/python3.9/site-packages/numpy/lib/npyio.py”, line 1067, in read\_data


     items = \[conv(val) for (conv, val) in zip(converters, vals)\]


     File “/Users/ronaldssebadduka/PycharmProjects/pythonProject1/venv/lib/python3.9/site-packages/numpy/lib/npyio.py”, line 1067, in


     items = \[conv(val) for (conv, val) in zip(converters, vals)\]


     File “/Users/ronaldssebadduka/PycharmProjects/pythonProject1/venv/lib/python3.9/site-packages/numpy/lib/npyio.py”, line 763, in floatconv


     return float(x)


     ValueError: could not convert string to float: ‘\\ufeff”6’



     I ˆget this error when i run your code!


     How can I fix it?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-608109)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 6, 2021 at 5:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-608169 "Direct link to this comment")





       Sorry to hear that, perhaps some of these tips will help:

       [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-608169)
441. ![](https://secure.gravatar.com/avatar/e4544b98db33f5276e620122dc7274ba362ae0a3d64738dd0ab8afa0a158387b?s=40&d=mm&r=g)



     [Shilpa](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)May 28, 2021 at 4:43 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611507 "Direct link to this comment")





     Contents are explained in a simple way and are so clear. Thanx Jason



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611507)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 28, 2021 at 6:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611547 "Direct link to this comment")





       You’re welcome.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611547)
442. ![](https://secure.gravatar.com/avatar/0dc343702f3d7d9ebba4a1bb588dbc186a7624332b212436c6988485e1e088f4?s=40&d=mm&r=g)



     Toni NehmeMay 28, 2021 at 7:56 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611596 "Direct link to this comment")





     Please please help me to build a Multilayer Perceptron to use it for regression problem. Thank you



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611596)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 29, 2021 at 6:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611636 "Direct link to this comment")





       Sure, see this:

       [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611636)
443. ![](https://secure.gravatar.com/avatar/e6b23171c0dcd001b2c02a7492d7d71d1acef4eb920683b7aa70c03cc178e962?s=40&d=mm&r=g)



     James MayrMay 29, 2021 at 11:02 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611689 "Direct link to this comment")





     Thank you sooo much for your tutorial! I struggled around with the input layer and the Keras help was not helpful. But your explanation gave me the insight and the things became total clear! That was very great, Thank you!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611689)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)May 30, 2021 at 5:50 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611716 "Direct link to this comment")





       You’re welcome!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-611716)
444. ![](https://secure.gravatar.com/avatar/5b919bf09e1b6ffe903e193b913e80d676105904bdcbb4e82ac3c0b79a635f2d?s=40&d=mm&r=g)



     MeenakshiJune 3, 2021 at 8:28 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612131 "Direct link to this comment")





     Great work Sir. Simple, detailed explanation of complex things.


     I would like to learn modelling for DDoS attacks detection in Neural networks. Please suggest the way.


     Tanks in advance.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612131)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 4, 2021 at 6:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612172 "Direct link to this comment")





       Perhaps the tutorials here will help if you are modeling your problem as a time series:

       [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612172)
445. ![](https://secure.gravatar.com/avatar/5b919bf09e1b6ffe903e193b913e80d676105904bdcbb4e82ac3c0b79a635f2d?s=40&d=mm&r=g)



     MeenakshiJune 5, 2021 at 11:34 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612345 "Direct link to this comment")





     Thank you very much. I will go through it Sir.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612345)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 6, 2021 at 5:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612383 "Direct link to this comment")





       You’re welcome.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-612383)
446. ![](https://secure.gravatar.com/avatar/c906a60aee9c11b5d421c9028bbd32c59ae6ab65c2da670dce1031a4c25aadbb?s=40&d=mm&r=g)



     JCJune 24, 2021 at 4:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-614278 "Direct link to this comment")





     The following are the outcome of the first 10 consecutive executions on my 8GB RAM 64bit Windows 10 platform:



     Accuracy: 65.49


     Accuracy: 70.70


     Accuracy: 75.91


     Accuracy: 76.04


     Accuracy: 78.26


     Accuracy: 76.04


     Accuracy: 77.86


     Accuracy: 79.17


     Accuracy: 78.52


     Accuracy: 78.91



     The computer does not have GPU. The script gives some warning messages. One of them is: “None of the MLIR Optimization Passes are enabled (registered 2)”



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-614278)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)June 24, 2021 at 6:06 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-614305 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-614305)
447. ![](https://secure.gravatar.com/avatar/76cd8e87caeb51dd2276f249f1756b12e9e49e0efba05a83af9369c48f523452?s=40&d=mm&r=g)



     [Sneha](http://snehaverma.me/)July 2, 2021 at 8:31 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615208 "Direct link to this comment")





     Hi,



     I have a question regarding the input amount. I am attempting to fit a neural network for a classification model. However, the features in my model are categorical so I need to one-hot encode them. For instance, if a categorical variable has 3 values and I one-hot encode it, would that make ‘input\_dim’ 1 or 3?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615208)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 3, 2021 at 6:05 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615286 "Direct link to this comment")





       Yes, categorical variables will need to be encoded.



       3 categories will become 3 binary input variables when using a one hot encoding.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615286)
448. ![](https://secure.gravatar.com/avatar/a67a75ccce3708f2cd68f242463a643343997a36405ea01386abc270150d7e0d?s=40&d=mm&r=g)



     RohanJuly 3, 2021 at 10:15 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615305 "Direct link to this comment")





     My results:


     Accuracy:75.78


     Accuracy:78.26


     Accuracy:76.30


     Accuracy:77.47


     Accuracy:77.47



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615305)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 4, 2021 at 5:58 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615379 "Direct link to this comment")





       Well done!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-615379)
449. ![](https://secure.gravatar.com/avatar/e470383d67365b7e322aa5b44f8fe41a9a3cc66596395f3e365531e6a1196ff4?s=40&d=mm&r=g)



     PatrickJuly 10, 2021 at 8:32 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-616086 "Direct link to this comment")





     Hi Jason,



     Thank you for all of your content. All very insightful for someone new to Keras and machine learning. If you could offer any guidance/insight into the below problem I’m trying to tackle, then it would be much appreciated.



     I am trying to replicate a similar Ball Prediction Model as discussed here:



     [https://towardsdatascience.com/predicting-t20-cricket-matches-with-a-ball-simulation-model-1e9cae5dea22](https://towardsdatascience.com/predicting-t20-cricket-matches-with-a-ball-simulation-model-1e9cae5dea22)



     This is a multiclassifcation problem (thank you for your article on this). There are 8 outputs that I am trying to predict (0, 1, 2, 3, 4, 6, Wide, Wicket) column H in my dataset ( [https://i.stack.imgur.com/DmTNb.png](https://i.stack.imgur.com/DmTNb.png)).



     This dataset is ball-by-ball (match) data of many cricket matches. Columns A-G are the input variables that should be used to predict the probability of each outcome (innings, over, batsman, bowler etc.)



     Model:



     X = my\_data\[:,0:7\]


     y = my\_data\[:,7\]



     model = Sequential()


     model.add(Dense(12, input\_dim=7, activation=’relu’))


     model.add(Dense(8, activation=’relu’))


     model.add(Dense(1, activation=’sigmoid’))



     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     model.fit(X, y, epochs=150, batch\_size=10, verbose=0)


     \_, accuracy = model.evaluate(X, y, verbose=0)


     print(‘Accuracy: %.2f’ % (accuracy\*100))



     Running the above model on the ball-by-ball dataset gives an accuracy of 30%. As the article suggests, I want to include more data i.e. the historical probability of each individual batsman and bowler achieving each of the 8 outcomes.



     This means I have 3 datasets which should be used to influence the probability of each outcome.



     How and when should I be trying to introduce these 3 linked datasets? I presumably want the model to consider all this information at the same time and not in isolation.



     Is it a case of trying to incorporate the batsman/bowler datasets into the match-by-match data? The only issue I have with this is that there are c. 200,000 rows of match data, whereas a player database will have c. 500 rows.



     Maybe I am wrong, and I should be running the multiple datasets through the model individually and then somehow pooling the outcomes – is this even possible? Although I doubt that this is even recommended/worthwhile



     If you have any suggestions on how to improve the above, or achieve the desired outcome, then it would be most welcomed.



     Thanks again for all your hard work in maintaining a great data science site.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-616086)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 11, 2021 at 5:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-616134 "Direct link to this comment")





       Defining the data/problem for a model is the real work in applied machine learning.



       There is no good/best way, I recommend reading papers on or related to the topic to get ideas, prototype, experiment, etc.



       Also, this may also help on defining the problem:

       [https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)



       Also, more generally, these tutorials explain how to get better performance from neural nets:

       [https://machinelearningmastery.com/start-here/#better](https://machinelearningmastery.com/start-here/#better)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-616134)
450. ![](https://secure.gravatar.com/avatar/a43b230c10778eba8d213d4d79907460b1fbe88c3d27588a1f88e316f0940189?s=40&d=mm&r=g)



     Jolene WangJuly 23, 2021 at 5:08 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617827 "Direct link to this comment")





     Hi Jason!



     Thank you for providing all of this content. I am trying to replicate this model by using my own csv file however it contains many NaN and thus can not be loaded through the loadtxt() function. As 0 is a very important number in my dataset, I cannot change my NAs to 0. What can I do?



     Thank you again for all of your help.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617827)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 23, 2021 at 6:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617853 "Direct link to this comment")





       You must impute the missing values first, there are many methods:

       [https://machinelearningmastery.com/?s=missing&post\_type=post&submit=Search](https://machinelearningmastery.com/?s=missing&post_type=post&submit=Search)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617853)
451. ![](https://secure.gravatar.com/avatar/a43b230c10778eba8d213d4d79907460b1fbe88c3d27588a1f88e316f0940189?s=40&d=mm&r=g)



     Jolene WangJuly 23, 2021 at 5:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617828 "Direct link to this comment")





     I forgot to mention but is there a way for me to keep the NaN in the dataset and have the model read it as just a missing value? It would be difficult for me to assign the NaNs a specific value as it could mess up the dataset.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617828)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)July 23, 2021 at 6:04 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617854 "Direct link to this comment")





       No. NaN will cause all computation to fail in a ml model, including a neural net.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-617854)
452. ![](https://secure.gravatar.com/avatar/328450bf2b7b5d36258b5d4adcdf0c14b04ef137946c7869d5ca5de98c888bbf?s=40&d=mm&r=g)



     Isiyaku SalehJuly 31, 2021 at 10:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619179 "Direct link to this comment")





     Thank very much Dr, Jason the tutorial has really served be well.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619179)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 1, 2021 at 4:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619292 "Direct link to this comment")





       You’re welcome!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619292)
453. ![](https://secure.gravatar.com/avatar/70745b56a6818c417da24957580ce3028c8f7dac693132174d0f327e5519f32f?s=40&d=mm&r=g)



     Tim PapaAugust 3, 2021 at 8:02 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619664 "Direct link to this comment")





     This tutorial builds a neural network, but what specifically this neural network is? Is it an ANN or CNN or RNN?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619664)




     - ![](https://secure.gravatar.com/avatar/5f1c8e7a708d04b1bd173e9120107d3fd43d8fad5be7c94796b877515b6d0357?s=40&d=mm&r=g)



       [Jason Brownlee](https://machinelearningmastery.com/)August 4, 2021 at 5:13 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619708 "Direct link to this comment")





       It is a multi-layer perceptron (MLP) which is a type of feed-forward neural network. It is not a CNN or RNN.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-619708)
454. ![](https://secure.gravatar.com/avatar/fece9c378f025d5fd24ed5af52c6d26f89e1ce1fb51d0bd3a6ebeeefac7e22aa?s=40&d=mm&r=g)



     Edwin BrownAugust 13, 2021 at 7:26 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-621568 "Direct link to this comment")





     First and foremost, thank you Jason Brownlee for getting me started with my first deep learning project. I followed step-by-step and found myself stuck for a while; however, after countless hours of researching I found my code below to work for Python 3.8.10, Tensorflow 2.5.0, IPython 7.26.0, and Keras 2.6.0 respected environments. I apologize if I over commented, I was taking notes as I was reading through Jason’s source codes and notes. I used Anaconda-Spyder and I wanted to see the results as well in Jupyter Notebook. I hope this helps:



     import sys


     import tensorflow as tf


     from tensorflow import keras


     from numpy import loadtxt


     from tensorflow.keras.models import Sequential


     from tensorflow.keras.layers import Dense



     \# Load the data and split the X(input) & y(output) variables


     \# Be sure your data is in the respected file as the project


     dataset = loadtxt(r’pima-indians-diabetes.csv’, delimiter=’,’)


     X = dataset\[:,0:8\]


     y = dataset\[:,8\]



     \# Create our sequential model



     \# input\_dim sets number of arguements for the number of input variables


     \# This structure as three layers


     \# Fully connected layers are defined by the dense class


     # for more on dense class view on Keras homepage


     \# ReLU on the first to layers and Sigmoid function on the output layer(third layer)


     \# Default threshold of 0.5 and better performance from ReLU


     \# ReLU measures output between 0 and 1 as seen in probability


     \# The model expects rows of data with 8 variables (the input\_dim=8 argument)


     \# The first hidden layer has 12 nodes and uses the relu activation function.


     \# The second hidden layer has 8 nodes and uses the relu activation function.


     \# The output layer has one node and uses the sigmoid activation function.



     model = Sequential()



     model.add(Dense(12, input\_dim=8, kernel\_initializer=’normal’, activation=’relu’))


     model.add(Dense(8, kernel\_initializer=’normal’, activation=’relu’))


     model.add(Dense(1, kernel\_initializer=’normal’, activation=’sigmoid’))



     \# Compile the model



     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])



     \# Fit the model onto the dataset



     \# Epoch: One pass through all of the rows in the training dataset.


     \# Batch: One or more samples considered by the model within an epoch before weights are updated.


     \# The CPU or GPU handles it from here, usually, larger datasets need the GPU



     model.fit(X, y, epochs=150, batch\_size=10, verbose=0)



     \# Evaluate the data



     \_, accuracy = model.evaluate(X, y, verbose=0)


     print(‘Accuracy: %.2f’ % (accuracy\*100))



     \# make probability predictions with the model


     predictions = model.predict(X)


     \# round predictions


     rounded = \[round(x\[0\]) for x in predictions\]



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-621568)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamAugust 14, 2021 at 2:33 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-621756 "Direct link to this comment")





       Good work!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-621756)
455. ![](https://secure.gravatar.com/avatar/333fdffdf1b39fcf00d2bb23f019ff3ac644efc1f0d2644f544df01bb31e1e4e?s=40&d=mm&r=g)



     Bonjour20August 15, 2021 at 9:43 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-621939 "Direct link to this comment")





     I use Windows system on my laptop , and I do not know if I should have a Linux destro > I am confused about where should I download the Dataset > He mentioned :” on the same place where ptyhon is installed” , what is this riddle ?


     It is a riddle for a beginner like me coming from non technological background .



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-621939)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamAugust 17, 2021 at 7:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-622154 "Direct link to this comment")





       Usually that means, you just need to place the data files and the python code file together at the same folder.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-622154)
456. ![](https://secure.gravatar.com/avatar/7e1593922625442904fed7a410faed7e8d80e712a38564c4479796ae7fb3778c?s=40&d=mm&r=g)



     sama samaanAugust 30, 2021 at 6:19 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-624854 "Direct link to this comment")





     Hello


     Thanks for this great tutorial 🙂



     Question no. 1: can we apply deep learning in Apache Spark?



     Question no. 2: I have the following dataset [https://www.kaggle.com/leandroecomp/sdn-traffic](https://www.kaggle.com/leandroecomp/sdn-traffic)


     I tried the multi-class classification code but it stop working. What could be the reason behind that fault?



     Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-624854)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamSeptember 1, 2021 at 7:39 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-625322 "Direct link to this comment")





       (1) yes (2) what specifically stopped working?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-625322)
457. ![](https://secure.gravatar.com/avatar/f0a1ad0dab27807978523c2b4a1478f637a30cd93c089840ddc4c9dd3861b33a?s=40&d=mm&r=g)



     MALAVIKASeptember 23, 2021 at 11:17 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627459 "Direct link to this comment")





     First of all, I am overwhelmed by the number of comments and prompt replies by the author. You are really a lifesaver to many, Jason.



     Now, I have a doubt. I have been searching for a simple feed-forward-back-propagation ANN code in python, and I could see only feed-forward neural networks everywhere. In your example, is backpropagation happening? Doesn’t ANN mean both the processes by default?



     Shouldn’t we apply back propagation in ANN, normally?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627459)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamSeptember 24, 2021 at 4:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627481 "Direct link to this comment")





       Feed-forward happens when you give input to the ANN. Backpropagation happens when you calculate the gradient and update the weights in each neuron.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627481)
458. ![](https://secure.gravatar.com/avatar/f0a1ad0dab27807978523c2b4a1478f637a30cd93c089840ddc4c9dd3861b33a?s=40&d=mm&r=g)



     MALAVIKASeptember 24, 2021 at 5:06 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627532 "Direct link to this comment")





     So, I suppose it’s (back-propagation) not happening in the above tutorial. Can you show us how to code the back-propagation in python, or direct me to any posts that show the same?



     Thank You.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627532)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamSeptember 25, 2021 at 4:36 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627576 "Direct link to this comment")





       When you call fit() function, backpropagation is used to update the model parameters. That’s part of the training process. We don’t normally do this explicitly. If you are interested, see a toy example here: [https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-627576)
459. ![](https://secure.gravatar.com/avatar/7e848674d08ead09a0f5b5c19283aaacb39e17e47c80f270e2652264b948eb80?s=40&d=mm&r=g)



     ElhamOctober 8, 2021 at 1:11 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-629011 "Direct link to this comment")





     Hi, Thanks a lot for this awesome tutorial. I’m using tensorflow version 2.6 and in making class predictions with the model with these lines of code,



     predict\_x = model.predict(X)


     classes\_x = np.argmax(predict\_x,axis=1)


     for i in range(5):


     print(‘%s => %d (expected %d)’ % (X\[i\].tolist(), classes\_x\[i\], y\[i\]))



     the outpout is:



     \[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0\] => 0 (expected 1)


     \[1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0\] => 0 (expected 0)


     \[8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0\] => 0 (expected 1)


     \[1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0\] => 0 (expected 0)


     \[0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0\] => 0 (expected 1)



     Why are all classes\_x zero?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-629011)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamOctober 13, 2021 at 5:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-629542 "Direct link to this comment")





       Because the prediction here is a binary one, hence predict\_x is Nx1 matrix which argmax will only report 0. Your syntax is correct for multi-class, which the neural network has output layer as Dense(n) with n>1



       I’ve updated the sample code here to reflect what you should do. Thanks for alerting me.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-629542)
460. ![](https://secure.gravatar.com/avatar/eebddfaf71403de4f1db4226809bedbfee68563b40125b85e9f3422bbd56bf2e?s=40&d=mm&r=g)



     christoperOctober 17, 2021 at 6:41 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-629996 "Direct link to this comment")





     hello this is helpful. I am studying neural networks and im just a beginner. You said this is mlp type of neural network right? I just want to ask, how about this? What kind of neural network architecture used here? is it rnn? ann? or ltstm? link below:



     [https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44](https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-629996)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamOctober 20, 2021 at 8:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-630535 "Direct link to this comment")





       MLP = Multilayer Perceptron, which usually means a neural network with 3 or more layers. The link you provided use Dense(), which is fully-connected layer. Hence it is also MLP.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-630535)
461. ![](https://secure.gravatar.com/avatar/39a11d721854525b37e18f0c318d532deb3c7d26a34c25fa97b43784348436f3?s=40&d=mm&r=g)



     FloOctober 25, 2021 at 11:29 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632226 "Direct link to this comment")





     Hi Jason and Adrian, I came across your very nice tutorial, because I have a quite similar problem.



     I have a couple of numerical process parameters of an engineering problem (similar to your input parameters here), which I want to check to an outcome value (which is different to your tutorial again a numerical value, not a classification). Can you tell me (or do you even now a accordingly handy tutorial like this one), how I need to modify the code?



     Thanks a lot!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632226)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamOctober 27, 2021 at 2:23 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632544 "Direct link to this comment")





       It sounds to me that it is a regression problem instead of classification problem. In this case, two things you may consider to change



       1\. The last Dense() layer, you may want a different activation (e.g., linear?) because sigmoidal is bounded between 0 and 1


       2\. model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\]) should have a loss and metric changed. For example, you may consider to use MSE because cross entropy and accuracy are measures specific to classification



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632544)
462. ![](https://secure.gravatar.com/avatar/680bf36b0e253c1d7165a87012cef8a2b8468ea68400875593136a3600383f74?s=40&d=mm&r=g)



     Dr Shazia SaqibOctober 28, 2021 at 3:14 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632855 "Direct link to this comment")





     awesome, great service, very helpful, am sharing with my students, Lord Bless you ameen



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-632855)

463. ![](https://secure.gravatar.com/avatar/eebddfaf71403de4f1db4226809bedbfee68563b40125b85e9f3422bbd56bf2e?s=40&d=mm&r=g)



     veejayNovember 5, 2021 at 11:23 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-635549 "Direct link to this comment")





     Awesome tutorial, very well-detailed. I have a question though,



     How to improve Validation Loss and Validation Accuracy? I am very new to Neural Network. I only scratched the surface of it. Weights, biases, activation function, loss function, architectures and how to build layers on keras and other fundamental terminologies (thanks from you and deeplizard tutorials from youtube.) I am studying and practicing it and I want to try and replicate some project and I came across this tutorial from Dataflair where he’s creating a chatbot and I tried to imitate it. LINK: [https://data-flair.training/blogs/python-chatbot-project/](https://data-flair.training/blogs/python-chatbot-project/) .


     So from what I have observed and based on my learnings, the model that he created is an ANN-MLP. My problem is, when I trained the model and set the validation\_split = 0.3, the training loss and accuracy are good but the validation loss and accuracy do the opposite. I know that it may be an overfitting problem so…


     Here’s what I did:


      -added regularization with L2


     – Slowed the learning rate and I also tried to speed it up


      -Dropouts (0.2-0.5)


      -Batch Size


      -Removing layers


      -Adding layers


      -Experimented different activation and loss functions (sigmoid, softplus, binary\_crossentropy)


      -I even tried to add data on my datasets (from 320 to 796 inputs)


     I tried all of this but val\_loss and val\_acc still high and low respectively.



     (Best that I did is loss: 0.1/accuracy 98 percent val loss: 1.9/val\_accuracy: 52 percent.



     while the worst is val\_loss: over 3.0 and val\_accuracy 35-40 percent )



     The dataset that i’m using is from dataflair but I expanded it. here’s my visualized model: [https://i.stack.imgur.com/HE1jU.png](https://i.stack.imgur.com/HE1jU.png)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-635549)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamNovember 7, 2021 at 10:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-636027 "Direct link to this comment")





       Can’t really tell what went wrong here. Did you verify the validation loss as you trained it? At first, the training loss and validation loss should be equally bad. How did they progressed in each training epoch? This may give you clues.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-636027)
464. ![](https://secure.gravatar.com/avatar/0ab5bbfb0e60d1c9f32ea49a9af556e6ee7d95710c583dde18768dd36bb18b0c?s=40&d=mm&r=g)



     VeejayNovember 9, 2021 at 6:54 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-636561 "Direct link to this comment")





     Yes I both trained and validate them. They are equally bad at first and as they progressed, the loss improved by miles but val\_loss and val\_accuracy improved an inch. T\_T



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-636561)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamNovember 14, 2021 at 1:36 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-638365 "Direct link to this comment")





       That’s expected. You model was looking at the training loss and try to improve itself, but it was not able to see the validation data so it is harder and slower to improve.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-638365)
465. ![](https://secure.gravatar.com/avatar/e3b483821db49f93aad219a4a30e0a4e6997648df000f04a7ece91cbbf607bbd?s=40&d=mm&r=g)



     MakNovember 17, 2021 at 6:24 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-639206 "Direct link to this comment")





     Your books helped me understand LSTMs greatly, I am having trouble with developing an attention layer, please can you do a tutorial on using Attention/ MultiheadAttention


     Thank you.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-639206)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamNovember 18, 2021 at 5:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-639308 "Direct link to this comment")





       Please see the series: [https://machinelearningmastery.com/category/attention/](https://machinelearningmastery.com/category/attention/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-639308)
466. ![](https://secure.gravatar.com/avatar/48fc1cc684b4a79a1053a45b8514f90774d6ec714b7dd544dcc43799dc3e21bc?s=40&d=mm&r=g)



     Nikhil GuptaNovember 25, 2021 at 5:47 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-641156 "Direct link to this comment")





     The accuracy from ANN for this data set is between 70-78%. Using Logistics Regression, we are getting 78% accuracy for the same dataset. So, what’s the advantage of using ANN?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-641156)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamNovember 26, 2021 at 2:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-641268 "Direct link to this comment")





       ANN is more flexible. Occam’s razor – you use the simplest model for the job. If logistic regression fits well, you have no reason to use ANN. It use more memory and runs slower.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-641268)
467. ![](https://secure.gravatar.com/avatar/39a11d721854525b37e18f0c318d532deb3c7d26a34c25fa97b43784348436f3?s=40&d=mm&r=g)



     FloDecember 3, 2021 at 8:38 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-642703 "Direct link to this comment")





     Thanks for the Tutorial



     I tried your approach and it worked nicely on my data. For a first shot I just used data, which is measured after the process (e.g. process time, temperature difference during the process, etc.). For a further, deeper investigation, I would like to use measured data curves, for example the development of the process temperature by time during the process itself. By use of these curves, I expect a higher degree of information.



     Could you provide a hint, how to work with this data? For the first shot I simply generated a table with my process parameters in the first 6 columns and my output value in column 7, which could be easily feeded into the modell.



     Thanks a lot!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-642703)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamDecember 8, 2021 at 6:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-643389 "Direct link to this comment")





       Everything sounds straightforward to me. Did you tried implemented this? Any error?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-643389)
468. ![](https://secure.gravatar.com/avatar/39a11d721854525b37e18f0c318d532deb3c7d26a34c25fa97b43784348436f3?s=40&d=mm&r=g)



     FloDecember 10, 2021 at 6:32 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-643907 "Direct link to this comment")





     To be honest, I have no clue how to provide the data. In the first case, I had a table with 7 columns: 6 Input process parameter and one column with output values.



     Now I would like to replace (are add) some input columns with time-recorded data curves, which are somehow tables (first column the timestamp, second column the time-specific process parameter) itself. How do I work with this?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-643907)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamDecember 15, 2021 at 5:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-644641 "Direct link to this comment")





       Usually I would have pandas to process data and convert it to numpy array before feeding to Keras model. Pandas allows you to manipulate tables easier



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-644641)
469. ![](https://secure.gravatar.com/avatar/32df0c00f6df36949e8c65ee9fed019fc0839e1021ba35d98d990378f98122f1?s=40&d=mm&r=g)



     RickDecember 28, 2021 at 7:46 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-647911 "Direct link to this comment")





     May need to adjust the import settings for compatibility with newer Tensorflow versions.



     Instead of:


     …


     from keras.models import Sequential


     from keras.layers import Dense



     Use:


     …


     from tensorflow.keras.models import Sequential


     from tensorflow.keras.layers import Dense



     Solved my issues with Conda.



     Thanks for the excellent tutorials and articles!!



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-647911)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelDecember 29, 2021 at 11:44 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-648186 "Direct link to this comment")





       Thank you for the feedback Rick! I also often try to run code in both Anaconda and Google Colab to identify and correct compatibility issues.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-648186)
470. ![](https://secure.gravatar.com/avatar/a77ada1d007e90a5b8f33cec378d0365f29bb0570c1d431e08e369393fbc76ff?s=40&d=mm&r=g)



     PreetiFebruary 10, 2022 at 4:18 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-654229 "Direct link to this comment")





     My Accuracy: 76.95



     Thank you for the code and detailed explanation



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-654229)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelFebruary 11, 2022 at 8:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-654325 "Direct link to this comment")





       You are very welcome, Preeti! Keep up the great wok!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-654325)
471. ![](https://secure.gravatar.com/avatar/da91f161e0d02053a3c6c0fb24c955f01b45c9afbc943fc66263bf3d24fd7cf7?s=40&d=mm&r=g)



     AlanMarch 9, 2022 at 8:46 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-658808 "Direct link to this comment")





     Hi James



     Great work



     Never mind neural networks, this is causing me a lot of deep thinking.



     I am running your tutorial on a pi 400 with 64bit OS on Thonny.



     Works reasonably well on this machine.



     However came across an error in one of your examples … Keras neural network using ‘ pima-indians-diabetes.csv’



     ” from tensorflow.python.eager.context import get\_config


     ImportError: cannot import name ‘get\_config’ from ‘tensorflow.python.eager.context’ (/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/context.py)”



     So discovered that the fault lay with Keras.models and layers and have rejigged the sketch as follows:-



     \# first neural network with keras tutorial


     from numpy import loadtxt


     from tensorflow.keras import models,layers #\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


     #from keras.models import Sequential #\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*



     #from keras.layers import Dense #\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


     \# load the dataset


     dataset = loadtxt(‘/home/pi/Documents/pima-indians-diabetes.csv’, delimiter=’,’)


     \# split into input (X) and output (y) variables


     X = dataset\[:,0:8\]


     y = dataset\[:,8\]


     \# define the keras model


     model = models.Sequential() #\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


     model.add(layers.Dense(12, input\_dim=8, activation=’relu’)) #\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


     model.add(layers.Dense(8, activation=’relu’)) #\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


     model.add(layers.Dense(1, activation=’sigmoid’)) #\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


     \# compile the keras model


     model.compile(loss=’binary\_crossentropy’, optimizer=’adam’, metrics=\[‘accuracy’\])


     \# fit the keras model on the dataset


     model.fit(X, y, epochs=150, batch\_size=10)


     \# evaluate the keras model


     \_, accuracy = model.evaluate(X, y)


     print(‘Accuracy: %.2f’ % (accuracy\*100))



     Now that produces


     Accuracy: 74.35



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-658808)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelMarch 10, 2022 at 10:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-658947 "Direct link to this comment")





       Hi Alan…Thank you for the feedback and support! Interesting application to the Raspberry Pi! Keep in mind that our implementations may not be fully compatible with the libraries that are developed for that platform. Keep up the great work!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-658947)
472. ![](https://secure.gravatar.com/avatar/f987ed3bcda6b24921bf8b85368a6849c94144691d78bd372e4709be6e410464?s=40&d=mm&r=g)



     NishanthMarch 14, 2022 at 3:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659559 "Direct link to this comment")





     Hi,



     Amazing tutorial! Simple and easy. I tried the same thing on my dataset but the last for loop does not seem to work. Could pls help me with it?



     Here is the for loop:


     for i in range(5):


     print(‘%s => %d (expected %d)’ % (X\[i\].tolist(), predictions\[i\], y\[i\]))



     Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659559)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelMarch 14, 2022 at 11:48 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659614 "Direct link to this comment")





       Hi Nishanth…are you copying and pasting the code or typing it in? Be careful regarding copying and pasting code and how it may affect the code layout as errors may be very difficult to spot visually.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659614)
473. ![](https://secure.gravatar.com/avatar/f987ed3bcda6b24921bf8b85368a6849c94144691d78bd372e4709be6e410464?s=40&d=mm&r=g)



     NishanthMarch 14, 2022 at 3:40 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659560 "Direct link to this comment")





     Hi here in the comment the print statement looks un-indented but in my code, I indent it and still does not work.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659560)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelMarch 14, 2022 at 11:52 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659616 "Direct link to this comment")





       Hi Nishanth…please see previous replies.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659616)
474. ![](https://secure.gravatar.com/avatar/f987ed3bcda6b24921bf8b85368a6849c94144691d78bd372e4709be6e410464?s=40&d=mm&r=g)



     NishanthMarch 14, 2022 at 3:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659561 "Direct link to this comment")





     Hi,



     Amazing Tutorial! Simple and Easy to follow. I tried it on my dataset but the last for loop that prints first 5 examples does not work. It gives me KeyError: 0



     Could you help me with it?



     Thanks



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659561)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelMarch 14, 2022 at 11:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659613 "Direct link to this comment")





       Hi Nishanth…please share the full error message so we can better assist you.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659613)
475. ![](https://secure.gravatar.com/avatar/f987ed3bcda6b24921bf8b85368a6849c94144691d78bd372e4709be6e410464?s=40&d=mm&r=g)



     NishanthMarch 14, 2022 at 11:41 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659699 "Direct link to this comment")





     Found a way out. Thing is that here the dataset is numpy array and mine was a pandas.DataFrame. Thanks for the help.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-659699)




     - ![](https://secure.gravatar.com/avatar/7e83300e92a2912863f1e742ff5c49ebbaf3ef752e2b5033cdf9544cf770409e?s=40&d=mm&r=g)



       MKNovember 6, 2022 at 6:02 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-683102 "Direct link to this comment")





       Hi Nishanth,



       Would you please share how you fix the Keyerror at last?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-683102)
476. ![](https://secure.gravatar.com/avatar/0893e645f23b68de42be2825caba56332c216781a760f34c1962fc9a28352687?s=40&d=mm&r=g)



     N V RamanApril 2, 2022 at 1:51 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-662566 "Direct link to this comment")





     Hello Jason,



     Really wonderful tutorial



     When I ran the code everything worked except while printing the predictions I get a key error.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-662566)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelApril 2, 2022 at 12:18 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-662638 "Direct link to this comment")





       Hi N V…Can you provide the exact error message so that we can better assist you?



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-662638)
477. ![](https://secure.gravatar.com/avatar/e38033539f65fca22a58e4a6b1c4c779d40bd026093b0545b4b15d5a17af18b5?s=40&d=mm&r=g)



     SusiaApril 9, 2022 at 1:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-663547 "Direct link to this comment")





     Hi, I’ve learned the same tutorial to develop the first neural net in Keras in one of your mini\_courses. To develop my own model on my own dataset, I’ve tried to adapt this tutorial. The problem is my target Y is count data (number of traffic flow for example). In my case, how to define the activation function for the output layer. Is it relu? How to choose the loss function? I’ve tried MeanSquaredError, the loss value is quite large, or categorical\_crossentropy, the loss value is nan. I am considering to order the complete book of Deep Learning With Python. What’s the difference of the tutorials inside the book and the mini\_course?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-663547)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelApril 9, 2022 at 8:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-663647 "Direct link to this comment")





       Hi Susia…The following resource may add clarity in how to choose an activation function:



       [https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-663647)
478. ![](https://secure.gravatar.com/avatar/54ad6854a47d3c7a7d9ace9f1ec65d1fa2f0e5691b708c8603d89ee047ca938d?s=40&d=mm&r=g)



     NasrinApril 23, 2022 at 4:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-665996 "Direct link to this comment")





     I sir, thanks a million for your awesome post


     could you please explain how we can divide X and y into the train and test sample in deep learning?


     this code is correct here?



     from sklearn.model\_selection import train\_test\_split


     X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.33, random\_state=42)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-665996)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelApril 24, 2022 at 3:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-666097 "Direct link to this comment")





       Hi Nasrin…the sample code you provided looks accurate. Feel free to implement it and let us know if you encounter any issues.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-666097)
479. ![](https://secure.gravatar.com/avatar/6b71ac0b42d2fe63d004e22557668bf64f7d08b3c0f790e4040093cf1182d15f?s=40&d=mm&r=g)



     Shiva ManharApril 23, 2022 at 3:25 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-666047 "Direct link to this comment")





     24/24 \[==============================\] – 0s 489us/step – loss: 0.4517 – accuracy: 0.7956


     Accuracy: 79.56



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-666047)

480. ![](https://secure.gravatar.com/avatar/388b6505f8a458833577ea13c2fd3851173105c4654ad1b968b11194c1176f4e?s=40&d=mm&r=g)



     Jack SparrowJune 3, 2022 at 5:38 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-671277 "Direct link to this comment")





     Deep Learning with keras mnist dataset:



     from cgi import test


     from pyexpat import model


     import numpy as np


     from keras.models import Sequential


     from keras import layers


     #from keras.layers import Convolution2D, MaxPooling2D #train on image data


     from keras.utils import np\_utils #veri dönüşümü için gerekli



     from keras.datasets import mnist #image data


     (X\_train, y\_train), (X\_test, y\_test) = mnist.load\_data()



     print(“Reshape öncesi”,X\_train.shape)



     X\_train = X\_train.reshape(-1, 28, 28, 1)


     X\_test = X\_test.reshape(-1, 28, 28, 1)



     print(“Reshape sonrası”,X\_train.shape)



     X\_train = X\_train.astype(‘float32’)


     X\_test = X\_test.astype(‘float32′)


     X\_train /= 255


     X\_test /= 255



     Y\_train = np\_utils.to\_categorical(y\_train)


     Y\_test = np\_utils.to\_categorical(y\_test)



     model = Sequential()



     model.add(layers.Convolution2D(32, 3, 3, activation=’relu’, input\_shape=(28,28,1)))


     model.add(layers.Convolution2D(32, 3, 3, activation=’relu’))


     model.add(layers.MaxPooling2D(pool\_size=(2,2)))


     model.add(layers.Dropout(0.25))



     model.add(layers.Flatten())


     model.add(layers.Dense(128, activation=’relu’))


     model.add(layers.Dropout(0.5))


     model.add(layers.Dense(10, activation=’softmax’))



     model.compile(loss=’categorical\_crossentropy’,


     optimizer=’adam’,


     metrics=\[‘accuracy’\])



     model.fit(X\_train, Y\_train,


     batch\_size=32, epochs=10, verbose=1)



     test\_loss, test\_acc = model.evaluate(X\_test, Y\_test, verbose=0)


     print(“Test Loss”, test\_loss)


     print(“Test Accuracy”,test\_acc)



     Deep Learning with data\_diagnosis dataset:



     import imp


     from pickletools import optimize


     from random import random


     from statistics import mode


     from tabnanny import verbose


     from warnings import filters


     from matplotlib.pyplot import axis


     import pandas as pd


     import numpy as np



     dataSet = pd.read\_csv(“.\\data\_diagnosis.csv”)


     dataSet.drop(\[“id”,”Unnamed: 32″\],axis=1,inplace=True)



     dataSet.diagnosis = \[1 if each == “M” else 0 for each in dataSet.diagnosis\]


     y=dataSet.diagnosis.values


     x\_data=dataSet.drop(\[“diagnosis”\],axis=1)


     x\_data.astype(“uint8”)



     from sklearn.preprocessing import StandardScaler


     scaler=StandardScaler()


     x=scaler.fit\_transform(x\_data)



     from keras.utils import to\_categorical


     Y=to\_categorical(y)



     from sklearn.model\_selection import train\_test\_split


     trainX,testX,trainy,testy=train\_test\_split(x,Y,test\_size=0.2,random\_state=42)



     trainX=trainX.reshape(trainX.shape\[0\],testX.shape\[1\],1)


     testX=testX.reshape(testX.shape\[0\],testX.shape\[1\],1)



     from keras import layers


     from keras import Sequential



     verbose,epochs,batch\_size=0,10,8


     n\_features,n\_outputs=trainX.shape\[1\],trainy.shape\[1\]



     model= Sequential()


     input\_shape=(trainX.shape\[1\],1)


     model.add(layers.Conv1D(filters=8,kernel\_size=5,activation=’relu’,input\_shape=input\_shape))


     model.add(layers.BatchNormalization())


     model.add(layers.MaxPooling1D(pool\_size=3))


     model.add(layers.Conv1D(filters=16,kernel\_size=5,activation=’relu’))


     model.add(layers.BatchNormalization())


     model.add(layers.MaxPooling1D(pool\_size=2))


     model.add(layers.Flatten())


     model.add(layers.Dense(200,activation=’relu’))


     model.add(layers.Dense(n\_outputs,activation=’softmax’))


     model.summary()


     print(‘başladı’)



     import keras


     import tensorflow


     #model.compile(loss=’categorical\_crossentropy’,optimizer=’adam’,metrics=\[‘accuracy’\])


     model.compile(loss=’binary\_crossentropy’,


     optimizer=tensorflow.keras.optimizers.Adam(),


     metrics=\[‘accuracy’\]) # 编译


     dataSet.info()


     model.fit(trainX,trainy,epochs=epochs,verbose=1)


     \_,accuracy=model.evaluate(testX,testy,verbose=0)



     print(accuracy)



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-671277)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelJune 3, 2022 at 9:12 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-671292 "Direct link to this comment")





       Thank you for the feedback Jack! Keep up the great work!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-671292)
481. ![](https://secure.gravatar.com/avatar/d010b707ec372170f5a7ab3ed6f111c6c684f2e2629a0a497fba6166d0eae0af?s=40&d=mm&r=g)



     JackJune 17, 2022 at 5:25 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-672593 "Direct link to this comment")





     24/24 \[==============================\] – 0s 1ms/step


     \[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0\] => 1 (expected 1)


     \[1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0\] => 0 (expected 0)


     \[8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0\] => 1 (expected 1)


     \[1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0\] => 0 (expected 0)


     \[0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0\] => 1 (expected 1)



     my accuracy is 77.99 but this shows it 100 is this right?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-672593)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelJune 17, 2022 at 9:28 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-672613 "Direct link to this comment")





       Thank you for the feedback Jack!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-672613)
482. ![](https://secure.gravatar.com/avatar/4d92fb4e4361d7ffab9e7e755a030898954d29a0ea137992c814c0c21c63edec?s=40&d=mm&r=g)



     Nicola MengaJune 22, 2022 at 5:53 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-673406 "Direct link to this comment")





     Hi.


     Thank you for this tutorial. It is very useful.


     I have a question. This is a tutorial for a binary classification purpose.


     However, I want to build a Feed Forward Neural Network which predicts more than one variable (more than one neuron in the output layer), which have a value between 0 and 1 (for example 0.956, 0.878, 0.897 and so on), unlike the case of this tutorial, in which the variable to be predicted takes only the values 0 or 1.


     I tried to apply the network developed in this tutorial for this purpose, but results are bad.


     My test dataset have 257 observations. If I apply this network, the prediction array is constituted by 257 values (one for each observation), but these values are all the same (for example 1: 0.985; 2: 0.985; 3: 0.985; …; 256: 0.985; 257: 0.985). I hope I explained.



     There is a keras model/function adequate for my problem (i.e. the prediction of a variable which is not 0 or 1)?



     Thank you for your help.



     Nicola Menga.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-673406)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelJune 23, 2022 at 10:59 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-673501 "Direct link to this comment")





       Hi Nicola…Please clarify and/or elaborate on your question so that we may better assist you.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-673501)
483. ![](https://secure.gravatar.com/avatar/3d29d274cfb34666ca8e4a99758fef6deba154fe5cd5c157f592acdd807ec380?s=40&d=mm&r=g)



     SadeghJuly 7, 2022 at 3:49 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-675108 "Direct link to this comment")





     Hi there,


     I always get warning when I’m using NN model that is made with keras in anaconda’s spyder consul .


     The warning is as follow:



     WARNING: AutoGraph could not transform <function Model.make\_test\_function..test\_function at 0x0000011A030555E0> and will run it as-is.


     Cause: Unable to locate the source code of <function Model.make\_test\_function..test\_function at 0x0000011A030555E0>. Note that functions defined in certain environments, like the interactive Python shell, do not expose their source code. If that is the case, you should define them in a .py source file. If you are certain the code is graph-compatible, wrap the call using @tf.autograph.experimental.do\_not\_convert. Original error: lineno is out of bounds


     To silence this warning, decorate the function with @tf.autograph.experimental.do\_not\_convert



     I really appreciate if you can help me out of this.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-675108)

484. ![](https://secure.gravatar.com/avatar/372dea4d936d347c675afc722c5b487a59ae007b3acc2cbbad57c2db7d11173b?s=40&d=mm&r=g)



     sukhAugust 12, 2022 at 6:22 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-680447 "Direct link to this comment")





     hello James Carmichael,



     Thanks for your all effort . as a beginner I manage to run your example code and read step by step the function of each line of code . very exiting journey started …..my query is i feed the different data in which first row have 12 variable input and 12th is the output result but in 5th or 6th column have under below. how i handle this types of input in dataset.my dataset error in reading .



     19 2 49 156 782 394 296.4 723.7 809.4 29.87 53.78 86


     740 366


     728 398


     659 161


     704 220


     795 173


     784 385


     732 282


     18 1 60 172 850 1455 794 670 28.44 80.74 90


     873


     842


     817


     749


     797


     849


     850


     847


     842



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-680447)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelAugust 13, 2022 at 6:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-680519 "Direct link to this comment")





       Hi sukh…You are very welcome! Are you receiving an error message that you can share? This will allow us to better assist you.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-680519)




       - ![](https://secure.gravatar.com/avatar/372dea4d936d347c675afc722c5b487a59ae007b3acc2cbbad57c2db7d11173b?s=40&d=mm&r=g)



         sukhAugust 13, 2022 at 10:34 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-680599 "Direct link to this comment")





         ok thanks. Actually my data file in csv format. i am able to read it . but facing problem in making array. my one input has multiple row and moreover spread into down column . means data is not in single row. Every input have same manner. kindly suggest me to possibility to make arrary in this type. or i need to put data in one cell of column E to make data in one row , last column K is output result. and my second input is started from row no 411. I hope you understand my data input relation. here under code and data.



         my query is …can we feed data in this manner ? and if yes then how I will declare my dataset to process further



         from numpy import loadtxt


         from tensorflow.keras.models import Sequential


         from tensorflow.keras.layers import Dense


         from google.colab import files


         uploaded = files.upload()



         import csv



         \# opening the CSV file


         with open(‘dataread.csv’, mode =’r’)as file:



         # reading the CSV file


         csvFile = csv.reader(file)


         print(csvFile)


         # displaying the contents of the CSV file


         for lines in csvFile:


         print(lines)



         19 2 49 156 782 296.4 723.7 809.4 29.87 53.78 86


         740


         728


         659


         704


         795


         784


         732


         744


         764


         777


         749


         700


         729


         722


         741


         790


         783


         736


         744


         781


         810


         745


         722


         734


         736


         750


         706


         744


         789


         851


         813


         750


         783


         786


         758


         731


         742


         708


         733


         720


         673


         689


         729


         700


         781


         786


         758


         717


         773


         802


         726


         719


         734


         707


         678


         754


         747


         715


         771


         830


         786


         751


         773


         811


         824


         820


         772


         760


         814


         735


         687


         726


         771


         733


         773


         822


         858


         806


         756


         783


         775


         776


         739


         730


         796


         775


         754


         721


         744


         764


         793


         742


         734


         774


         802


         759


         735


         744


         767


         735


         723


         691


         748


         719


         749


         846


         822


         749


         753


         825


         854


         817


         754


         737


         785


         803


         785


         746


         736


         783


         741


         737


         694


         814


         754


         761


         814


         823


         785


         733


         759


         786


         814


         763


         792


         851


         813


         795


         751


         759


         780


         760


         738


         760


         801


         767


         738


         673


         697


         673


         664


         691


         783


         821


         823


         807


         746


         775


         822


         827


         763


         732


         756


         750


         814


         766


         733


         772


         813


         792


         722


         777


         793


         813


         757


         747


         817


         805


         788


         802


         754


         772


         788


         847


         781


         749


         763


         814


         838


         748


         749


         760


         788


         720


         685


         697


         658


         684


         807


         843


         759


         730


         750


         807


         774


         748


         715


         779


         803


         818


         755


         768


         800


         787


         759


         798


         838


         843


         775


         801


         814


         750


         716


         745


         758


         779


         721


         717


         768


         744


         773


         758


         724


         730


         774


         744


         772


         733


         663


         671


         654


         762


         820


         818


         797


         770


         847


         827


         818


         751


         726


         760


         779


         804


         790


         755


         768


         820


         812


         852


         759


         787


         825


         782


         766


         746


         808


         793


         791


         745


         787


         800


         844


         733


         739


         780


         783


         739


         726


         745


         796


         800


         752


         796


         804


         813


         735


         726


         739


         699


         665


         648


         678


         779


         801


         798


         822


         772


         824


         837


         795


         739


         714


         771


         802


         761


         727


         773


         789


         917


         876


         788


         788


         810


         790


         770


         789


         787


         771


         743


         796


         848


         853


         769


         807


         817


         831


         817


         766


         817


         766


         707


         668


         702


         821


         817


         828


         799


         765


         795


         817


         798


         751


         792


         832


         831


         776


         764


         806


         811


         760


         747


         802


         823


         755


         754


         800


         823


         792


         750


         805


         818


         793


         752


         748


         741


         736


         736


         685


         749


         719


         766


         905


         857


         760


         741


         774


         815


         773


         746


         778


         846


         825


         775


         800


         819


         767


         780


         804


         896


         812


         757


         811


         819


         817


         779


         774


         791


         818


         770


         754


         771


         786


         753


         744


         793


         805


         799



         18 1 79 159 532 1182 1486 1744 51.75 83.64 76


         354


         831


         848


         466


         442


         837


         842


         401


         347


         721


         699


         945


         1001


         869


         837


         889


         935


         823


         876


         817


         821


         951


         878


         929


         799


         790


         849


         838


         822


         957


         933


         803


         767


         840


         905


         794


         710


         756


         1004


         966


         858


         809


         955


         930


         944


         820


         809


         823


         821


         905


         894


         890


         869


         856


         819


         762


         724


         695


         797


         794


         745


         894


         966


         923


         875


         896


         911


         859


         925


         863


         862


         884


         900


         827


         937


         936


         912


         932


         819


         800


         770


         1008


         921


         806


         924


         881


         848


         953


         893


         871


         926


         991


         889


         867


         913


         815


         901


         888


         815


         834


         876


         899


         849


         982


         886


         883


         867


         914


         928


         986


         868


         888


         957


         922


         895


         861


         828


         874


         834


         798


         862


         1016


         864


         904


         926


         838


         939


         924


         885


         890


         941


         897


         863


         1034


         906


         842


         866


         862


         832


         896


         913


         881


         875


         916


         914


         878


         957


         890


         793


         759


         804


         1003


         786


         868


         955


         840


         848


         938


         884


         886


         928


         889


         873


         966


         927


         913


         884


         868


         846


         900


         882


         836


         847


         910


         901


         874


         835


         870


         882


         814


         761


         857


         742


         719


         729


         947


         823


         822


         782


         914


         858


         850


         891


         1003


         836


         1034


         873


         867


         846


         799


         860


         772


         784


         787


         991


         936


         909


         1071


         1039


         1037


         1065


         966


         1022


         1023


         963


         959


         897


         870


         886


         881


         854


         943


         975


         869


         918


         900


         890


         960


         995


         853


         927


         926


         892


         970


         956


         881


         901


         997


         858


         924


         840


         852


         995


         1076


         896


         967


         942


         910


         1050


         994


         993


         1024


         915


         972


         942


         866


         866


         854


         837


         945


         955


         912


         930


         914


         927


         995


         987


         850


         838


         757


         727


         705


         744


         962


         859


         854


         919


         905


         900


         1002


         868


         858


         945


         890


         831


         863


         854


         901


         980


         917


         886


         944


         898


         977


         817


         747


         728


         777


         834


         908


         850


         792


         811


         964


         872


         834


         870


         937


         849


         910


         858


         834


         874


         936


         867


         825


         831


         891


         890


         912


         907


         938


         873


         873


         893


         891


         875


         959


         914


         872


         946


         875


         797


         888


         893


         810


         1069


         977


         925


         900


         874



         18 1 60 172 850 1455 794 670 28.44 80.74 90


         873


         842


         817


         749


         797


         849


         850


         847


         842


         809


         779


         739


         737


         746


         763


         854


         935


         911


         863


         832


         820


         775


         756


         819


         820


         810


         787


         766


         837


         843


         867


         820


         749


         726


         759


         823


         763


         761


         769


         767


         767


         736


         796


         864


         871


         833


         780


         785


         741


         697


         659


         659


         696


         794


         975


         866


         784


         820


         825


         800


         780


         752


         812


         775


         741


         709


         676


         675


         656


         674


         686


         691


         694


         714


         707


         743


         753


         741


         712


         717


         733


         730


         735


         735


         759


         750


         746


         750


         739


         775


         757


         715


         703


         730


         831


         844


         811


         749


         775


         795


         826


         819


         812


         820


         878


         925


         885


         840


         796


         794


         830


         870


         876


         863


         846


         815


         825


         919


         910


         859


         803


         795


         839


         887


         844


         813


         841


         891


         854


         836


         806


         785


         813


         855


         880


         816


         854


         886


         897


         811


         811


         847


         873


         841


         774


         735


         750


         820


         805


         824


         832


         828


         832


         916


         903


         894


         854


         817


         846


         859


         891


         891


         852


         836


         841


         840


         820


         839


         845


         871


         894


         856


         850


         869


         876


         859


         858


         812


         738


         745


         843


         860


         836


         847


         841


         845


         856


         910


         969


         953


         923


         860


         835


         821


         814


         844


         895


         936


         914


         866


         841


         824


         804


         844


         921


         935


         915


         855


         860


         884


         881


         850


         824


         821


         861


         941


         869


         825


         852


         868


         865


         854


         872


         898


         888


         868


         839


         835


         841


         822


         792


         825


         829


         806


         757


         763


         790


         868


         782


         776


         785


         729


         719


         716


         805


         761


         754


         825


         755


         724


         742


         766


         763


         743


         823


         889


         851


         825


         873


         837


         790


         813


         822


         869


         871


         824


         825


         893


         859


         881


         853


         810


         824


         835


         835


         851


         843


         806


         746


         730


         716


         753


         885


         886


         829


         795


         816


         849


         831


         870


         854


         808


         754


         783


         820


         740


         770


         787


         830


         858


         820


         805


         820


         847


         834


         855


         862


         837


         841


         824


         799


         751


         770


         773


         774


         865


         1019


         1005


         1028


         993


         939


         900


         897


         873


         829


         836


         875


         884


         916


         937


         892


         829


         812


         825


         801


         824


         1010


         924


         905


         877


         865


         968


         934


         843


         862


         846


         855


         847


         848


         825


         821


         821


         805


         814


         879


         847


         814


         766


         853


         850


         826


         780


         831


         795


         874


         845


         814


         850


         895


         886


         892


         843


         800


         819


         836


         833


         786


         832


         880


         863


         828


         836


         887


         918


         19 2 67 161 837 380.5 385.9 314.9 86


         825


         800


         745


         749


         819


         856


         818


         800


         816


         796


         747


         716


         674


         702


         776


         788


         724


         740


         768


         751


         715


         712


         722


         717


         721


         717


         747


         793


         745


         743


         776


         755


         724


         740


         750


         736


         740


         756


         761


         727


         729


         741


         764


         733


         761


         798


         765


         730


         726


         761


         779


         737


         713


         762


         781


         757


         739


         726


         737


         740


         728


         706


         720


         736


         754


         752


         766


         752


         743


         708


         717


         717


         723


         714


         718


         770


         797


         774


         774


         806


         782


         740


         734


         740


         736


         723


         751


         774


         740


         720


         720


         740


         715


         705


         728


         742


         725


         712


         753


         765


         728


         721


         743


         712


         700


         704


         734


         746


         703


         708


         727


         736


         702


         698


         730


         728


         700


         701


         731


         720


         704


         709


         730


         730


         698


         712


         716


         660


         643


         648


         656


         667


         689


         844


         881


         848


         853


         832


         794


         761


         753


         719


         721


         762


         788


         806


         830


         776


         734


         730


         746


         790


         785


         766


         771


         795


         769


         771


         735


         745


         790


         832


         823


         748


         746


         786


         788


         779


         756


         772


         761


         785


         755


         765


         795


         806


         798


         759


         793


         805


         777


         749


         774


         800


         797


         762


         773


         777


         727


         735


         772


         773


         732


         783


         810


         828


         745


         738


         735


         726


         734


         757


         756


         761


         750


         739


         755


         751


         729


         750


         760


         742


         733


         803


         829


         764


         753


         773


         756


         736


         730


         742


         758


         756


         759


         764


         777


         728


         757


         771


         759


         737


         767


         784


         765


         786


         801


         750


         744


         798


         762


         733


         760


         778


         750


         743


         774


         779


         747


         794


         780


         752


         784


         799


         752


         733


         766


         769


         727


         734


         757


         726


         713


         739


         764


         751


         712


         713


         745


         755


         717


         713


         753


         760


         736


         761


         776


         765


         733


         742


         777


         758


         714


         732


         750


         736


         724


         720


         747


         784


         763


         732


         738


         737


         723


         706


         720


         750


         753


         722


         723


         730


         733


         712


         712


         719


         733


         704


         701


         743


         765


         744


         725


         735


         725


         747


         703


         687


         686


         651


         650


         670


         709


         721


         775


         748


         730


         727


         769


         781


         750


         723


         736


         762


         740


         766


         789


         752


         726


         747


         797


         761


         746


         778


         760


         747


         777


         784


         808


         769


         773


         753


         737


         747


         775


         761


         739


         743


         760


         737


         714


         724


         739


         725


         707


         704


         740


         773


         727


         743


         761


         825


         742


         736


         756


         712


         716


         746


         737


         720


         761


         785


         744


         716


         725


         755


         728


         700


         704


         717


         740


         716


         732


         763


         756


         746


         746


         757


         750


         721


         721


         735


         769


         780


         794


         802


         815


         749


         746


         783


         791


         745


         760


         796


         761


         745


         766


         788


         742


         735


         743


         784


         750


         735


         775


         781


         751


         742



         [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-680599)
485. ![](https://secure.gravatar.com/avatar/372dea4d936d347c675afc722c5b487a59ae007b3acc2cbbad57c2db7d11173b?s=40&d=mm&r=g)



     sukhAugust 19, 2022 at 9:38 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-681621 "Direct link to this comment")





     hello James Carmichael,



     I put long data in this panel, looks do not nice. I apologies for this. will take care for future.



     further I studied numpy array now and understood.



     My query is, my output result is not 0 and 1 like your given programm. if I have output variable like 90 ,110, 112, ……..and i want to trained my model by giving output . and later want to incash the output. would you suggest which model is ok for this type of programm



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-681621)

486. ![](https://secure.gravatar.com/avatar/947ec4d1fb73aa21e61c2d2ca603deb9e1e1c3bb9dddcbe3253b9a49f33b053c?s=40&d=mm&r=g)



     J JaraOctober 9, 2022 at 4:58 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-682471 "Direct link to this comment")





     This is a binary classifier. How to create a classifier for data with several classes?



     Obviously, I could use one-hot encoding for the classes, and create as many binary classifiers as there are classes, but is there any better alternative?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-682471)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelOctober 10, 2022 at 11:09 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-682475 "Direct link to this comment")





       Hi J Jara…The following resource may be of interest:



       [https://machinelearningmastery.com/multi-label-classification-with-deep-learning/](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-682475)
487. ![](https://secure.gravatar.com/avatar/b3170ac3537be6a7af54d34726feb21f21dd3c28de16e16cfea7630ae7941eab?s=40&d=mm&r=g)



     ElNovember 25, 2022 at 12:47 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-686636 "Direct link to this comment")





     Hello


     I can’t download the dataset, its a lot of numbers, but I didn’t understand how can I download them.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-686636)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelNovember 25, 2022 at 9:18 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-686690 "Direct link to this comment")





       Hi El…Please clarify what you have done to download the dataset so that we may better assist you.



       The following link may link be helpful:



       [https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-686690)
488. ![](https://secure.gravatar.com/avatar/66fa18cbb379776b9c2c2b3e039d2a96470cb10b1e1dc95c617e84534103faf4?s=40&d=mm&r=g)



     suraDecember 10, 2022 at 7:50 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-687847 "Direct link to this comment")





     HI



     I use keras model conv1d for raw dataset X\_train= (142315, 23)


     Y\_train = (142315,)


     my code



     n\_timesteps = X\_train.shape\[1\] #23



     input\_layer = tensorflow.keras.layers.Input(shape=(n\_timesteps,1))


     conv\_layer1 = tensorflow.keras.layers.Conv1D(filters=5,


     kernel\_size=7,


     activation=”relu”)(input\_layer)


     max\_pool1 = tensorflow.keras.layers.MaxPooling1D(pool\_size=2, strides=5)(conv\_layer1)



     conv\_layer2 = tensorflow.keras.layers.Conv1D(filters=3,


     kernel\_size=3,


     activation=”relu”)(max\_pool1)


     flatten\_layer = tensorflow.keras.layers.Flatten()(conv\_layer2)


     dense\_layer = tensorflow.keras.layers.Dense(15, activation=”relu”)(flatten\_layer)


     output\_layer = tensorflow.keras.layers.Dense(6, activation=”softmax”)(dense\_layer)



     model = tensorflow.keras.Model(inputs=input\_layer, outputs=output\_layer)


     \# Prints a string summary of the network.


     model.summary()



     and after that i use optimization technological for hyperprameters and when # Returning the details of the best solution. print this error can helpe me?????



     error



     5121 # Use logits whenever they are available. `softmax` and `sigmoid`



     ValueError: Shapes (142315,) and (142315, 2) are incompatible



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-687847)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelDecember 11, 2022 at 9:35 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-687859 "Direct link to this comment")





       Hi sura…Thanks for asking.



       I’m eager to help, but I just don’t have the capacity to debug code for you.



       I am happy to make some suggestions:



       Consider aggressively cutting the code back to the minimum required. This will help you isolate the problem and focus on it.


       Consider cutting the problem back to just one or a few simple examples.


       Consider finding other similar code examples that do work and slowly modify them to meet your needs. This might expose your misstep.


       Consider posting your question and code to StackOverflow.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-687859)
489. ![](https://secure.gravatar.com/avatar/3e703f3fb8ad6abb15990a2a176042b37ce8f57c6ecedb03b2e3add2fda3688a?s=40&d=mm&r=g)



     NiallJanuary 5, 2023 at 3:45 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688154 "Direct link to this comment")





     Accuracy : 86% if I run preprocessing transformation with scaler on the dataset and use full dataset for train/prediction.


     Accuracy : 84% on train and 81% on test using train:test split (only gets above 77 for me with with scaler on data input).



     Great article, clear concise explanation of every line of code and found the extension tips at end of article really helpful and you link to a tutorial guide for each extension suggestion. Love the comprehensive approach taken on this site.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688154)

490. ![](https://secure.gravatar.com/avatar/448b4f95d3c828ee3a8065d34f74f201ff74eb8677891f133fbd7a715f9df90a?s=40&d=mm&r=g)



     Jun HoJanuary 19, 2023 at 5:24 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688425 "Direct link to this comment")





     Hi Jason, may I know what is this type of Neural Network? is a Feedforward, Multilayer Perceptron or else? I feel like it could be Feedforward.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688425)




     - ![Adrian Tam](https://machinelearningmastery.com/wp-content/uploads/2024/04/adrian-e1713809353338-150x150.jpeg)



       Adrian TamJanuary 20, 2023 at 6:37 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688434 "Direct link to this comment")





       This is multilayer perceptron network. But also feedforward network because it is always moving in the forward direction. Sometimes, we use different names to mean the same thing.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688434)
491. ![](https://secure.gravatar.com/avatar/0d1cf44aa6d4e47bcb9d909247e126ddf61a5552ed4f6c72193c0d1556f1a0c8?s=40&d=mm&r=g)



     AbdullahFebruary 22, 2023 at 6:10 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688830 "Direct link to this comment")





     In “Load data” you should import the “loadtxt” from “numpy”


     Because beginners like me are use to run every piece of code 1 by 1.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688830)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelFebruary 23, 2023 at 8:24 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688840 "Direct link to this comment")





       Thank you for your feedback and suggestions Abdullah!



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-688840)
492. ![](https://secure.gravatar.com/avatar/1d01869d669af9ab1c728747f22c6721429650689eebe9a71f412a21eeb55325?s=40&d=mm&r=g)



     DEEP HAZRAAugust 14, 2023 at 11:11 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-696264 "Direct link to this comment")





     thanks for knowledge sharing.



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-696264)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelAugust 15, 2023 at 10:20 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-696280 "Direct link to this comment")





       Thank you for your feedback and support Deep Hazra! We appreciate it.



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-696280)
493. ![](https://secure.gravatar.com/avatar/28dd6fbdb69dc7fc48c645ea06d269b4d2dc2d8960924cb5ba40414c745228f0?s=40&d=mm&r=g)



     Sharon ManoSeptember 12, 2023 at 12:34 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-697892 "Direct link to this comment")





     Hi Jason,



     It is a great tutorial. I appreciate the way you had put it together.


     Do you have a post on how to couple the trained network to an optimization algorithm to use the network to find the input parameter that results in maximized output value?



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-697892)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelSeptember 12, 2023 at 10:32 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-697921 "Direct link to this comment")





       Hi Sharon…The following course may be of interest to you:



       [https://machinelearningmastery.com/optimization-for-machine-learning-crash-course/](https://machinelearningmastery.com/optimization-for-machine-learning-crash-course/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-697921)
494. ![](https://secure.gravatar.com/avatar/971a6499d7483dd22aacece2fb7dbcbb912174823a3325d91fc6d4bcdfae9470?s=40&d=mm&r=g)



     AlexSeptember 12, 2023 at 10:42 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-697926 "Direct link to this comment")





     I read the publication by Smith, 1988, titled ‘Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,’ where ‘The diabetes pedigree function’ is used as part of the neural network training. Can you explain the relationship of this function in training deep learning models using Keras?”



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-697926)




     - ![](https://secure.gravatar.com/avatar/9489ebe21e231593130db8f6494ec78291a6c9dee89713b86d2001f54ee1698a?s=40&d=mm&r=g)



       James CarmichaelSeptember 14, 2023 at 9:30 am[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-698056 "Direct link to this comment")





       Hi Alex…That is a great question! The following resource may be of interest:



       [https://www.analyticsvidhya.com/blog/2021/07/diabetes-prediction-with-pycaret/](https://www.analyticsvidhya.com/blog/2021/07/diabetes-prediction-with-pycaret/)



       [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-698056)
495. ![](https://secure.gravatar.com/avatar/76099abe173dff3e436394a0ce93cbd6992a5c5c1ef40c582a011b529130b1f5?s=40&d=mm&r=g)



     RobFebruary 10, 2024 at 4:38 pm[#](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-709090 "Direct link to this comment")





     Hi



     I’ve been getting a error when importing from tensorflow.keras.model


     from tensorflow.keras.model import Sequential


     gives me the error ‘No module named ”tensorflow.keras.model”’



     I’ve had to change the imports to:



     from keras.models import Sequential


     from keras.layers import Dense



     but now not sure if what I’m dong is equivalent, I should note that I am not at the end of the tutorial yet.



     I have installed tensorflow 2.15 and keras 2.15. Maybe this is a version mismatch? I tried it with 2.12,2.12 but had the same problem, couldn’t go back any further without downgrading pip



     [Reply](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/#comment-709090)


### Leave a Reply [Click here to cancel reply.](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/\#respond)

Comment \*

Name (required)

Email (will not be published) (required)

Δ