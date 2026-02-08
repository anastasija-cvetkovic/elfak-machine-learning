![(Image by Author)](https://towardsdatascience.com/wp-content/uploads/2020/09/1kyYft8Tq0xEXquFgT_k1sg.png)

(Image by [Author](https://medium.com/@beginningofthefuture))

### ULTIMATE DATA SCIENCE GUIDE

Data is a collection of facts and figures, observations, or descriptions of things in an unorganized or organized form. Data can exist as images, words, numbers, characters, videos, audios, and etcetera.

### What is data preprocessing

To analyze our data and extract the insights out of it, it is necessary to process the data before we start building up our machine learning model i.e. we need to convert our data in the form which our model can understand. Since the machines cannot understand data in the form of images, audios, etc.

> **Data is processed in the form (an efficient format) that it can be easily interpreted by the algorithm and produce the required output accurately.**

The data we use in the real world is not perfect and it is incomplete, inconsistent (with outliers and noisy values), and in an unstructured form. Preprocessing the raw data helps to organize, scaling, clean (remove outliers), standardize i.e. simplifying it to feed the data to the machine learning algorithm.

**The process of data preprocessing involves a few steps:**

-   **Data cleaning:** the data we use may have some missing points (like rows or columns which does not contain any values) or have noisy data (irrelevant data that is difficult to interpret by the machine). To solve the above problems we can delete the empty rows and columns or fill them with some other values and we can use methods like regression and clustering for noisy data.
-   **Data transformation:** this the process of transforming the raw data into the format that is ready to suitable for the model. It may include steps like- categorical encoding, scaling, normalization, standardization, etc.
-   **Data reduction:** this helps to reduce the size of the data we are working on (for easy analysis) while maintaining the integrity of the original data.

### Scikit-learn library for data preprocessing

[Scikit-learn](https://scikit-learn.org/) is a popular machine learning library available as an open-source. This library provides us various essential tools including algorithms for random forests, classification, regression, and of course for **data preprocessing as well.** This library is built on the top of NumPy and SciPy and it is easy to learn and understand.

We can use the following code to import the library in the workspace:

```
import sklearn
```

For including the features for preprocessing we can use the following code:

```
from sklearn import preprocessing
```

> In this article, we will be focussing on some essential data preprocessing features like **standardization, normalization, categorical encoding, discretization, imputation of missing values, generating polynomial features,** and **custom transformers.**

So, now let’s get started with these functions!

### Standardization

Standardization is a technique used to scale the data such that the mean of the data becomes zero and the standard deviation becomes one. Here the values are not restricted to a particular range. We can use standardization when features of input data set have large differences between their ranges.

![(Image by Author) The formula for standardization of data](https://towardsdatascience.com/wp-content/uploads/2020/09/1OUcXeN0RRE7n0lg6O8BRcw.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) The formula for standardization of data

Let us consider the following example:

```
from sklearn import preprocessing
import numpy as np
x = np.array([[1, 2, 3],
[ 4,  5,  6],
[ 7,  8, 9]])
y_scaled = preprocessing.scale(x)
print(y_scaled)
```

Here we have an input array of dimension 3×3 with its values ranging from one to nine. Using the `scale`function available in the `preprocessing`we can quickly scale our data.

![(Image by Author) Scaled data](https://towardsdatascience.com/wp-content/uploads/2020/09/1hyGdFujQmf4kgSwsc2Yh-w.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Scaled data

There is another function available in this library `StandardScaler`, this helps us to compute mean and standard deviation to the training set of data and reapplying the same transformation to the training dataset by implementing the `Transformer API` .

If we want to scale our features in a given range we can use the `MinMaxScaler`(using parameter `feature_range=(min,max)`) or `MinAbsScaler`(the difference is that the maximum absolute value of each feature is scaled to unit size in `MinAbsScaler`)

```
from sklearn.preprocessing import MinMaxScaler
import numpy as np
x = MinMaxScaler(feature_range=(0,8))
y = np.array([[1, 2, 3],
[ 4,  -5,  -6],
[ 7,  8, 9]])
scale = x.fit_transform(y)
scale
```

Here the values of an array of dimension 3×3 are scaled in a given range of `(0,8)`and we have used the `.fit_transform()`function which will help us to apply the same transformation to another dataset later.

![(Image by Author) Scaled data in a specified range](https://towardsdatascience.com/wp-content/uploads/2020/09/1cwI7Uln-QyZrgEeLWYUdpQ.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Scaled data in a specified range

### Normalization

Normalization is the process where the values are scaled in a range of **\-1,1** i.e. converting the values to a common scale. This ensures that the large values in the data set do not influence the learning process and have a similar impact on the model’s learning process. Normalization can be used when we want to quantify the similarity of any pair of samples such as dot-product.

```
from sklearn import preprocessing
import numpy as np
X = [[1,2,3],
[4,-5,-6],
[7,8,9]]
y = preprocessing.normalize(X)
y
```

![(Image by Author) Normalized data](https://towardsdatascience.com/wp-content/uploads/2020/09/1pQZ1Wj01cDHKrRcIQXPwgg.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Normalized data

This module also provides us an alternative for `Transformer API`, by using the `Normalizer` function which implements the same operation.

### Encoding categorical features

Many times the data we use may not have the features values in a continuous form, but instead the forms of categories with text labels. To get this data processed by the machine learning model, it is necessary for converting these categorical features into a machine-understandable form.

There are two functions available in this module through which we can encode our categorical features:

-   **OrdinalEncoder:** this is to convert categorical features to integer values such that the function converts each categorical feature to one new feature of integers (0 to n\_categories – 1).

```
import sklearn.preprocessing
import numpy as np
enc = preprocessing.OrdinalEncoder()
X = [['a','b','c','d'], ['e', 'f', 'g', 'h'],['i','j','k','l']]
enc.fit(X)
enc.transform([['a', 'f', 'g','l']])
```

Here, three categories are encoded as `0,1,2` and the output result for the above input is:

![(Image by Author) Encoded data](https://towardsdatascience.com/wp-content/uploads/2020/09/1Zb0cv_TaR4ETp2kWzDXRRQ.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Encoded data

-   **OneHotEncode:** this encoder function transforms each categorical feature with `n_categories` possible values into `n_categories` binary features, with one of them 1, and all others 0. Check the following example for a better understanding.

```
import sklearn.preprocessing
import numpy as np
enc = preprocessing.OneHotEncoder()
X = [['a','b','c','d'], ['e', 'f', 'g', 'h'],['i','j','k','l']]
enc.fit(X)
enc.transform([['a', 'f', 'g','l']]).toarray().reshape(4,3)
```

![(Image by Author) Encoded data](https://towardsdatascience.com/wp-content/uploads/2020/09/1EIVGDVhfnyPZIyw44LYnFg.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Encoded data

### Discretization

The process of discretization helps us to separate the continuous features of data into discrete values (also known as binning or quantization). This is similar to creating a histogram using continuous data (where discretization focuses on assigning feature values to these bins). Discretization can help us introduce non-linearity in linear models in some cases.

```
import sklearn.preprocessing 
import numpy as np
X = np.array([[ 1,2,3],
              [-4,-5,6],
              [7,8,9]])
dis = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal')
dis.fit_transform(X)
```

Using the `KBinsDiscretizer()`, the function discretizes the features into `k` bins. By default, the output is one-hot encoded, which we can change with the `encode` parameter.

![(Image by Author) Data discretization](https://towardsdatascience.com/wp-content/uploads/2020/09/1vAhxv4OnZy5WQoAjo5kDMg.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Data discretization

### Imputation of missing values

This process is used to process the missing values in the data (NaNs, blanks, etcetera) by assigning a value to them (imputing- based on the known part of the dataset) so that the data can be processed by the model. Let’s understand this with an example:

```
from sklearn.impute import SimpleImputer
import numpy as np
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
X = [[np.nan, 1,2], [3,4, np.nan], [5, np.nan, 6]]
impute.fit_transform(X)
```

Here, we have used `SimpleImputer()` function for imputing the missing values. The parameters used in this function are `missing_values` to specify the missing values to be imputed, `strategy` to specify how we want to impute the value, like in the above example we have used `mean`, this means that the missing values will be replaced by the mean of column values. We can use other parameters for `strategy`, like median, mode, `most_frequent` (based on the frequency of occurrence of particular value in a column), or `constant` (a constant value).

![(Image by Author) Imputing missing values](https://towardsdatascience.com/wp-content/uploads/2020/09/1fq3BGYTQVUpQLCbnE69GUQ.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Imputing missing values

### Generating polynomial features

To get greater accuracy in the results of our machine learning model, sometimes it is good to introduce complexity in the model (by adding non-linearity). We can simply implement this by using the function `PolynomialFeatures()`.

```
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
x = np.array([[1,2],
              [3,4]])
nonl = PolynomialFeatures(2)
nonl.fit_transform(x)
```

![(Image by Author) Generating polynomial features](https://towardsdatascience.com/wp-content/uploads/2020/09/16l-x0q813qmw4d33vxzZCg.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Generating polynomial features

In the example above, we have specified the degree of the non-linear model required to `2` in the `PolynomialFeatures()` function. The feature values of the input array are transformed from \*_(X1, X2) to (1, X1, X2, X1², X1_X2, X2²).\*\*

### Custom transformers

If it is required to transform the entire data using a particular function (existing in python) for any purpose like data processing or cleaning, we can create a custom transformer by implementing the function `FunctionTransformer()` and passing the required function through it.

```
import sklearn.preprocessing 
import numpy as np
transformer = preprocessing.FunctionTransformer(np.log1p, validate=True)
X = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
transformer.transform(X)
```

In this example, we have used the log function to transform our dataset values.

![(Image by Author) Implementing custom transformers](https://towardsdatascience.com/wp-content/uploads/2020/09/1jYOqF8FUDgZyKSxsQbDNTg.png)

(Image by [Author](https://medium.com/@beginningofthefuture)) Implementing custom transformers

### Conclusion

I hope with this article you would have understood the concepts and need of data preprocessing in machine learning models and will be able to apply these concepts in the real data sets.

For a better understanding of these concepts, I will recommend you try implementing these concepts on your once. Keep exploring, and I am sure you will discover new features along the way.

If you have any questions or comments, please post them in the comment section.

> Check out the complete data visualization guide and essential functions of NumPy:
> 
> [**Data Visualization with Python**](https://towardsdatascience.com/data-visualization-with-python-8bc988e44f22)
> 
> [**Cheatsheet for NumPy: Essential and Lesser-Known Functions**](https://towardsdatascience.com/numpy-cheatsheet-for-essential-functions-python-2e7d8618d688)

___

_**Originally published at: [www.patataeater.blogspot.com](http://www.patataeater.blogspot.com/)**_

```
Resources:
https://scikit-learn.org/stable/modules/preprocessing.html#
```