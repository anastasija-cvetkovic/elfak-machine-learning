In the era of big data and [artificial intelligence](https://realpython.com/python-ai-neural-network/), [data science](https://realpython.com/tutorials/data-science/) and [machine learning](https://realpython.com/tutorials/machine-learning/) have become essential in many fields of science and technology. A necessary aspect of working with data is the ability to describe, summarize, and represent data visually. **Python statistics libraries** are comprehensive, popular, and widely used tools that will assist you in working with data.

**In this tutorial, you’ll learn:**

-   What **numerical quantities** you can use to describe and summarize your datasets
-   How to **calculate** descriptive statistics in pure Python
-   How to get **descriptive statistics** with available Python libraries
-   How to **visualize** your datasets

## Understanding Descriptive Statistics[](https://realpython.com/python-statistics/#understanding-descriptive-statistics "Permanent link")

**Descriptive statistics** is about describing and summarizing data. It uses two main approaches:

1.  **The quantitative approach** describes and summarizes data numerically.
2.  **The visual approach** illustrates data with charts, plots, histograms, and other graphs.

You can apply descriptive statistics to one or many datasets or [variables](https://realpython.com/python-variables/). When you describe and summarize a single variable, you’re performing **univariate analysis**. When you search for statistical relationships among a pair of variables, you’re doing a **bivariate analysis**. Similarly, a **multivariate analysis** is concerned with multiple variables at once.

### Types of Measures[](https://realpython.com/python-statistics/#types-of-measures "Permanent link")

In this tutorial, you’ll learn about the following types of measures in descriptive statistics:

-   **Central tendency** tells you about the centers of the data. Useful measures include the mean, median, and mode.
-   **Variability** tells you about the spread of the data. Useful measures include variance and standard deviation.
-   **Correlation or joint variability** tells you about the relation between a pair of variables in a dataset. Useful measures include covariance and the [correlation coefficient](https://realpython.com/numpy-scipy-pandas-correlation-python/).

You’ll learn how to understand and calculate these measures with Python.

### Population and Samples[](https://realpython.com/python-statistics/#population-and-samples "Permanent link")

In statistics, the **population** is a set of all elements or items that you’re interested in. Populations are often vast, which makes them inappropriate for collecting and analyzing data. That’s why statisticians usually try to make some conclusions about a population by choosing and examining a representative subset of that population.

This subset of a population is called a **sample**. Ideally, the sample should preserve the essential statistical features of the population to a satisfactory extent. That way, you’ll be able to use the sample to glean conclusions about the population.

### Outliers[](https://realpython.com/python-statistics/#outliers "Permanent link")

An **outlier** is a data point that differs significantly from the majority of the data taken from a sample or population. There are many possible causes of outliers, but here are a few to start you off:

-   **Natural variation** in data
-   **Change** in the behavior of the observed system
-   **Errors** in data collection

Data collection errors are a particularly prominent cause of outliers. For example, the limitations of measurement instruments or procedures can mean that the correct data is simply not obtainable. Other errors can be caused by miscalculations, data contamination, human error, and more.

There isn’t a precise mathematical definition of outliers. You have to rely on experience, knowledge about the subject of interest, and common sense to determine if a data point is an outlier and how to handle it.

## Choosing Python Statistics Libraries[](https://realpython.com/python-statistics/#choosing-python-statistics-libraries "Permanent link")

There are many Python statistics libraries out there for you to work with, but in this tutorial, you’ll be learning about some of the most popular and widely used ones:

-   **Python’s [`statistics`](https://docs.python.org/3/library/statistics.html)** is a built-in Python library for descriptive statistics. You can use it if your datasets are not too large or if you can’t rely on importing other libraries.
    
-   **[NumPy](https://docs.scipy.org/doc/numpy/user/index.html)** is a third-party library for numerical computing, optimized for working with single- and multi-dimensional arrays. Its primary type is the array type called [`ndarray`](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html). This library contains many [routines](https://docs.scipy.org/doc/numpy/reference/routines.statistics.html) for statistical analysis.
    
-   **[SciPy](https://www.scipy.org/getting-started.html)** is a third-party library for scientific computing based on NumPy. It offers additional functionality compared to NumPy, including [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) for statistical analysis.
    
-   **[pandas](https://pandas.pydata.org/pandas-docs/stable/)** is a third-party library for numerical computing based on NumPy. It excels in handling labeled one-dimensional (1D) data with [`Series`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) objects and two-dimensional (2D) data with [`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) objects.
    
-   **[Matplotlib](https://matplotlib.org/)** is a third-party library for data visualization. It works well in combination with NumPy, SciPy, and pandas.
    

Note that, in many cases, `Series` and [`DataFrame`](https://realpython.com/pandas-dataframe/) objects can be used in place of NumPy arrays. Often, you might just pass them to a NumPy or [SciPy](https://realpython.com/python-scipy-cluster-optimize/) statistical function. In addition, you can get the unlabeled data from a `Series` or `DataFrame` as a `np.ndarray` object by calling [`.values`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html) or [`.to_numpy()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html).

## Getting Started With Python Statistics Libraries[](https://realpython.com/python-statistics/#getting-started-with-python-statistics-libraries "Permanent link")

The built-in Python `statistics` library has a relatively small number of the most important statistics functions. The [official documentation](https://docs.python.org/3/library/statistics.html) is a valuable resource to find the details. If you’re limited to pure Python, then the Python `statistics` library might be the right choice.

A good place to start learning about NumPy is the official [User Guide](https://docs.scipy.org/doc/numpy/user/index.html), especially the [quickstart](https://docs.scipy.org/doc/numpy/user/quickstart.html) and [basics](https://docs.scipy.org/doc/numpy/user/basics.html) sections. The [official reference](https://docs.scipy.org/doc/numpy/reference/) can help you refresh your memory on specific NumPy concepts. While you read this tutorial, you might want to check out the [statistics](https://docs.scipy.org/doc/numpy/reference/routines.statistics.html) section and the official [`scipy.stats` reference](https://docs.scipy.org/doc/scipy/reference/stats.html) as well.

If you want to learn pandas, then the [official Getting Started page](https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html) is an excellent place to begin. The [introduction to data structures](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html) can help you learn about the fundamental data types, `Series` and `DataFrame`. Likewise, the excellent [official introductory tutorial](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) aims to give you enough information to start effectively using pandas in practice.

`matplotlib` has a comprehensive [official User’s Guide](https://matplotlib.org/users/index.html) that you can use to dive into the details of using the library. [Anatomy of Matplotlib](https://github.com/matplotlib/AnatomyOfMatplotlib) is an excellent resource for beginners who want to start working with `matplotlib` and its related libraries.

Let’s start using these Python statistics libraries!

## Calculating Descriptive Statistics[](https://realpython.com/python-statistics/#calculating-descriptive-statistics "Permanent link")

Start by importing all the packages you’ll need:

These are all the packages you’ll need for Python statistics calculations. Usually, you won’t use Python’s built-in `math` package, but it’ll be useful in this tutorial. Later, you’ll import `matplotlib.pyplot` for data visualization.

Let’s create some data to work with. You’ll start with Python lists that contain some arbitrary numeric data:

Now you have the lists `x` and `x_with_nan`. They’re almost the same, with the difference that `x_with_nan` contains a `nan` value. It’s important to understand the behavior of the Python statistics routines when they come across a **[not-a-number value (`nan`)](https://en.wikipedia.org/wiki/NaN)**. In data science, missing values are common, and you’ll often replace them with `nan`.

Now, create `np.ndarray` and `pd.Series` objects that correspond to `x` and `x_with_nan`:

You now have two NumPy arrays (`y` and `y_with_nan`) and two pandas `Series` (`z` and `z_with_nan`). All of these are 1D sequences of values.

You can optionally specify a label for each value in `z` and `z_with_nan`.

### Measures of Central Tendency[](https://realpython.com/python-statistics/#measures-of-central-tendency "Permanent link")

The **measures of central tendency** show the central or middle values of datasets. There are several definitions of what’s considered to be the center of a dataset. In this tutorial, you’ll learn how to identify and calculate these measures of central tendency:

-   Mean
-   Weighted mean
-   Geometric mean
-   Harmonic mean
-   Median
-   Mode

#### Mean[](https://realpython.com/python-statistics/#mean "Permanent link")

The **sample mean**, also called the **sample arithmetic mean** or simply the **average**, is the arithmetic average of all the items in a dataset. The mean of a dataset 𝑥 is mathematically expressed as Σᵢ𝑥ᵢ/𝑛, where 𝑖 = 1, 2, …, 𝑛. In other words, it’s the sum of all the elements 𝑥ᵢ divided by the number of items in the dataset 𝑥.

This figure illustrates the mean of a sample with five data points:

[![Python Statistics](https://files.realpython.com/media/py-stats-01.3254dbfe6b9a.png)](https://files.realpython.com/media/py-stats-01.3254dbfe6b9a.png)

The green dots represent the data points 1, 2.5, 4, 8, and 28. The red dashed line is their mean, or (1 + 2.5 + 4 + 8 + 28) / 5 = 8.7.

You can calculate the mean with pure Python using [`sum()`](https://realpython.com/python-sum-function/) and [`len()`](https://realpython.com/len-python-function/), without importing libraries:

Although this is clean and elegant, you can also apply built-in Python statistics functions:

You’ve called the functions [`mean()`](https://docs.python.org/3/library/statistics.html#statistics.mean) and [`fmean()`](https://docs.python.org/3/library/statistics.html#statistics.fmean) from the built-in Python `statistics` library and got the same result as you did with pure Python. `fmean()` is introduced in [Python 3.8](https://realpython.com/python38-new-features/) as a faster alternative to `mean()`. It always returns a floating-point number.

However, if there are `nan` values among your data, then `statistics.mean()` and `statistics.fmean()` will return `nan` as the output:

This result is consistent with the behavior of `sum()`, because `sum(x_with_nan)` also returns `nan`.

If you use NumPy, then you can get the mean with [`np.mean()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html):

In the example above, `mean()` is a function, but you can use the corresponding method [`.mean()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.mean.html) as well:

The function `mean()` and method `.mean()` from NumPy return the same result as `statistics.mean()`. This is also the case when there are `nan` values among your data:

You often don’t need to get a `nan` value as a result. If you prefer to ignore `nan` values, then you can use [`np.nanmean()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html):

`nanmean()` simply ignores all `nan` values. It returns the same value as `mean()` if you were to apply it to the dataset without the `nan` values.

`pd.Series` objects also have the method [`.mean()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mean.html):

As you can see, it’s used similarly as in the case of NumPy. However, `.mean()` from pandas ignores `nan` values by default:

This behavior is the result of the default value of the optional parameter `skipna`. You can change this parameter to modify the behavior.

#### Weighted Mean[](https://realpython.com/python-statistics/#weighted-mean "Permanent link")

The **weighted mean**, also called the **weighted arithmetic mean** or **weighted average**, is a generalization of the arithmetic mean that enables you to define the relative contribution of each data point to the result.

You define one **weight 𝑤ᵢ** for each data point 𝑥ᵢ of the dataset 𝑥, where 𝑖 = 1, 2, …, 𝑛 and 𝑛 is the number of items in 𝑥. Then, you multiply each data point with the corresponding weight, sum all the products, and divide the obtained sum with the sum of weights: Σᵢ(𝑤ᵢ𝑥ᵢ) / Σᵢ𝑤ᵢ.

The weighted mean is very handy when you need the mean of a dataset containing items that occur with given relative frequencies. For example, say that you have a set in which 20% of all items are equal to 2, 50% of the items are equal to 4, and the remaining 30% of the items are equal to 8. You can calculate the mean of such a set like this:

Here, you take the frequencies into account with the weights. With this method, you don’t need to know the total number of items.

You can implement the weighted mean in pure Python by combining `sum()` with either [`range()`](https://realpython.com/courses/python-range-function/) or [`zip()`](https://realpython.com/python-zip-function/):

Again, this is a clean and elegant implementation where you don’t need to import any libraries.

However, if you have large datasets, then NumPy is likely to provide a better solution. You can use [`np.average()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.average.html) to get the weighted mean of NumPy arrays or pandas `Series`:

The result is the same as in the case of the pure Python implementation. You can also use this method on ordinary lists and tuples.

Another solution is to use the element-wise product `w * y` with [`np.sum()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html) or [`.sum()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.sum.html):

That’s it! You’ve calculated the weighted mean.

However, be careful if your dataset contains `nan` values:

In this case, `average()` returns `nan`, which is consistent with `np.mean()`.

#### Harmonic Mean[](https://realpython.com/python-statistics/#harmonic-mean "Permanent link")

The **harmonic mean** is the reciprocal of the mean of the reciprocals of all items in the dataset: 𝑛 / Σᵢ(1/𝑥ᵢ), where 𝑖 = 1, 2, …, 𝑛 and 𝑛 is the number of items in the dataset 𝑥. One variant of the pure Python implementation of the harmonic mean is this:

It’s quite different from the value of the arithmetic mean for the same data `x`, which you calculated to be 8.7.

You can also calculate this measure with [`statistics.harmonic_mean()`](https://docs.python.org/3/library/statistics.html#statistics.harmonic_mean):

The example above shows one implementation of `statistics.harmonic_mean()`. If you have a `nan` value in a dataset, then it’ll return `nan`. If there’s at least one `0`, then it’ll return `0`. If you provide at least one negative number, then you’ll get [`statistics.StatisticsError`](https://docs.python.org/3/library/statistics.html#statistics.StatisticsError):

Keep these three scenarios in mind when you’re using this method!

A third way to calculate the harmonic mean is to use [`scipy.stats.hmean()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hmean.html):

Again, this is a pretty straightforward implementation. However, if your dataset contains `nan`, `0`, a negative number, or anything but positive [numbers](https://realpython.com/python-numbers/), then you’ll get a [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)!

#### Geometric Mean[](https://realpython.com/python-statistics/#geometric-mean "Permanent link")

The **geometric mean** is the 𝑛-th root of the product of all 𝑛 elements 𝑥ᵢ in a dataset 𝑥: ⁿ√(Πᵢ𝑥ᵢ), where 𝑖 = 1, 2, …, 𝑛. The following figure illustrates the arithmetic, harmonic, and geometric means of a dataset:

[![Python Statistics](https://files.realpython.com/media/py-stats-02.ec1ca0f9a9ac.png)](https://files.realpython.com/media/py-stats-02.ec1ca0f9a9ac.png)

Again, the green dots represent the data points 1, 2.5, 4, 8, and 28. The red dashed line is the mean. The blue dashed line is the harmonic mean, and the yellow dashed line is the geometric mean.

You can implement the geometric mean in pure Python like this:

As you can see, the value of the geometric mean, in this case, differs significantly from the values of the arithmetic (8.7) and harmonic (2.76) means for the same dataset `x`.

Python 3.8 introduced [`statistics.geometric_mean()`](https://docs.python.org/3/library/statistics.html#statistics.geometric_mean), which converts all values to floating-point numbers and returns their geometric mean:

You’ve got the same result as in the previous example, but with a minimal rounding error.

If you pass data with `nan` values, then `statistics.geometric_mean()` will behave like most similar functions and return `nan`:

Indeed, this is consistent with the behavior of `statistics.mean()`, `statistics.fmean()`, and `statistics.harmonic_mean()`. If there’s a zero or negative number among your data, then `statistics.geometric_mean()` will raise the `statistics.StatisticsError`.

You can also get the geometric mean with [`scipy.stats.gmean()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html):

You obtained the same result as with the pure Python implementation.

If you have `nan` values in a dataset, then `gmean()` will return `nan`. If there’s at least one `0`, then it’ll return `0.0` and give a warning. If you provide at least one negative number, then you’ll get `nan` and the warning.

#### Median[](https://realpython.com/python-statistics/#median "Permanent link")

The **sample median** is the middle element of a sorted dataset. The dataset can be sorted in increasing or decreasing order. If the number of elements 𝑛 of the dataset is odd, then the median is the value at the middle position: 0.5(𝑛 + 1). If 𝑛 is even, then the median is the arithmetic mean of the two values in the middle, that is, the items at the positions 0.5𝑛 and 0.5𝑛 + 1.

For example, if you have the data points 2, 4, 1, 8, and 9, then the median value is 4, which is in the middle of the sorted dataset (1, 2, 4, 8, 9). If the data points are 2, 4, 1, and 8, then the median is 3, which is the average of the two middle elements of the sorted sequence (2 and 4). The following figure illustrates this:

[![Python Statistics](https://files.realpython.com/media/py-stats-04.f7b39a21dd2d.png)](https://files.realpython.com/media/py-stats-04.f7b39a21dd2d.png)

The data points are the green dots, and the purple lines show the median for each dataset. The median value for the upper dataset (1, 2.5, 4, 8, and 28) is 4. If you remove the outlier 28 from the lower dataset, then the median becomes the arithmetic average between 2.5 and 4, which is 3.25.

The figure below shows both the mean and median of the data points 1, 2.5, 4, 8, and 28:

[![Python Statistics](https://files.realpython.com/media/py-stats-03.33356e86aa97.png)](https://files.realpython.com/media/py-stats-03.33356e86aa97.png)

Again, the mean is the red dashed line, while the median is the purple line.

The main difference between the behavior of the mean and median is related to dataset **outliers** or **extremes**. The mean is heavily affected by outliers, but the median only depends on outliers either slightly or not at all. Consider the following figure:

[![Python Statistics](https://files.realpython.com/media/py-stats-05.b5c3dba0cd5f.png)](https://files.realpython.com/media/py-stats-05.b5c3dba0cd5f.png)

The upper dataset again has the items 1, 2.5, 4, 8, and 28. Its mean is 8.7, and the median is 5, as you saw earlier. The lower dataset shows what’s going on when you move the rightmost point with the value 28:

-   **If you increase its value (move it to the right)**, then the mean will rise, but the median value won’t ever change.
-   **If you decrease its value (move it to the left)**, then the mean will drop, but the median will remain the same until the value of the moving point is greater than or equal to 4.

You can compare the mean and median as one way to detect outliers and asymmetry in your data. Whether the mean value or the median value is more useful to you depends on the context of your particular problem.

Here is one of many possible pure Python implementations of the median:

Two most important steps of this implementation are as follows:

1.  **Sorting** the elements of the dataset
2.  **Finding** the middle element(s) in the sorted dataset

You can get the median with [`statistics.median()`](https://docs.python.org/3/library/statistics.html#statistics.median):

The sorted version of `x` is `[1, 2.5, 4, 8.0, 28.0]`, so the element in the middle is `4`. The sorted version of `x[:-1]`, which is `x` without the last item `28.0`, is `[1, 2.5, 4, 8.0]`. Now, there are two middle elements, `2.5` and `4`. Their average is `3.25`.

[`median_low()`](https://docs.python.org/3/library/statistics.html#statistics.median_low) and [`median_high()`](https://docs.python.org/3/library/statistics.html#statistics.median_high) are two more functions related to the median in the Python `statistics` library. They always return an element from the dataset:

-   **If the number of elements is odd**, then there’s a single middle value, so these functions behave just like `median()`.
-   **If the number of elements is even**, then there are two middle values. In this case, `median_low()` returns the lower and `median_high()` the higher middle value.

You can use these functions just as you’d use `median()`:

Again, the sorted version of `x[:-1]` is `[1, 2.5, 4, 8.0]`. The two elements in the middle are `2.5` (low) and `4` (high).

Unlike most other functions from the Python `statistics` library, `median()`, `median_low()`, and `median_high()` don’t return `nan` when there are `nan` values among the data points:

Beware of this behavior because it might not be what you want!

You can also get the median with [`np.median()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html):

You’ve obtained the same values with `statistics.median()` and `np.median()`.

However, if there’s a `nan` value in your dataset, then `np.median()` issues the [`RuntimeWarning`](https://docs.python.org/3.7/library/exceptions.html#RuntimeWarning) and returns `nan`. If this behavior is not what you want, then you can use [`nanmedian()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmedian.html) to ignore all `nan` values:

The obtained results are the same as with `statistics.median()` and `np.median()` applied to the datasets `x` and `y`.

pandas `Series` objects have the method [`.median()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.median.html) that ignores `nan` values by default:

The behavior of `.median()` is consistent with `.mean()` in pandas. You can change this behavior with the optional parameter `skipna`.

#### Mode[](https://realpython.com/python-statistics/#mode "Permanent link")

The **sample mode** is the value in the dataset that occurs most frequently. If there isn’t a single such value, then the set is **multimodal** since it has multiple modal values. For example, in the set that contains the points 2, 3, 2, 8, and 12, the number 2 is the mode because it occurs twice, unlike the other items that occur only once.

This is how you can get the mode with pure Python:

You use `u.count()` to get the number of occurrences of each item in `u`. The item with the maximal number of occurrences is the mode. Note that you don’t have to use `set(u)`. Instead, you might replace it with just `u` and iterate over the entire list.

You can obtain the mode with [`statistics.mode()`](https://docs.python.org/3/library/statistics.html#statistics.mode) and [`statistics.multimode()`](https://docs.python.org/3/library/statistics.html#statistics.multimode):

As you can see, `mode()` returned a single value, while `multimode()` returned the list that contains the result. This isn’t the only difference between the two functions, though. If there’s more than one modal value, then `mode()` raises `StatisticsError`, while `multimode()` returns the list with all modes:

You should pay special attention to this scenario and be careful when you’re choosing between these two functions.

`statistics.mode()` and `statistics.multimode()` handle `nan` values as regular values and can return `nan` as the modal value:

In the first example above, the number `2` occurs twice and is the modal value. In the second example, `nan` is the modal value since it occurs twice, while the other values occur only once.

You can also get the mode with [`scipy.stats.mode()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html):

This function returns the object with the modal value and the number of times it occurs. If there are multiple modal values in the dataset, then only the **smallest** value is returned.

You can get the mode and its number of occurrences as NumPy arrays with dot notation:

This code uses `.mode` to return the smallest mode (`12`) in the array `v` and `.count` to return the number of times it occurs (`3`). `scipy.stats.mode()` is also flexible with `nan` values. It allows you to define desired behavior with the optional parameter `nan_policy`. This parameter can take on the values `'propagate'`, `'raise'` (an error), or `'omit'`.

pandas `Series` objects have the method [`.mode()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mode.html#pandas.Series.mode) that handles multimodal values well and ignores `nan` values by default:

As you can see, `.mode()` returns a new `pd.Series` that holds all modal values. If you want `.mode()` to take `nan` values into account, then just pass the optional argument `dropna=False`.

### Measures of Variability[](https://realpython.com/python-statistics/#measures-of-variability "Permanent link")

The measures of central tendency aren’t sufficient to describe data. You’ll also need the **measures of variability** that quantify the spread of data points. In this section, you’ll learn how to identify and calculate the following variability measures:

-   Variance
-   Standard deviation
-   Skewness
-   Percentiles
-   Ranges

#### Variance[](https://realpython.com/python-statistics/#variance "Permanent link")

The **sample variance** quantifies the spread of the data. It shows numerically how far the data points are from the mean. You can express the sample variance of the dataset 𝑥 with 𝑛 elements mathematically as 𝑠² = Σᵢ(𝑥ᵢ − mean(𝑥))² / (𝑛 − 1), where 𝑖 = 1, 2, …, 𝑛 and mean(𝑥) is the sample mean of 𝑥. If you want to understand deeper why you divide the sum with 𝑛 − 1 instead of 𝑛, then you can dive deeper into [Bessel’s correction](https://en.wikipedia.org/wiki/Bessel%27s_correction).

The following figure shows you why it’s important to consider the variance when describing datasets:

[![Python Statistics](https://files.realpython.com/media/py-stats-06.2cafb41d561e.png)](https://files.realpython.com/media/py-stats-06.2cafb41d561e.png)

There are two datasets in this figure:

1.  **Green dots:** This dataset has a smaller variance or a smaller average difference from the mean. It also has a smaller range or a smaller difference between the largest and smallest item.
2.  **White dots:** This dataset has a larger variance or a larger average difference from the mean. It also has a bigger range or a bigger difference between the largest and smallest item.

Note that these two datasets have the same mean and median, even though they appear to differ significantly. Neither the mean nor the median can describe this difference. That’s why you need the measures of variability.

Here’s how you can calculate the sample variance with pure Python:

This approach is sufficient and calculates the sample variance well. However, the shorter and more elegant solution is to call the existing function [`statistics.variance()`](https://docs.python.org/3/library/statistics.html#statistics.variance):

You’ve obtained the same result for the variance as above. `variance()` can avoid calculating the mean if you provide the mean explicitly as the second argument: `statistics.variance(x, mean_)`.

If you have `nan` values among your data, then `statistics.variance()` will return `nan`:

This behavior is consistent with `mean()` and most other functions from the Python `statistics` library.

You can also calculate the sample variance with NumPy. You should use the function [`np.var()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html) or the corresponding method [`.var()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.var.html):

It’s very important to specify the parameter `ddof=1`. That’s how you set the [delta degrees of freedom](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)) to `1`. This parameter allows the proper calculation of 𝑠², with (𝑛 − 1) in the denominator instead of 𝑛.

If you have `nan` values in the dataset, then `np.var()` and `.var()` will return `nan`:

This is consistent with `np.mean()` and `np.average()`. If you want to skip `nan` values, then you should use [`np.nanvar()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanvar.html):

`np.nanvar()` ignores `nan` values. It also needs you to specify `ddof=1`.

`pd.Series` objects have the method [`.var()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.var.html) that skips `nan` values by default:

It also has the parameter `ddof`, but its default value is `1`, so you can omit it. If you want a different behavior related to `nan` values, then use the optional parameter `skipna`.

You calculate the **population variance** similarly to the sample variance. However, you have to use 𝑛 in the denominator instead of 𝑛 − 1: Σᵢ(𝑥ᵢ − mean(𝑥))² / 𝑛. In this case, 𝑛 is the number of items in the entire population. You can get the population variance similar to the sample variance, with the following differences:

-   **Replace** `(n - 1)` with `n` in the pure Python implementation.
-   **Use** [`statistics.pvariance()`](https://docs.python.org/3/library/statistics.html#statistics.pvariance) instead of `statistics.variance()`.
-   **Specify** the parameter `ddof=0` if you use NumPy or pandas. In NumPy, you can omit `ddof` because its default value is `0`.

Note that you should always be aware of whether you’re working with a sample or the entire population whenever you’re calculating the variance!

#### Standard Deviation[](https://realpython.com/python-statistics/#standard-deviation "Permanent link")

The **sample standard deviation** is another measure of data spread. It’s connected to the sample variance, as standard deviation, 𝑠, is the positive square root of the sample variance. The standard deviation is often more convenient than the variance because it has the same unit as the data points. Once you get the variance, you can calculate the standard deviation with pure Python:

Although this solution works, you can also use [`statistics.stdev()`](https://docs.python.org/3/library/statistics.html#statistics.stdev):

Of course, the result is the same as before. Like `variance()`, `stdev()` doesn’t calculate the mean if you provide it explicitly as the second argument: `statistics.stdev(x, mean_)`.

You can get the standard deviation with NumPy in almost the same way. You can use the function [`std()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html) and the corresponding method [`.std()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.std.html) to calculate the standard deviation. If there are `nan` values in the dataset, then they’ll return `nan`. To ignore `nan` values, you should use [`np.nanstd()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanstd.html). You use `std()`, `.std()`, and `nanstd()` from NumPy as you would use `var()`, `.var()`, and `nanvar()`:

Don’t forget to set the delta degrees of freedom to `1`!

`pd.Series` objects also have the method [`.std()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.std.html) that skips `nan` by default:

The parameter `ddof` defaults to `1`, so you can omit it. Again, if you want to treat `nan` values differently, then apply the parameter `skipna`.

The **population standard deviation** refers to the entire population. It’s the positive square root of the population variance. You can calculate it just like the sample standard deviation, with the following differences:

-   **Find** the square root of the population variance in the pure Python implementation.
-   **Use** [`statistics.pstdev()`](https://docs.python.org/3/library/statistics.html#statistics.pstdev) instead of `statistics.stdev()`.
-   **Specify** the parameter `ddof=0` if you use NumPy or pandas. In NumPy, you can omit `ddof` because its default value is `0`.

As you can see, you can determine the standard deviation in Python, NumPy, and pandas in almost the same way as you determine the variance. You use different but analogous functions and methods with the same arguments.

#### Skewness[](https://realpython.com/python-statistics/#skewness "Permanent link")

The **sample skewness** measures the asymmetry of a data sample.

There are several mathematical definitions of skewness. One common expression to calculate the skewness of the dataset 𝑥 with 𝑛 elements is (𝑛² / ((𝑛 − 1)(𝑛 − 2))) (Σᵢ(𝑥ᵢ − mean(𝑥))³ / (𝑛𝑠³)). A simpler expression is Σᵢ(𝑥ᵢ − mean(𝑥))³ 𝑛 / ((𝑛 − 1)(𝑛 − 2)𝑠³), where 𝑖 = 1, 2, …, 𝑛 and mean(𝑥) is the sample mean of 𝑥. The skewness defined like this is called the **adjusted Fisher-Pearson standardized moment coefficient**.

The previous figure showed two datasets that were quite symmetrical. In other words, their points had similar distances from the mean. In contrast, the following image illustrates two asymmetrical sets:

[![Python Statistics](https://files.realpython.com/media/py-stats-07.92abf9f362b0.png)](https://files.realpython.com/media/py-stats-07.92abf9f362b0.png)

The first set is represented by the green dots and the second with the white ones. Usually, **negative skewness** values indicate that there’s a dominant tail on the left side, which you can see with the first set. **Positive skewness values** correspond to a longer or fatter tail on the right side, which you can see in the second set. If the skewness is close to 0 (for example, between −0.5 and 0.5), then the dataset is considered quite symmetrical.

Once you’ve calculated the size of your dataset `n`, the sample mean `mean_`, and the standard deviation `std_`, you can get the sample skewness with pure Python:

The skewness is positive, so `x` has a right-side tail.

You can also calculate the sample skewness with [`scipy.stats.skew()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html):

The obtained result is the same as the pure Python implementation. The parameter `bias` is set to `False` to enable the corrections for statistical bias. The optional parameter `nan_policy` can take the values `'propagate'`, `'raise'`, or `'omit'`. It allows you to control how you’ll handle `nan` values.

pandas `Series` objects have the method [`.skew()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.skew.html) that also returns the skewness of a dataset:

Like other methods, `.skew()` ignores `nan` values by default, because of the default value of the optional parameter `skipna`.

#### Percentiles[](https://realpython.com/python-statistics/#percentiles "Permanent link")

The **sample 𝑝 percentile** is the element in the dataset such that 𝑝% of the elements in the dataset are less than or equal to that value. Also, (100 − 𝑝)% of the elements are greater than or equal to that value. If there are two such elements in the dataset, then the sample 𝑝 percentile is their arithmetic mean. Each dataset has three **quartiles**, which are the percentiles that divide the dataset into four parts:

-   **The first quartile** is the sample 25th percentile. It divides roughly 25% of the smallest items from the rest of the dataset.
-   **The second quartile** is the sample 50th percentile or the **median**. Approximately 25% of the items lie between the first and second quartiles and another 25% between the second and third quartiles.
-   **The third quartile** is the sample 75th percentile. It divides roughly 25% of the largest items from the rest of the dataset.

Each part has approximately the same number of items. If you want to divide your data into several intervals, then you can use [`statistics.quantiles()`](https://docs.python.org/3/library/statistics.html#statistics.quantiles):

In this example, `8.0` is the median of `x`, while `0.1` and `21.0` are the sample 25th and 75th percentiles, respectively. The parameter `n` defines the number of resulting equal-probability percentiles, and `method` determines how to calculate them.

You can also use [`np.percentile()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.percentile.html) to determine any sample percentile in your dataset. For example, this is how you can find the 5th and 95th percentiles:

`percentile()` takes several arguments. You have to provide the dataset as the first argument and the percentile value as the second. The dataset can be in the form of a NumPy array, list, tuple, or similar data structure. The percentile can be a number between 0 and 100 like in the example above, but it can also be a sequence of numbers:

This code calculates the 25th, 50th, and 75th percentiles all at once. If the percentile value is a sequence, then `percentile()` returns a NumPy array with the results. The first statement returns the array of quartiles. The second statement returns the median, so you can confirm it’s equal to the 50th percentile, which is `8.0`.

If you want to ignore `nan` values, then use [`np.nanpercentile()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanpercentile.html) instead:

That’s how you can avoid `nan` values.

NumPy also offers you very similar functionality in [`quantile()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.quantile.html) and [`nanquantile()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanquantile.html). If you use them, then you’ll need to provide the quantile values as the numbers between 0 and 1 instead of percentiles:

The results are the same as in the previous examples, but here your arguments are between 0 and 1. In other words, you passed `0.05` instead of `5` and `0.95` instead of `95`.

`pd.Series` objects have the method [`.quantile()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.quantile.html):

`.quantile()` also needs you to provide the quantile value as the argument. This value can be a number between 0 and 1 or a sequence of numbers. In the first case, `.quantile()` returns a scalar. In the second case, it returns a new `Series` holding the results.

#### Ranges[](https://realpython.com/python-statistics/#ranges "Permanent link")

The **range of data** is the difference between the maximum and minimum element in the dataset. You can get it with the function [`np.ptp()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html):

This function returns `nan` if there are `nan` values in your NumPy array. If you use a pandas `Series` object, then it will return a number.

Alternatively, you can use built-in Python, [NumPy](https://realpython.com/numpy-max-maximum/), or pandas functions and methods to calculate the maxima and minima of sequences:

-   [`max()`](https://docs.python.org/3/library/functions.html#max) and [`min()`](https://docs.python.org/3/library/functions.html#min) from the Python standard library
-   [`amax()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html) and [`amin()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html) from NumPy
-   [`nanmax()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmax.html) and [`nanmin()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmin.html) from NumPy to ignore `nan` values
-   [`.max()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.max.html) and [`.min()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.min.html) from NumPy
-   [`.max()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.max.html) and [`.min()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.min.html) from pandas to ignore `nan` values by default

Here are some examples of how you would use these routines:

That’s how you get the range of data.

The **interquartile range** is the difference between the first and third quartile. Once you calculate the quartiles, you can take their difference:

Note that you access the values in a pandas `Series` object with the labels `0.75` and `0.25`.

### Summary of Descriptive Statistics[](https://realpython.com/python-statistics/#summary-of-descriptive-statistics "Permanent link")

SciPy and pandas offer useful routines to quickly get descriptive statistics with a single function or method call. You can use [scipy.stats.describe()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html) like this:

You have to provide the dataset as the first argument. The argument can be a NumPy array, list, tuple, or similar data structure. You can omit `ddof=1` since it’s the default and only matters when you’re calculating the variance. You can pass `bias=False` to force correcting the skewness and [kurtosis](https://en.wikipedia.org/wiki/Kurtosis) for statistical bias.

`describe()` returns an object that holds the following descriptive statistics:

-   **`nobs`**: the number of observations or elements in your dataset
-   **`minmax`**: the tuple with the minimum and maximum values of your dataset
-   **`mean`**: the mean of your dataset
-   **`variance`**: the variance of your dataset
-   **`skewness`**: the skewness of your dataset
-   **`kurtosis`**: the kurtosis of your dataset

You can access particular values with dot notation:

With SciPy, you’re just one function call away from a descriptive statistics summary for your dataset.

pandas has similar, if not better, functionality. `Series` objects have the method [`.describe()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.describe.html):

It returns a new `Series` that holds the following:

-   **`count`:** the number of elements in your dataset
-   **`mean`:** the mean of your dataset
-   **`std`:** the standard deviation of your dataset
-   **`min` and `max`:** the minimum and maximum values of your dataset
-   **`25%`, `50%`, and `75%`:** the quartiles of your dataset

If you want the resulting `Series` object to contain other percentiles, then you should specify the value of the optional parameter `percentiles`. You can access each item of `result` with its label:

That’s how you can get descriptive statistics of a `Series` object with a single method call using pandas.

### Measures of Correlation Between Pairs of Data[](https://realpython.com/python-statistics/#measures-of-correlation-between-pairs-of-data "Permanent link")

You’ll often need to examine the relationship between the corresponding elements of two variables in a dataset. Say there are two variables, 𝑥 and 𝑦, with an equal number of elements, 𝑛. Let 𝑥₁ from 𝑥 correspond to 𝑦₁ from 𝑦, 𝑥₂ from 𝑥 to 𝑦₂ from 𝑦, and so on. You can then say that there are 𝑛 pairs of corresponding elements: (𝑥₁, 𝑦₁), (𝑥₂, 𝑦₂), and so on.

You’ll see the following **measures of correlation** between pairs of data:

-   **Positive correlation** exists when larger values of 𝑥 correspond to larger values of 𝑦 and vice versa.
-   **Negative correlation** exists when larger values of 𝑥 correspond to smaller values of 𝑦 and vice versa.
-   **Weak or no correlation exists** if there is no such apparent relationship.

The following figure shows examples of negative, weak, and positive correlation:

[![Python Statistics](https://files.realpython.com/media/py-stats-08.5a1e9f3e3aa4.png)](https://files.realpython.com/media/py-stats-08.5a1e9f3e3aa4.png)

The plot on the left with the red dots shows negative correlation. The plot in the middle with the green dots shows weak correlation. Finally, the plot on the right with the blue dots shows positive correlation.

The two statistics that measure the correlation between datasets are **covariance** and the **correlation coefficient**. Let’s define some data to work with these measures. You’ll create two Python lists and use them to get corresponding NumPy arrays and pandas `Series`:

Now that you have the two variables, you can start exploring the relationship between them.

#### Covariance[](https://realpython.com/python-statistics/#covariance "Permanent link")

The **sample covariance** is a measure that quantifies the strength and direction of a relationship between a pair of variables:

-   **If the correlation is positive,** then the covariance is positive, as well. A stronger relationship corresponds to a higher value of the covariance.
-   **If the correlation is negative,** then the covariance is negative, as well. A stronger relationship corresponds to a lower (or higher [absolute](https://realpython.com/python-absolute-value)) value of the covariance.
-   **If the correlation is weak,** then the covariance is close to zero.

The covariance of the variables 𝑥 and 𝑦 is mathematically defined as 𝑠ˣʸ = Σᵢ (𝑥ᵢ − mean(𝑥)) (𝑦ᵢ − mean(𝑦)) / (𝑛 − 1), where 𝑖 = 1, 2, …, 𝑛, mean(𝑥) is the sample mean of 𝑥, and mean(𝑦) is the sample mean of 𝑦. It follows that the covariance of two identical variables is actually the variance: 𝑠ˣˣ = Σᵢ(𝑥ᵢ − mean(𝑥))² / (𝑛 − 1) = (𝑠ˣ)² and 𝑠ʸʸ = Σᵢ(𝑦ᵢ − mean(𝑦))² / (𝑛 − 1) = (𝑠ʸ)².

This is how you can calculate the covariance in pure Python:

First, you have to find the mean of `x` and `y`. Then, you apply the mathematical formula for the covariance.

NumPy has the function [`cov()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html) that returns the **covariance matrix**:

Note that `cov()` has the optional parameters `bias`, which defaults to `False`, and `ddof`, which defaults to `None`. Their default values are suitable for getting the sample covariance matrix. The upper-left element of the covariance matrix is the covariance of `x` and `x`, or the variance of `x`. Similarly, the lower-right element is the covariance of `y` and `y`, or the variance of `y`. You can check to see that this is true:

As you can see, the variances of `x` and `y` are equal to `cov_matrix[0, 0]` and `cov_matrix[1, 1]`, respectively.

The other two elements of the covariance matrix are equal and represent the actual covariance between `x` and `y`:

You’ve obtained the same value of the covariance with `np.cov()` as with pure Python.

pandas `Series` have the method [`.cov()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cov.html) that you can use to calculate the covariance:

Here, you call `.cov()` on one `Series` object and pass the other object as the first argument.

#### Correlation Coefficient[](https://realpython.com/python-statistics/#correlation-coefficient "Permanent link")

The **correlation coefficient**, or **Pearson product-moment correlation coefficient**, is denoted by the symbol 𝑟. The coefficient is another measure of the correlation between data. You can think of it as a standardized covariance. Here are some important facts about it:

-   **The value 𝑟 > 0** indicates positive correlation.
-   **The value 𝑟 < 0** indicates negative correlation.
-   **The value r = 1** is the maximum possible value of 𝑟. It corresponds to a perfect positive linear relationship between variables.
-   **The value r = −1** is the minimum possible value of 𝑟. It corresponds to a perfect negative linear relationship between variables.
-   **The value r ≈ 0**, or when 𝑟 is around zero, means that the correlation between variables is weak.

The mathematical formula for the correlation coefficient is 𝑟 = 𝑠ˣʸ / (𝑠ˣ𝑠ʸ) where 𝑠ˣ and 𝑠ʸ are the standard deviations of 𝑥 and 𝑦 respectively. If you have the means (`mean_x` and `mean_y`) and standard deviations (`std_x`, `std_y`) for the datasets `x` and `y`, as well as their covariance `cov_xy`, then you can calculate the correlation coefficient with pure Python:

You’ve got the variable `r` that represents the correlation coefficient.

`scipy.stats` has the routine [`pearsonr()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) that calculates the correlation coefficient and the [𝑝-value](https://en.wikipedia.org/wiki/P-value):

`pearsonr()` returns a tuple with two numbers. The first one is 𝑟 and the second is the 𝑝-value.

Similar to the case of the covariance matrix, you can apply [`np.corrcoef()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html) with `x_` and `y_` as the arguments and get the **correlation coefficient matrix**:

The upper-left element is the correlation coefficient between `x_` and `x_`. The lower-right element is the correlation coefficient between `y_` and `y_`. Their values are equal to `1.0`. The other two elements are equal and represent the actual correlation coefficient between `x_` and `y_`:

Of course, the result is the same as with pure Python and `pearsonr()`.

You can get the correlation coefficient with [`scipy.stats.linregress()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html):

`linregress()` takes `x_` and `y_`, performs [linear regression](https://realpython.com/linear-regression-in-python/), and returns the results. `slope` and `intercept` define the equation of the regression line, while `rvalue` is the correlation coefficient. To access particular values from the result of `linregress()`, including the correlation coefficient, use dot notation:

That’s how you can perform linear regression and obtain the correlation coefficient.

pandas `Series` have the method [`.corr()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.corr.html) for calculating the correlation coefficient:

You should call `.corr()` on one `Series` object and pass the other object as the first argument.

## Working With 2D Data[](https://realpython.com/python-statistics/#working-with-2d-data "Permanent link")

Statisticians often work with 2D data. Here are some examples of 2D data formats:

-   [Database](https://realpython.com/tutorials/databases/) tables
-   [CSV files](https://realpython.com/python-csv/)
-   [Excel](https://realpython.com/working-with-large-excel-files-in-pandas/), Calc, and Google [spreadsheets](https://realpython.com/openpyxl-excel-spreadsheets-python/)

NumPy and SciPy provide a comprehensive means to work with 2D data. pandas has the class `DataFrame` specifically to handle 2D labeled data.

### Axes[](https://realpython.com/python-statistics/#axes "Permanent link")

Start by creating a 2D NumPy array:

Now you have a 2D dataset, which you’ll use in this section. You can apply Python statistics functions and methods to it just as you would to 1D data:

As you can see, you get statistics (like the mean, median, or variance) across all data in the array `a`. Sometimes, this behavior is what you want, but in some cases, you’ll want these quantities calculated for each row or column of your 2D array.

The functions and methods you’ve used so far have one optional parameter called **`axis`**, which is essential for handling 2D data. `axis` can take on any of the following values:

-   **`axis=None`** says to calculate the statistics across all data in the array. The examples above work like this. This behavior is often the default in NumPy.
-   **`axis=0`** says to calculate the statistics across all rows, that is, for each column of the array. This behavior is often the default for SciPy statistical functions.
-   **`axis=1`** says to calculate the statistics across all columns, that is, for each row of the array.

Let’s see `axis=0` in action with `np.mean()`:

The two statements above return new NumPy arrays with the mean for each column of `a`. In this example, the mean of the first column is `6.2`. The second column has the mean `8.2`, while the third has `1.8`.

If you provide `axis=1` to `mean()`, then you’ll get the results for each row:

As you can see, the first row of `a` has the mean `1.0`, the second `2.0`, and so on.

The parameter `axis` works the same way with other NumPy functions and methods:

You’ve got the medians and sample variations for all columns (`axis=0`) and rows (`axis=1`) of the array `a`.

This is very similar when you work with SciPy statistics functions. But remember that in this case, the default value for `axis` is `0`:

If you omit `axis` or provide `axis=0`, then you’ll get the result across all rows, that is, for each column. For example, the first column of `a` has a geometric mean of `4.0`, and so on.

If you specify `axis=1`, then you’ll get the calculations across all columns, that is for each row:

In this example, the geometric mean of the first row of `a` is `1.0`. For the second row, it’s approximately `1.82`, and so on.

If you want statistics for the entire dataset, then you have to provide `axis=None`:

The geometric mean of all the items in the array `a` is approximately `2.83`.

You can get a Python statistics summary with a single function call for 2D data with [scipy.stats.describe()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html). It works similar to 1D arrays, but you have to be careful with the parameter `axis`:

When you provide `axis=None`, you get the summary across all data. Most results are scalars. If you set `axis=0` or omit it, then the return value is the summary for each column. So, most results are the arrays with the same number of items as the number of columns. If you set `axis=1`, then `describe()` returns the summary for all rows.

You can get a particular value from the summary with dot notation:

That’s how you can see a statistics summary for a 2D array with a single function call.

### DataFrames[](https://realpython.com/python-statistics/#dataframes "Permanent link")

The class `DataFrame` is one of the fundamental pandas data types. It’s very comfortable to work with because it has labels for rows and columns. Use the array `a` and create a `DataFrame`:

In practice, the names of the columns matter and should be descriptive. The names of the rows are sometimes specified automatically as `0`, `1`, and so on. You can specify them explicitly with the parameter `index`, though you’re free to omit `index` if you like.

`DataFrame` methods are very similar to `Series` methods, though the behavior is different. If you call Python statistics methods without arguments, then the `DataFrame` will return the results for each column:

What you get is a new `Series` that holds the results. In this case, the `Series` holds the mean and variance for each column. If you want the results for each row, then just specify the parameter `axis=1`:

The result is a `Series` with the desired quantity for each row. The labels `'first'`, `'second'`, and so on refer to the different rows.

You can isolate each column of a `DataFrame` like this:

Now, you have the column `'A'` in the form of a `Series` object and you can apply the appropriate methods:

That’s how you can obtain the statistics for a single column.

Sometimes, you might want to use a `DataFrame` as a NumPy array and apply some function to it. It’s possible to get all data from a `DataFrame` with `.values` or `.to_numpy()`:

`df.values` and `df.to_numpy()` give you a NumPy array with all items from the `DataFrame` without row and column labels. Note that `df.to_numpy()` is more flexible because you can specify the data type of items and whether you want to use the existing data or copy it.

Like `Series`, `DataFrame` objects have the method [`.describe()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) that returns another `DataFrame` with the statistics summary for all columns:

The summary contains the following results:

-   **`count`:** the number of items in each column
-   **`mean`:** the mean of each column
-   **`std`:** the standard deviation
-   **`min` and `max`:** the minimum and maximum values
-   **`25%`, `50%`, and `75%`:** the percentiles

If you want the resulting `DataFrame` object to contain other percentiles, then you should specify the value of the optional parameter `percentiles`.

You can access each item of the summary like this:

That’s how you can get descriptive Python statistics in one `Series` object with a single pandas method call.

## Visualizing Data[](https://realpython.com/python-statistics/#visualizing-data "Permanent link")

In addition to calculating the numerical quantities like mean, median, or variance, you can use visual methods to present, describe, and summarize data. In this section, you’ll learn how to present your data visually using the following graphs:

-   Box plots
-   Histograms
-   Pie charts
-   Bar charts
-   X-Y plots
-   Heatmaps

`matplotlib.pyplot` is a very convenient and widely-used library, though it’s not the only Python library available for this purpose. You can import it like this:

Now, you have `matplotlib.pyplot` imported and ready for use. The second statement sets the style for your plots by choosing colors, line widths, and other stylistic elements. You’re free to omit these if you’re satisfied with the default style settings.

You’ll use [pseudo-random numbers](https://realpython.com/courses/generating-random-data-python/) to get data to work with. You don’t need knowledge on [random numbers](https://realpython.com/lessons/randomness-modeling-and-simulation/) to be able to understand this section. You just need some arbitrary numbers, and pseudo-random generators are a convenient tool to get them. The module [`np.random`](https://docs.scipy.org/doc/numpy-1.16.0/reference/routines.random.html) generates arrays of pseudo-random numbers:

-   [Normally distributed numbers](https://realpython.com/numpy-random-normal) are generated with [`np.random.randn()`](https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.randn.html).
-   [Uniformly distributed integers](https://en.wikipedia.org/wiki/Discrete_uniform_distribution) are generated with [`np.random.randint()`](https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.randint.html).

NumPy 1.17 introduced another [module](https://numpy.org/devdocs/release/1.17.0-notes.html#new-extensible-numpy-random-module-with-selectable-random-number-generators) for pseudo-random number generation. To learn more about it, check the [official documentation](https://docs.scipy.org/doc/numpy/reference/random/generator.html).

### Box Plots[](https://realpython.com/python-statistics/#box-plots "Permanent link")

The **box plot** is an excellent tool to visually represent descriptive statistics of a given dataset. It can show the range, interquartile range, median, mode, outliers, and all quartiles. First, create some data to represent with a box plot:

The first statement sets the seed of the NumPy random number generator with [`seed()`](https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.seed.html), so you can get the same results each time you run the code. You don’t have to set the seed, but if you don’t specify this value, then you’ll get different results each time.

The other statements generate three NumPy arrays with normally distributed pseudo-random numbers. `x` refers to the array with 1000 items, `y` has 100, and `z` contains 10 items. Now that you have the data to work with, you can apply [`.boxplot()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html) to get the box plot:

The parameters of `.boxplot()` define the following:

-   **`x`** is your data.
-   **`vert`** sets the plot orientation to horizontal when `False`. The default orientation is vertical.
-   **`showmeans`** shows the mean of your data when `True`.
-   **`meanline`** represents the mean as a line when `True`. The default representation is a point.
-   **`labels`:** the labels of your data.
-   **`patch_artist`** determines how to draw the graph.
-   **`medianprops`** denotes the properties of the line representing the median.
-   **`meanprops`** indicates the properties of the line or dot representing the mean.

There are other parameters, but their analysis is beyond the scope of this tutorial.

The code above produces an image like this:

[![Python Statistics](https://files.realpython.com/media/py-stats-09.bbe925f1a3e3.png)](https://files.realpython.com/media/py-stats-09.bbe925f1a3e3.png)

You can see three box plots. Each of them corresponds to a single dataset (`x`, `y`, or `z`) and show the following:

-   **The mean** is the red dashed line.
-   **The median** is the purple line.
-   **The first quartile** is the left edge of the blue rectangle.
-   **The third quartile** is the right edge of the blue rectangle.
-   **The interquartile range** is the length of the blue rectangle.
-   **The range** contains everything from left to right.
-   **The outliers** are the dots to the left and right.

A box plot can show so much information in a single figure!

### Histograms[](https://realpython.com/python-statistics/#histograms "Permanent link")

[Histograms](https://realpython.com/python-histograms/) are particularly useful when there are a large number of unique values in a dataset. The histogram divides the values from a sorted dataset into intervals, also called **bins**. Often, all bins are of equal width, though this doesn’t have to be the case. The values of the lower and upper bounds of a bin are called the **bin edges**.

The **frequency** is a single value that corresponds to each bin. It’s the number of elements of the dataset with the values between the edges of the bin. By convention, all bins but the rightmost one are half-open. They include the values equal to the lower bounds, but exclude the values equal to the upper bounds. The rightmost bin is closed because it includes both bounds. If you divide a dataset with the bin edges 0, 5, 10, and 15, then there are three bins:

1.  **The first and leftmost bin** contains the values greater than or equal to 0 and less than 5.
2.  **The second bin** contains the values greater than or equal to 5 and less than 10.
3.  **The third and rightmost bin** contains the values greater than or equal to 10 and less than or equal to 15.

The function [`np.histogram()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html) is a convenient way to get data for histograms:

It takes the array with your data and the number (or edges) of bins and returns two NumPy arrays:

1.  **`hist`** contains the frequency or the number of items corresponding to each bin.
2.  **`bin_edges`** contains the edges or bounds of the bin.

What `histogram()` calculates, [`.hist()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html) can show graphically:

The first argument of `.hist()` is the sequence with your data. The second argument defines the edges of the bins. The third disables the option to create a histogram with cumulative values. The code above produces a figure like this:

[![Python Statistics](https://files.realpython.com/media/py-stats-10.47c60c3e5c75.png)](https://files.realpython.com/media/py-stats-10.47c60c3e5c75.png)

You can see the bin edges on the horizontal axis and the frequencies on the vertical axis.

It’s possible to get the histogram with the cumulative numbers of items if you provide the argument `cumulative=True` to `.hist()`:

This code yields the following figure:

[![Python Statistics](https://files.realpython.com/media/py-stats-11.2d63bac53eb9.png)](https://files.realpython.com/media/py-stats-11.2d63bac53eb9.png)

It shows the histogram with the cumulative values. The frequency of the first and leftmost bin is the number of items in this bin. The frequency of the second bin is the sum of the numbers of items in the first and second bins. The other bins follow this same pattern. Finally, the frequency of the last and rightmost bin is the total number of items in the dataset (in this case, 1000). You can also directly draw a histogram with [`pd.Series.hist()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.hist.html) using `matplotlib` in the background.

### Pie Charts[](https://realpython.com/python-statistics/#pie-charts "Permanent link")

**Pie charts** represent data with a small number of labels and given relative frequencies. They work well even with the labels that can’t be ordered (like nominal data). A pie chart is a circle divided into multiple slices. Each slice corresponds to a single distinct label from the dataset and has an area proportional to the relative frequency associated with that label.

Let’s define data associated to three labels:

Now, create a pie chart with [`.pie()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.pie.html):

The first argument of `.pie()` is your data, and the second is the sequence of the corresponding labels. `autopct` defines the format of the relative frequencies shown on the figure. You’ll get a figure that looks like this:

[![Python Statistics](https://files.realpython.com/media/py-stats-12.85291860060a.png)](https://files.realpython.com/media/py-stats-12.85291860060a.png)

The pie chart shows `x` as the smallest part of the circle, `y` as the next largest, and then `z` as the largest part. The percentages denote the relative size of each value compared to their sum.

### Bar Charts[](https://realpython.com/python-statistics/#bar-charts "Permanent link")

**Bar charts** also illustrate data that correspond to given labels or discrete numeric values. They can show the pairs of data from two datasets. Items of one set are the **labels**, while the corresponding items of the other are their **frequencies**. Optionally, they can show the errors related to the frequencies, as well.

The bar chart shows parallel rectangles called **bars**. Each bar corresponds to a single label and has a height proportional to the frequency or relative frequency of its label. Let’s generate three datasets, each with 21 items:

You use [`np.arange()`](https://realpython.com/how-to-use-numpy-arange/) to get `x`, or the array of consecutive integers from `0` to `20`. You’ll use this to represent the labels. `y` is an array of uniformly distributed random integers, also between `0` and `20`. This array will represent the frequencies. `err` contains normally distributed floating-point numbers, which are the errors. These values are optional.

You can create a bar chart with [`.bar()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.bar.html) if you want vertical bars or [`.barh()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.barh.html) if you’d like horizontal bars:

This code should produce the following figure:

[![Python Statistics](https://files.realpython.com/media/py-stats-13.86e4d6acf1bd.png)](https://files.realpython.com/media/py-stats-13.86e4d6acf1bd.png)

The heights of the red bars correspond to the frequencies `y`, while the lengths of the black lines show the errors `err`. If you don’t want to include the errors, then omit the parameter `yerr` of `.bar()`.

### X-Y Plots[](https://realpython.com/python-statistics/#x-y-plots "Permanent link")

The **x-y plot** or **scatter plot** represents the pairs of data from two datasets. The horizontal x-axis shows the values from the set `x`, while the vertical y-axis shows the corresponding values from the set `y`. You can optionally include the regression line and the correlation coefficient. Let’s generate two datasets and perform linear regression with `scipy.stats.linregress()`:

The dataset `x` is again the array with the integers from 0 to 20. `y` is calculated as a linear function of `x` distorted with some random noise.

`linregress` returns several values. You’ll need the `slope` and `intercept` of the regression line, as well as the correlation coefficient `r`. Then you can apply [`.plot()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.plot.html) to get the x-y plot:

The result of the code above is this figure:

[![Python Statistics](https://files.realpython.com/media/py-stats-14.33b9d9b32eb4.png)](https://files.realpython.com/media/py-stats-14.33b9d9b32eb4.png)

You can see the data points (x-y pairs) as red squares, as well as the blue regression line.

### Heatmaps[](https://realpython.com/python-statistics/#heatmaps "Permanent link")

A **heatmap** can be used to visually show a matrix. The colors represent the numbers or elements of the matrix. Heatmaps are particularly useful for illustrating the covariance and correlation matrices. You can create the heatmap for a covariance matrix with [`.imshow()`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.imshow.html):

Here, the heatmap contains the labels `'x'` and `'y'` as well as the numbers from the covariance matrix. You’ll get a figure like this:

[![Python Statistics](https://files.realpython.com/media/py-stats-15.432905d1b05a.png)](https://files.realpython.com/media/py-stats-15.432905d1b05a.png)

The yellow field represents the largest element from the matrix `130.34`, while the purple one corresponds to the smallest element `38.5`. The blue squares in between are associated with the value `69.9`.

You can obtain the heatmap for the correlation coefficient matrix following the same logic:

The result is the figure below:

[![Python Statistics](https://files.realpython.com/media/py-stats-16.c0240902890d.png)](https://files.realpython.com/media/py-stats-16.c0240902890d.png)

The yellow color represents the value `1.0`, and the purple color shows `0.99`.

## Conclusion[](https://realpython.com/python-statistics/#conclusion "Permanent link")

You now know the quantities that describe and summarize datasets and how to calculate them in Python. It’s possible to get **descriptive statistics** with pure Python code, but that’s rarely necessary. Usually, you’ll use some of the libraries created especially for this purpose:

-   **Use Python’s `statistics`** for the most important Python statistics functions.
-   **Use NumPy** to handle arrays efficiently.
-   **Use SciPy** for additional Python statistics routines for NumPy arrays.
-   **Use pandas** to work with labeled datasets.
-   **Use Matplotlib** to visualize data with plots, charts, and histograms.

In the era of big data and artificial intelligence, you must know how to calculate descriptive statistics measures. Now you’re ready to dive deeper into the world of [data science](https://realpython.com/tutorials/data-science/) and [machine learning](https://realpython.com/tutorials/machine-learning/)! If you have questions or comments, then please put them in the comments section below.