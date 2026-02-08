Data preparation plays an important role in your workflow. You need to transform the data in a way that a computer will be able to work with it.

### Steps in Data Preprocessing

Any database is a collection of data objects. You can also call them data samples, events, observations, or records. However, each of them is described with the help of different characteristics. In data science lingo, they are called attributes or features.

Data preprocessing is a necessary step before [building a model](https://serokell.io/blog/ai-ml-dl-difference#how-can-machines-learn%3F) with these features.

![Graphic showing the hierarchical relationship among the main headings and subheadings of this article](https://miro.medium.com/v2/resize:fit:700/0*6EHvrIjcckwDplb5.jpg)

Image source: Author

It usually happens in stages. Let's have a closer look at each of them.

### Data Quality Assessment

First of all, you need to have a good look at your database and perform a data quality assessment. A random collection of data often has irrelevant bits. Here are some examples.

#### Mismatching in data types

Quite often, you might mix together datasets that use different data formats; hence, the mismatching: integer vs. float or UTF8 vs ASCII.

#### Different dimensions of data arrays

When you aggregate data from different datasets, for example, from five different arrays of data for voice recognition, three fields that are present in one of them can be missing in four other arrays.

#### Mixture of data values

Let's imagine that you have data collected from two independent sources. As a result, the gender field has two different values for women: _woman_ and _female_.

To clean this dataset, you have to make sure that the same name is used as the descriptor within the dataset (it can be _woman_ in our case).

#### Outliers in the dataset

Within 200 years of daily temperature observations for New York, there were several days with [very low temperatures in summer](https://thestarryeye.typepad.com/weather/cold/).

Outliers are very dangerous. They can strongly influence the output of a machine learning model. Usually, the researchers evaluate the outliers to identify whether each particular record is the result of an error in the data collection or a unique phenomenon which should be taken into consideration for data processing.

#### Missing data

You may also notice that some important values are missing. These problems arise due to the human factor, program errors, or other reasons. They will affect the accuracy of the predictions, so before going any further with your database, you need to do data cleaning.

### Why Do We Need to Preprocess Data?

By preprocessing data, we:

-   **Make our database more accurate.** We eliminate the incorrect or missing values that are there as a result of the human factor or bugs.
-   **Boost consistency.** When there are inconsistencies in data or duplicates, it affects the accuracy of the results.
-   **Make the database more complete.** We can fill in the attributes that are missing if needed.
-   **Smooth the data.** This way we make it easier to use and interpret.

### Data Cleaning

The goal of data cleaning is to provide simple, complete, and clear sets of examples for machine learning.

#### Missing data

The situation where you have missing data in your dataset is quite common. In this case, you are looking for additional datasets or collecting more observations.

When you concatenate two or more datasets into one database to get a bigger training set, some data field mismatches are quite common.

When not all the fields are represented in the joined massives, it is better to delete such fields in advance before merging.

**What to do:** If more than 50% of values are missing for any of the database rows or columns, you have to delete the whole row/column unless it is possible to fill in the missing values.

Imagine you make a database of Haskell lovers. The values for the gender column are missing for several records: Nik, Jane, Julia, and Helen. In this case, the researcher can add the missing data based on their conclusions. However, this method has flaws, and the model has to bear the risk of being inaccurate.

#### Noisy data

A large amount of additional meaningless data is called _noise._

![Photo collage of different objects and scenes illustrating the concept of noise in data](https://miro.medium.com/v2/resize:fit:700/0*5tQHP7MJD2M7EQeP.jpg)

Image source: Author

This can be:

-   duplicates or semi-duplicates of the data records
-   data segments, which have no value for a particular research study
-   unnecessary information fields for each of the variables

For example, suppose you need to know whether the person speaks English or not. But you have a whole set of features, including the color of their eyes, shoe size, pulse and blood pressure, etc.

You can apply one of the following methods to solve this problem:

-   **Binning**. Use _[binning](https://en.wikipedia.org/wiki/Data_binning)_ if you have a pool of sorted data. Divide all the data into smaller segments of the same size and apply your dataset preparation methods separately on each segment. For example, you can bin the values for Age into categories such as 21–35, 36–59, and 60–79.
-   **Regression.** Regression analysis helps to decide what variables do indeed have an impact. Apply regression analysis to smooth large volumes of data. This will allow you to only work with the key features instead of trying to analyze an overwhelming number of variables. In our [post about regression](https://serokell.io/blog/regression-analysis-overview), you can learn more about how to conduct a regression analysis step-by-step.
-   **Clustering.** Finally, you can apply clustering algorithms to group the data. Here you need to be careful with the [outliers](https://datascience.stackexchange.com/questions/63695/how-to-handle-outliers-for-clustering-algorithms).

The outliers are the singular data points dissimilar to the rest of the domain.

![A single white penguin among a group of black penguins](https://miro.medium.com/v2/resize:fit:700/0*jLY34NvKbm6iEF1k.jpg)

Image source: Author

It's important not to substitute the outliers by taking them as noise. For example, we are building an algorithm that sorts out different sorts of apples. We can encounter two types of outliers in our dataset:

-   The images contain exotic fruits like pineapples and kiwi. They can be found in your data due to a sampling mistake and represent noise in your dataset.
-   There also can be photos of some weird apples, for example, those that have a strange shape. When our goal is to teach the machine to recognize the apple sorts, deviation from groups is important. Such outliers will help to teach the ML model to recognize special characters and increase the accuracy of the forecast.

When we are not talking about obvious things like apples and pineapples, it is quite complicated to decide whether the item is important or just noise. Here, the expertise of the data scientist has a great influence on the success of ML modeling.

### Data Transformation

In fact, by cleaning and smoothing the data, we have already performed data modification. However, by [data transformation](https://www.geeksforgeeks.org/data-transformation-in-data-mining/?ref=rp), we understand the methods of turning the data into an appropriate format for the computer to learn from.

**Example**: For research about smog around the globe, you have data about wind speeds. However, the data got mixed, and we have three variants of figures: meters per second, miles per second, and kilometers per hour. We need to transform these data to the same scale for ML modeling.

Here are the techniques for data transformation or data scaling:

#### Aggregation

In the case of [data aggregation](https://searchsqlserver.techtarget.com/definition/data-aggregation?_ga=2.169373090.2026354342.1601061799-1619241326.1601061799), the data is pooled together and presented in a unified format for data analysis.

![Photos of a group of cats, a group of dogs, and a group of monkeys](https://miro.medium.com/v2/resize:fit:700/0*6r7nrQZBbkH9noVg.jpg)

Working with a large amount of high-quality data allows for getting more reliable results from the ML model.

If we want to build a neural network algorithm that simulates the style of Vincent Van Gogh, we need to provide as many paintings by this famous artist as we can to provide enough material for training. The images need to have the same digital format, and we will use data transformation techniques to achieve that.

#### Normalization

[Normalization](https://en.wikipedia.org/wiki/Normalization_(statistics)) helps you to scale the data within a range to avoid building incorrect ML models while training and/or executing data analysis. If the data range is very wide, it will be hard to compare the figures. With various normalization techniques, you can transform the original data linearly, [perform decimal scaling or Z-score normalization](https://www.geeksforgeeks.org/data-transformation-in-data-mining/?ref=rp).

For example, to compare the population growth of city X (one+ million citizens) to one thousand new citizens in city Y, we need to normalize these figures.

![Populations of six cities represented by circles of different sizes](https://miro.medium.com/v2/resize:fit:700/0*N5fFPourF98qeGod.jpg)

Image source: Author

#### Feature selection

Feature selection is the selection of variables in data that are the best predictors for the variable we want to predict.

![Feature selection techniques broken down in to categories of unsupervised vs. supervised.](https://miro.medium.com/v2/resize:fit:700/0*2sgUUGruWNxbknFe.jpg)

Image source: Author

If there are a lot of features, then the classifier operation time increases. In addition, prediction accuracy often decreases, especially if there are a lot of garbage features in the data (that are not correlated with the target variable). In the Machine Learning Mastery blog, you can learn [how to perform feature selection](https://machinelearningmastery.com/perform-feature-selection-machine-learning-data-weka/) for your ML database.

#### Discretization

During [discretization](https://www.computer.org/csdl/proceedings-article/icdm/2007/30180183/12OmNqJ8tw1), a programmer transforms the data into sets of small intervals. For example, putting people in categories _young_, _middle age_, and _senior_ rather than working with continuous age values. Discretization helps to improve efficiency.

#### Concept hierarchy generation

If you use the [concept hierarchy generation method](https://www.sciencedirect.com/topics/computer-science/concept-hierarchy#:~:text=Concept%20hierarchy%20generation%20based%20on,hierarchical%20ordering%20among%20the%20attributes.), you can generate a hierarchy between the attributes where it was not specified. For example, if you have the location information that includes a street, city, province, and country but they have no hierarchical order, this method can help you transform the data.

#### Generalization

With the help of generalization, it is possible to convert low-level data features to high-level data features. For example, house addresses can be generalized to higher-level definitions, such as town or country.

### Data Reduction

When you work with large amounts of data, it becomes harder to come up with reliable solutions. Data reduction can be used to reduce the amount of data and decrease the costs of analysis.

Researchers really need data reduction when working with verbal speech datasets. Massive arrays contain individual features of the speakers, for example, interjections and filling words. In this case, huge databases can be decreased to a representative sampling for the analysis.

Here are a few techniques for data reduction:

#### Attribute feature selection

Techniques for data transformation can also be used for data reduction. If you construct a new feature combining the given features in order to make the data mining process more efficient, it is called an _attribute selection_. For example, the features _male/female_ and _student_ can be constructed into _male student/female student_. This can be useful if we conduct research about how many men and/or women are students, but their study field doesn't interest us.

#### Dimensionality reduction

Datasets that are used to solve real-life tasks have a huge number of features. Computer vision, speech generation, translation, and many other tasks cannot sacrifice the speed of operation for the sake of quality. It's possible to use [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) to cut the number of features used.

![Graphic illustrating the relative number of positions in one, two, and three dimensions](https://miro.medium.com/v2/resize:fit:700/0*lwv8VSdcXm2tGX3L.jpg)

Image source: Author

#### Numerosity reduction

Numerosity reduction is a method of data reduction that replaces the original data by a smaller form of data representation. There are two types of numerosity reduction methods: [parametric and non-parametric](https://www.geeksforgeeks.org/numerosity-reduction-in-data-mining/#:~:text=Numerosity%20Reduction%20is%20a%20data,Parametric%20and%20Non%2DParametric%20methods.).

**Parametric Methods**

Parametric methods use models to represent data. Commonly, [regression](http://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1870-90442016000100031) is used to build such models.

**Non-parametric methods**

These techniques allow for storing reduced representations of the data through [histograms](https://en.wikipedia.org/wiki/Histogram), [data sampling](https://searchbusinessanalytics.techtarget.com/definition/data-sampling?_ga=2.181832228.2026354342.1601061799-1619241326.1601061799), and [data cube aggregation](https://en.wikipedia.org/wiki/Data_cube).

A good resource to explore if you are interested in data reduction techniques is [GeeksforGeeks](https://www.geeksforgeeks.org/data-reduction-in-data-mining/?ref=rp).

### Final Thoughts

Now you know all the steps you need to take to preprocess your data for analysis. Afterward, you can start working on finding the best dataset and choosing the perfect ML algorithm.