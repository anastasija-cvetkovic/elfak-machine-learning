In my decade plus as a data scientist, my experience largely agrees with Andrew Ng’s statement, "Applied machine learning is basically feature engineering." From the very start of my career, building credit card fraud models at SAS, most of my value as a data scientist came from my ability to engineer new features and capture both business insights and behavior observed in the data to help the model identify the target. Also aligning with this sentiment is Dr. Pedro Domingos’ statement, "At the end of the day, some machine learning projects succeed and some fail. What makes the difference? Easily the most important factor is the features used."

### Common feature engineering content

Despite both this experience and the statements by leaders in the field, feature engineering content is still lacking. While a search for "feature engineering" on Google returns many pages, the underlying content is very similar and focuses on only a few topics:

-   handling missing values
-   handling outliers
-   binning numeric variables
-   encoding categorical features
-   numerical transformations
-   scaling numerical features
-   extracting parts of a date

### The problem with this content

These are important topics and, in many cases, necessary for the underlying machine learning algorithm to even be able to process the data. However, they are lacking in three primary ways. First, in some cases, they blindly apply the techniques without understanding why. Many blogs discuss how numeric features need to be scaled before modeling. This is not true. Tree-based methods, e.g. XGBoost, LightGBM, etc. are invariant to scaling. [Praveen Thenraj](https://medium.com/@praveenmec67) gave a nice example of this in his [post on Towards Data Science](https://towardsdatascience.com/do-decision-trees-need-feature-scaling-97809eaa60c6).

Second, while the technique is needed, there are large classes of problems where the technique will not be practical and other techniques are needed. An example of this is encoding categorical features. Most discussions of feature engineering explain one-hot or dummy encoding and often explain label encoding. Rarely do they discuss the challenge of high cardinality categorical variables. Most often, I have seen this occur when dealing with ZIP codes or Postcodes, but this can also occur when working with medical data and ICD-9 or ICD-10 diagnosis codes. One-hot encoding ZIP codes could result in over 40,000 new features. This is not a practical solution. Understanding the structure of these codes can allow them to be rolled up to higher levels of aggregation. Even after aggregation, there may be too many categories for either one-hot or label encoding to be useful. Further, some categorical variables do not have an easily identifiable aggregation that can reduce the number of categories needed. Target encoding or leave-one-out encoding can handle these high cardinality categorical features. While some feature engineering blogs discuss these techniques, they often fail to highlight their usefulness in this case or the impracticality of one-hot encoding.

### Feature engineering is more than this.

Finally, and most crucially, feature engineering is far more than this. When people talk about the power of feature engineering or the art of feature engineering, they are not referring to these standard techniques. Instead, it is the work recognizing that ZIP codes can be aggregated to cities, states, DMAs, etc. and performing those aggregations. This engineering can be further performed by combining these aggregations with all purchases in that location to capture a better sense of the local market. Finally, comparisons between an individual’s purchases and the local market overall. This is just the tip of the iceberg and is my favorite part of the data science process.

To be fair, it is hard to show examples of this sort of feature engineering without also discussing an actual modeling problem. But to neglect it altogether when discussing feature engineering may leave a false sense of its scope. In addition, many powerful techniques exist that can be applied across a broad set of problems.

### Feature engineering with time series data

For example, many data science problems have a time series character. This is not to say that the problems are time series problems, rather there are repeated observations that need to be aggregated together to represent the underlying behavior of interest. In building models for customer analytics (e.g. churn, lifetime value), the individual transaction and the way they change over time are important. A feature engineering blog could share code showing how to calculate common features for this as follows.

When working with data about a single customer with multiple observations over time, one of the most common techniques to aggregate this data is through rolling windows followed by an aggregation in pandas. In this case, a weekly aggregation is created and the minimum, mean, maximum and standard deviation are created.

```
weekly_resample = df.rolling('7D')
aggregated_df = weekly_resample.agg(['min', 'mean', 'max', 'std'])
aggregated_df.columns = ['_'.join(col).strip() + '_week' for col in 
                         aggregated_df.columns.values]
```

Note that for this to work, the dataframe `df` needs the date set as its index.

```
df.set_index('date', inplace=True)
```

Using rolling windows, `aggregated_df` contains a record for each transaction with the aggregation of all transactions from the prior seven days. If, instead, a prediction was needed after every week, additional work would be needed to filter out just the last record prior to that week. pandas supports this use case with `resample`. If the following code is run instead, one record per week is returned.

```
weekly_resample = df.resample('7D', 
                              origin=pd.to_datetime('2021-01-03'))
aggregated_df = weekly_resample.agg(['min', 'mean', 'max', 'std'])
aggregated_df.columns = ['_'.join(col).strip() + '_week' for col in
                         aggregated_df.columns.values]
```

Since January 3, 2021 is a Sunday, the aggregation starts on each Sunday and includes all transactions in the rest of the week. If the seven days prior to Sunday was desired, the following change could be made.

```
weekly_resample = df.resample('7D', 
                              origin=pd.to_datetime('2021-01-03'),
                              lavel='right')
```

Once these weekly aggregations are created (of either type), rate of change of these aggregations can be exceedingly powerful. From a physics point of view, this is the velocity of the underlying feature and we can look at the rate of change over multiple time periods.

```
aggregated_df['velocity_1wk'] = (aggregated_df['mean_week'] - 
                              aggregated_df['mean_week'].shift(1))/7 aggregated_df['velocity_4wk'] = (aggregated_df['mean_week'] - 
                             aggregated_df['mean_week'].shift(4))/28
```

This gives us the rate of change of the weekly mean over one week and four weeks. Next, the rate of change of this rate of change (acceleration) can be calculated.

```
aggregated_df['acceleration_1wk'] = (aggregated_df['velocity_1wk'] - 
                           aggregated_df['velocity_1wk'].shift(1))/7
```

### 50+ Free Feature Engineering Tutorials

Techniques like these can be used to build features that can represent the underlying behavior of interest. At [Rasgo](https://www.rasgoml.com/), we wanted to help the data science community by sharing these techniques. To do this, we created a [free repository devoted to feature engineering tutorials](https://hubs.la/Q0100qK30) that contains feature engineering code, including the examples above, as Jupyter notebooks. In addition to feature engineering, there are examples for:

-   Feature Profiling and EDA
-   Data Cleaning
-   Train-Test Splits
-   Feature Importance
-   Feature Selection