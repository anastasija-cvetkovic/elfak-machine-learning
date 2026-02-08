**20 | Chapter 1: Working with Vectors, Matrices, and Arrays in NumPy**

**CHAPTER 2**

### Loading Data

### 2.0 Introduction

The first step in any machine learning endeavor is to get the raw data into our system.

The raw data might be a logfile, dataset file, database, or cloud blob store such as

Amazon S3. Furthermore, often we will want to retrieve data from multiple sources.

The recipes in this chapter look at methods of loading data from a variety of sources,

including CSV files and SQL databases. We also cover methods of generating simula‐

ted data with desirable properties for experimentation. Finally, while there are many

ways to load data in the Python ecosystem, we will focus on using the pandas library’s

extensive set of methods for loading external data, and using scikit-learn—an open

source machine learning library in Python—for generating simulated data.

### 2.1 Loading a Sample Dataset

**Problem**

You want to load a preexisting sample dataset from the scikit-learn library.

**Solution**

scikit-learn comes with a number of popular datasets for you to use:

```
# Load scikit-learn's datasets
from sklearn import datasets
```

```
# Load digits dataset
digits = datasets.load_digits()
```

```
# Create features matrix
```

##### 21

```
features = digits.data
```

```
# Create target vector
target = digits.target
```

```
# View first observation
features[0]
```

```
array([ 0., 0., 5., 13., 9., 1., 0., 0., 0., 0., 13.,
15., 10., 15., 5., 0., 0., 3., 15., 2., 0., 11.,
8., 0., 0., 4., 12., 0., 0., 8., 8., 0., 0.,
5., 8., 0., 0., 9., 8., 0., 0., 4., 11., 0.,
1., 12., 7., 0., 0., 2., 14., 5., 10., 12., 0.,
0., 0., 0., 6., 13., 10., 0., 0., 0.])
```

**Discussion**

Often we do not want to go through the work of loading, transforming, and cleaning

a real-world dataset before we can explore some machine learning algorithm or

method. Luckily, scikit-learn comes with some common datasets we can quickly load.

These datasets are often called “toy” datasets because they are far smaller and cleaner

than a dataset we would see in the real world. Some popular sample datasets in

scikit-learn are:

load_iris

```
Contains 150 observations on the measurements of iris flowers. It is a good
dataset for exploring classification algorithms.
```

load_digits

```
Contains 1,797 observations from images of handwritten digits. It is a good
dataset for teaching image classification.
```

To see more details on any of these datasets, you can print the DESCR attribute:

```
# Load scikit-learn's datasets
from sklearn import datasets
```

```
# Load digits dataset
digits = datasets.load_digits()
```

```
# Print the attribute
print(digits.DESCR)
```

```
.. _digits_dataset:
```

```
Optical recognition of handwritten digits dataset
--------------------------------------------------
```

```
**Data Set Characteristics:**
```

```
:Number of Instances: 1797
```

**22 | Chapter 2: Loading Data**

```
:Number of Attributes: 64
:Attribute Information: 8x8 image of integer pixels in the range 0..16.
:Missing Attribute Values: None
:Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
:Date: July; 1998
```

**See Also**

- •scikit-learn toy datasets
- •The Digit Dataset

### 2.2 Creating a Simulated Dataset

**Problem**

You need to generate a dataset of simulated data.

**Solution**

scikit-learn offers many methods for creating simulated data. Of those, three methods

are particularly useful: make_regression, make_classification, and make_blobs.

When we want a dataset designed to be used with linear regression, make_regression

is a good choice:

```
# Load library
from sklearn.datasets import make_regression
```

```
# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples = 100,
n_features = 3,
n_informative = 3,
n_targets = 1,
noise = 0.0,
coef = True ,
random_state = 1)
```

```
# View feature matrix and target vector
print('Feature Matrix \n ', features[:3])
print('Target Vector \n ', target[:3])
```

```
Feature Matrix
[[ 1.29322588 -0.61736206 -0.11044703]
[-2.793085 0.36633201 1.93752881]
[ 0.80186103 -0.18656977 0.0465673 ]]
Target Vector
[-10.37865986 25.5124503 19.67705609]
```

```
2.2 Creating a Simulated Dataset | 23
```

If we are interested in creating a simulated dataset for classification, we can use

make_classification:

```
# Load library
from sklearn.datasets import make_classification
```

```
# Generate features matrix and target vector
features, target = make_classification(n_samples = 100,
n_features = 3,
n_informative = 3,
n_redundant = 0,
n_classes = 2,
weights = [.25, .75],
random_state = 1)
```

```
# View feature matrix and target vector
print('Feature Matrix \n ', features[:3])
print('Target Vector \n ', target[:3])
```

```
Feature Matrix
[[ 1.06354768 -1.42632219 1.02163151]
[ 0.23156977 1.49535261 0.33251578]
[ 0.15972951 0.83533515 -0.40869554]]
Target Vector
[1 0 0]
```

Finally, if we want a dataset designed to work well with clustering techniques, scikit-

learn offers make_blobs:

```
# Load library
from sklearn.datasets import make_blobs
```

```
# Generate features matrix and target vector
features, target = make_blobs(n_samples = 100,
n_features = 2,
centers = 3,
cluster_std = 0.5,
shuffle = True ,
random_state = 1)
```

```
# View feature matrix and target vector
print('Feature Matrix \n ', features[:3])
print('Target Vector \n ', target[:3])
```

```
Feature Matrix
[[ -1.22685609 3.25572052]
[ -9.57463218 -4.38310652]
[-10.71976941 -4.20558148]]
Target Vector
[0 1 1]
```

**24 | Chapter 2: Loading Data**

**Discussion**

As might be apparent from the solutions, make_regression returns a feature matrix

of float values and a target vector of float values, while make_classification and

make_blobs return a feature matrix of float values and a target vector of integers

representing membership in a class.

scikit-learn’s simulated datasets offer extensive options to control the type of data

generated. scikit-learn’s documentation contains a full description of all the parame‐

ters, but a few are worth noting.

In make_regression and make_classification, n_informative determines the

number of features that are used to generate the target vector. If n_informative

is less than the total number of features (n_features), the resulting dataset will have

redundant features that can be identified through feature selection techniques.

In addition, make_classification contains a weights parameter that allows us to

simulate datasets with imbalanced classes. For example, weights = [.25, .75]

would return a dataset with 25% of observations belonging to one class and 75% of

observations belonging to a second class.

For make_blobs, the centers parameter determines the number of clusters generated.

Using the matplotlib visualization library, we can visualize the clusters generated by

make_blobs:

```
# Load library
import matplotlib.pyplot as plt
```

```
# View scatterplot
plt.scatter(features[:,0], features[:,1], c=target)
plt.show()
```

```
2.2 Creating a Simulated Dataset | 25
```

**See Also**

- •make_regression documentation
- •make_classification documentation
- •make_blobs documentation

### 2.3 Loading a CSV File

**Problem**

You need to import a comma-separated value (CSV) file.

**Solution**

Use the pandas library’s read_csv to load a local or hosted CSV file into a pandas

DataFrame:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.csv'
```

```
# Load dataset
dataframe = pd.read_csv(url)
```

```
# View first two rows
dataframe.head(2)
```

```
integer datetime category
0 5 2015-01-01 00:00:00 0
1 5 2015-01-01 00:00:01 0
```

**Discussion**

There are two things to note about loading CSV files. First, it is often useful to take

a quick look at the contents of the file before loading. It can be very helpful to see

how a dataset is structured beforehand and what parameters we need to set to load in

the file. Second, read_csv has over 30 parameters and therefore the documentation

can be daunting. Fortunately, those parameters are mostly there to allow it to handle a

wide variety of CSV formats.

CSV files get their names from the fact that the values are literally separated by com‐

mas (e.g., one row might be 2,"2015-01-01 00:00:00",0); however, it is common

for CSV files to use other separators, such as tabs (which are referred to as TSV files).

**26 | Chapter 2: Loading Data**

The pandas sep parameter allows us to define the delimiter used in the file. Although

it is not always the case, a common formatting issue with CSV files is that the first

line of the file is used to define column headers (e.g., integer, datetime, category

in our solution). The header parameter allows us to specify if or where a header row

exists. If a header row does not exist, we set header=None.

The read_csv function returns a pandas DataFrame: a common and useful object for

working with tabular data that we’ll cover in more depth throughout this book.

### 2.4 Loading an Excel File

**Problem**

You need to import an Excel spreadsheet.

**Solution**

Use the pandas library’s read_excel to load an Excel spreadsheet:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.xlsx'
```

```
# Load data
dataframe = pd.read_excel(url, sheet_name=0, header=0)
```

```
# View the first two rows
dataframe.head(2)
```

```
integer datetime category
5 2015-01-01 00:00:00 0
0 5 2015-01-01 00:00:01 0
1 9 2015-01-01 00:00:02 0
```

**Discussion**

This solution is similar to our solution for reading CSV files. The main difference

is the additional parameter, sheet_name, that specifies which sheet in the Excel

file we wish to load. sheet_name can accept both strings, containing the name of

the sheet, and integers, pointing to sheet positions (zero-indexed). If we need to

load multiple sheets, we include them as a list. For example, sheet_name=[0,1,2,

"Monthly Sales"] will return a dictionary of pandas DataFrames containing the

first, second, and third sheets, and the sheet named Monthly Sales.

```
2.4 Loading an Excel File | 27
```

### 2.5 Loading a JSON File

**Problem**

You need to load a JSON file for data preprocessing.

**Solution**

The pandas library provides read_json to convert a JSON file into a pandas object:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.json'
```

```
# Load data
dataframe = pd.read_json(url, orient='columns')
```

```
# View the first two rows
dataframe.head(2)
```

```
category datetime integer
0 0 2015-01-01 00:00:00 5
1 0 2015-01-01 00:00:01 5
```

**Discussion**

Importing JSON files into pandas is similar to the last few recipes we have seen. The

key difference is the orient parameter, which indicates to pandas how the JSON

file is structured. However, it might take some experimenting to figure out which

argument (split, records, index, columns, or values) is the right one. Another

helpful tool pandas offers is json_normalize, which can help convert semistructured

JSON data into a pandas DataFrame.

**See Also**

- •json_normalize documentation

### 2.6 Loading a Parquet File

**Problem**

You need to load a Parquet file.

**28 | Chapter 2: Loading Data**

**Solution**

The pandas read_parquet function allows us to read in Parquet files:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://machine-learning-python-cookbook.s3.amazonaws.com/data.parquet'
```

```
# Load data
dataframe = pd.read_parquet(url)
```

```
# View the first two rows
dataframe.head(2)
```

```
category datetime integer
0 0 2015-01-01 00:00:00 5
1 0 2015-01-01 00:00:01 5
```

**Discussion**

Parquet is a popular data storage format in the large data space. It is often used with

big data tools such as Hadoop and Spark. While PySpark is outside the focus of

this book, it’s highly likely companies operating on a large scale will use an efficient

data storage format such as Parquet, and it’s valuable to know how to read it into a

dataframe and manipulate it.

**See Also**

- •Apache Parquet documentation

### 2.7 Loading an Avro File

**Problem**

You need to load an Avro file into a pandas DataFrame.

**Solution**

The use the pandavro library’s read_avro method:

```
# Load library
import requests
import pandavro as pdx
```

```
# Create URL
url = 'https://machine-learning-python-cookbook.s3.amazonaws.com/data.avro'
```

```
2.7 Loading an Avro File | 29
```

```
# Download file
r = requests.get(url)
open('data.avro', 'wb').write(r.content)
```

```
# Load data
dataframe = pdx.read_avro('data.avro')
```

```
# View the first two rows
dataframe.head(2)
```

```
category datetime integer
0 0 2015-01-01 00:00:00 5
1 0 2015-01-01 00:00:01 5
```

**Discussion**

Apache Avro is an open source, binary data format that relies on schemas for the

data structure. At the time of writing, it is not as common as Parquet. However,

large binary data formats such as Avro, thrift, and Protocol Buffers are growing in

popularity due to their efficient nature. If you work with large data systems, you’re

likely to run into one of these formats in the near future.

**See Also**

- •Apache Avro documentation

### 2.8 Querying a SQLite Database

**Problem**

You need to load data from a database using structured query language (SQL).

**Solution**

pandas’ read_sql_query allows us to make an SQL query to a database and load it:

```
# Load libraries
import pandas as pd
from sqlalchemy import create_engine
```

```
# Create a connection to the database
database_connection = create_engine('sqlite:///sample.db')
```

```
# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
```

**30 | Chapter 2: Loading Data**

```
# View first two rows
dataframe.head(2)
```

```
first_name last_name age preTestScore postTestScore
0 Jason Miller 42 4 25
1 Molly Jacobson 52 24 94
```

**Discussion**

SQL is the lingua franca for pulling data from databases. In this recipe we first use

create_engine to define a connection to an SQL database engine called SQLite. Next

we use pandas’ read_sql_query to query that database using SQL and put the results

in a DataFrame.

SQL is a language in its own right and, while beyond the scope of this book, it is

certainly worth knowing for anyone wanting to learn about machine learning. Our

SQL query, SELECT _ FROM data, asks the database to give us all columns (_) from

the table called data.

Note that this is one of a few recipes in this book that will not run without extra

code. Specifically, create_engine('sqlite:///sample.db') assumes that an SQLite

database already exists.

**See Also**

- •SQLite
- •W3Schools SQL Tutorial

### 2.9 Querying a Remote SQL Database

**Problem**

You need to connect to, and read from, a remote SQL database.

**Solution**

Create a connection with pymysql and read it into a dataframe with pandas:

```
# Import libraries
import pymysql
import pandas as pd
```

```
# Create a DB connection
# Use the following example to start a DB instance
# https://github.com/kylegallatin/mysql-db-example
conn = pymysql.connect(
```

```
2.9 Querying a Remote SQL Database | 31
```

```
host='localhost',
user='root',
password = "",
db='db',
)
```

```
# Read the SQL query into a dataframe
dataframe = pd.read_sql("select * from data", conn)
```

```
# View the first two rows
dataframe.head(2)
```

```
integer datetime category
0 5 2015-01-01 00:00:00 0
1 5 2015-01-01 00:00:01 0
```

**Discussion**

Of all of the recipes presented in this chapter, this is probably the one we will use

most in the real world. While connecting and reading from an example sqlite

database is useful, it’s likely not representative of tables you’ll need to connect to in

an enterprise environment. Most SQL instances that you’ll connect to will require

you to connect to the host and port of a remote machine, specifying a username

and password for authentication. This example requires you to start a running SQL

instance locally that mimics a remote server on localhost so that you can get a sense

of the workflow.

**See Also**

- •PyMySQL documentation
- •pandas Read SQL documentation

### 2.10 Loading Data from a Google Sheet

**Problem**

You need to read in data directly from a Google Sheet.

**Solution**

Use pandas read_CSV and pass a URL that exports the Google Sheet as a CSV:

```
# Import libraries
import pandas as pd
```

```
# Google Sheet URL that downloads the sheet as a CSV
```

**32 | Chapter 2: Loading Data**

```
url = "https://docs.google.com/spreadsheets/d/"\
"1ehC-9otcAuitqnmWksqt1mOrTRCL38dv0K9UjhwzTOA/export?format=csv"
```

```
# Read the CSV into a dataframe
dataframe = pd.read_csv(url)
```

```
# View the first two rows
dataframe.head(2)
```

```
integer datetime category
0 5 2015-01-01 00:00:00 0
1 5 2015-01-01 00:00:01 0
```

**Discussion**

While Google Sheets can easily be downloaded, it’s sometimes helpful to

be able to read them directly into Python without any intermediate steps.

The /export?format=csv query parameter at the end of the URL above creates an

endpoint from which we can either download the file or read it into pandas.

**See Also**

- •Google Sheets API

### 2.11 Loading Data from an S3 Bucket

**Problem**

You need to read a CSV file from an S3 bucket you have access to.

**Solution**

Add storage options to pandas giving it access to the S3 object:

```
# Import libraries
import pandas as pd
```

```
# S3 path to CSV
s3_uri = "s3://machine-learning-python-cookbook/data.csv"
```

```
# Set AWS credentials (replace with your own)
ACCESS_KEY_ID = " xxxxxxxxxxxxx "
SECRET_ACCESS_KEY = " xxxxxxxxxxxxxxxx "
```

```
# Read the CSV into a dataframe
dataframe = pd.read_csv(s3_uri,storage_options={
"key": ACCESS_KEY_ID,
"secret": SECRET_ACCESS_KEY,
```

```
2.11 Loading Data from an S3 Bucket | 33
```

##### }

##### )

```
# View first two rows
dataframe.head(2)
```

```
integer datetime category
0 5 2015-01-01 00:00:00 0
1 5 2015-01-01 00:00:01 0
```

**Discussion**

Many enterprises now keep data in cloud provider blob stores such as Amazon S3

or Google Cloud Storage (GCS). It’s common for machine learning practitioners

to connect to these sources to retrieve data. Although the S3 URI (s3://machine-

learning-python-cookbook/data.csv) is public, it still requires you to provide your

own AWS access credentials to access it. It’s worth noting that public objects also have

HTTP URLs from which they can download files, such as this one for the CSV file.

**See Also**

- •Amazon S3
- •AWS Security Credentials

### 2.12 Loading Unstructured Data

**Problem**

You need to load unstructured data like text or images.

**Solution**

Use the base Python open function to load the information:

```
# Import libraries
import requests
```

```
# URL to download the txt file from
txt_url = "https://machine-learning-python-cookbook.s3.amazonaws.com/text.txt"
```

```
# Get the txt file
r = requests.get(txt_url)
```

```
# Write it to text.txt locally
with open('text.txt', 'wb') as f:
f.write(r.content)
```

**34 | Chapter 2: Loading Data**

```
# Read in the file
with open('text.txt', 'r') as f:
text = f.read()
```

```
# Print the content
print(text)
```

```
Hello there!
```

**Discussion**

While structured data can easily be read in from CSV, JSON, or various databases,

unstructured data can be more challenging and may require custom processing down

the line. Sometimes it’s helpful to open and read in files using Python’s basic open

function. This allows us to open files and then read the content of that file.

**See Also**

- •Python’s open function
- •Context managers in Python

```
2.12 Loading Unstructured Data | 35
```

**CHAPTER 3**

### Data Wrangling

### 3.0 Introduction

Data wrangling is a broad term used, often informally, to describe the process of

transforming raw data into a clean, organized format ready for use. For us, data

wrangling is only one step in preprocessing our data, but it is an important step.

The most common data structure used to “wrangle” data is the dataframe, which

can be both intuitive and incredibly versatile. Dataframes are tabular, meaning that

they are based on rows and columns like you would see in a spreadsheet. Here is a

dataframe created from data about passengers on the Titanic:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data as a dataframe
dataframe = pd.read_csv(url)
```

```
# Show first five rows
dataframe.head(5)
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.00 female 1 1
1 Allison, Miss Helen Loraine 1st 2.00 female 0 1
2 Allison, Mr Hudson Joshua Creighton 1st 30.00 male 0 0
3 Allison, Mrs Hudson JC (Bessie Waldo Daniels) 1st 25.00 female 0 1
4 Allison, Master Hudson Trevor 1st 0.92 male 1 0
```

##### 37

There are three important things to notice in this dataframe.

First, in a dataframe each row corresponds to one observation (e.g., a passenger) and

each column corresponds to one feature (gender, age, etc.). For example, by looking

at the first observation we can see that Miss Elisabeth Walton Allen stayed in first

class, was 29 years old, was female, and survived the disaster.

Second, each column contains a name (e.g., Name, PClass, Age) and each row contains

an index number (e.g., 0 for the lucky Miss Elisabeth Walton Allen). We will use these

to select and manipulate observations and features.

Third, two columns, Sex and SexCode, contain the same information in different

formats. In Sex, a woman is indicated by the string female, while in SexCode, a

woman is indicated by using the integer 1. We will want all our features to be unique,

and therefore we will need to remove one of these columns.

In this chapter, we will cover a wide variety of techniques to manipulate dataframes

using the pandas library with the goal of creating a clean, well-structured set of

observations for further preprocessing.

### 3.1 Creating a Dataframe

**Problem**

You want to create a new dataframe.

**Solution**

pandas has many methods for creating a new DataFrame object. One easy method is

to instantiate a DataFrame using a Python dictionary. In the dictionary, each key is a

column name and the value is a list, where each item corresponds to a row:

```
# Load library
import pandas as pd
```

```
# Create a dictionary
dictionary = {
"Name": ['Jacky Jackson', 'Steven Stevenson'],
"Age": [38, 25],
"Driver": [ True , False ]
}
```

```
# Create DataFrame
dataframe = pd.DataFrame(dictionary)
```

```
# Show DataFrame
dataframe
```

**38 | Chapter 3: Data Wrangling**

```
Name Age Driver
0 Jacky Jackson 38 True
1 Steven Stevenson 25 False
```

It’s easy to add new columns to any dataframe using a list of values:

```
# Add a column for eye color
dataframe["Eyes"] = ["Brown", "Blue"]
```

```
# Show DataFrame
dataframe
```

```
Name Age Driver Eyes
0 Jacky Jackson 38 True Brown
1 Steven Stevenson 25 False Blue
```

**Discussion**

pandas offers what can feel like an infinite number of ways to create a DataFrame. In

the real world, creating an empty DataFrame and then populating it will almost never

happen. Instead, our DataFrames will be created from real data we have loaded from

other sources (e.g., a CSV file or database).

### 3.2 Getting Information about the Data

**Problem**

You want to view some characteristics of a DataFrame.

**Solution**

One of the easiest things we can do after loading the data is view the first few rows

using head:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Show two rows
dataframe.head(2)
```

```
3.2 Getting Information about the Data | 39
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
```

We can also take a look at the number of rows and columns:

```
# Show dimensions
dataframe.shape
```

```
(1313, 6)
```

We can get descriptive statistics for any numeric columns using describe:

```
# Show statistics
dataframe.describe()
```

```
Age Survived SexCode
count 756.000000 1313.000000 1313.000000
mean 30.397989 0.342727 0.351866
std 14.259049 0.474802 0.477734
min 0.170000 0.000000 0.000000
25% 21.000000 0.000000 0.000000
50% 28.000000 0.000000 0.000000
75% 39.000000 1.000000 1.000000
max 71.000000 1.000000 1.000000
```

Additionally, the info method can show some helpful information:

```
# Show info
dataframe.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1313 entries, 0 to 1312
Data columns (total 6 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 Name 1313 non-null object
1 PClass 1313 non-null object
2 Age 756 non-null float64
3 Sex 1313 non-null object
4 Survived 1313 non-null int64
5 SexCode 1313 non-null int64
dtypes: float64(1), int64(2), object(3)
memory usage: 61.7+ KB
```

**Discussion**

After we load some data, it’s a good idea to understand how it’s structured and what

kind of information it contains. Ideally, we would view the full data directly. But with

**40 | Chapter 3: Data Wrangling**

most real-world cases, the data could have thousands to hundreds of thousands to

millions of rows and columns. Instead, we have to rely on pulling samples to view

small slices and calculating summary statistics of the data.

In our solution, we are using a toy dataset of the passengers of the Titanic. Using

head, we can look at the first few rows (five by default) of the data. Alternatively, we

can use tail to view the last few rows. With shape we can see how many rows and

columns our DataFrame contains. With describe we can see some basic descriptive

statistics for any numerical column. And, finally, info displays a number of helpful

data points about the DataFrame, including index and column data types, non-null

values, and memory usage.

It is worth noting that summary statistics do not always tell the full story. For

example, pandas treats the columns Survived and SexCode as numeric columns

because they contain 1s and 0s. However, in this case the numerical values represent

categories. For example, if Survived equals 1, it indicates that the passenger survived

the disaster. For this reason, some of the summary statistics provided don’t make

sense, such as the standard deviation of the SexCode column (an indicator of the

passenger’s gender).

### 3.3 Slicing DataFrames

**Problem**

You need to select a specific subset data or slices of a DataFrame.

**Solution**

Use loc or iloc to select one or more rows or values:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Select first row
dataframe.iloc[0]
```

```
Name Allen, Miss Elisabeth Walton
PClass 1st
Age 29
Sex female
Survived 1
```

```
3.3 Slicing DataFrames | 41
```

```
SexCode 1
Name: 0, dtype: object
```

We can use : to define the slice of rows we want, such as selecting the second, third,

and fourth rows:

```
# Select three rows
dataframe.iloc[1:4]
```

```
Name PClass Age Sex Survived SexCode
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
2 Allison, Mr Hudson Joshua Creighton 1st 30.0 male 0 0
3 Allison, Mrs Hudson JC (Bessie Waldo Daniels) 1st 25.0 female 0 1
```

We can even use it to get all rows up to a point, such as all rows up to and including

the fourth row:

```
# Select four rows
dataframe.iloc[:4]
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
2 Allison, Mr Hudson Joshua Creighton 1st 30.0 male 0 0
3 Allison, Mrs Hudson JC (Bessie Waldo Daniels) 1st 25.0 female 0 1
```

DataFrames do not need to be numerically indexed. We can set the index of a

DataFrame to any value where the value is unique to each row. For example, we can

set the index to be passenger names and then select rows using a name:

```
# Set index
dataframe = dataframe.set_index(dataframe['Name'])
```

```
# Show row
dataframe.loc['Allen, Miss Elisabeth Walton']
```

```
Name Allen, Miss Elisabeth Walton
PClass 1st
Age 29
Sex female
Survived 1
SexCode 1
Name: Allen, Miss Elisabeth Walton, dtype: object
```

**Discussion**

All rows in a pandas DataFrame have a unique index value. By default, this index is

an integer indicating the row position in the DataFrame; however, it does not have

**42 | Chapter 3: Data Wrangling**

to be. DataFrame indexes can be set to be unique alphanumeric strings or customer

numbers. To select individual rows and slices of rows, pandas provides two methods:

- •loc is useful when the index of the DataFrame is a label (e.g., a string).
- •iloc works by looking for the position in the DataFrame. For example, iloc[0]
  will return the first row regardless of whether the index is an integer or a label.

It is useful to be comfortable with both loc and iloc since they will come up a lot

during data cleaning.

### 3.4 Selecting Rows Based on Conditionals

**Problem**

You want to select DataFrame rows based on some condition.

**Solution**

This can be done easily in pandas. For example, if we wanted to select all the women

on the Titanic:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Show top two rows where column 'sex' is 'female'
dataframe[dataframe['Sex'] == 'female'].head(2)
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
```

Take a moment to look at the format of this solution. Our conditional statement is

dataframe['Sex'] == 'female'; by wrapping that in dataframe[] we are telling

pandas to “select all the rows in the DataFrame where the value of dataframe['Sex']

is 'female'.” These conditions result in a pandas series of booleans.

Multiple conditions are easy as well. For example, here we select all the rows where

the passenger is a female 65 or older:

```
3.4 Selecting Rows Based on Conditionals | 43
```

```
# Filter rows
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]
```

```
Name PClass Age Sex Survived SexCode
73 Crosby, Mrs Edward Gifford (Catherine Elizabet... 1st 69.0 female 1 1
```

**Discussion**

Conditionally selecting and filtering data is one of the most common tasks in data

wrangling. You rarely want all the raw data from the source; instead, you are interes‐

ted in only some subset of it. For example, you might only be interested in stores in

certain states or the records of patients over a certain age.

### 3.5 Sorting Values

**Problem**

You need to sort a dataframe by the values in a column.

**Solution**

Use the pandas sort_values function:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Sort the dataframe by age, show two rows
dataframe.sort_values(by=["Age"]).head(2)
```

```
Name PClass Age Sex Survived SexCode
763 Dean, Miss Elizabeth Gladys (Millvena) 3rd 0.17 female 1 1
751 Danbom, Master Gilbert Sigvard Emanuel 3rd 0.33 male 0 0
```

**Discussion**

During data analysis and exploration, it’s often useful to sort a DataFrame by a

particular column or set of columns. The by argument to sort_values takes a list of

columns by which to sort the DataFrame and will sort based on the order of column

names in the list.

**44 | Chapter 3: Data Wrangling**

By default, the ascending argument is set to True, so it will sort the values lowest to

highest. If we wanted the oldest passengers instead of the youngest, we could set it to

False.

### 3.6 Replacing Values

**Problem**

You need to replace values in a DataFrame.

**Solution**

The pandas replace method is an easy way to find and replace values. For example,

we can replace any instance of "female" in the Sex column with "Woman":

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Replace values, show two rows
dataframe['Sex'].replace("female", "Woman").head(2)
```

```
0 Woman
1 Woman
Name: Sex, dtype: object
```

We can also replace multiple values at the same time:

```
# Replace "female" and "male" with "Woman" and "Man"
dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5)
```

```
0 Woman
1 Woman
2 Man
3 Woman
4 Man
Name: Sex, dtype: object
```

We can also find and replace across the entire DataFrame object by specifying the

whole dataframe instead of a single column:

```
# Replace values, show two rows
dataframe.replace(1, "One").head(2)
```

```
3.6 Replacing Values | 45
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29 female One One
1 Allison, Miss Helen Loraine 1st 2 female 0 One
```

replace also accepts regular expressions:

```
# Replace values, show two rows
dataframe.replace(r"1st", "First", regex= True ).head(2)
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton First 29.0 female 1 1
1 Allison, Miss Helen Loraine First 2.0 female 0 1
```

**Discussion**

replace is a tool we use to replace values. It is simple and yet has the powerful ability

to accept regular expressions.

### 3.7 Renaming Columns

**Problem**

You want to rename a column in a pandas DataFrame.

**Solution**

Rename columns using the rename method:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Rename column, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)
```

```
Name Passenger Class Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
```

Notice that the rename method can accept a dictionary as a parameter. We can use the

dictionary to change multiple column names at once:

**46 | Chapter 3: Data Wrangling**

```
# Rename columns, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)
```

```
Name Passenger Class Age Gender Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
```

**Discussion**

Using rename with a dictionary as an argument to the columns parameter is my

preferred way to rename columns because it works with any number of columns.

If we want to rename all columns at once, this helpful snippet of code creates a

dictionary with the old column names as keys and empty strings as values:

```
# Load library
import collections
```

```
# Create dictionary
column_names = collections.defaultdict(str)
```

```
# Create keys
for name in dataframe.columns:
column_names[name]
```

```
# Show dictionary
column_names
```

```
defaultdict(str,
{'Age': '',
'Name': '',
'PClass': '',
'Sex': '',
'SexCode': '',
'Survived': ''})
```

### 3.8 Finding the Minimum, Maximum, Sum, Average, and Count

**Problem**

You want to find the min, max, sum, average, or count of a numeric column.

**Solution**

pandas comes with some built-in methods for commonly used descriptive statistics

such as min, max, mean, sum, and count:

```
3.8 Finding the Minimum, Maximum, Sum, Average, and Count | 47
```

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())
```

```
Maximum: 71.0
Minimum: 0.17
Mean: 30.397989417989415
Sum: 22980.879999999997
Count: 756
```

**Discussion**

In addition to the statistics used in the solution, pandas offers variance (var), stan‐

dard deviation (std), kurtosis (kurt), skewness (skew), standard error of the mean

(sem), mode (mode), median (median), value counts, and a number of others.

Furthermore, we can also apply these methods to the whole DataFrame:

```
# Show counts
dataframe.count()
```

```
Name 1313
PClass 1313
Age 756
Sex 1313
Survived 1313
SexCode 1313
dtype: int64
```

### 3.9 Finding Unique Values

**Problem**

You want to select all unique values in a column.

**Solution**

Use unique to view an array of all unique values in a column:

**48 | Chapter 3: Data Wrangling**

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Select unique values
dataframe['Sex'].unique()
```

```
array(['female', 'male'], dtype=object)
```

Alternatively, value_counts will display all unique values with the number of times

each value appears:

```
# Show counts
dataframe['Sex'].value_counts()
```

```
male 851
female 462
Name: Sex, dtype: int64
```

**Discussion**

Both unique and value_counts are useful for manipulating and exploring categorical

columns. Very often in categorical columns there will be classes that need to be

handled in the data wrangling phase. For example, in the Titanic dataset, PClass is

a column indicating the class of a passenger’s ticket. There were three classes on the

Titanic; however, if we use value_counts we can see a problem:

```
# Show counts
dataframe['PClass'].value_counts()
```

```
3rd 711
1st 322
2nd 279
* 1
Name: PClass, dtype: int64
```

While almost all passengers belong to one of three classes as expected, a single

passenger has the class \*. There are a number of strategies for handling this type of

issue, which we will address in Chapter 5, but for now just realize that “extra” classes

are common in categorical data and should not be ignored.

Finally, if we simply want to count the number of unique values, we can use nunique:

```
# Show number of unique values
dataframe['PClass'].nunique()
```

```
4
```

```
3.9 Finding Unique Values | 49
```

### 3.10 Handling Missing Values

**Problem**

You want to select missing values in a DataFrame.

**Solution**

isnull and notnull return booleans indicating whether a value is missing:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
## Select missing values, show two rows
dataframe[dataframe['Age'].isnull()].head(2)
```

```
Name PClass Age Sex Survived SexCode
12 Aubert, Mrs Leontine Pauline 1st NaN female 1 1
13 Barkworth, Mr Algernon H 1st NaN male 1 0
```

**Discussion**

Missing values are a ubiquitous problem in data wrangling, yet many underestimate

the difficulty of working with missing data. pandas uses NumPy’s NaN (Not a Num‐

ber) value to denote missing values, but it is important to note that NaN is not fully

implemented natively in pandas. For example, if we wanted to replace all strings

containing male with missing values, we get an error:

```
# Attempt to replace values with NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)
```

```
---------------------------------------------------------------------------
```

```
NameError Traceback (most recent call last)
```

```
<ipython-input-7-5682d714f87d> in <module>()
1 # Attempt to replace values with NaN
----> 2 dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)
```

```
NameError: name 'NaN' is not defined
---------------------------------------------------------------------------
```

**50 | Chapter 3: Data Wrangling**

To have full functionality with NaN we need to import the NumPy library first:

```
# Load library
import numpy as np
```

```
# Replace values with NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)
```

Oftentimes a dataset uses a specific value to denote a missing observation, such as

NONE, -999, or ... The pandas read_csv function includes a parameter allowing us to

specify the values used to indicate missing values:

```
# Load data, set missing values
dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])
```

We can also use the pandas fillna function to impute the missing values of a

column. Here, we show the places where Age is null using the isna function and then

fill those values with the mean age of passengers.

```
# Get a single null row
null_entry = dataframe[dataframe["Age"].isna()].head(1)
```

```
print(null_entry)
```

```
Name PClass Age Sex Survived SexCode
12 Aubert, Mrs Leontine Pauline 1st NaN female 1 1
```

```
# Fill all null values with the mean age of passengers
null_entry.fillna(dataframe["Age"].mean())
```

```
Name PClass Age Sex Survived SexCode
12 Aubert, Mrs Leontine Pauline 1st 30.397989 female 1 1
```

### 3.11 Deleting a Column

**Problem**

You want to delete a column from your DataFrame.

**Solution**

The best way to delete a column is to use drop with the parameter axis=1 (i.e., the

column axis):

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
3.11 Deleting a Column | 51
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Delete column
dataframe.drop('Age', axis=1).head(2)
```

```
Name PClass Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st female 1 1
1 Allison, Miss Helen Loraine 1st female 0 1
```

You can also use a list of column names as the main argument to drop multiple

columns at once:

```
# Drop columns
dataframe.drop(['Age', 'Sex'], axis=1).head(2)
```

```
Name PClass Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 1 1
1 Allison, Miss Helen Loraine 1st 0 1
```

If a column does not have a name (which can sometimes happen), you can drop it by

its column index using dataframe.columns:

```
# Drop column
dataframe.drop(dataframe.columns[1], axis=1).head(2)
```

```
Name Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 29.0 female 1 1
1 Allison, Miss Helen Loraine 2.0 female 0 1
```

**Discussion**

drop is the idiomatic method of deleting a column. An alternative method is

del dataframe['Age'], which works most of the time but is not recommended

because of how it is called within pandas (the details of which are outside the scope of

this book).

I recommend that you avoid using the pandas inplace=True argument. Many pandas

methods include an inplace parameter that, when set to True, edits the DataFrame

directly. This can lead to problems in more complex data processing pipelines

because we are treating the DataFrames as mutable objects (which they technically

are). I recommend treating DataFrames as immutable objects. For example:

```
# Create a new DataFrame
dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)
```

**52 | Chapter 3: Data Wrangling**

In this example, we are not mutating the DataFrame dataframe but instead

are making a new DataFrame that is an altered version of dataframe called

dataframe_name_dropped. If you treat your DataFrames as immutable objects, you

will save yourself a lot of headaches down the road.

### 3.12 Deleting a Row

**Problem**

You want to delete one or more rows from a DataFrame.

**Solution**

Use a boolean condition to create a new DataFrame excluding the rows you want to

delete:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Delete rows, show first three rows of output
dataframe[dataframe['Sex'] != 'male'].head(3)
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
3 Allison, Mrs Hudson JC (Bessie Waldo Daniels) 1st 25.00 female 0 1
```

**Discussion**

While technically you can use the drop method (for example, dataframe.drop([0,

1], axis=0) to drop the first two rows), a more practical method is simply to

wrap a boolean condition inside dataframe[]. This enables us to use the power of

conditionals to delete either a single row or (far more likely) many rows at once.

We can use boolean conditions to easily delete single rows by matching a unique

value:

```
# Delete row, show first two rows of output
dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)
```

```
3.12 Deleting a Row | 53
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
2 Allison, Mr Hudson Joshua Creighton 1st 30.0 male 0 0
```

We can even use it to delete a single row by specifying the row index:

```
# Delete row, show first two rows of output
dataframe[dataframe.index != 0].head(2)
```

```
Name PClass Age Sex Survived SexCode
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
2 Allison, Mr Hudson Joshua Creighton 1st 30.0 male 0 0
```

### 3.13 Dropping Duplicate Rows

**Problem**

You want to drop duplicate rows from your DataFrame.

**Solution**

Use drop_duplicates, but be mindful of the parameters:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Drop duplicates, show first two rows of output
dataframe.drop_duplicates().head(2)
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
1 Allison, Miss Helen Loraine 1st 2.0 female 0 1
```

**Discussion**

A keen reader will notice that the solution didn’t actually drop any rows:

```
# Show number of rows
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))
```

**54 | Chapter 3: Data Wrangling**

```
Number Of Rows In The Original DataFrame: 1313
Number Of Rows After Deduping: 1313
```

This is because drop_duplicates defaults to dropping only rows that match perfectly

across all columns. Because every row in our DataFrame is unique, none will be

dropped. However, often we want to consider only a subset of columns to check for

duplicate rows. We can accomplish this using the subset parameter:

```
# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'])
```

```
Name PClass Age Sex Survived SexCode
0 Allen, Miss Elisabeth Walton 1st 29.0 female 1 1
2 Allison, Mr Hudson Joshua Creighton 1st 30.0 male 0 0
```

Take a close look at the preceding output: we told drop_duplicates to only consider

any two rows with the same value for Sex to be duplicates and to drop them. Now we

are left with a DataFrame of only two rows: one woman and one man. You might be

asking why drop_duplicates decided to keep these two rows instead of two different

rows. The answer is that drop_duplicates defaults to keeping the first occurrence of

a duplicated row and dropping the rest. We can control this behavior using the keep

parameter:

```
# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'], keep='last')
```

```
Name PClass Age Sex Survived SexCode
1307 Zabour, Miss Tamini 3rd NaN female 0 1
1312 Zimmerman, Leo 3rd 29.0 male 0 0
```

A related method is duplicated, which returns a boolean series denoting whether

a row is a duplicate or not. This is a good option if you don’t want to simply drop

duplicates:

```
dataframe.duplicated()
```

```
0 False
1 False
2 False
3 False
4 False
...
1308 False
1309 False
1310 False
1311 False
1312 False
Length: 1313, dtype: bool
```

```
3.13 Dropping Duplicate Rows | 55
```

### 3.14 Grouping Rows by Values

**Problem**

You want to group individual rows according to some shared value.

**Solution**

groupby is one of the most powerful features in pandas:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Group rows by the values of the column 'Sex', calculate mean # of each group
dataframe.groupby('Sex').mean(numeric_only= True )
```

```
Sex Age Survived SexCode
female 29.396424 0.666667 1.0
male 31.014338 0.166863 0.0
```

**Discussion**

groupby is where data wrangling really starts to take shape. It is very common to

have a DataFrame where each row is a person or an event and we want to group

them according to some criterion and then calculate a statistic. For example, you can

imagine a DataFrame where each row is an individual sale at a national restaurant

chain and we want the total sales per restaurant. We can accomplish this by grouping

rows by individual restaurants and then calculating the sum of each group.

Users new to groupby often write a line like this and are confused by what is returned:

```
# Group rows
dataframe.groupby('Sex')
```

```
<pandas.core.groupby.DataFrameGroupBy object at 0x10efacf28>
```

Why didn’t it return something more useful? The reason is that groupby needs to be

paired with some operation that we want to apply to each group, such as calculating

an aggregate statistic (e.g., mean, median, sum). When talking about grouping we

often use shorthand and say “group by gender,” but that is incomplete. For grouping

to be useful, we need to group by something and then apply a function to each of

those groups:

**56 | Chapter 3: Data Wrangling**

```
# Group rows, count rows
dataframe.groupby('Survived')['Name'].count()
```

```
Survived
0 863
1 450
Name: Name, dtype: int64
```

Notice Name added after the groupby? That is because particular summary statistics

are meaningful only to certain types of data. For example, while calculating the

average age by gender makes sense, calculating the total age by gender does not. In

this case, we group the data into survived or not, and then count the number of

names (i.e., passengers) in each group.

We can also group by a first column, then group that grouping by a second column:

```
# Group rows, calculate mean
dataframe.groupby(['Sex','Survived'])['Age'].mean()
```

```
Sex Survived
female 0 24.901408
1 30.867143
male 0 32.320780
1 25.951875
Name: Age, dtype: float64
```

### 3.15 Grouping Rows by Time

**Problem**

You need to group individual rows by time periods.

**Solution**

Use resample to group rows by chunks of time:

```
# Load libraries
import pandas as pd
import numpy as np
```

```
# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')
```

```
# Create DataFrame
dataframe = pd.DataFrame(index=time_index)
```

```
# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)
```

```
# Group rows by week, calculate sum per week
dataframe.resample('W').sum()
```

```
3.15 Grouping Rows by Time | 57
```

```
Sale_Amount
2017-06-11 86423
2017-06-18 101045
2017-06-25 100867
2017-07-02 100894
2017-07-09 100438
2017-07-16 10297
```

**Discussion**

Our standard Titanic dataset does not contain a datetime column, so for this recipe

we have generated a simple DataFrame where each row is an individual sale. For each

sale we know its date and time and its dollar amount (this data isn’t realistic because

the sales take place precisely 30 seconds apart and are exact dollar amounts, but for

the sake of simplicity let’s pretend).

The raw data looks like this:

```
# Show three rows
dataframe.head(3)
```

```
Sale_Amount
2017-06-06 00:00:00 7
2017-06-06 00:00:30 2
2017-06-06 00:01:00 7
```

Notice that the date and time of each sale is the index of the DataFrame; this is

because resample requires the index to be a datetime-like value.

Using resample we can group the rows by a wide array of time periods (offsets) and

then we can calculate statistics on each time group:

```
# Group by two weeks, calculate mean
dataframe.resample('2W').mean()
```

```
Sale_Amount
2017-06-11 5.001331
2017-06-25 5.007738
2017-07-09 4.993353
2017-07-23 4.950481
```

```
# Group by month, count rows
dataframe.resample('M').count()
```

**58 | Chapter 3: Data Wrangling**

```
Sale_Amount
2017-06-30 72000
2017-07-31 28000
```

You might notice that in the two outputs the datetime index is a date even though

we are grouping by weeks and months, respectively. The reason is that by default

resample returns the label of the right “edge” (the last label) of the time group. We

can control this behavior using the label parameter:

```
# Group by month, count rows
dataframe.resample('M', label='left').count()
```

```
Sale_Amount
2017-05-31 72000
2017-06-30 28000
```

**See Also**

- •List of pandas time offset aliases

### 3.16 Aggregating Operations and Statistics

**Problem**

You need to aggregate an operation over each column (or a set of columns) in a

dataframe.

**Solution**

Use the pandas agg method. Here, we can easily get the minimum value of every

column:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Get the minimum of every column
dataframe.agg("min")
```

```
Name Abbing, Mr Anthony
PClass *
Age 0.17
```

```
3.16 Aggregating Operations and Statistics | 59
```

```
Sex female
Survived 0
SexCode 0
dtype: object
```

Sometimes, we want to apply specific functions to specific sets of columns:

```
# Mean Age, min and max SexCode
dataframe.agg({"Age":["mean"], "SexCode":["min", "max"]})
```

```
Age SexCode
mean 30.397989 NaN
min NaN 0.0
max NaN 1.0
```

We can also apply aggregate functions to groups to get more specific, descriptive

statistics:

```
# Number of people who survived and didn't survive in each class
dataframe.groupby(
["PClass","Survived"]).agg({"Survived":["count"]}
).reset_index()
```

```
PClass Survived Count
0 * 0 1
1 1st 0 129
2 1st 1 193
3 2nd 0 160
4 2nd 1 119
5 3rd 0 573
6 3rd 1 138
```

**Discussion**

Aggregate functions are especially useful during exploratory data analysis to learn

information about different subpopulations of data and the relationship between

variables. By grouping the data and applying aggregate statistics, you can view

patterns in the data that may prove useful during the machine learning or feature

engineering process. While visual charts are also helpful, it’s often useful to have such

specific, descriptive statistics as a reference to better understand the data.

**See Also**

- •pandas agg documentation

**60 | Chapter 3: Data Wrangling**

### 3.17 Looping over a Column

**Problem**

You want to iterate over every element in a column and apply some action.

**Solution**

You can treat a pandas column like any other sequence in Python and loop over it

using the standard Python syntax:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Print first two names uppercased
for name in dataframe['Name'][0:2]:
print(name.upper())
```

```
ALLEN, MISS ELISABETH WALTON
ALLISON, MISS HELEN LORAINE
```

**Discussion**

In addition to loops (often called for loops), we can also use list comprehensions:

```
# Show first two names uppercased
[name.upper() for name in dataframe['Name'][0:2]]
```

```
['ALLEN, MISS ELISABETH WALTON', 'ALLISON, MISS HELEN LORAINE']
```

Despite the temptation to fall back on for loops, a more Pythonic solution would use

the pandas apply method, described in Recipe 3.18.

### 3.18 Applying a Function over All Elements in a Column

**Problem**

You want to apply some function over all elements in a column.

**Solution**

Use apply to apply a built-in or custom function on every element in a column:

```
3.17 Looping over a Column | 61
```

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Create function
def uppercase(x):
return x.upper()
```

```
# Apply function, show two rows
dataframe['Name'].apply(uppercase)[0:2]
```

```
0 ALLEN, MISS ELISABETH WALTON
1 ALLISON, MISS HELEN LORAINE
Name: Name, dtype: object
```

**Discussion**

apply is a great way to do data cleaning and wrangling. It is common to write a

function to perform some useful operation (separate first and last names, convert

strings to floats, etc.) and then map that function to every element in a column.

### 3.19 Applying a Function to Groups

**Problem**

You have grouped rows using groupby and want to apply a function to each group.

**Solution**

Combine groupby and apply:

```
# Load library
import pandas as pd
```

```
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
```

```
# Load data
dataframe = pd.read_csv(url)
```

```
# Group rows, apply function to groups
dataframe.groupby('Sex').apply( lambda x: x.count())
```

**62 | Chapter 3: Data Wrangling**

```
Sex Name PClass Age Sex Survived SexCode
female 462 462 288 462 462 462
male 851 851 468 851 851 851
```

**Discussion**

In Recipe 3.18 I mentioned apply. apply is particularly useful when you want to

apply a function to groups. By combining groupby and apply we can calculate

custom statistics or apply any function to each group separately.

### 3.20 Concatenating DataFrames

**Problem**

You want to concatenate two DataFrames.

**Solution**

Use concat with axis=0 to concatenate along the row axis:

```
# Load library
import pandas as pd
```

```
# Create DataFrame
data_a = {'id': ['1', '2', '3'],
'first': ['Alex', 'Amy', 'Allen'],
'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])
```

```
# Create DataFrame
data_b = {'id': ['4', '5', '6'],
'first': ['Billy', 'Brian', 'Bran'],
'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])
```

```
# Concatenate DataFrames by rows
pd.concat([dataframe_a, dataframe_b], axis=0)
```

```
id first last
0 1 Alex Anderson
1 2 Amy Ackerman
2 3 Allen Ali
0 4 Billy Bonder
1 5 Brian Black
2 6 Bran Balwner
```

```
3.20 Concatenating DataFrames | 63
```

You can use axis=1 to concatenate along the column axis:

```
# Concatenate DataFrames by columns
pd.concat([dataframe_a, dataframe_b], axis=1)
```

```
id first last id first last
0 1 Alex Anderson 4 Billy Bonder
1 2 Amy Ackerman 5 Brian Black
2 3 Allen Ali 6 Bran Balwner
```

**Discussion**

Concatenating is not a word you hear much outside of computer science and pro‐

gramming, so if you have not heard it before, do not worry. The informal definition

of concatenate is to glue two objects together. In the solution we glued together two

small DataFrames using the axis parameter to indicate whether we wanted to stack

the two DataFrames on top of each other or place them side by side.

### 3.21 Merging DataFrames

**Problem**

You want to merge two DataFrames.

**Solution**

To inner join, use merge with the on parameter to specify the column to merge on:

```
# Load library
import pandas as pd
```

```
# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
'name'])
```

```
# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
'total_sales'])
```

```
# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id')
```

**64 | Chapter 3: Data Wrangling**

```
employee_id name total_sales
0 3 Alice Bees 23456
1 4 Tim Horton 2512
```

merge defaults to inner joins. If we want to do an outer join, we can specify that with

the how parameter:

```
# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')
```

```
employee_id name total_sales
0 1 Amy Jones NaN
1 2 Allen Keys NaN
2 3 Alice Bees 23456.0
3 4 Tim Horton 2512.0
4 5 NaN 2345.0
5 6 NaN 1455.0
```

The same parameter can be used to specify left and right joins:

```
# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')
```

```
employee_id name total_sales
0 1 Amy Jones NaN
1 2 Allen Keys NaN
2 3 Alice Bees 23456.0
3 4 Tim Horton 2512.0
```

We can also specify the column name in each DataFrame to merge on:

```
# Merge DataFrames
pd.merge(dataframe_employees,
dataframe_sales,
left_on='employee_id',
right_on='employee_id')
```

```
employee_id name total_sales
0 3 Alice Bees 23456
1 4 Tim Horton 2512
```

If, instead of merging on two columns, we want to merge on the indexes of

each DataFrame, we can replace the left_on and right_on parameters with

left_index=True and right_index=True.

```
3.21 Merging DataFrames | 65
```

**Discussion**

The data we need to use is often complex; it doesn’t always come in one piece.

Instead, in the real world, we’re usually faced with disparate datasets from multiple

database queries or files. To get all that data into one place, we can load each data

query or data file into pandas as individual DataFrames and then merge them into a

single DataFrame.

This process might be familiar to anyone who has used SQL, a popular language

for doing merging operations (called joins). While the exact parameters used by

pandas will be different, they follow the same general patterns used by other software

languages and tools.

There are three aspects to specify with any merge operation. First, we have to

specify the two DataFrames we want to merge. In the solution, we named them

dataframe_employees and dataframe_sales. Second, we have to specify the name(s)

of the columns to merge on—that is, the columns whose values are shared between

the two DataFrames. For example, in our solution both DataFrames have a column

named employee_id. To merge the two DataFrames we will match the values in each

DataFrame’s employee_id column. If these two columns use the same name, we can

use the on parameter. However, if they have different names, we can use left_on and

right_on.

What is the left and right DataFrame? The left DataFrame is the first one we specified

in merge, and the right DataFrame is the second one. This language comes up again

in the next sets of parameters we will need.

The last aspect, and most difficult for some people to grasp, is the type of merge

operation we want to conduct. This is specified by the how parameter. merge supports

the four main types of joins:

Inner

```
Return only the rows that match in both DataFrames (e.g., return any row
with an employee_id value appearing in both dataframe_employees and
dataframe_sales).
```

Outer

```
Return all rows in both DataFrames. If a row exists in one DataFrame but not in
the other DataFrame, fill NaN values for the missing values (e.g., return all rows
in both dataframe_employee and dataframe_sales).
```

**66 | Chapter 3: Data Wrangling**

Left

```
Return all rows from the left DataFrame but only rows from the right
DataFrame that match with the left DataFrame. Fill NaN values for the miss‐
ing values (e.g., return all rows from dataframe_employees but only rows
from dataframe_sales that have a value for employee_id that appears in
dataframe_employees).
```

Right

```
Return all rows from the right DataFrame but only rows from the left Data‐
Frame that match with the right DataFrame. Fill NaN values for the miss‐
ing values (e.g., return all rows from dataframe_sales but only rows from
dataframe_employees that have a value for employee_id that appears in
dataframe_sales).
```

If you did not understand all of that, I encourage you to play around with the how

parameter in your code and see how it affects what merge returns.

**See Also**

- •A Visual Explanation of SQL Joins
- •pandas documentation: Merge, join, concatenate and compare

```
3.21 Merging DataFrames | 67
```

**CHAPTER 4**

### Handling Numerical Data

### 4.0 Introduction

Quantitative data is the measurement of something—whether class size, monthly

sales, or student scores. The natural way to represent these quantities is numerically

(e.g., 29 students, $529,392 in sales). In this chapter, we will cover numerous strate‐

gies for transforming raw numerical data into features purpose-built for machine

learning algorithms.

### 4.1 Rescaling a Feature

**Problem**

You need to rescale the values of a numerical feature to be between two values.

**Solution**

Use scikit-learn’s MinMaxScaler to rescale a feature array:

```
# Load libraries
import numpy as np
from sklearn import preprocessing
```

```
# Create feature
feature = np.array([[-500.5],
[-100.1],
[0],
[100.1],
[900.9]])
```

```
# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
```

##### 69

```
# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)
```

```
# Show feature
scaled_feature
```

```
array([[ 0. ],
[ 0.28571429],
[ 0.35714286],
[ 0.42857143],
[ 1. ]])
```

**Discussion**

Rescaling is a common preprocessing task in machine learning. Many of the algo‐

rithms described later in this book will assume all features are on the same scale,

typically 0 to 1 or –1 to 1. There are a number of rescaling techniques, but one of

the simplest is called min-max scaling. Min-max scaling uses the minimum and max‐

imum values of a feature to rescale values to within a range. Specifically, min-max

calculates:

```
xi′=
```

```
xi− minx
maxx − minx
```

where x is the feature vector, xi is an individual element of feature x, and xi′ is

the rescaled element. In our example, we can see from the outputted array that the

feature has been successfully rescaled to between 0 and 1:

```
array([[ 0. ],
[ 0.28571429],
[ 0.35714286],
[ 0.42857143],
[ 1. ]])
```

scikit-learn’s MinMaxScaler offers two options to rescale a feature. One option is to

use fit to calculate the minimum and maximum values of the feature, and then

use transform to rescale the feature. The second option is to use fit_transform to

do both operations at once. There is no mathematical difference between the two

options, but there is sometimes a practical benefit to keeping the operations separate

because it allows us to apply the same transformation to different sets of the data.

**See Also**

- •Feature scaling, Wikipedia
- •About Feature Scaling and Normalization, Sebastian Raschka

**70 | Chapter 4: Handling Numerical Data**

### 4.2 Standardizing a Feature

**Problem**

You want to transform a feature to have a mean of 0 and a standard deviation of 1.

**Solution**

scikit-learn’s StandardScaler performs both transformations:

```
# Load libraries
import numpy as np
from sklearn import preprocessing
```

```
# Create feature
x = np.array([[-1000.1],
[-200.2],
[500.5],
[600.6],
[9000.9]])
```

```
# Create scaler
scaler = preprocessing.StandardScaler()
```

```
# Transform the feature
standardized = scaler.fit_transform(x)
```

```
# Show feature
standardized
```

```
array([[-0.76058269],
[-0.54177196],
[-0.35009716],
[-0.32271504],
[ 1.97516685]])
```

**Discussion**

A common alternative to the min-max scaling discussed in Recipe 4.1 is rescaling of

features to be approximately standard normally distributed. To achieve this, we use

standardization to transform the data such that it has a mean, x, of 0 and a standard

deviation, σ, of 1. Specifically, each element in the feature is transformed so that:

```
xi′=
```

```
xi−x
σ
```

```
4.2 Standardizing a Feature | 71
```

where xi′ is our standardized form of xi. The transformed feature represents the

number of standard deviations of the original value from the feature’s mean value

(also called a z-score in statistics).

Standardization is a common go-to scaling method for machine learning preprocess‐

ing and, in my experience, is used more often than min-max scaling. However, it

depends on the learning algorithm. For example, principal component analysis often

works better using standardization, while min-max scaling is often recommended for

neural networks (both algorithms are discussed later in this book). As a general rule,

I’d recommend defaulting to standardization unless you have a specific reason to use

an alternative.

We can see the effect of standardization by looking at the mean and standard devia‐

tion of our solution’s output:

```
# Print mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())
```

```
Mean: 0.0
Standard deviation: 1.0
```

If our data has significant outliers, it can negatively impact our standardization by

affecting the feature’s mean and variance. In this scenario, it is often helpful to instead

rescale the feature using the median and quartile range. In scikit-learn, we do this

using the RobustScaler method:

```
# Create scaler
robust_scaler = preprocessing.RobustScaler()
```

```
# Transform feature
robust_scaler.fit_transform(x)
```

```
array([[ -1.87387612],
[ -0.875 ],
[ 0. ],
[ 0.125 ],
[ 10.61488511]])
```

### 4.3 Normalizing Observations

**Problem**

You want to rescale the feature values of observations to have unit norm (a total

length of 1).

**72 | Chapter 4: Handling Numerical Data**

**Solution**

Use Normalizer with a norm argument:

```
# Load libraries
import numpy as np
from sklearn.preprocessing import Normalizer
```

```
# Create feature matrix
features = np.array([[0.5, 0.5],
[1.1, 3.4],
[1.5, 20.2],
[1.63, 34.4],
[10.9, 3.3]])
```

```
# Create normalizer
normalizer = Normalizer(norm="l2")
```

```
# Transform feature matrix
normalizer.transform(features)
```

```
array([[ 0.70710678, 0.70710678],
[ 0.30782029, 0.95144452],
[ 0.07405353, 0.99725427],
[ 0.04733062, 0.99887928],
[ 0.95709822, 0.28976368]])
```

**Discussion**

Many rescaling methods (e.g., min-max scaling and standardization) operate on

features; however, we can also rescale across individual observations. Normalizer

rescales the values on individual observations to have unit norm (the sum of their

lengths is 1). This type of rescaling is often used when we have many equivalent

features (e.g., text classification when every word or n-word group is a feature).

Normalizer provides three norm options with Euclidean norm (often called L2)

being the default argument:

```
∥x∥ 2 = x 12 +x 22 +⋯+xn^2
```

where x is an individual observation and xn is that observation’s value for the nth

feature.

```
# Transform feature matrix
features_l2_norm = Normalizer(norm="l2").transform(features)
```

```
# Show feature matrix
features_l2_norm
```

```
4.3 Normalizing Observations | 73
```

```
array([[ 0.70710678, 0.70710678],
[ 0.30782029, 0.95144452],
[ 0.07405353, 0.99725427],
[ 0.04733062, 0.99887928],
[ 0.95709822, 0.28976368]])
```

Alternatively, we can specify Manhattan norm (L1):

```
∥x∥ 1 = ∑
i= 1
```

```
n
xi.
```

```
# Transform feature matrix
features_l1_norm = Normalizer(norm="l1").transform(features)
```

```
# Show feature matrix
features_l1_norm
```

```
array([[ 0.5 , 0.5 ],
[ 0.24444444, 0.75555556],
[ 0.06912442, 0.93087558],
[ 0.04524008, 0.95475992],
[ 0.76760563, 0.23239437]])
```

Intuitively, L2 norm can be thought of as the distance between two points in New

York for a bird (i.e., a straight line), while L1 can be thought of as the distance for a

human walking on the street (walk north one block, east one block, north one block,

east one block, etc.), which is why it is called “Manhattan norm” or “Taxicab norm.”

Practically, notice that norm="l1" rescales an observation’s values so they sum to 1,

which can sometimes be a desirable quality:

```
# Print sum
print("Sum of the first observation \' s values:",
features_l1_norm[0, 0] + features_l1_norm[0, 1])
```

```
Sum of the first observation's values: 1.0
```

### 4.4 Generating Polynomial and Interaction Features

**Problem**

You want to create polynomial and interaction features.

**Solution**

Even though some choose to create polynomial and interaction features manually,

scikit-learn offers a built-in method:

```
# Load libraries
import numpy as np
```

**74 | Chapter 4: Handling Numerical Data**

```
from sklearn.preprocessing import PolynomialFeatures
```

```
# Create feature matrix
features = np.array([[2, 3],
[2, 3],
[2, 3]])
```

```
# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias= False )
```

```
# Create polynomial features
polynomial_interaction.fit_transform(features)
```

```
array([[ 2., 3., 4., 6., 9.],
[ 2., 3., 4., 6., 9.],
[ 2., 3., 4., 6., 9.]])
```

The degree parameter determines the maximum degree of the polynomial. For

example, degree=2 will create new features raised to the second power:

```
x 1 ,x 2 ,x 12 ,x 12 ,x 22
```

while degree=3 will create new features raised to the second and third power:

```
x 1 ,x 2 ,x 12 ,x 22 ,x 13 ,x 23 ,x 12 ,x 13 ,x 23
```

Furthermore, by default PolynomialFeatures includes interaction features:

```
x 1 x 2
```

We can restrict the features created to only interaction features by setting

interaction_only to True:

```
interaction = PolynomialFeatures(degree=2,
interaction_only= True , include_bias= False )
```

```
interaction.fit_transform(features)
```

```
array([[ 2., 3., 6.],
[ 2., 3., 6.],
[ 2., 3., 6.]])
```

**Discussion**

Polynomial features are often created when we want to include the notion that there

exists a nonlinear relationship between the features and the target. For example, we

might suspect that the effect of age on the probability of having a major medical

```
4.4 Generating Polynomial and Interaction Features | 75
```

condition is not constant over time but increases as age increases. We can encode that

nonconstant effect in a feature, x, by generating that feature’s higher-order forms (x
2
,

x
3
, etc.).

Additionally, often we run into situations where the effect of one feature is dependent

on another feature. A simple example would be if we were trying to predict whether

or not our coffee was sweet, and we had two features: (1) whether or not the coffee

was stirred, and (2) whether or not we added sugar. Individually, each feature does

not predict coffee sweetness, but the combination of their effects does. That is, a

coffee would only be sweet if the coffee had sugar and was stirred. The effects of each

feature on the target (sweetness) are dependent on each other. We can encode that

relationship by including an interaction feature that is the product of the individual

features.

### 4.5 Transforming Features

**Problem**

You want to make a custom transformation to one or more features.

**Solution**

In scikit-learn, use FunctionTransformer to apply a function to a set of features:

```
# Load libraries
import numpy as np
from sklearn.preprocessing import FunctionTransformer
```

```
# Create feature matrix
features = np.array([[2, 3],
[2, 3],
[2, 3]])
```

```
# Define a simple function
def add_ten(x: int) -> int:
return x + 10
```

```
# Create transformer
ten_transformer = FunctionTransformer(add_ten)
```

```
# Transform feature matrix
ten_transformer.transform(features)
```

```
array([[12, 13],
[12, 13],
[12, 13]])
```

We can create the same transformation in pandas using apply:

**76 | Chapter 4: Handling Numerical Data**

```
# Load library
import pandas as pd
```

```
# Create DataFrame
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
```

```
# Apply function
df.apply(add_ten)
```

```
feature_1 feature_2
0 12 13
1 12 13
2 12 13
```

**Discussion**

It is common to want to make some custom transformations to one or more features.

For example, we might want to create a feature that is the natural log of the values

of a different feature. We can do this by creating a function and then mapping it

to features using either scikit-learn’s FunctionTransformer or pandas’ apply. In the

solution we created a very simple function, add_ten, which added 10 to each input,

but there is no reason we could not define a much more complex function.

### 4.6 Detecting Outliers

**Problem**

You want to identify extreme observations.

**Solution**

Detecting outliers is unfortunately more of an art than a science. However, a common

method is to assume the data is normally distributed and, based on that assumption,

“draw” an ellipse around the data, classifying any observation inside the ellipse as

an inlier (labeled as 1 ) and any observation outside the ellipse as an outlier (labeled

as -1):

```
# Load libraries
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
```

```
# Create simulated data
features, _ = make_blobs(n_samples = 10,
n_features = 2,
centers = 1,
random_state = 1)
```

```
4.6 Detecting Outliers | 77
```

```
# Replace the first observation's values with extreme values
features[0,0] = 10000
features[0,1] = 10000
```

```
# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)
```

```
# Fit detector
outlier_detector.fit(features)
```

```
# Predict outliers
outlier_detector.predict(features)
```

```
array([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

In these arrays, values of -1 refer to outliers whereas values of 1 refer to inliers. A

major limitation of this approach is the need to specify a contamination parameter,

which is the proportion of observations that are outliers—a value that we don’t

know. Think of contamination as our estimate of the cleanliness of our data. If we

expect our data to have few outliers, we can set contamination to something small.

However, if we believe that the data is likely to have outliers, we can set it to a higher

value.

Instead of looking at observations as a whole, we can instead look at individual

features and identify extreme values in those features using interquartile range (IQR):

```
# Create one feature
feature = features[:,0]
```

```
# Create a function to return index of outliers
def indicies_of_outliers(x: int) -> np.array(int):
q1, q3 = np.percentile(x, [25, 75])
iqr = q3 - q1
lower_bound = q1 - (iqr * 1.5)
upper_bound = q3 + (iqr * 1.5)
return np.where((x > upper_bound) | (x < lower_bound))
```

```
# Run function
indicies_of_outliers(feature)
```

```
(array([0]),)
```

IQR is the difference between the first and third quartile of a set of data. You can

think of IQR as the spread of the bulk of the data, with outliers being observations far

from the main concentration of data. Outliers are commonly defined as any value 1.5

IQRs less than the first quartile, or 1.5 IQRs greater than the third quartile.

**78 | Chapter 4: Handling Numerical Data**

**Discussion**

There is no single best technique for detecting outliers. Instead, we have a collection

of techniques all with their own advantages and disadvantages. Our best strategy

is often trying multiple techniques (e.g., both EllipticEnvelope and IQR-based

detection) and looking at the results as a whole.

If at all possible, we should look at observations we detect as outliers and try to

understand them. For example, if we have a dataset of houses and one feature is

number of rooms, is an outlier with 100 rooms really a house or is it actually a hotel

that has been misclassified?

**See Also**

- •Three Ways to Detect Outliers (and the source of the IQR function used in this
  recipe)

### 4.7 Handling Outliers

**Problem**

You have outliers in your data that you want to identify and then reduce their impact

on the data distribution.

**Solution**

Typically we can use three strategies to handle outliers. First, we can drop them:

```
# Load library
import pandas as pd
```

```
# Create DataFrame
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]
```

```
# Filter observations
houses[houses['Bathrooms'] < 20]
```

```
Price Bathrooms Square_Feet
0 534433 2.0 1500
1 392333 3.5 2500
2 293222 2.0 1500
```

```
4.7 Handling Outliers | 79
```

Second, we can mark them as outliers and include “Outlier” as a feature:

```
# Load library
import numpy as np
```

```
# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
```

```
# Show data
houses
```

```
Price Bathrooms Square_Feet Outlier
0 534433 2.0 1500 0
1 392333 3.5 2500 0
2 293222 2.0 1500 0
3 4322032 116.0 48000 1
```

Finally, we can transform the feature to dampen the effect of the outlier:

```
# Log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
```

```
# Show data
houses
```

```
Price Bathrooms Square_Feet Outlier Log_Of_Square_Feet
0 534433 2.0 1500 0 7.313220
1 392333 3.5 2500 0 7.824046
2 293222 2.0 1500 0 7.313220
3 4322032 116.0 48000 1 10.778956
```

**Discussion**

Similar to detecting outliers, there is no hard-and-fast rule for handling them. How

we handle them should be based on two aspects. First, we should consider what

makes them outliers. If we believe they are errors in the data, such as from a broken

sensor or a miscoded value, then we might drop the observation or replace outlier

values with NaN since we can’t trust those values. However, if we believe the outliers

are genuine extreme values (e.g., a house [mansion] with 200 bathrooms), then

marking them as outliers or transforming their values is more appropriate.

Second, how we handle outliers should be based on our goal for machine learning.

For example, if we want to predict house prices based on features of the house, we

might reasonably assume the price for mansions with over 100 bathrooms is driven

by a different dynamic than regular family homes. Furthermore, if we are training a

**80 | Chapter 4: Handling Numerical Data**

model to use as part of an online home loan web application, we might assume that

our potential users will not include billionaires looking to buy a mansion.

So what should we do if we have outliers? Think about why they are outliers, have an

end goal in mind for the data, and, most importantly, remember that not making a

decision to address outliers is itself a decision with implications.

One additional point: if you do have outliers, standardization might not be appropri‐

ate because the mean and variance might be highly influenced by the outliers. In this

case, use a rescaling method more robust against outliers, like RobustScaler.

**See Also**

- •RobustScaler documentation

### 4.8 Discretizating Features

**Problem**

You have a numerical feature and want to break it up into discrete bins.

**Solution**

Depending on how we want to break up the data, there are two techniques we can

use. First, we can binarize the feature according to some threshold:

```
# Load libraries
import numpy as np
from sklearn.preprocessing import Binarizer
```

```
# Create feature
age = np.array([[6],
[12],
[20],
[36],
[65]])
```

```
# Create binarizer
binarizer = Binarizer(threshold=18)
```

```
# Transform feature
binarizer.fit_transform(age)
```

```
array([[0],
[0],
[1],
[1],
[1]])
```

```
4.8 Discretizating Features | 81
```

Second, we can break up numerical features according to multiple thresholds:

```
# Bin feature
np.digitize(age, bins=[20,30,64])
```

```
array([[0],
[0],
[1],
[2],
[3]])
```

Note that the arguments for the bins parameter denote the left edge of each bin. For

example, the 20 argument does not include the element with the value of 20, only

the two values smaller than 20. We can switch this behavior by setting the parameter

right to True:

```
# Bin feature
np.digitize(age, bins=[20,30,64], right= True )
```

```
array([[0],
[0],
[0],
[2],
[3]])
```

**Discussion**

Discretization can be a fruitful strategy when we have reason to believe that a numer‐

ical feature should behave more like a categorical feature. For example, we might

believe there is very little difference in the spending habits of 19- and 20-year-olds,

but a significant difference between 20- and 21-year-olds (the age in the United States

when young adults can consume alcohol). In that example, it could be useful to break

up individuals in our data into those who can drink alcohol and those who cannot.

Similarly, in other cases it might be useful to discretize our data into three or more

bins.

In the solution, we saw two methods of discretization—scikit-learn’s Binarizer for

two bins and NumPy’s digitize for three or more bins—however, we can also use

digitize to binarize features like Binarizer by specifying only a single threshold:

```
# Bin feature
np.digitize(age, bins=[18])
```

```
array([[0],
[0],
[1],
[1],
[1]])
```

**82 | Chapter 4: Handling Numerical Data**

**See Also**

- •digitize documentation

### 4.9 Grouping Observations Using Clustering

**Problem**

You want to cluster observations so that similar observations are grouped together.

**Solution**

If you know that you have k groups, you can use k-means clustering to group

similar observations and output a new feature containing each observation’s group

membership:

```
# Load libraries
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
```

```
# Make simulated feature matrix
features, _ = make_blobs(n_samples = 50,
n_features = 2,
centers = 3,
random_state = 1)
```

```
# Create DataFrame
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
```

```
# Make k-means clusterer
clusterer = KMeans(3, random_state=0)
```

```
# Fit clusterer
clusterer.fit(features)
```

```
# Predict values
dataframe["group"] = clusterer.predict(features)
```

```
# View first few observations
dataframe.head(5)
```

```
feature_1 feature_2 group
0 –9.877554 –3.336145 0
1 –7.287210 –8.353986 2
2 –6.943061 –7.023744 2
3 –7.440167 –8.791959 2
4 –6.641388 –8.075888 2
```

```
4.9 Grouping Observations Using Clustering | 83
```

**Discussion**

We are jumping ahead of ourselves a bit and will go into much more depth about

clustering algorithms later in the book. However, I wanted to point out that we can

use clustering as a preprocessing step. Specifically, we use unsupervised learning algo‐

rithms like k-means to cluster observations into groups. The result is a categorical

feature with similar observations being members of the same group.

Don’t worry if you did not understand all of that: just file away the idea that cluster‐

ing can be used in preprocessing. And if you really can’t wait, feel free to flip to

Chapter 19 now.

### 4.10 Deleting Observations with Missing Values

**Problem**

You need to delete observations containing missing values.

**Solution**

Deleting observations with missing values is easy with a clever line of NumPy:

```
# Load library
import numpy as np
```

```
# Create feature matrix
features = np.array([[1.1, 11.1],
[2.2, 22.2],
[3.3, 33.3],
[4.4, 44.4],
[np.nan, 55]])
```

```
# Keep only observations that are not (denoted by ~) missing
features[~np.isnan(features).any(axis=1)]
```

```
array([[ 1.1, 11.1],
[ 2.2, 22.2],
[ 3.3, 33.3],
[ 4.4, 44.4]])
```

Alternatively, we can drop missing observations using pandas:

```
# Load library
import pandas as pd
```

```
# Load data
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
```

```
# Remove observations with missing values
dataframe.dropna()
```

**84 | Chapter 4: Handling Numerical Data**

```
feature_1 feature_2
0 1.1 11.1
1 2.2 22.2
2 3.3 33.3
3 4.4 44.4
```

**Discussion**

Most machine learning algorithms cannot handle any missing values in the target and

feature arrays. For this reason, we cannot ignore missing values in our data and must

address the issue during preprocessing.

The simplest solution is to delete every observation that contains one or more

missing values, a task quickly and easily accomplished using NumPy or pandas.

That said, we should be very reluctant to delete observations with missing values.

Deleting them is the nuclear option, since our algorithm loses access to the informa‐

tion contained in the observation’s nonmissing values.

Just as important, depending on the cause of the missing values, deleting observations

can introduce bias into our data. There are three types of missing data:

Missing completely at random (MCAR)

```
The probability that a value is missing is independent of everything. For example,
a survey respondent rolls a die before answering a question: if she rolls a six, she
skips that question.
```

Missing at random (MAR)

```
The probability that a value is missing is not completely random but depends
on the information captured in other features. For example, a survey asks about
gender identity and annual salary, and women are more likely to skip the salary
question; however, their nonresponse depends only on information we have
captured in our gender identity feature.
```

Missing not at random (MNAR)

```
The probability that a value is missing is not random and depends on informa‐
tion not captured in our features. For example, a survey asks about annual salary,
and women are more likely to skip the salary question, and we do not have a
gender identity feature in our data.
```

It is sometimes acceptable to delete observations if they are MCAR or MAR. How‐

ever, if the value is MNAR, the fact that a value is missing is itself information.

Deleting MNAR observations can inject bias into our data because we are removing

observations produced by some unobserved systematic effect.

```
4.10 Deleting Observations with Missing Values | 85
```

**See Also**

- •Identifying the 3 Types of Missing Data
- •Missing-Data Imputation

### 4.11 Imputing Missing Values

**Problem**

You have missing values in your data and want to impute them via a generic method

or prediction.

**Solution**

You can impute missing values using k-nearest neighbors (KNN) or the scikit-learn

SimpleImputer class. If you have a small amount of data, predict and impute the

missing values using k-nearest neighbors:

```
# Load libraries
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
```

```
# Make a simulated feature matrix
features, _ = make_blobs(n_samples = 1000,
n_features = 2,
random_state = 1)
```

```
# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)
```

```
# Replace the first feature's first value with a missing value
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan
```

```
# Predict the missing values in the feature matrix
knn_imputer = KNNImputer(n_neighbors=5)
features_knn_imputed = knn_imputer.fit_transform(standardized_features)
```

```
# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_knn_imputed[0,0])
```

```
True Value: 0.8730186114
Imputed Value: 1.09553327131
```

**86 | Chapter 4: Handling Numerical Data**

Alternatively, we can use scikit-learn’s SimpleImputer class from the imputer module

to fill in missing values with the feature’s mean, median, or most frequent value.

However, we will typically get worse results than with KNN:

```
# Load libraries
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
```

```
# Make a simulated feature matrix
features, _ = make_blobs(n_samples = 1000,
n_features = 2,
random_state = 1)
```

```
# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)
```

```
# Replace the first feature's first value with a missing value
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan
```

```
# Create imputer using the "mean" strategy
mean_imputer = SimpleImputer(strategy="mean")
```

```
# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)
```

```
# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])
```

```
True Value: 0.8730186114
Imputed Value: -3.05837272461
```

**Discussion**

There are two main strategies for replacing missing data with substitute values,

each of which has strengths and weaknesses. First, we can use machine learning to

predict the values of the missing data. To do this we treat the feature with missing

values as a target vector and use the remaining subset of features to predict missing

values. While we can use a wide range of machine learning algorithms to impute

values, a popular choice is KNN. KNN is addressed in depth in Chapter 15, but the

short explanation is that the algorithm uses the k nearest observations (according to

some distance metric) to predict the missing value. In our solution we predicted the

missing value using the five closest observations.

The downside to KNN is that in order to know which observations are the closest to

the missing value, it needs to calculate the distance between the missing value and

```
4.11 Imputing Missing Values | 87
```

every single observation. This is reasonable in smaller datasets but quickly becomes

problematic if a dataset has millions of observations. In such cases, approximate

nearest neighbors (ANN) is a more feasible approach. We will discuss ANN in

Recipe 15.5.

An alternative and more scalable strategy than KNN is to fill in the missing values

of numerical data with the mean, median, or mode. For example, in our solution we

used scikit-learn to fill in missing values with a feature’s mean value. The imputed

value is often not as close to the true value as when we used KNN, but we can scale

mean-filling to data containing millions of observations more easily.

If we use imputation, it is a good idea to create a binary feature indicating whether

the observation contains an imputed value.

**See Also**

- •scikit-learn documentation: Imputation of Missing Values
- •A Study of K-Nearest Neighbour as an Imputation Method

**88 | Chapter 4: Handling Numerical Data**

**CHAPTER 5**

### Handling Categorical Data

### 5.0 Introduction

It is often useful to measure objects not in terms of their quantity but in terms of

some quality. We frequently represent qualitative information in categories such as

gender, colors, or brand of car. However, not all categorical data is the same. Sets

of categories with no intrinsic ordering are called nominal. Examples of nominal

categories include:

- •Blue, Red, Green
- •Man, Woman
- •Banana, Strawberry, Apple

In contrast, when a set of categories has some natural ordering we refer to it as

ordinal. For example:

- •Low, Medium, High
- •Young, Old
- •Agree, Neutral, Disagree

Furthermore, categorical information is often represented in data as a vector or

column of strings (e.g., "Maine", "Texas", "Delaware"). The problem is that most

machine learning algorithms require inputs to be numerical values.

The k-nearest neighbors algorithm is an example of an algorithm that requires

numerical data. One step in the algorithm is calculating the distances between obser‐

vations—often using Euclidean distance:

##### 89

∑i= 1

```
n
xi−yi
2
```

where x and y are two observations and subscript i denotes the value for the obser‐

vations’ ith feature. However, the distance calculation obviously is impossible if the

value of xi is a string (e.g., "Texas"). Instead, we need to convert the string into some

numerical format so that it can be input into the Euclidean distance equation. Our

goal is to transform the data in a way that properly captures the information in the

categories (ordinality, relative intervals between categories, etc.). In this chapter we

will cover techniques for making this transformation as well as overcoming other

challenges often encountered when handling categorical data.

### 5.1 Encoding Nominal Categorical Features

**Problem**

You have a feature with nominal classes that has no intrinsic ordering (e.g., apple,

pear, banana), and you want to encode the feature into numerical values.

**Solution**

One-hot encode the feature using scikit-learn’s LabelBinarizer:

```
# Import libraries
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
```

```
# Create feature
feature = np.array([["Texas"],
["California"],
["Texas"],
["Delaware"],
["Texas"]])
```

```
# Create one-hot encoder
one_hot = LabelBinarizer()
```

```
# One-hot encode feature
one_hot.fit_transform(feature)
```

```
array([[0, 0, 1],
[1, 0, 0],
[0, 0, 1],
[0, 1, 0],
[0, 0, 1]])
```

We can use the classes\_ attribute to output the classes:

**90 | Chapter 5: Handling Categorical Data**

```
# View feature classes
one_hot.classes_
```

```
array(['California', 'Delaware', 'Texas'],
dtype='<U10')
```

If we want to reverse the one-hot encoding, we can use inverse_transform:

```
# Reverse one-hot encoding
one_hot.inverse_transform(one_hot.transform(feature))
```

```
array(['Texas', 'California', 'Texas', 'Delaware', 'Texas'],
dtype='<U10')
```

We can even use pandas to one-hot encode the feature:

```
# Import library
import pandas as pd
```

```
# Create dummy variables from feature
pd.get_dummies(feature[:,0])
```

```
California Delaware Texas
0 0 0 1
1 1 0 0
2 0 0 1
3 0 1 0
4 0 0 1
```

One helpful feature of scikit-learn is the ability to handle a situation where each

observation lists multiple classes:

```
# Create multiclass feature
multiclass_feature = [("Texas", "Florida"),
("California", "Alabama"),
("Texas", "Florida"),
("Delaware", "Florida"),
("Texas", "Alabama")]
```

```
# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()
```

```
# One-hot encode multiclass feature
one_hot_multiclass.fit_transform(multiclass_feature)
```

```
array([[0, 0, 0, 1, 1],
[1, 1, 0, 0, 0],
[0, 0, 0, 1, 1],
[0, 0, 1, 1, 0],
[1, 0, 0, 0, 1]])
```

Once again, we can see the classes with the classes\_ method:

```
5.1 Encoding Nominal Categorical Features | 91
```

```
# View classes
one_hot_multiclass.classes_
```

```
array(['Alabama', 'California', 'Delaware', 'Florida', 'Texas'], dtype=object)
```

**Discussion**

We might think the proper strategy is to assign each class a numerical value (e.g.,

Texas = 1, California = 2). However, when our classes have no intrinsic ordering

(e.g., Texas isn’t “less” than California), our numerical values erroneously create an

ordering that is not present.

The proper strategy is to create a binary feature for each class in the original feature.

This is often called one-hot encoding (in machine learning literature) or dummying

(in statistical and research literature). Our solution’s feature was a vector containing

three classes (i.e., Texas, California, and Delaware). In one-hot encoding, each class

becomes its own feature with 1s when the class appears and 0s otherwise. Because

our feature had three classes, one-hot encoding returned three binary features (one

for each class). By using one-hot encoding we can capture the membership of an

observation in a class while preserving the notion that the class lacks any sort of

hierarchy.

Finally, it is often recommended that after one-hot encoding a feature, we drop one of

the one-hot encoded features in the resulting matrix to avoid linear dependence.

**See Also**

- •Dummy Variable Trap in Regression Models, Algosome
- •Dropping one of the columns when using one-hot encoding, Cross Validated

### 5.2 Encoding Ordinal Categorical Features

**Problem**

You have an ordinal categorical feature (e.g., high, medium, low), and you want to

transform it into numerical values.

**Solution**

Use the pandas DataFrame replace method to transform string labels to numerical

equivalents:

```
# Load library
import pandas as pd
```

```
# Create features
```

**92 | Chapter 5: Handling Categorical Data**

```
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})
```

```
# Create mapper
scale_mapper = {"Low":1,
"Medium":2,
"High":3}
```

```
# Replace feature values with scale
dataframe["Score"].replace(scale_mapper)
```

```
0 1
1 1
2 2
3 2
4 3
Name: Score, dtype: int64
```

**Discussion**

Often we have a feature with classes that have some kind of natural ordering. A

famous example is the Likert scale:

- •Strongly Agree
- •Agree
- •Neutral
- •Disagree
- •Strongly Disagree

When encoding the feature for use in machine learning, we need to transform the

ordinal classes into numerical values that maintain the notion of ordering. The most

common approach is to create a dictionary that maps the string label of the class to a

number and then apply that map to the feature.

It is important that our choice of numeric values is based on our prior information on

the ordinal classes. In our solution, high is literally three times larger than low. This

is fine in many instances but can break down if the assumed intervals between the

classes are not equal:

```
dataframe = pd.DataFrame({"Score": ["Low",
"Low",
"Medium",
"Medium",
"High",
"Barely More Than Medium"]})
```

```
scale_mapper = {"Low":1,
"Medium":2,
"Barely More Than Medium":3,
```

```
5.2 Encoding Ordinal Categorical Features | 93
```

```
"High":4}
```

```
dataframe["Score"].replace(scale_mapper)
```

```
0 1
1 1
2 2
3 2
4 4
5 3
Name: Score, dtype: int64
```

In this example, the distance between Low and Medium is the same as the distance

between Medium and Barely More Than Medium, which is almost certainly not accu‐

rate. The best approach is to be conscious about the numerical values mapped to

classes:

```
scale_mapper = {"Low":1,
"Medium":2,
"Barely More Than Medium":2.1,
"High":3}
```

```
dataframe["Score"].replace(scale_mapper)
```

```
0 1.0
1 1.0
2 2.0
3 2.0
4 3.0
5 2.1
Name: Score, dtype: float64
```

### 5.3 Encoding Dictionaries of Features

**Problem**

You have a dictionary and want to convert it into a feature matrix.

**Solution**

Use DictVectorizer:

```
# Import library
from sklearn.feature_extraction import DictVectorizer
```

```
# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
{"Red": 4, "Blue": 3},
{"Red": 1, "Yellow": 2},
{"Red": 2, "Yellow": 2}]
```

**94 | Chapter 5: Handling Categorical Data**

```
# Create dictionary vectorizer
dictvectorizer = DictVectorizer(sparse= False )
```

```
# Convert dictionary to feature matrix
features = dictvectorizer.fit_transform(data_dict)
```

```
# View feature matrix
features
```

```
array([[ 4., 2., 0.],
[ 3., 4., 0.],
[ 0., 1., 2.],
[ 0., 2., 2.]])
```

By default DictVectorizer outputs a sparse matrix that only stores elements with

a value other than 0. This can be very helpful when we have massive matrices

(often encountered in natural language processing) and want to minimize the mem‐

ory requirements. We can force DictVectorizer to output a dense matrix using

sparse=False.

We can get the names of each generated feature using the get_feature_names

method:

```
# Get feature names
feature_names = dictvectorizer.get_feature_names()
```

```
# View feature names
feature_names
```

```
['Blue', 'Red', 'Yellow']
```

While not necessary, for the sake of illustration we can create a pandas DataFrame to

view the output better:

```
# Import library
import pandas as pd
```

```
# Create dataframe from features
pd.DataFrame(features, columns=feature_names)
```

```
Blue Red Yellow
0 4.0 2.0 0.0
1 3.0 4.0 0.0
2 0.0 1.0 2.0
3 0.0 2.0 2.0
```

```
5.3 Encoding Dictionaries of Features | 95
```

**Discussion**

A dictionary is a popular data structure used by many programming languages;

however, machine learning algorithms expect the data to be in the form of a matrix.

We can accomplish this using scikit-learn’s DictVectorizer.

This is a common situation when working with natural language processing. For

example, we might have a collection of documents and for each document we have

a dictionary containing the number of times every word appears in the document.

Using DictVectorizer, we can easily create a feature matrix where every feature is

the number of times a word appears in each document:

```
# Create word count dictionaries for four documents
doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}
```

```
# Create list
doc_word_counts = [doc_1_word_count,
doc_2_word_count,
doc_3_word_count,
doc_4_word_count]
```

```
# Convert list of word count dictionaries into feature matrix
dictvectorizer.fit_transform(doc_word_counts)
```

```
array([[ 4., 2., 0.],
[ 3., 4., 0.],
[ 0., 1., 2.],
[ 0., 2., 2.]])
```

In our toy example there are only three unique words (Red, Yellow, Blue) so there

are only three features in our matrix; however, you can imagine that if each document

was actually a book in a university library our feature matrix would be very large (and

then we would want to set sparse to True).

**See Also**

- •How to Create Dictionaries in Python
- •SciPy Sparse Matrices

### 5.4 Imputing Missing Class Values

**Problem**

You have a categorical feature containing missing values that you want to replace with

predicted values.

**96 | Chapter 5: Handling Categorical Data**

**Solution**

The ideal solution is to train a machine learning classifier algorithm to predict the

missing values, commonly a k-nearest neighbors (KNN) classifier:

```
# Load libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
```

```
# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45],
[1, 1.18, 1.33],
[0, 1.22, 1.27],
[1, -0.21, -1.19]])
```

```
# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31],
[np.nan, -0.67, -0.22]])
```

```
# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])
```

```
# Predict class of missing values
imputed_values = trained_model.predict(X_with_nan[:,1:])
```

```
# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))
```

```
# Join two feature matrices
np.vstack((X_with_imputed, X))
```

```
array([[ 0. , 0.87, 1.31],
[ 1. , -0.67, -0.22],
[ 0. , 2.1 , 1.45],
[ 1. , 1.18, 1.33],
[ 0. , 1.22, 1.27],
[ 1. , -0.21, -1.19]])
```

An alternative solution is to fill in missing values with the feature’s most frequent

value:

```
from sklearn.impute import SimpleImputer
```

```
# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))
```

```
imputer = SimpleImputer(strategy='most_frequent')
```

```
imputer.fit_transform(X_complete)
```

```
array([[ 0. , 0.87, 1.31],
[ 0. , -0.67, -0.22],
```

```
5.4 Imputing Missing Class Values | 97
```

##### [ 0. , 2.1 , 1.45],

##### [ 1. , 1.18, 1.33],

##### [ 0. , 1.22, 1.27],

##### [ 1. , -0.21, -1.19]])

**Discussion**

When we have missing values in a categorical feature, our best solution is to open

our toolbox of machine learning algorithms to predict the values of the missing

observations. We can accomplish this by treating the feature with the missing values

as the target vector and the other features as the feature matrix. A commonly used

algorithm is KNN (discussed in depth in Chapter 15), which assigns to the missing

value the most frequent class of the k nearest observations.

Alternatively, we can fill in missing values with the most frequent class of the feature

or even discard the observations with missing values. While less sophisticated than

KNN, these options are much more scalable to larger data. In any case, it is advisable

to include a binary feature indicating which observations contain imputed values.

**See Also**

- •scikit-learn documentation: Imputation of Missing Values
- •Overcoming Missing Values in a Random Forest Classifier
- •A Study of K-Nearest Neighbour as an Imputation Method

### 5.5 Handling Imbalanced Classes

**Problem**

You have a target vector with highly imbalanced classes, and you want to make

adjustments so that you can handle the class imbalance.

**Solution**

Collect more data. If that isn’t possible, change the metrics used to evaluate your

model. If that doesn’t work, consider using a model’s built-in class weight parame‐

ters (if available), downsampling, or upsampling. We cover evaluation metrics in a

later chapter, so for now let’s focus on class weight parameters, downsampling, and

upsampling.

To demonstrate our solutions, we need to create some data with imbalanced classes.

Fisher’s Iris dataset contains three balanced classes of 50 observations, each indicating

the species of flower (Iris setosa, Iris virginica, and Iris versicolor). To unbalance the

dataset, we remove 40 of the 50 Iris setosa observations and then merge the Iris

**98 | Chapter 5: Handling Categorical Data**

virginica and Iris versicolor classes. The end result is a binary target vector indicating

if an observation is an Iris setosa flower or not. The result is 10 observations of Iris

setosa (class 0) and 100 observations of not Iris setosa (class 1):

```
# Load libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
```

```
# Load iris data
iris = load_iris()
```

```
# Create feature matrix
features = iris.data
```

```
# Create target vector
target = iris.target
```

```
# Remove first 40 observations
features = features[40:,:]
target = target[40:]
```

```
# Create binary target vector indicating if class 0
target = np.where((target == 0), 0, 1)
```

```
# Look at the imbalanced target vector
target
```

```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

Many algorithms in scikit-learn offer a parameter to weight classes during training

to counteract the effect of their imbalance. While we have not covered it yet, Random

ForestClassifier is a popular classification algorithm and includes a class_weight

parameter; learn more about the RandomForestClassifier in Recipe 14.4. You can

pass an argument explicitly specifying the desired class weights:

```
# Create weights
weights = {0: 0.9, 1: 0.1}
```

```
# Create random forest classifier with weights
RandomForestClassifier(class_weight=weights)
```

```
RandomForestClassifier(class_weight={0: 0.9, 1: 0.1})
```

Or you can pass balanced, which automatically creates weights inversely propor‐

tional to class frequencies:

```
5.5 Handling Imbalanced Classes | 99
```

```
# Train a random forest with balanced class weights
RandomForestClassifier(class_weight="balanced")
```

```
RandomForestClassifier(class_weight='balanced')
```

Alternatively, we can downsample the majority class or upsample the minority class.

In downsampling, we randomly sample without replacement from the majority class

(i.e., the class with more observations) to create a new subset of observations equal

in size to the minority class. For example, if the minority class has 10 observations,

we will randomly select 10 observations from the majority class and use those 20

observations as our data. Here we do exactly that using our unbalanced iris data:

```
# Indicies of each class's observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]
```

```
# Number of observations in each class
n_class0 = len(i_class0)
n_class1 = len(i_class1)
```

```
# For every observation of class 0, randomly sample
# from class 1 without replacement
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace= False )
```

```
# Join together class 0's target vector with the
# downsampled class 1's target vector
np.hstack((target[i_class0], target[i_class1_downsampled]))
```

```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

```
# Join together class 0's feature matrix with the
# downsampled class 1's feature matrix
np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5]
```

```
array([[ 5. , 3.5, 1.3, 0.3],
[ 4.5, 2.3, 1.3, 0.3],
[ 4.4, 3.2, 1.3, 0.2],
[ 5. , 3.5, 1.6, 0.6],
[ 5.1, 3.8, 1.9, 0.4]])
```

Our other option is to upsample the minority class. In upsampling, for every observa‐

tion in the majority class, we randomly select an observation from the minority class

with replacement. The result is the same number of observations from the minority

and majority classes. Upsampling is implemented very similarly to downsampling,

just in reverse:

```
# For every observation in class 1, randomly sample from class 0 with
# replacement
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace= True )
```

```
# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate((target[i_class0_upsampled], target[i_class1]))
```

**100 | Chapter 5: Handling Categorical Data**

```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

```
# Join together class 0's upsampled feature matrix with class 1's feature matrix
np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5]
```

```
array([[ 5. , 3.5, 1.6, 0.6],
[ 5. , 3.5, 1.6, 0.6],
[ 5. , 3.3, 1.4, 0.2],
[ 4.5, 2.3, 1.3, 0.3],
[ 4.8, 3. , 1.4, 0.3]])
```

**Discussion**

In the real world, imbalanced classes are everywhere—most visitors don’t click the

buy button, and many types of cancer are thankfully rare. For this reason, handling

imbalanced classes is a common activity in machine learning.

Our best strategy is simply to collect more observations—especially observations

from the minority class. However, often this is just not possible, so we have to resort

to other options.

A second strategy is to use a model evaluation metric better suited to imbalanced

classes. Accuracy is often used as a metric for evaluating the performance of a model,

but when imbalanced classes are present, accuracy can be ill suited. For example,

if only 0.5% of observations have some rare cancer, then even a naive model that

predicts nobody has cancer will be 99.5% accurate. Clearly this is not ideal. Some

better metrics we discuss in later chapters are confusion matrices, precision, recall, F 1

scores, and ROC curves.

A third strategy is to use the class weighing parameters included in implementations

of some models. This allows the algorithm to adjust for imbalanced classes. Fortu‐

nately, many scikit-learn classifiers have a class_weight parameter, making it a good

option.

The fourth and fifth strategies are related: downsampling and upsampling. In down‐

sampling we create a random subset of the majority class of equal size to the minority

class. In upsampling we repeatedly sample with replacement from the minority class

to make it of equal size as the majority class. The decision between using downsam‐

pling and upsampling is context-specific, and in general we should try both to see

which produces better results.

```
5.5 Handling Imbalanced Classes | 101
```
