---
layout: post
title: End to End Data Science Workflow - Popular Recipe Insights
image: "/posts/end-to-end1.jpg"
tags: [Data Science, Business Requirements, Python, OOP, Business Insights]
---
In this end-to-end data science project, we investigated recipe popularity for a meal subscription service to see what insights could be uncovered to drive further website traffic and subscriptions to the business! 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Data Validation & Preparation](#data-overview)
- [02. Exploratory Data Analysis](#eda)
- [03. Model Development and Evaluation](#model-overview)
- [04. Discussion](#discussion)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The product team at a meal subscription service provider want our help with analysing and then predicting which recipes will be popular within their available catalogue.

Picking a popular recipe to display on the homepage of the website can increase traffic to the rest of the website by up to 40%, according to testing carried out by the Product Team!

The asks from the product team are to:
- Predict which recipes will lead to high traffic for the website.
- Correctly predict these high traffic recipes 80% of the time.

<br>

### Actions <a name="overview-actions"></a>

After initial data cleaning and validation to ensure that entries were in the desired form, exploratory data analysis was carried out to gain a better look at different factors of the data.

Some of these included which recipes were most popular and what correlation there was between features of the data and high traffic being driven to the main website of the company.

With exploratory analysis completed, the dataset was manipulated into a form that would be suitable for passing through a range of machine learning classification models, with the goal of producing a model with a precision score of at least 80%.

Precision was the chosen metric as the product team was concerned with correctly identifying the recipes that drove high traffic to the rest of the website from the recipes that did not. This will be expanded upon later on!

<br>

### Results & Discussion <a name="overview-results"></a>

A Logistic Regression model was found to be the best performing of teh tested models, with a precision score of ~83%, and was the recommended model to be implemented currently.

Insights into the nature of popular recipes were also gleaned: pork, potato and vegetable recipes were the most popular to be displayed on the home page of the website, whilst neither serving size nor nutritional information had any real impact on recipe popularity.

Recommendations were made to evaluate the suitability of current recipes for display on the homepage, as well as the suggestion to productionise the logistic regression model to predict likelihood of popularity for newly added recipes.

The potential for A/B testing of recipe display on the homepage and the effect that it could have on driving traffic to the rest of the website was aos offered as another avenue of exploration for the product team.

____
<br>

# Data Validation & Preparation  <a name="data-overview"></a>

First things first, we need to take a look at the data that has been supplied to us. This data contains a snapshot of information about the recipes available to the product team and should contain 8 distinct columns.
1. recipe - a numeric column identifying each recipe
2. calories - a numeric column detailing the number of calories within a recipe
3. carbohydrate - a numeric column giving the weight of carbohydrates in the recipe, in grams
4. sugar - a numeric column giving the weight of sugar in the recipe, in grams
5. protein - a numeric column giving the weight of protein in the recipe, in grams
6. category - a categorical column detailing the type of recipe present from 10 possible options
7. servings - a numeric column providing the number of servings each recipe produces
8. high_traffic - a categorical column detailing whether traffic was high to the website when the recipe was shown on the homepage

We can load in this data via python and begin to check it over.

<br>

```python
#import packages for data validation
import pandas as pd
import numpy as np

#read in recipe data
recipes = pd.read_csv(...)
#look at summary information on the dataframe
print(recipes.info())
```
<br>
Information on each column within the dataset can be seen below:
<br>
<br>

| **#** | **Column** | **Non-Null Count** | **Dtype** |
|---|---|---|---|
| 0 | recipe | 947 non-null | int64 |
| 1 | calories | 895 non-null | float64 |
| 2 | carbohydrate | 895 non-null | float64 |
| 3 | sugar | 895 non-null | float64 |
| 4 | protein | 895 non-null | float64 |
| 5 | category | 947 non-null | object |
| 6 | servings | 947 non-null | object |
| 7 | high_traffic | 574 non-null | object |

___
<br>

#### 0. recipe

The recipe column seems to be in the form we want it, containing 947 unique integer numbers each corresponding to a different recipe. We should check whether there are any duplicated recipes within this dataset though.

```python
#check for duplicates
print(f"{recipes.duplicated(subset='recipe').sum()} Duplicates")

>>0 Duplicates
```
<br>

#### 1. calories

The calories column is also numeric, this time in decimal form, but with missing values present in the column. What percentage of the column is composed of missing values?

```python
#check percentage of missing values in calorie column
missing_percentage = round(100*recipes['calories'].isna().sum()/len(recipes),2)

#display missing percentage
print(f"Calorie Column Mising/Null Values: {missing_percentage}%")

>>Calorie Column Missing/Null Values: 5.49%
```
We have ~5.5% of our calorie column being taken by missing/null values (52 rows), but from the initial information on the dataset it seems that all of the nutritional columns have the same number of missing values! Lets check the data frame for null values to see what the link is.

```python
#check where values are missing, and display the first five rows
recipes[recipes['calories'].isna().head()]
```
| # | **recipe** | **calories** | **carbohydrate** | **sugar** | **protein** | **category** | **servings** | **high_traffic** |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | NaN | NaN | NaN | NaN | Pork | 6 | High |
| 23 | 24 | NaN | NaN | NaN | NaN | Meat | 6 | NaN |
| 48 | 49 | NaN | NaN | NaN | NaN | Chicken Breat | 6 | NaN |
| 82 | 83 | NaN | NaN | NaN | NaN | Meat | 6 | High |
| 89 | 84 | NaN | NaN | NaN | NaN | Pork | 6 | High |

This initial look is replicated across the remaining null values: Every missing/null entry within the calories column is matched with missing values across carbohydrate, sugar and protein columns too! With this in mind, it makes sense to drop these rows where null values are present to avoid mass imputation across four of our columns.

```python
#drop missing rows across nutritional information
recipes = recipes[~recipes['calories'].isna()]

#check the information on our dataframe
recipes.info()
```

| **#** | **Column** | **Non-Null Count** | **Dtype** |
|---|---|---|---|
| 0 | recipe | 895 non-null | int64 |
| 1 | calories | 895 non-null | float64 |
| 2 | carbohydrate | 895 non-null | float64 |
| 3 | sugar | 895 non-null | float64 |
| 4 | protein | 895 non-null | float64 |
| 5 | category | 895 non-null | object |
| 6 | servings | 895 non-null | object |
| 7 | high_traffic | 535 non-null | object |

With these rows dropped, we can move onto the other columns!

<br>

#### 2. carbohydrate, 3. sugar, 4. protein

Thanks to our prior checks when validating the calories column, we don't need to do any further changes to these three columns!
They are in numeric, decimal form as we want with no missing values now that we have removed the null values that were flagged in the process of cleaning the calories column.

<br>

#### 5. category

The category column is currently present as a categorical column, which should have 10 possible entries. Let's check.

```python
#check category column for possible entries
print(recipes['category'].value_counts())
```

| **category** | **count** |
|---|---|
| Breakfast | 106 |
| Chicken Breast | 94 |
| Beverages | 92 |
| Potato | 83 |
| Lunch/Snacks | 82 |
| Vegetable | 78 |
| Dessert | 77 |
| Meat | 74 |
| Pork | 73 |
| Chicken | 69 |
| One Dish Meal | 67 |

We have 11 entries in this column, and we can see that there has been some overlap with the chicken recipes, with one entry for chicken and one entry for chicken breast.

We can combine these into the expected Chicken category.

```python
#update category column
recipes['category'].replace('Chicken Breast','Chicken',inplace=True)
#cast category column as category
recipes['category'] = recipes['category'].astype('category')
#check entries within category column
recipes['category'].value_counts()
```

| **category** | **count** |
|---|---|
| Chicken | 163 |
| Breakfast | 106 |
| Beverages | 92 |
| Potato | 83 |
| Lunch/Snacks | 82 |
| Vegetable | 78 |
| Dessert | 77 |
| Meat | 74 |
| Pork | 73 |
| One Dish Meal | 67 |

<br>

#### 6. servings

The vast majority of entries are numbers in string form, apart from three entries with text in them about snacks!

We can combine these entries with their respective numerical points; '4 as a snack' with '4' and '6 as a snack' with '6', and then convert the remaining entries into integers to satisfy the numeric condition for the column.

```python
#update servings row to string form and then remove 'as a snack' section
recipes['servings'] = recipes['servings'].astype('str')
recipes['servings'] = recipes['servings'].str.replace(' as a snack','')

#update servings column as integer type
recipes['servings'] = recipes['servings'].astype('int64')

print(recipes.info())
```

| **#** | **Column** | **Non-Null Count** | **Dtype** |
|---|---|---|---|
| 0 | recipe | 895 non-null | int64 |
| 1 | calories | 895 non-null | float64 |
| 2 | carbohydrate | 895 non-null | float64 |
| 3 | sugar | 895 non-null | float64 |
| 4 | protein | 895 non-null | float64 |
| 5 | category | 895 non-null | object |
| 6 | servings | 895 non-null | int64 |
| 7 | high_traffic | 535 non-null | object |

<br>

#### 7. high_traffic

This is expected as a categorical column, with a 'High' marker if traffic was high to the website.

This is present within the column but it leaves us with null values in the column too! To counteract this, the column will be transformed into a boolean form, using 'True' to represent High and 'False' to represent the null values, which are assumed to correspond to a recipe that doesn't drive high traffic to the website.

```python
#converting high_traffic column to boolean form
recipes['high_traffic'] = np.where(recipes['high_traffic']=='High','True','False')

#convert to category
recipes['high_traffic'] = recipes['high_traffic'].astype('category')

#check values within high_traffic column
recipes['high_traffic'].value_counts()
```

| **high_traffic**| **count** |
|---|---|
| True | 535 |
| False | 360 |

Our dataframe has now been cleaned and validated into the desired form!
<br>
___
<br>

# Exploratory Data Analysis  <a name="eda"></a>

With our data cleaned for the moment, we can begin to explore connections that may be present within the data.

The business purpose for this exercise is to be able to accurately deduce (~80% accuracy) which recipes are popular and driving high traffic to the rest of the website.

To start off with, let us examine the distribution of each recipe type within the dataset.

#### Single Variable Analysis

```python
#import plotting functionality
import seaborn as sns
import matplotlib.pyplot as plt

#set style for plots
sns.set_style('darkgrid')
sns.color_palette('CMRmap')

#countplot showing the number of recipes of each category in the dataset
sns.countplot(recipes['category'])
plt.title('Count of recipe type')
plt.show()
```
<br>
![alt text](/img/posts/recipe_countplot.png "Count of Recipes Within Dataset")
<br>

Chicken recipes are comfortably the largest proportion of recipes, with the other nine categories on a relatively even footing ranging from 60 to 100.

This could be because chicken recipes are seen as easier/less time consuming to prepare, in which case it may be reflected in the high traffic count.

There is a good spread of meal types within this dataset, now we can check the spread of data across the numeric columns utilising histograms.

<br>

```python
#histogram for float columns
float_columns = recipes.select_dtypes(include='float').columns
#number of rows for plot
row_num = len(float_columns)

#set up figures and axes for subsequent plots
fig, axes = plt.subplots(row_num,1,figsize = (12,5*row_num))

#iterate over columns for histograms
for i,col in enumerate(float_columns):
    ax = axes[i] if row_num > 1 else axes
    sns.histplot(data=recipes,x=col,kde=True,ax=ax)
    ax.set_title(f"Distribution of Data within {col.capitalize()} column")

#adjust for spacing between plots and display
plt.tight_layout()
plt.show()
```

<br>
![alt text](/img/posts/recipe_data_histplot.png "Data Spread within Nutritional Columns")
<br>

Looking at the distribution of the data across our numerical columns, we can see a right-tailed distribution for all four, and so an approach involving the median and inter-quartile range for outlier calculation should be applied instead of utilising the mean and standard deviation.

Mean and standard deviation are susceptible to influence by outliers within a dataset and tend to be used when the data approximates a normal distribution.

Median and inter-quartile range can better ignore these outliers and provide a more representative measure of central tendency and spread respectively.

We will come onto outlier calculation within the model development section.

<br>

#### Multi-Variable Analysis

We've looked at individual variables here, now let's see how they are related.

Grouping by category and finding the median numerical values would be a good place to start.

```python
#group by category
median_by_category = recipes.groupby('category')[float_columns].median().reset_index()

#plot out the median values for each category across each food group
for col in float_columns:
    #create the plot
    plt.figure()
    median_by_category.plot(kind='bar',x='category',y=col)
    
    #add labels
    plt.xlabel('Category')
    plt.ylabel(f"{col.capitalize()}")
    plt.title(f"Median {col.capitalize()} Across Category")

#show the plots
plt.show()
```

The collection of these plots is shown below.

<br>
![alt text](/img/posts/median_category_combo.png "Median Values Across Nutritional Columns")
<br>

The medians across each category and nutritional type are not symmetrical, with lots of variance across categories and between food groups.

Protein sees spikes across the meat dishes (Chicken, Meat, Pork) as you would expect along with One Dish Meals, likely to contain some form of protein or meat.

Conversely, Carbohydrate sees the highest values across meals without a specific emphasis on some form of meat (Breakfast, Potato, Dessert) with One Dish Meal appearing as well, pointing towards a well-rounded approach to the recipes used within One Dish Meals!

Sugar is topped by dessert by a long margin, but these peak values across the food groups do not map directly onto a higher calorie value, with protein being the closest approximation.

___
<br>

Coming back to the business objective in this assignment, we should compare what drives high and low traffic to the Tasty Bytes website, using the different categories and servings for each recipe.

First, we can look at the different categories.

```python
#Compare categories to see what drives high traffic
sns.countplot(data=recipes,y='category',hue='high_traffic')
plt.ylabel('Category')
plt.xlabel('Count')
plt.title('Recipe Categories Driving High Traffic')
plt.show()
```

<br>
![alt text](/img/posts/high_traffic_by_category.png "High Traffic Recipe Categories")
<br>

Interesting information to be gleaned here! Beverages and Breakfast recipes seem to be much less popular where Vegetable, Potato and Pork recipes have a large number of High traffic flags.

We should check the normalised values for each category to see what the proportional popularity is.

```python
#pivot table with categories
categories_table = recipes.pivot_table(index='category',columns='high_traffic',values='recipe',aggfunc='count')
#normalise table for proportional values
norm_categories_table = categories_table.div(categories_table.sum(axis=1),axis=0)

#look at produced pivot table
print(norm_categories_table)
```

| **high_traffic** | **False** | **True** |
|---|---|---|
| **category** |---|---|
| Beverages | 0.945652 | 0.054348 |
| Breakfast | 0.688679 | 0.311321 |
| Chicken | 0.576687 | 0.423313 |
| Dessert | 0.376623 | 0.623377 |
| Lunch/Snacks | 0.365854 | 0.634146 |
| Meat | 0.243243 | 0.756757 |
| One Dish Meal | 0.238806 | 0.761194 |
| Pork | 0.095890 | 0.904110 |
| Potato | 0.060241 | 0.939759 |
| Vegetable | 0.012821 | 0.987179 |

This proportional look confirms what we supposed based on the count plot; that Vegetable, Potato, and Pork recipes have the highest proportion of High traffic responses.

Beverages and Breakfast recipes seem much more unpopular!

Interestingly, while on a magnitude basis Chicken recipes have the third highest High traffic responses, on a proportional level they are the third lowest category.

Now to look at the breakdown of servings by traffic flag!

```python
#Compare servings to see what drives high traffic
sns.countplot(data=recipes,y='servings',hue='high_traffic')
plt.ylabel('Servings')
plt.xlabel('Count')
plt.title('Recipe Servings Driving High Traffic')
plt.show()
```

<br>
![alt text](/img/posts/high_traffic_by_servings.png "High Traffic Recipe Servings")
<br>

On a magnitude level, 4 servings and 6 servings are the most popular recipes, with 4 comfortably largest. 1 and 2 servings are at a similar level of popularity to each other, behind 4 and 6.

How does that translate into a proportional view?

```python
#pivot table with categories
servings_table = recipes.pivot_table(index='servings',columns='high_traffic',values='recipe',aggfunc='count')
#normalise table for proportional values
norm_servings_table = servings_table.div(servings_table.sum(axis=1),axis=0)

#look at produced pivot table
print(norm_servings_table)
```

| **high_traffic** | **False** | **True** |
|---|---|---|
| **servings** |---|---|
| 1 | 0.414201 | 0.585799 |
| 2 | 0.436782 | 0.563218 |
| 4 | 0.400545 | 0.599455 |
| 6 | 0.362162 | 0.637838 |

Proportionally, we see that 6 servings is the highest in popularity for driving traffic to Tasty Bytes.

However the difference between the highest and lowest proportion is only ~7%, so perhaps there is not too much of a link between the serving portions per recipe and the traffic driven to the Tasty Bytes website.

___
<br>

Before we begin modelling, we should check if there is any correlation between features within this dataset that we should be concerned about.

Links between data would point to multi-colinearity - one feature of a dataset predicting another feature of the same dataset - which would need to be dealt with.

We will cover multi-colinearity in more depth within our final data pre-processing steps before we begin to model.

```python
#pairplot of numeric features within dataset
interested_columns = ['calories', 'carbohydrate', 'sugar', 'protein', 'category', 'servings', 'high_traffic']
#create figure object to enable title addition
g = sns.pairplot(recipes[interested_columns],hue='high_traffic')
#add title
g.fig.suptitle('Relationships Between Features and Popularity',y=1.02)
#show plot
plt.show()
```
<br>
![alt text](/img/posts/pairplot.png "Correlation Between Features in Dataset")
<br>
<br>

This pairplot confirms for us that there is no strong correlation between our numeric features and whether high traffic is driven to the website.

___
<br>

# Model Development and Evaluation <a name="model-overview"></a>

The product department want a model that is able to predict which recipes will and won't drive high traffic to the main website.

This is a quintessential binary classification problem; we need our model to predict whether a recipe is high traffic or not based on the dataset that we have been supplied.

Before we begin to consider what we should choose for our binary classification model, there is some data preprocessing that needs to be done to turn that existing dataset into one that can be ingested properly for modelling and predicting.

#### Data Pre-Processing

One thing to consider with our data is the presence of outliers.

We saw earlier that for each of Calories, Carbohydrate, Protein and Sugar, the data skewed into a right tailed distribution.

Plotting box plots for each of these variables will demonstrate this skew well.

```python
#define float columns
float_columns = recipes.select_dtypes(include='float').columns

#number of rows for plotting subplots
row_num = len(float_columns)

#set up figures and axes for subsequent plots
fig_box, axes_box = plt.subplots(row_num,1,figsize = (12,5*row_num))

#iterate over columns for histograms
for i,col in enumerate(float_columns):
    ax = axes_box[i] if row_num > 1 else axes_box
    sns.boxplot(data=recipes,x=col,ax=ax)
    ax.set_title(f"Box Plot: {col.capitalize()}")
    ax.set_xlabel(f"{col.capitalize()}")

#adjust for spacing between plots and display
plt.tight_layout()
plt.show()
```

<br>
![alt text](/img/posts/boxplot.png "Boxplots for Nutritional Columns")
<br>
<br>

The whiskers of each box plot are at the 95th percentile and 5th percentile respectively.

We can see that quite a few data points in each category fall outside of these bounds, so we need to deal with these outliers.

Removal would be suboptimal, leading to a high degree of data loss. Let's see what the lower and upper bounds are for each column here.

```python
#function to calculate outlier bounds for dataframe columns
def find_outlier_bounds(df,col):
    #define lower and upper quartiles
    lq = df[col].quantile(0.25)
    uq = df[col].quantile(0.75)
    #calculate inter quartile range
    iqr = uq-lq
    #calculate upper and lower bounds for outliers
    u_thresh = uq + 1.5*iqr
    l_thresh = lq - 1.5*iqr
    #return these bounds
    return u_thresh, l_thresh

#define dictionary to store each column name and associated outliers
outlier_limits = {}

#loop over float columns to pull out outliers
for col in float_columns:
    upper,lower =find_outlier_bounds(recipes,col)
    outlier_limits[col] = (upper,lower)

#print out outliers for each column
for col,bounds in outlier_limits.items():
    print(f"{col.capitalize()}\nUpper Limit: {round(bounds[0],2)}\nLower Limit: {round(bounds[1],2)}\nUpper Limit Outliers: {len(recipes[recipes[col]>bounds[0]])}\nLower Limit Outliers: {len(recipes[recipes[col]<bounds[1]])}\n========")
```

A function was defined to calculate the bounds for outliers within each column, this is displayed in a table below:

| **Column** | **Upper Limit** | **Lower Limit** | **Upper Limit Outliers** | **Lower Limit Outliers** |
|---|---|---|---|---|
| **Calories** | 1328.48 | -620.4 | 47 | 0 |
| **Carbohydrate** | 99.85 | -46.51 | 58 | 0 |
| **Sugar** | 21.97 | -10.48 | 79 | 0 |
| **Protein** | 70.71 | -27.31 | 77 | 0 |

<br>

The negative values for the lower limits are not really of concern to us: we have no values for any of these columns below 0.

The upper limits are of interest to us though, with a 47, 58, 79 and 77 points counted as outliers from the Calories, Carbohydrate, Sugar and Protein columns respectively.

We don't want to remove all of these values from our dataset, but we do need to appropriately deal with them.

We don't want to cap, or windsorise, our data as this may affect the ability of our model to generalise on data points that sit outside of these values, as well as maintaining the same type of data distribution that we want to change.

There are a few options available to us in the form of data transformations that we will look at: Logarithmic, Yeo-Johnson, Square Root.

Box-Cox transformations require that all data must be strictly positive values ABOVE 0; we have recipes with protein values equalling zero that we could change to very small values instead but this may introduce a level of bias to the transformation.

This is why we have opted for the Yeo-Johnson transformation instead, as this can take in positive and negative values without the need for us to change zero values. Lets code this up!

```python
#import statistical transformation capability
from scipy.stats import yeojohnson
#set up copy dataframes to perform transformations over
yeo_transformed_data = recipes.copy()
sq_root_transformed_data = recipes.copy()
log_transformed_data = recipes.copy()

#iterate through to transform data
for col in float_columns:
    yeo_transformed_data[col] = yeojohnson(yeo_transformed_data[col])[0]
    sq_root_transformed_data[col] = np.sqrt(sq_root_transformed_data[col])
    log_transformed_data[col] = np.log1p(log_transformed_data[col])
    
#plot out transformed distributions to see changes
row_num = len(float_columns)
fig_trans, axes_trans = plt.subplots(row_num,3,figsize=(17,15))

#loop over columns to plot transformations
for i,col in enumerate(float_columns):
    #yeo-johnson transformation
    ax = axes_trans[i][0]
    sns.histplot(data=yeo_transformed_data,x=col,kde=True,ax=ax)
    ax.set_title(f"Yeo Transformed Distribution of Data within {col.capitalize()} column")
    ax.set_xlabel(f"{col.capitalize()}")
    #square root transformation
    ax_2 = axes_trans[i][1]
    sns.histplot(data=sq_root_transformed_data,x=col,kde=True,ax=ax_2)
    ax_2.set_title(f"Square Root Transformed Distribution of Data within {col.capitalize()} column")
    ax_2.set_xlabel(f"{col.capitalize()}")
    #log transformation
    ax_3 = axes_trans[i][2]
    sns.histplot(data=log_transformed_data,x=col,kde=True,ax=ax_3)
    ax_3.set_title(f"Log Transformed Distribution of Data within {col.capitalize()} column")
    ax_3.set_xlabel(f"{col.capitalize()}")

#sort out layout of figures
plt.tight_layout()
plt.show()
```

<br>
![alt text](/img/posts/transformed_data_histplot.png "Transformed Data")
<br>
<br>

A busy visual! The important thing to note is that the Yeo-Johnson distributions, the left most column of histograms, much more closely approximate a normal distribution across each column.

This covers off our outlier calculations, now to convert our categorical data.

___
<br>

#### Converting Categorical Variables

With our data in the form we expected it to be, we now want to change our categorical columns into a format that will be usable for modelling, through the process of creating dummy variables! These dummy variables encode a single column into a selection of binary ones, returning a 1 if the category is present and a 0 if not.

```python
#create dummy variables from category column
category_dummies = pd.get_dummies(recipes['category'],drop_first=True,dtype='int64')
category_dummies.head()
```

| | **Breakfast** | **Chicken** | **Dessert** | **Lunch/Snacks** | **Meat** | **One Dish Meal** | **Pork** | **Potato** | **Vegetable** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

We have our dummy variables created now! Standard practice is to drop the first newly created column, in this case Beverages, to avoid multi-colinearity.

Multi-colinearity is where one of or a combination of the newly created variables will exactly predict another variable within the dataset.

In this case if the recipe had a value of 0 for Breakfast, Chicken, Dessert, Lunch/Snacks, Meat, One Dish Meal, Pork, Potato and Vegetable then it would have to have a 1 for Beverages: the data on the other variables exactly predicts the data on one variable.

Avoiding this ensures that our model will be more robust in testing and training when we get to creating it, whilst retaining as much information as we can.

We will concatenate our transformed data with this dataframe of dummy variables, dropping the category column and the recipe column, as well as converting the high_traffic column into 1s and 0s now that we have performed our exploratory analysis.

```python
#create usable dataframe for modelling
recipes_model = pd.concat([yeo_transformed_data,category_dummies],axis=1)

#now to drop the category and recipe columns
recipes_model.drop(['recipe','category'],axis=1,inplace=True)

#convert the high_traffic column into Boolean: 1 and 0
recipes_model['high_traffic'] = np.where(recipes_model['high_traffic']=='True',1,0)

#look at resultant dataframe
recipes_model.info()
```
<br>

| **#** | **Column** | **Non-Null Count** | **Dtype** |
|---|---|---|---|
| 0 | calories | 895 non-null | float64 |
| 1 | carbohydrate | 895 non-null | float64 |
| 2 | sugar | 895 non-null | float64 |
| 3 | protein | 895 non-null | float64 |
| 4 | servings | 895 non-null | int64 |
| 5 | high_traffic | 535 non-null | int64 |
| 6 | Breakfast | 895 non-null | int64 |
| 7 | Chicken | 895 non-null | int64 |
| 8 | Dessert | 895 non-null | int64 |
| 9 | Lunch/Snacks | 895 non-null | int64 |
| 10 | Meat | 895 non-null | int64 |
| 11 | One Dish Meal | 895 non-null | int64 |
| 12 | Pork | 895 non-null | int64 |
| 13 | Potato | 895 non-null | int64 |
| 14 | Vegetable | 895 non-null | int64 |

Our data is now in the form we need it to be for modeling, which means that it is time to consider the approach we will take for modeling.

<br>

# Model Development and Evaluation  <a name="model-overview"></a>

The baseline model that we will employ will be a logistic regression model, with a classification threshold at 50%. Some other options that we will consider are a decision tree, random forest and k-nearest neighbours models. Before we begin, we will need to import the relevant modules for our model creation, training, validation and testing. After that we can split our data into training and testing


```python
#import models for model development
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

#import metrics for evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#import data splitting functionality
from sklearn.model_selection import train_test_split

#define features and target, X and y
X = recipes_model.drop('high_traffic',axis=1)
y = recipes_model['high_traffic']

#split out into training and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
```
<br>

With our data split and ready for modeling, we can create a function to evaluate the models to see which performs best for the business use case provided by the product team.

```python
#function to train and predict from out model
def model_fitting_metrics(model, X_train, X_test, y_train, y_test):
    """A function to take in a model instance and data sets and return a batch of metrics based on that model's performance: accuracy, precision, recall, f1 score, confusion matrix"""
    eval_metrics = {}
    #fit our model
    model.fit(X_train,y_train)
    #predict y values
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    #carry out metric calculations for training
    train_model_accuracy = accuracy_score(y_train,y_pred_train)
    train_model_precision = precision_score(y_train,y_pred_train)
    train_model_recall = recall_score(y_train,y_pred_train)
    train_model_f1 = f1_score(y_train,y_pred_train)
    train_model_confusion_matrix = confusion_matrix(y_train,y_pred_train)
    #add metrics to dictionary
    eval_metrics['train'] = [train_model_accuracy,train_model_precision,train_model_recall,train_model_f1,train_model_confusion_matrix]
    
    #carry out metric calculations for testing
    test_model_accuracy = accuracy_score(y_test,y_pred_test)
    test_model_precision = precision_score(y_test,y_pred_test)
    test_model_recall = recall_score(y_test,y_pred_test)
    test_model_f1 = f1_score(y_test,y_pred_test)
    test_model_confusion_matrix = confusion_matrix(y_test,y_pred_test)
    #add metrics to dictionary
    eval_metrics['test'] = [test_model_accuracy,test_model_precision,test_model_recall,test_model_f1,test_model_confusion_matrix]
    
    #return eval_metrics for scrutiny
    return eval_metrics

#function to print out model metrics
def print_metrics(metrics,model_type):
    print(f"{model_type}\n======\nAccuracy Score: {metrics[0]}\nPrecision Score: {metrics[1]}\nRecall Score: {metrics[2]}\nF1 Score: {metrics[3]}\nConfusion Matrix:\n {metrics[4]}\n")    
```
<br>

#### Model Choices

A wide selection of models were chosen for this testing, 9 in total! These range from a simple logistic regression model up through to hyper parameter tuning models, that iterate through a selection of model parameters to find the most optimal selection of values to produce the best model score. The model score that we are concerned with is precision, so that will be our metric of choice. I'll give an example of fitting and predicting the baseline model, after which the results will be summarised.
<br>

```python
#Logistic Regression model
logm = LogisticRegression()
#create object for evaluation metrics
logm_metrics = model_fitting_metrics(logm,X_train,X_test,y_train,y_test)
#inspect metrics
print_metrics(logm_metrics['train'],'Logistic Regression - Train')
print_metrics(logm_metrics['test'],'Logistic Regression - Test')
```
<br>

| **Logistic Regression - Train** | **Score** |
|---|---|
| Accuracy | 0.75139 |
| Precision | 0.79206 |
| Recall | 0.79206 |
| F1 | 0.79206 |
|---|---|
| **Logistic Regression - Test** | **Score** |
| Accuracy | 0.74860 |
| Precision | 0.82979 |
| Recall | 0.72897 |
| F1 | 0.77612 |

<br>
Our initial model already meets the requirements laid out by the product team, with a precision score over 80%! Let's compare this to the other model precision scores.

#### Model Evaluation

As previously mentioned, the main metric that the product team is concerned with for their predictions is Precision: the fraction of positive responses correctly identified by a model from the total number of positive responses identified by the model.

In other words, the number of True Positive (TP) predictions divided by the sum of True Positive and Flse Positive (FP) predictions.

$`Precision = \frac{TP}{TP+FP}`$

Let's have a look at the precision scores for each of the models that we have tested. We will include the F1 score as well - a more generalised score of model performance that balances accurate predictions of correct values **AND** incorrect values, just to see how each model compares.

<br>

| **Model** | **Precision Score (%)** | **F1 Score (%)** |
|---|---|---|
| Logistic Regression | 82.98 | 77.61 |
| Tuned Gradient Boosting | 82.95 | 74.87 |
| Tuned Decision Tree | 81.18 | 71.88 |
| Gradient Boosting | 79.17 | 74.88 |
| Decision Tree | 78.64 | 77.14 |
| Tuned Random Forest | 75.21 | 78.57 |
| Random Forest | 74.59 | 79.48 |
| Tuned K Nearest Neighbours | 71.01 | 55.68 |
| K Nearest Neighbours | 68.27 | 67.30 |

<br>
Out of the 9 models tested, including ones specifically tuned to try to maximise performance, the best three models for the requirements of the product department are the tuned Gradient Boosting Classifier, the tuned Decision Tree and the baseline Logistic Regression models!

The Decision Tree model has a precision score of 81.2% but it is beaten, by both the Gradient Boosting model and the Logistic Regression model, with precision scores of 82.95% and 82.98%.

Virtually nothing separates the logistic Regression and Tuned Gradient Boosting models!

Interestingly if we look at F1 scores, a weighted average of precision and recall, the three best performers are the Tuned Random Forest, the initial Random Forest and our baseline model.

For the purposes of our investigation, Logistic Regression or the Tuned Gradient Boosting model would do but for ease of interpretability and explainability the **Logistic Regression model would be favoured!**

___
<br>

# Discussion <a name="discussion"></a>

As previously stated, the product team are looking for a model that has high precision, being able to correctly identify at least 80% of recipes that, when displayed on the home page, will drive high traffic to the rest of the website.

To further investigate this, we can look at the ratio between True Positives (recipes correctly identified as high traffic) and False Positives (recipes incorrectly identified as high traffic) to further check how efficient our models were at identifying correct incidents of high traffic recipes.

This ratio, which we can term High Traffic Conversion Rate, can act as our Key Performance Indicator (KPI) alongside the precision scores we have seen in our model evaluation.

We are looking for a value above 4 in this KPI: a 4 to 1 ratio converts to an 80% correct identification for our models.

<br>

| **Model** | **High Traffic Conversion Rate** |
|---|---|
| Logistic Regression | 4.88 |
| Tuned Gradient Boosting | 4.87 |
| Tuned Decision Tree | 4.31 |
| Gradient Boosting | 3.80 |
| Decision Tree | 3.68 |
| Tuned Random Forest | 3.03 |
| Random Forest | 2.94 |
| Tuned K Nearest Neighbours | 2.45 |
| K Nearest Neighbours | 2.15 |

<br>

It's a very close call between the Tuned Gradient Boosting Classifier and the Logistic Regression models: 4.87 to 4.88.

Either of these would be a good fit for the demands of the product team, but I would recommend the Logistic Regression model for the simplicity of deployment and increased ease of explainability to stakeholders.

<br>
___

In summary, we have taken the provided data on recipes given to us by the product team, then cleaned and validated it to carry out some initial exploratory analysis.

This initial analysis highlighted some interesting connections: whilst the highest number of recipes were chicken based, chicken recipes had the third lowest percentage of recipes driving high traffic to the main website.

There was a wide range of values across the Calories, Carbohydrate, Protein and Sugar columns that didn't seem to correlate to higher traffic from the home page, with the exception of a small, weak correlation between protein and driving high traffic.

After the initial analysis, further data preprocessing was required to get the data in a form that allowed for modelling. Five different types of model were explored; Logistic Regression as a baseline, Decision Tree, Random Forest, Gradient Boosting and a K Nearest Neighbors model.

From this selection, tuning was carried out to try and maximise the performance of each model if possible.

In the end, the baseline Logistic Regression was the best performer, with a precision score of 82.98% just beating out the tuned Gradient Boosting Classifier (82.95%).

The Key Performance Indicator of High Traffic Conversion Rate supports this case, with the Logistic Regression model (4.88) again beating out the Tuned Gradient Boosting Classifier (4.87).

Logistic Regression is the model that I would recommend to the product team based on these factors.

Next steps from this project would be to gather more recipe data to train and test a wider selection of models on, potentially using image data of the recipes in conjunction with numerical features to train a Convolutional Neural Network to classify which recipes are popular within the product team's catalogue.

The low volume of data available for testing each model would be another consideration for the product team, as further data within each training and testing set would make for a more reliable and reproducible modelling outcome across the board.

A recommendation for the product team moving forwards would be to productionise the most successful model in this trial so that new recipes can be passed through and the probability of driving high traffic to the rest of the Tasty Bytes website can be established for each new recipe.

Alongside this, it would be worth experimenting with different displayed forms of a recipe on the homepage in an A/B test to determine if there are any improvements or tweaks that can be made to drive further traffic to the main website.
