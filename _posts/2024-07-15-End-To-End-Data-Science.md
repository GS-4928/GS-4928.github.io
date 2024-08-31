---
layout: post
title: End-To-End Data Science Project: 
image: "/posts/ab-testing-title-img.png"
tags: [Data Science, Data Validation, Business Requirements, Python]
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

The product team at Tasty Bytes want our help with analysing and then predicting which recipes will be popular within their available catalogue. Picking a popular recipe to display on the homepage of the Tasty Bytes website can increase traffic to the rest of the website by up to 40%! The asks from the product team are to:
- Predict which recipes will lead to high traffic for the website.
- Correctly predict these high traffic recipes 80% of the time.

<br>
<br>

### Actions <a name="overview-actions"></a>

After initial data cleaning and validation to ensure that entries were in the desired form, exploratory data analysis was carried out to gain a better look at different factors of the data.

Some of these included which recipes were most popular and what correlation there was between features of the data and high traffic being driven to the main website of the company.

With exploratory analysis completed, the dataset was manipulated into a form that would be suitable for passing through a range of machine learning classification models, with the goal of producing a model with a precision score of at least 80%.

Precision was the chosen metric as the product team was concerned with correctly identifying the recipes that drove high traffic to the rest of the website from the recipes that did not. This will be expanded upon later on!

<br>
<br>

### Results & Discussion <a name="overview-results"></a>

A Logistic Regression model was found to be the best performing of teh tested models, with a precision score of ~83%, and was the recommended model to be implemented currently.

Insights into the nature of popular recipes were also gleaned: pork, potato and vegetable recipes were the most popular to be displayed on the home page of the website, whilst neither serving size nor nutritional information had any real impact on recipe popularity.

Recommendations were made to evaluate the suitability of current recipes for display on the homepage, as well as the suggestion to productionise the logistic regression model to predict likelihood of popularity for newly added recipes.

The potential for A/B testing of recipe display on the homepage and the effect that it could have on driving traffic to the rest of the website was aos offered as another avenue of exploration for the product team.

<br>
<br>
___

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
|  | **recipe** | **calories** | **carbohydrate** | **sugar** | **protein** | **category** | **servings** | **high_traffic** |
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

| **category** |  |
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

| **category** |  |
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

| **high_traffic**|
|---|---|
| True | 535 |
| False | 360 |

Our dataframe has now been cleaned and validated into the desired form!

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
| category |---|---|
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
![alt text](/img/posts/high_traffic_by_serving.png "High Traffic Recipe Servings")
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
| servings |---|---|
| 1 | 0.414201 | 0.585799 |
| 2 | 0.436782 | 0.563218 |
| 4 | 0.400545 | 0.599455 |
| 6 | 0.362162 | 0.637838 |

Proportionally, we see that 6 servings is the highest in popularity for driving traffic to Tasty Bytes.

However the difference between the highest and lowest proportion is only ~7%, so perhaps there is not too much of a link between the serving portions per recipe and the traffic driven to the Tasty Bytes website.

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
