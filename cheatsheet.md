# CheatSheet for ADS

```python
# Read CSV
df = pd.read_csv("filename")
# For Numeric Columns Only
df_numerics_only = df.select_dtypes(include=np.number)
```
```python
# Load Dataset directly from SciKitLearn
from sklearn import datasets

iris = datasets.load_iris()
df1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
```
## Exp 1: Descriptive Statistics
```python

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
%matplotlob inline

# Count/Mean/Median/Mode/Min/Max/Sum/Std/Var/Corelation Coeff: 
df.column_name.count()
df.column_name.mean()
df.column_name.median()
df.column_name.mode()
df.column_name.min()
df.column_name.max()
df.column_name.sum()
df.column_name.std()
df.column_name.var()
df.column_name1.corr(df.column_name2) # df.corr() might work

# Quartiles/Min/Max/Mean/Count/Std:
df.describe()

# Standard Error of Mean
stats.sem(df.column_name) 
# OR
df.column_name.std / (len(df.column_name)) ** 0.5

# Coeff of Variation
(df.column_name.std / df.column_name.mean) * 100

# Null Values in every column
df.isnull().sum()

# Cumulative Percentage
(df.column_name.cumsum() / df.column_name.sum()) * 100

# Skewness/Kurtosis
df.skew()
df.kurtosis()

# Trimmed Mean
stats.trim_mean(df.column_name)

# Sum of squares
ssColumn = 0
for i in range(len(df.column_name)):
    ssColumn += df.column_name[i] ** 2
print(ssColumn)

# Box and Whisker plot
df.boxplot('column1')

# Scatter Plot
plt.scatter(df.column1, df.column2)
pd.plotting.scatter_matrix(df) # for cool effects

# Correlation Matrix
df.corr()
```
## Exp 2: Data Imputation
```python
# Find missing rows
df.isnull().sum()

# Remove null row
df1 = df.dropna()
print(df1)

# Mean/Median/Mode/Arbitary Value/Mode Imputation
df1 = df.fillna(df.mean())
df2 = df.fillna(df.median())
df4 = df.fillna(df.mode().iloc[0])
df3 = df.fillna(27)
print(dfx)

# End Of Tail Imputation
df1 = df[['column1', 'column2', ....]]
for i in df1:
    eod = df[i].mean() + 3*df[i].std()
    df[i] = df[i].fillna(eod)
print(df1) 

# Random Sample Imputation
def random_imputation(df1,column_name):
    number_missing = df1.column_name.isnull().sum()
    observed_values = df1.loc[df1.column_name.notnull(),column_name]
    df1.loc[df1.column_name.isnull(),column_name + '_imp'] = np.random.choice(observed_values,number_missing,replace = True)
    return df1

df1 = df[['column1', 'column2', ....]]
for i in df1:
    df1[column_name + '_imp'] = df1.column_name
    df1 = random_imputation(df1, column_name)
print(df1)

# OR
df1 = df[['column1', 'column2', ....]]
for i in df1:
    df1[i].dropna().sample(df1[i].isnull().sum(),random_state=0)
print(df1)

# Linear Regression Imputation
from sklearn.model_selection import train_test_split
from  sklearn.impute import SimpleImputer, MissingIndicator
X = new_df[['Evaporation', 'Cloud9am', 'Cloud3pm']]
y = new_df['Pressure3pm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 22)
X_train_original = X_train.copy()
X_test_original = X_test.copy()
for var in ['Evaporation', 'Cloud9am', 'Cloud3pm']:
    X_train[var + '_missing'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var + '_missing'] = np.where(X_test[var].isnull(), 1, 0)
X_train


imputer = SimpleImputer(strategy="mean")
X_train_original_transformed = imputer.fit_transform(X_train_original)
X_test_original_transformed = imputer.transform(X_test_original)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train_original_transformed, y_train)
y_pred = reg.predict(X_test_original_transformed)
y_pred
```

## Exp 3: Exploratory Data Analysis
```python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px

# Scatter plot
plt.xlabel("distance")
plt.ylabel("fare_amount")
plt.scatter(df['distance'], df['fare_amount'], c ="blue")

# Scatter Matrix
pd.plotting.scatter_matrix(df)

# Boxplot
df.boxplot('column1', 'column2')

#Distribution Chart / Distplot
plt.figure(figsize=[15,4])
sns.histplot(data=df,  x="column_name"
             , bins=40, kde=True)

#JoinPlot
sns.jointplot(x ='distance', y ='fare_amount', data = df)

# Histogram
df['passenger_count'].hist()
plt.show()

#Pie Chart
plt.pie(df.coumn_name.value_counts(),labels=df.column_name.unique(),autopct ='% 1.1f %%')
plt.pie(values,labels=lst)
plt.show()

#Bubble Chart
fig = px.scatter(df, x="distance_range", y="fare_amount", size="distance",color="passenger_count",  log_x=True, size_max=60)
fig.show()

#Density Chart
df['passenger_count'].plot.density(color='green')
plt.title('Density plot for passenger count')
plt.show()

#Parallel Chart
df1 =df.sample(n=1000)
fig = px.parallel_coordinates(df1, color="passenger_count",
                              dimensions=['distance_range','fare_amount',],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()

# Creating Andrews curves
df1 = df[['distance','passenger_count','fare_amount']]
df1=df1.sample(n=100)
x = pd.plotting.andrews_curves(df1,'passenger_count')
x.plot()
plt.show()

# Heatmap
df_numeric = df.select_dtypes(include=np.number)
corr = df_numeric_corr
hm = sns.heatmap(corr, cmap="Blues", annot=True)
plt.show()

```
