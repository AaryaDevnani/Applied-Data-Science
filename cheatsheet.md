# CheatSheet for ADS

## Libraries for each plot
1. Matplotlib:
    * Scatter plot
    * Pie Chart

2. Pandas
    * Scatter Matrix
    * Box Plot
    * Histogram
    * Density Chart
    * Andrews Curve

3. Seaborn
    * Distplot / Histplot
    * Joint Plot

4. Plotly
    * Line Chart
    * Bubble Chart
    * Parallel Chart

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

#Line Chart
fig = px.line(df1, x="distance_range", y="fare_amount",color='passenger_count')
fig.show()

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
## Exp 4: Supervised Models Performance Metrics
```python

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

y = df['column_name'] # Dependent Variable
x = df[['column_name1','column_name2',...]] #Independent Variables
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.67)

dt = DecisionTreeClassifier()
model = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Prediction 1", "Prediction 2"]))

cm = confusion_matrix(y_test,y_pred)
cmDisply = ConfusionMatrixDisplay(cm)
cmDisply.plot()
plt.show()

```
## Exp 5: Unsupervised Models Performance Metrics
```python

#K Means
from sklearn.cluster import KMeans
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(df)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Adjusted Rand Score
from sklearn.metrics.cluster import adjusted_rand_score
labels_true = target.values
labels_pred = labels
print(adjusted_rand_score(labels_true, labels_pred))

# Mutual Info Score
from sklearn.metrics import mutual_info_score
print(mutual_info_score (labels_true, labels_pred))

# Silhoutte Score
from sklearn.metrics import silhouette_score
print(silhouette_score(df, labels))

```
## Exp 6: Skip

## Exp 7: Outlier Detection
```python

#DBSCAN
from sklearn.cluster import DBSCAN

scaler = StandardScaler()
X = scaler.fit_transform(df)
dbs = DBSCAN(eps=12.5, min_samples=4)
model = dbs.fit(X)
dbscanDs = df.copy()
dbscanDs['Cluster'] = model.labels_ 
# Identify outliers based on their cluster label (-1 indicates an outlier)
outliers = df[dbs.labels_ == -1]
print(outliers.count())
dbscanDs.Cluster.value_counts()

#KNN
from sklearn.neighbors import NearestNeighbors
k = 5
# Fit the KNN model on the High, Low, Open, Close, and Adj Close columns
X = df[['fare_amount', 'distance']]
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
# Calculate the average distance to the k nearest neighbors for each data point
avg_distances = distances[:,1:].mean(axis=1)
# Set a threshold for outlier detection (e.g. 2 standard deviations above the mean)
threshold = avg_distances.mean() + 2 * avg_distances.std()
# Identify outliers based on the threshold
outliers = df[avg_distances > threshold]
# Print the outliers count
print(outliers.count())
```

## Exp 8: SMOTE
```python
import numpy as np 
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

x = data[['fare_amount','passenger_count']]
y = data['distance']

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size= 0.7, random_state=25)

smote = SMOTE(random_state=25,k_neighbors=1)
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)

lr =LogisticRegression()
modelOversampled = LogisticRegression.fit(X_oversample,y_oversample)
model = LogisticRegression.fit(X_train,y_train)
predOversampled = modelOversampled.pred(X_test)
pred = model.pred(X_test)
cr = classification_report(y_oversample,pred) # This is smote Classification Report
cr = classification_report(y_test,pred) # This is without SMOTE classification Report
```

# Exp 9: Inferential Statistics

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind, ttest_1samp, f_oneway
from statsmodels.stats.weightstats import ztest

# Z-test
sample_mean = df['fare_amount'].mean()
pop_mean =  7 # assumed population mean
sample_std = df['fare_amount'].std()
n = len(df)
z_score = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))
p_value = stats.norm.sf(abs(z_score)) * 2  # two-tailed test
print('Z-score:', z_score)
print('P-value:', p_value)

# T-test
sample_mean = df['fare_amount'].mean()
pop_mean = 7  # assumed population mean
sample_std = df['fare_amount'].std()
n = len(df)
t_score, p_value = stats.ttest_1samp(df['fare_amount'], pop_mean)
print('T-score:', t_score)
print('P-value:', p_value)

model = ols('fare_amount ~ C(passenger_count)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 1 Sample T-Test
t_stat, p_value = ttest_1samp(df['fare_amount'], 7)
print("1-sample t-test results:")
print("t-statistic: ", t_stat)
print("p-value: ", p_value)

# 2 Sample T-Test
sample1 = df[df['passenger_count'] == 5]['fare_amount']
sample2 = df[df['passenger_count'] == 6]['fare_amount']
t_stat, p_value = ttest_ind(sample1, sample2)
print("2-sample t-test results:")
print("t-statistic: ", t_stat)
print("p-value: ", p_value)

# ANOVA
group1 = df[df['passenger_count'] == 1]['fare_amount']
group2 = df[df['passenger_count'] == 2]['fare_amount']
group3 = df[df['passenger_count'] == 3]['fare_amount']
f_stat, p_value = f_oneway(group1, group2, group3)
print("ANOVA results:")
print("f-statistic: ", f_stat)
print("p-value: ", p_value)
```
