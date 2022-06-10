#gopiprasad

,adras.ipynb
,adras.ipynb_
Files
..
Drop files to upload them to session storage
Disk
69.37 GB available
[ ]

[4]
0s
# Most basic stuff for EDA.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

[5]
0s
#import the dataset from system and for EDA analytics
df = pd.read_csv("/content/train-chennai-sale.csv")
df.head(10)

[6]
0s
df.columns
#----------------
#checking the columns name

Index(['PRT_ID', 'AREA', 'INT_SQFT', 'DATE_SALE', 'DIST_MAINROAD', 'N_BEDROOM',
       'N_BATHROOM', 'N_ROOM', 'SALE_COND', 'PARK_FACIL', 'DATE_BUILD',
       'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE', 'QS_ROOMS',
       'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'REG_FEE', 'COMMIS',
       'SALES_PRICE'],
      dtype='object')
[7]
0s
#We'll check columns and rows that were automatically guessed by pandas library
df.shape
(7109, 22)
[8]
0s
#We'll check dtypes that were automatically guessed by pandas library
df.dtypes

#We'll check information that were automatically guessed by pandas library
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7109 entries, 0 to 7108
Data columns (total 22 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   PRT_ID         7109 non-null   object 
 1   AREA           7109 non-null   object 
 2   INT_SQFT       7109 non-null   int64  
 3   DATE_SALE      7109 non-null   object 
 4   DIST_MAINROAD  7109 non-null   int64  
 5   N_BEDROOM      7108 non-null   float64
 6   N_BATHROOM     7104 non-null   float64
 7   N_ROOM         7109 non-null   int64  
 8   SALE_COND      7109 non-null   object 
 9   PARK_FACIL     7109 non-null   object 
 10  DATE_BUILD     7109 non-null   object 
 11  BUILDTYPE      7109 non-null   object 
 12  UTILITY_AVAIL  7109 non-null   object 
 13  STREET         7109 non-null   object 
 14  MZZONE         7109 non-null   object 
 15  QS_ROOMS       7109 non-null   float64
 16  QS_BATHROOM    7109 non-null   float64
 17  QS_BEDROOM     7109 non-null   float64
 18  QS_OVERALL     7061 non-null   float64
 19  REG_FEE        7109 non-null   int64  
 20  COMMIS         7109 non-null   int64  
 21  SALES_PRICE    7109 non-null   int64  
dtypes: float64(6), int64(6), object(10)
memory usage: 1.2+ MB
Double-click (or enter) to edit

*gathering the information datatype of data avaible in dataset *

[9]
0s
df.isnull().value_counts()
#checking the null value in the dataset 
PRT_ID  AREA   INT_SQFT  DATE_SALE  DIST_MAINROAD  N_BEDROOM  N_BATHROOM  N_ROOM  SALE_COND  PARK_FACIL  DATE_BUILD  BUILDTYPE  UTILITY_AVAIL  STREET  MZZONE  QS_ROOMS  QS_BATHROOM  QS_BEDROOM  QS_OVERALL  REG_FEE  COMMIS  SALES_PRICE
False   False  False     False      False          False      False       False   False      False       False       False      False          False   False   False     False        False       False       False    False   False          7056
                                                                                                                                                                                                  True        False    False   False            47
                                                              True        False   False      False       False       False      False          False   False   False     False        False       False       False    False   False             4
                                                                                                                                                                                                  True        False    False   False             1
                                                   True       False       False   False      False       False       False      False          False   False   False     False        False       False       False    False   False             1
dtype: int64
Seprating the catergical data from the dataset

[10]
0s
df_selected = df.select_dtypes(include = ["object"])
df[df_selected.columns] = df_selected.apply(lambda x: x.str.strip())
df_selected

*The describe() method is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame. It analyzes both numeric and object series and also the DataFrame column sets of mixed data, T belong to transport the columns *

[11]
0s
df.describe().T

DATA CLEANING

[12]
0s
# change the date and time to pandas datetime 
df['DATE_SALE'] = pd.to_datetime(df['DATE_SALE'])
df['DATE_BUILD'] = pd.to_datetime(df['DATE_BUILD'])
From dataframe correcting the spelling and exploring the data with one columns with another columns that support for the further precision and from the dataframe catergical columns one by one checking whether their is any spelling mistake in every columns and finding the null value from the particular column italicized text

Area column

[13]
0s
df.AREA.value_counts()
Chrompet      1681
Karapakkam    1363
KK Nagar       996
Velachery      979
Anna Nagar     783
Adyar          773
T Nagar        496
Chrompt          9
Chrmpet          6
Chormpet         6
TNagar           5
Karapakam        3
Ana Nagar        3
Velchery         2
Ann Nagar        2
Adyr             1
KKNagar          1
Name: AREA, dtype: int64
[14]
0s
correction={'Chrompt':'Chrompet',
            'Chrmpet':'Chrompet',
            'Chormpet':'Chrompet',
            'Karapakam':'Karapakkam',
            'KKNagar':'KK Nagar',
            'Velchery':'Velachery',
            'Ann Nagar':'Anna Nagar',
            'Ana Nagar':'Anna Nagar',
             'Adyr':'Adyar' ,
            'TNagar':'T Nagar'}
df['AREA']=df.AREA.replace(correction)     
[15]
0s
df.AREA.value_counts().isnull()
Chrompet      False
Karapakkam    False
KK Nagar      False
Velachery     False
Anna Nagar    False
Adyar         False
T Nagar       False
Name: AREA, dtype: bool
sales condition column

[16]
0s
df.SALE_COND.value_counts()
AdjLand        1433
Partial        1429
Normal Sale    1423
AbNormal       1406
Family         1403
Adj Land          6
Ab Normal         5
Partiall          3
PartiaLl          1
Name: SALE_COND, dtype: int64
[17]
0s
sale_corr={'Adj Land':'AdjLand',
          'Partiall':'Partial',
          'PartiaLl':'Partial',
           'Ab Normal':'AbNormal'}
df['SALE_COND']=df.SALE_COND.replace(sale_corr)     
[18]
0s
df.SALE_COND.value_counts().isnull()
AdjLand        False
Partial        False
Normal Sale    False
AbNormal       False
Family         False
Name: SALE_COND, dtype: bool
building type

[19]
0s
df.BUILDTYPE.value_counts().isnull()
House         False
Commercial    False
Others        False
Other         False
Comercial     False
Name: BUILDTYPE, dtype: bool
[20]
0s
build_corr={'Comercial':'Commercial','Other':'Others'}
df['BUILDTYPE']=df.BUILDTYPE.replace(build_corr)     
[21]
0s
df.BUILDTYPE.value_counts()
House         2444
Others        2336
Commercial    2329
Name: BUILDTYPE, dtype: int64
Street column

[22]
0s
df.STREET.value_counts()
Paved        2560
Gravel       2520
No Access    2010
Pavd           12
NoAccess        7
Name: STREET, dtype: int64
[23]
0s
street_corr={'Pavd':'Paved','NoAccess':'No Access'}
df['STREET']=df.STREET.replace(street_corr) 
[24]
0s
df.STREET.value_counts().isnull()
Paved        False
Gravel       False
No Access    False
Name: STREET, dtype: bool
MZzone columns

[25]
0s
df.MZZONE.value_counts()
RL    1858
RH    1822
RM    1817
C      550
A      537
I      525
Name: MZZONE, dtype: int64
[26]
0s
df.MZZONE.value_counts().isnull()
RL    False
RH    False
RM    False
C     False
A     False
I     False
Name: MZZONE, dtype: bool
[27]
0s
df.PARK_FACIL.value_counts()
Yes    3587
No     3520
Noo       2
Name: PARK_FACIL, dtype: int64
[28]
0s
df['PARK_FACIL']=df.PARK_FACIL.replace({'Noo':'No'})
[29]
0s
df.PARK_FACIL.value_counts().isnull()
Yes    False
No     False
Name: PARK_FACIL, dtype: bool
Utility column

# This is formatted as code
[30]
0s
df.UTILITY_AVAIL.value_counts()
AllPub     1886
NoSeWa     1871
NoSewr     1829
ELO        1522
All Pub       1
Name: UTILITY_AVAIL, dtype: int64
[31]
0s
utili_corr={'All Pub':'AllPub'}
df['UTILITY_AVAIL']=df.UTILITY_AVAIL.replace(utili_corr) 
[32]
0s
df.UTILITY_AVAIL.value_counts().isnull()
AllPub    False
NoSeWa    False
NoSewr    False
ELO       False
Name: UTILITY_AVAIL, dtype: bool
Numerical column contain only the numerical like float and integer value that are seprating from the main data frame for EDA purpose

[33]
0s
numerical = df.select_dtypes(include = ["float64","int64"])
numerical

[34]
0s
numerical.isnull().value_counts()
INT_SQFT  DIST_MAINROAD  N_BEDROOM  N_BATHROOM  N_ROOM  QS_ROOMS  QS_BATHROOM  QS_BEDROOM  QS_OVERALL  REG_FEE  COMMIS  SALES_PRICE
False     False          False      False       False   False     False        False       False       False    False   False          7056
                                                                                           True        False    False   False            47
                                    True        False   False     False        False       False       False    False   False             4
                                                                                           True        False    False   False             1
                         True       False       False   False     False        False       False       False    False   False             1
dtype: int64
[35]
0s
# from the dataframe were filling the null value with the  format of forwardfilling method  
df.fillna(method='ffill',axis=0,inplace=True)
[36]
0s
df.isnull().value_counts()
PRT_ID  AREA   INT_SQFT  DATE_SALE  DIST_MAINROAD  N_BEDROOM  N_BATHROOM  N_ROOM  SALE_COND  PARK_FACIL  DATE_BUILD  BUILDTYPE  UTILITY_AVAIL  STREET  MZZONE  QS_ROOMS  QS_BATHROOM  QS_BEDROOM  QS_OVERALL  REG_FEE  COMMIS  SALES_PRICE
False   False  False     False      False          False      False       False   False      False       False       False      False          False   False   False     False        False       False       False    False   False          7109
dtype: int64
[37]
0s
#checking the duplicated columns in the dataset
df.duplicated().any()
False
[38]
0s
df.columns
Index(['PRT_ID', 'AREA', 'INT_SQFT', 'DATE_SALE', 'DIST_MAINROAD', 'N_BEDROOM',
       'N_BATHROOM', 'N_ROOM', 'SALE_COND', 'PARK_FACIL', 'DATE_BUILD',
       'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE', 'QS_ROOMS',
       'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'REG_FEE', 'COMMIS',
       'SALES_PRICE'],
      dtype='object')
[39]
0s
df['N_BEDROOM']=df['N_BEDROOM'].apply(int)
df['N_BATHROOM']=df['N_BATHROOM'].apply(int)
df['N_ROOM']=df['N_ROOM'].apply(int)

[40]
0s
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7109 entries, 0 to 7108
Data columns (total 22 columns):
 #   Column         Non-Null Count  Dtype         
---  ------         --------------  -----         
 0   PRT_ID         7109 non-null   object        
 1   AREA           7109 non-null   object        
 2   INT_SQFT       7109 non-null   int64         
 3   DATE_SALE      7109 non-null   datetime64[ns]
 4   DIST_MAINROAD  7109 non-null   int64         
 5   N_BEDROOM      7109 non-null   int64         
 6   N_BATHROOM     7109 non-null   int64         
 7   N_ROOM         7109 non-null   int64         
 8   SALE_COND      7109 non-null   object        
 9   PARK_FACIL     7109 non-null   object        
 10  DATE_BUILD     7109 non-null   datetime64[ns]
 11  BUILDTYPE      7109 non-null   object        
 12  UTILITY_AVAIL  7109 non-null   object        
 13  STREET         7109 non-null   object        
 14  MZZONE         7109 non-null   object        
 15  QS_ROOMS       7109 non-null   float64       
 16  QS_BATHROOM    7109 non-null   float64       
 17  QS_BEDROOM     7109 non-null   float64       
 18  QS_OVERALL     7109 non-null   float64       
 19  REG_FEE        7109 non-null   int64         
 20  COMMIS         7109 non-null   int64         
 21  SALES_PRICE    7109 non-null   int64         
dtypes: datetime64[ns](2), float64(4), int64(8), object(8)
memory usage: 1.2+ MB
Additing the reg_fee and commis column with price sale for the usage of EDA

[41]
0s
df["SALES_PRICE"]= df.SALES_PRICE+df.COMMIS+df.REG_FEE
calculating the age of the build

[42]
0s
df["Age_build"]=(df.DATE_SALE)-(df.DATE_BUILD)
[43]
0s
year =[round(df.Age_build[x].days/365.25,1)for x in range (df.shape[0])]
[44]
0s
df["Age"]=year
df["Age"]

0       43.9
1       11.0
2       19.6
3       22.0
4       29.6
        ... 
7104    49.1
7105     8.8
7106    28.6
7107    31.3
7108    44.0
Name: Age, Length: 7109, dtype: float64
[45]
0s
df.drop(['PRT_ID','Age_build','DATE_SALE','SALE_COND','REG_FEE', 'COMMIS'],axis=1,inplace=True)
[46]
0s
df.drop([   'DATE_BUILD'],axis=1,inplace=True)
For EDA final dataset is ready

[47]
0s
df

EDA with the data set by using the graphical analytics in that i have explain the salesprice and other columns

bold text

[48]
0s

sns.barplot(y = "SALES_PRICE", x ="AREA", data=df,order=df[['AREA','SALES_PRICE']].groupby('AREA').mean().sort_values('SALES_PRICE').reset_index().AREA)


[49]
1s
plt.figure(figsize=(15,7))
sns.relplot(x='AREA',y='SALES_PRICE', hue='PARK_FACIL',data=df)
plt.show()

[50]
1s
sns.boxplot(x="AREA", y="INT_SQFT", hue="BUILDTYPE",
            data=df)

[51]
0s
sns.catplot(x="AREA", y="N_BEDROOM",hue="BUILDTYPE",kind="boxen", data=df)
plt.figure(figsize=(7,14))
plt.show()


[52]
1s
sns.catplot(x="UTILITY_AVAIL", y="SALES_PRICE",hue='AREA', data=df,kind="boxen")
plt.figure(figsize=(7,14))
plt.show()


[53]
0s
sns.catplot(x="AREA", y="Age", data=df)

[54]
0s
sns.catplot(x="AREA", y="Age", hue="N_BEDROOM", kind="bar", data=df)

[55]
0s
sns.catplot(x="AREA", y="SALES_PRICE", hue="N_BEDROOM", kind="bar", data=df)

[56]
0s
sns.catplot(x="AREA", y="SALES_PRICE", hue="N_BATHROOM", kind="bar", data=df)

[57]
0s
sns.catplot(x="N_BEDROOM", y="SALES_PRICE", hue="N_BATHROOM", kind="bar", data=df)

[58]
2s
sns.relplot(x="Age", y="SALES_PRICE", hue="AREA", data=df)

[ ]

[ ]

[59]
0s
import plotly.graph_objects as go


date_cols = ['Age', 'QS_OVERALL', 'BUILDTYPE']
fig = go.Figure()
for col in date_cols:
    fig.add_trace(go.Scatter(x=df[col], y=df['SALES_PRICE'], mode='markers', name=col))
fig.update_layout( width=1300, title='SalePrice Distribution with Date columns', yaxis_title='SalePrice', xaxis_title='Year')
fig.show()

[60]
1s
fig = make_subplots(rows=(df.shape[1]-1)//3, cols=3)
for i, col in enumerate(df.columns[:-1]):
    _col = (i+1)%3
    _col = _col if _col != 0 else 3 
    series = df.groupby(col)['SALES_PRICE'].mean()
    fig.add_trace(go.Scatter(y=series, x=series.index, name=col ), row=(i//5)+1, col=_col)
    fig.update_yaxes(title='SALES_PRICE')
        
fig.update_layout( height=1500, width=1300)
fig.show()

library for the model selection and making the precidition report from the model selection and importing the library

[61]
0s
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import xgboost as xgb
[70]

[63]
0s
y_test
array([2413, 2453, 1034, ..., 1815, 1574, 1087])
[64]
0s
X_train.shape, X_test.shape
((4976, 16), (2133, 16))
[65]
0s
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['AREA']= label_encoder.fit_transform(df['AREA'])
df['PARK_FACIL']= label_encoder.fit_transform(df['PARK_FACIL'])
df['BUILDTYPE']= label_encoder.fit_transform(df['BUILDTYPE'])
df['MZZONE']= label_encoder.fit_transform(df['MZZONE'])
df['UTILITY_AVAIL']= label_encoder.fit_transform(df['UTILITY_AVAIL'])
df['STREET']= label_encoder.fit_transform(df['STREET'])


[ ]

[66]
0s
df

[71]
0s
X = df.iloc[:, :-1].values # select all rows and select all columns except the last column as my feature
y = df.iloc[:, 1].values # target as arrays
# Syntax : dataset.loc[:, :-1]
from sklearn.model_selection import train_test_split #import the required function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 7)
[72]
0s
  
# create linear regression object
reg = linear_model.LinearRegression()
[73]
0s
# train the model using the training sets
reg.fit(X_train, y_train)
LinearRegression()
[74]
0s
# regression coefficients
print('Coefficients: ', reg.coef_)
# variance score: 1 means perfect prediction

print('Variance score: {}'.format(reg.score(X_test, y_test)))

Coefficients:  [-4.15955830e-11  1.00000000e+00  8.66007095e-14  1.41398216e-10
 -9.36072732e-11 -5.03147593e-11  2.39746849e-10 -6.78593463e-11
  5.02043057e-13  2.21524132e-11  4.00049093e-11  1.10697693e-11
  1.32596903e-11  1.53298822e-11 -4.68003412e-11  0.00000000e+00]
Variance score: 1.0
[75]
0s
plt.style.use('fivethirtyeight')
  
## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  
## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
# plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
## plotting legend
plt.legend(loc = 'upper right')
  
## plot title
plt.title("Residual errors")

[76]
0s
from sklearn.metrics import mean_squared_error, r2_score
import math

# create linear regression object
reg = LinearRegression()
reg.fit(X_train, y_train)
# testing
y_pred = reg.predict(X_test)

# evaluating
print('Root Mean Square Error of the model is: ',math.sqrt(mean_squared_error(y_test, y_pred)))
print('Fitting score: ',reg.score(X_train,y_train))
print('R-squared score: ',r2_score(y_test, y_pred))
Root Mean Square Error of the model is:  2.557307789618561e-10
Fitting score:  1.0
R-squared score:  1.0
[77]
1s
# training
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(n_estimators=100,n_jobs=-1, random_state=0)
rfreg.fit(X_train, y_train.ravel())

# testing
y_pred = rfreg.predict(X_test)

# evaluating
from sklearn.metrics import mean_squared_error
print('Root Mean Square Error of the model is: ',math.sqrt(mean_squared_error(y_test, y_pred)))
print('Fitting score: ',rfreg.score(X_train,y_train))
print('R-squared score: ',r2_score(y_test, y_pred))
Root Mean Square Error of the model is:  0.5880032482664239
Fitting score:  0.9999997631397567
R-squared score:  0.9999983785129228
[78]
0s
# training
from xgboost import XGBRegressor
xgbreg = XGBRegressor(random_state = 0)
xgbreg.fit(X_train, y_train.ravel())

# testing
y_pred = xgbreg.predict(X_test)

# evaluating
from sklearn.metrics import mean_squared_error,r2_score
print('Root Mean Square Error of the model is: ',math.sqrt(mean_squared_error(y_test, y_pred)))
print('Fitting score: ',xgbreg.score(X_train,y_train))
print('R-squared score: ',r2_score(y_test, y_pred))
[05:12:27] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
Root Mean Square Error of the model is:  3.576313128019238
Fitting score:  0.9999431240248041
R-squared score:  0.9999400174236917
[79]
0s
predictions = xgbreg.predict(X_test)
predictions[:10]
array([2416.0732, 2453.203 , 1032.8038, 1151.6139, 1170.6245, 2280.3438,
       1112.1606, 1236.8163,  884.6944, 1637.8175], dtype=float32)
[80]
0s
submission_df = pd.DataFrame(columns=[ 'SALES_PRICE'])
#submission_df['PRT_ID'] = ID
submission_df['SALES_PRICE'] = predictions
submission_df.to_csv('submission.csv', header=True, index=False)
submission_df.head(10)

for cross checking the linear regrssion i have done the cross calidition if my model is under fitting or over fitting

[82]
0s
from sklearn.model_selection import cross_validate
regressor = LinearRegression(normalize = True) #untrained model
[83]
0s
cv_results = cross_validate(regressor, X, y, cv=10, scoring = "r2")
cv_results['test_score'].mean()

1.0
[84]
0s
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Lasso, Ridge
for alpha in [0.001,0.002,0.003,0.005,0.01,0.02,0.03,0.04,0.1,0.2,0.5,1]:
  regressor = Ridge(normalize=True,alpha = alpha)
  cv_results = cross_validate(regressor, X, y, cv=5, scoring = "r2" , return_train_score=True)
  print("Alpha : ", alpha, cv_results['test_score'].mean(), cv_results['train_score'].mean())
Alpha :  0.001 0.9999874167515971 0.9999874942578852
Alpha :  0.002 0.999951801616486 0.999952098049876
Alpha :  0.003 0.9998960296816172 0.9998966682385619
Alpha :  0.005 0.9997336610346064 0.9997352926894958
Alpha :  0.01 0.999115046192222 0.9991204392621255
Alpha :  0.02 0.9974157412063199 0.9974313631205938
Alpha :  0.03 0.9955191316087844 0.9955460348202335
Alpha :  0.04 0.9936214669588599 0.9936595202545948
Alpha :  0.1 0.9838349921444763 0.9839278142569396
Alpha :  0.2 0.9716481311991112 0.971800187456572
Alpha :  0.5 0.942493535567311 0.9427447401685441
Alpha :  1 0.8970498957459265 0.8973884643987937
[ ]


check
0s
completed at 11:05 AM
Made 1 formatting edit on line 10
Could not connect to the reCAPTCHA service. Please check your internet connection and reload to get a reCAPTCHA challenge.


Runtime disconnected




