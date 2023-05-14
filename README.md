# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(data.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

# Change to categoric column to numeric
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values
data['Embarked']=data['Embarked'].fillna('S')

# Change to categoric column to numeric
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

#heatmap for train dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

#Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

#Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(11)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))

```
# OUPUT
![Screenshot 2023-05-14 135151](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/e0e17710-19d5-41d3-b55e-3c5037d79999)


![Screenshot 2023-05-14 135159](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/e8315363-2348-4b16-8df8-ba797d39d207)

![Screenshot 2023-05-14 135204](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/8c3fac0c-8c96-4766-b35f-fdda2be6f765)

![Screenshot 2023-05-14 135211](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/4ca6339e-91c3-4030-b48f-468686e42300)

![Screenshot 2023-05-14 135217](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/026824c6-0687-4e89-b405-9c208f1b249e)
![Screenshot 2023-05-14 135225](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/53c93e15-5dad-4fcd-be3f-6bbdd1494e00)

![Screenshot 2023-05-14 135233](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/25c22b85-dbc7-4d87-a831-2fc420482773)
![Screenshot 2023-05-14 135238](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/52d646c5-ee89-47fc-b0db-4763d73d427e)

![Screenshot 2023-05-14 135247](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/cc20652a-af45-4278-ba11-e908d232ea58)

![Screenshot 2023-05-14 135253](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/49401b1d-03ac-4021-ad22-833fb0feb2e4)

![Screenshot 2023-05-14 135306](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/9f5f9170-0d46-4427-a394-29d626b5e37f)

![Screenshot 2023-05-14 135312](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/ab42b721-a164-4629-964b-7672cfc0e73d)

![Screenshot 2023-05-14 135321](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/8e0fad7b-7945-4bd9-80cb-55a8fc47cb4c)


![Screenshot 2023-05-14 135325](https://github.com/Dharshan011/Ex-07-Feature-Selection/assets/113497491/ea80d1de-7a2c-462d-b5b2-8817217bf610)

## RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.























