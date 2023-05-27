# EX-05-Feature-Generation


# AIM:
To read the given data and perform Feature Generation process and save the data to a file. 

# EXPLANATION:
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM:
## STEP 1:
Read the given Data
## STEP 2:
Clean the Data Set using Data Cleaning Process
## STEP 3:
Apply Feature Generation techniques to all the feature of the data set
## STEP 4:
Save the data to the file


# PROGRAM:
## Data.csv
~~~
Program Developed by: CHARU DHARSHINI K
Register number:212221220008

import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

# Feature scaling:
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
~~~

## Encoding.csv
~~~
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
~~~

## Titanic.csv
~~~
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
~~~

# OUTPUT:

![image](https://user-images.githubusercontent.com/94828147/194209465-9441f5e0-a26e-4f74-b283-fce38528cb02.png)


![image](https://user-images.githubusercontent.com/94828147/194209513-d3254563-aab3-4c40-bcf2-46048f85bfd4.png)

![image](https://user-images.githubusercontent.com/94828147/194209529-dbf46a95-d95f-4bbc-acf2-a3271ca093c0.png)


![image](https://user-images.githubusercontent.com/94828147/194209556-000ac824-d1a3-467f-ac33-b9e02d904141.png)


![image](https://user-images.githubusercontent.com/94828147/194209578-e869f9f2-4de0-422c-925a-672cbf8b5edb.png)


![image](https://user-images.githubusercontent.com/94828147/194209598-8c11623c-e88d-43da-96ce-049c74844266.png)

![image](https://user-images.githubusercontent.com/94828147/194209695-7c1a371b-e83b-417f-9c4a-69a3d804868c.png)

![image](https://user-images.githubusercontent.com/94828147/194209728-c246de60-85cf-488e-921b-dcca3b494088.png)


![image](https://user-images.githubusercontent.com/94828147/194209999-5f106da5-bf8e-422b-a521-278350b56d4e.png)

![image](https://user-images.githubusercontent.com/94828147/194210031-806c038c-1123-4b43-8f73-898476dd2905.png)

![image](https://user-images.githubusercontent.com/94828147/194210063-5e9b4dae-e1fa-42c9-8404-4214b1aef530.png)

![image](https://user-images.githubusercontent.com/94828147/194210081-cf238326-8329-458c-bfbb-bc3c7d714461.png)

![image](https://user-images.githubusercontent.com/94828147/194210108-c977c29e-e284-43e7-9b38-92553bd81066.png)

![image](https://user-images.githubusercontent.com/94828147/194210127-9940fdd3-9ca2-4115-b011-9336e4243cef.png)

![image](https://user-images.githubusercontent.com/94828147/194210143-a839bb80-70f7-447f-8c33-f48d53021218.png)

![image](https://user-images.githubusercontent.com/94828147/194210158-9dd876ce-72db-4df9-b5b0-0982c5211550.png)


![image](https://user-images.githubusercontent.com/94828147/194210537-f5486aea-8e7e-42c3-ba7a-cd9d89f1f6fe.png)

![image](https://user-images.githubusercontent.com/94828147/194210630-ba27f724-61f2-4120-a50a-bc9c0fc5759d.png)

![image](https://user-images.githubusercontent.com/94828147/194210660-18384160-ac5a-4719-8ee3-7af42d5ae3b0.png)

![image](https://user-images.githubusercontent.com/94828147/194210676-a4aa6783-fba1-4522-ad29-61695d39328e.png)

![image](https://user-images.githubusercontent.com/94828147/194210796-735bb602-95a1-44bb-a63b-47c56515cf24.png)

![image](https://user-images.githubusercontent.com/94828147/194210834-72b61140-2569-484e-9b6c-029c1131be4f.png)

![image](https://user-images.githubusercontent.com/94828147/194210858-7bd4b1c6-642b-4b5b-b005-09bc365d3fa9.png)

![image](https://user-images.githubusercontent.com/94828147/194210909-ef280d03-d9d5-4765-9d92-58e18afa7f68.png)

![image](https://user-images.githubusercontent.com/94828147/194210947-fc993f9a-7fe1-4156-a74b-18a52f466ed5.png)

![image](https://user-images.githubusercontent.com/94828147/194210974-b5e87751-4df6-40a7-8d24-92c43f003c80.png)


# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.


