## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
~~~
![image](https://github.com/user-attachments/assets/6e7483c7-c80d-4577-9533-a10fcc424ba6)

~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

~~~
![image](https://github.com/user-attachments/assets/76b695c7-73d7-4a05-bdaf-1867a88faba2)

~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![image](https://github.com/user-attachments/assets/89e0f6c5-d428-4124-b5ac-f9555e3bbdaf)

~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~

![image](https://github.com/user-attachments/assets/b65c6650-21f8-4bc7-bb05-4b713240d66d)

~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
~~~

![image](https://github.com/user-attachments/assets/5696cae0-49c4-4f88-abef-66c4712b7e44)

~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~

![image](https://github.com/user-attachments/assets/cbf34f7c-f7af-4eae-8892-b8a00510ba47)

~~~
pip install --upgrade category_encoders
~~~

![image](https://github.com/user-attachments/assets/d2697434-d676-4a23-a29a-a4e68093e0db)

~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
~~~

![image](https://github.com/user-attachments/assets/f0d63853-de49-4684-bdc3-054318c8584a)

~~~
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
~~~

![image](https://github.com/user-attachments/assets/fcd8f683-e2a1-42fd-8643-be25a1cb9eef)

~~~
dfb=pd.concat([df,nd],axis=1)
dfb
~~~

![image](https://github.com/user-attachments/assets/767dee95-e799-4d5f-b28d-42e3bf5fdad2)

~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
~~~

![image](https://github.com/user-attachments/assets/d6eadc18-2161-490b-853d-2953769c427f)

~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
~~~

![image](https://github.com/user-attachments/assets/d3589260-b23d-497d-828a-0211c184986b)

~~~
df.skew()
~~~

![image](https://github.com/user-attachments/assets/e88af397-4b99-4d66-93b2-125f6709b3b2)

~~~
np.log(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/8831fee3-a581-4b65-abc6-03e52131948c)

~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~

![image](https://github.com/user-attachments/assets/7a86718f-50d4-44d0-bd8d-8e3e423489d8)

~~~
np.square(df["Highly Positive Skew"])
~~~

![image](https://github.com/user-attachments/assets/3596132e-6b17-499c-a705-b4232e9a336e)

~~~
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~

![image](https://github.com/user-attachments/assets/4474fbc6-45ae-4113-b268-52c9fcf03f68)

~~~
df.skew()
~~~

![image](https://github.com/user-attachments/assets/7b2a1a6a-9aaf-40b9-8341-b7d9ab9332cb)

~~~
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
~~~

![image](https://github.com/user-attachments/assets/c2de5ad1-b8b5-4ec0-a6db-7f9fc096425f)

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~

![image](https://github.com/user-attachments/assets/f85f47d6-9b82-44ad-ac25-ac78a7e0c62f)

~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~

![image](https://github.com/user-attachments/assets/20a65696-779b-4dc0-a144-9fb4779d1d24)

~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~


![image](https://github.com/user-attachments/assets/d6e51929-71cb-47e8-b11a-0024d60cee00)

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~


![image](https://github.com/user-attachments/assets/fddd10e0-3747-4916-b61b-c91df90d02f6)

~~~
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~


![image](https://github.com/user-attachments/assets/27de13f2-75a6-4b09-9252-2a484963df80)

~~~
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
~~~


![image](https://github.com/user-attachments/assets/e2a1fe5d-6e08-41a6-b69c-781485f2579d)

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
~~~


![image](https://github.com/user-attachments/assets/ef45f6eb-a9c2-46a9-a28b-d3d50589dbf4)

~~~
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
~~~


![image](https://github.com/user-attachments/assets/1ea75130-d47c-42f1-80e6-7e8813a4435c)




# RESULT:
       # INCLUDE YOUR RESULT HERE

       
