#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libary needed 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier


# In[2]:


#reading the csv file

df = pd.read_csv('C:\\Users\\Tathagata\\Downloads\\data crop\\Crop_recommendation.csv') 


# In[3]:


#info of first 5 row 
df.head()


# In[4]:


df


# In[5]:


print("Shape of the dataframe: ",df.shape) #info of  the shape
df.isna().sum() #Pandas DataFrame.isna() function is used to check the missing values in a given DataFrame.


# In[6]:


df.info() #info of the data frame


# In[7]:


df.describe()


# In[8]:


df.dtypes #type of value in every coloumns


# In[9]:


sns.displot(x=df['N'], bins=100,edgecolor="black",color='black',facecolor='b')
plt.title("Nitrogen",size=30)
plt.show()


# In[10]:


sns.displot(x=df['P'],bins=100,color='black',edgecolor='black',kde=True,facecolor='brown') 
plt.title("Phosphorus", size=20)
plt.xticks(range(0,150,20))
plt.show()


# In[11]:


sns.displot(x=df['humidity'],bins=50, color='black',facecolor='#ffb03b',kde=True,edgecolor='black')
plt.title("Humidity",size=20)
plt.show()


# In[12]:


sns.displot(x=df['rainfall'], bins=100, color='black',facecolor='pink',kde=True,edgecolor='black')
plt.title("Rainfall",size=20)
plt.show()


# In[13]:


sns.displot(x=df['temperature'], bins=100,kde=True,edgecolor="black",color='black',facecolor='#ffb03b')
plt.title("Temperature",size=20)
plt.show()


# In[14]:


sns.displot(x=df['K'],kde=True, bins=50, facecolor='green',edgecolor='black', color='black')
plt.title("Potassium",size=20)
plt.show()


# In[15]:


sns.relplot(x='rainfall',y='temperature',data=df,kind='scatter',hue='label',height=5)
plt.show()


# In[16]:


sns.relplot(x='K',y='rainfall',data=df,kind='scatter',hue='label',height=5)
plt.show()


# In[17]:


sns.pairplot(data=df,hue='label')
plt.show()


# In[150]:


# Unique values in the label column

crops = df['label'].unique()
print(len(crops))
print(crops)
print(pd.value_counts(df['label']))


# In[19]:


# Filtering each unique label and store it in a list df2 for to plot the box plot

df2=[]
for i in crops:
    df2.append(df[df['label'] == i])
df2[1].head()


# In[20]:


#for detecting outlier in the dataset

def detect_outlier(x):
    q1 = x.quantile(0.25) #This line calculates the first quartile (25th percentile) of the dataset. It represents the value below which 25% of the data falls.
    q3 = x.quantile(0.75)#This line calculates the third quartile (75th percentile) of the dataset .
    IQR = q3-q1 #the interquartile range (IQR), which is the range between the first and third quartiles.
    lower_limit = q1 - (1.5*IQR)
    upper_limit = q3 + (1.5*IQR)
    print(f"Lower limit: {lower_limit} Upper limit: {upper_limit}")
    print(f"Minimum value: {x.min()}   MAximum Value: {x.max()}")
    for i in [x.min(),x.max()]:
        if i == x.min():
            if lower_limit > x.min():
                print("Lower limit failed - Need to remove minimum value")
            elif lower_limit < x.min():
                print("Lower limit passed - No need to remove outlier")
        elif i == x.max():
            if upper_limit > x.max():
                print("Upper limit passed - No need to remove outlier")
            elif upper_limit < x.max():
                print("Upper limit failed - Need to remove maximum value")
detect_outlier(df['K'][df['label']=='grapes'])


# In[21]:


for i in df['label'].unique():
    detect_outlier(df['K'][df['label']==i])
    print('---------------------------------------------')


# In[22]:


x = df.drop(['label'], axis=1)
x.head()


# In[23]:


Y = df['label']

# used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.
encode = preprocessing.LabelEncoder()

y = encode.fit_transform(Y)
print("Label length: ",len(y))


# In[24]:


y


# In[25]:


Y


# In[26]:


#splitting the dataset for training and testing
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

print(len(x_train),len(y_train),len(x_test),len(y_test))


# In[27]:


x_train


# In[28]:


y_train


# In[29]:


# Best model choosing
a={'decision tree' : {
        'model' : DecisionTreeClassifier(criterion='gini'),
        'params':{'decisiontreeclassifier__splitter':['best','random']}
    },
    'svm': {
        'model': SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
   'k classifier':{
       'model':KNeighborsClassifier(),
       'params':{'kneighborsclassifier__n_neighbors':[5,10,20,25],'kneighborsclassifier__weights':['uniform','distance']}
   }
}


# In[30]:


score=[]
details = []
best_param = {}
for mdl,par in a.items():
    pipe = make_pipeline(preprocessing.StandardScaler(),par['model'])
    res = model_selection.GridSearchCV(pipe,par['params'],cv=5)
    res.fit(x_train,y_train)
    score.append({
        'Model name':mdl,
        'Best score':res.best_score_,
        'Best param':res.best_params_
    })
    details.append(pd.DataFrame(res.cv_results_))
    best_param[mdl]=res.best_estimator_
pd.DataFrame(score)


# In[31]:


score


# In[32]:


predicted = best_param['random_forest'].predict(x_test)
predicted


# In[33]:


classifier= RandomForestClassifier(criterion='gini')
classifier.fit(x_train,y_train)#fitting the randomForest Algorithm


# In[34]:


classifier.score(x_test,y_test)


# In[35]:


classifier_en= DecisionTreeClassifier( criterion='entropy')
classifier_en.fit(x_train,y_train)


# In[36]:


classifier_en.score(x_test,y_test)


# In[38]:


prediction = classifier.predict(t1)
prediction


# In[39]:


# value mapping ,Eg: If predicted value id 20 then its belongs to Crop rice. So on...
dha2 =pd.DataFrame(Y)
code = pd.DataFrame(dha2['label'].unique())


# In[40]:


dha = pd.DataFrame(y)
encode = pd.DataFrame(dha[0].unique())
refer = pd.DataFrame()
refer['code']=code
refer['encode']=encode
refer


# In[44]:


from sklearn.svm import SVC


# In[46]:


clf_svc = SVC(kernel = 'rbf', random_state = 0)
clf_svc.fit(x_train,y_train)


# In[47]:


clf_svc.score(x_test,y_test)


# In[53]:


knn = KNeighborsClassifier(n_neighbors=7) #K-NEAREST NEIGHBOUR ALGORITHM
knn.fit(x_train, y_train)


# In[54]:


knn.score(x_test,y_test)


# In[49]:


clf_svm=SVM()
clf_svm.fit(x_train,y_train)


# In[61]:


predicted_knn = knn.predict(x_test)
predicted_knn


# In[59]:


predicted_svc = clf_svc.predict(x_test)
predicted_svc


# In[60]:


predicted_dec = classifier_en.predict(x_test)
predicted_dec


# In[71]:


predic_random=classifier.predict(x_test)
predic_random


# In[64]:


plt.scatter(predicted_dec,predicted_knn)
plt.xlabel("predicted_decisionTree")
plt.ylabel("predicted_Knn")

plt.show()


# In[63]:


plt.plot(predicted_dec,predicted_knn)
plt.xlabel("predicted_decisionTree")
plt.ylabel("predicted_Knn")

plt.show()


# In[68]:


plt.scatter(predicted_dec,predicted_svc)
plt.xlabel("predicted_decisionTree")
plt.ylabel("predicted_Svc")

plt.show()


# In[ ]:





# In[72]:


plt.scatter(predicted_dec,predic_random)
plt.xlabel("predicted_decisionTree")
plt.ylabel("predicted_randomForest")

plt.show()


# In[73]:


dha = pd.DataFrame(y)
encode = pd.DataFrame(dha[0].unique())
refer = pd.DataFrame()
refer['code']=code
refer['encode']=encode
refer


# In[78]:


val=refer.sort_values(by="encode")
val


# In[148]:


t1=np.array([[107,34,32,26.774637,66.413269,6.78006,177.774507],[12,25,43,43.004459,82.320763,7.840207,263.964248]])
t1


# In[ ]:





# In[155]:


pre = classifier.predict(x_test)
class_labels = ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", "grapes", "jute", "kidneybeans", 
                "lentil", "maize", "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
                "rice", "watermelon"]

for prediction in pre:
    if prediction >= 0 and prediction < len(class_labels):
        print(class_labels[prediction])

   
    


# In[154]:


sns.catplot(data=df, x='label', y='ph', kind='box', height=10, aspect=20/8.27)
plt.title("Nitrogen",size=30)
plt.show()


# In[ ]:




