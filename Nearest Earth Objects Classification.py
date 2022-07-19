#!/usr/bin/env python
# coding: utf-8

# *Importing libraries* that will prove useful in the analyzing, visualizing and classificaton process.

# In[218]:


import numpy as np
import pandas as pd
import math, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
import joblib


# ### Loading the dataset

# In[2]:


df = pd.read_csv('archive/neo_v2.csv', delimiter=',')


# ### Data Analysis

# In[3]:


df.head()


# In[4]:


df.shape


# In[6]:


df.info()


# In[9]:


df.hazardous.value_counts()


# #### Checking for Null values

# In[10]:


df.isnull().sum()


# The descriptions above indicate that there are no NULL values present in our dataset.

# We can also draw some interesting observations from the table of statistical expressions on the numerical columns.

# In[11]:


df.describe()


# In[44]:


df.corr()


# In[45]:


sns.heatmap(df.corr())


# In[49]:


df.orbiting_body.unique()


# We can infer from the information above that the feature sentry_object has no correlation with the other features. Similarily, the oribiting_body feature is set to a the same value over the entire dataset so its usefulness in our prediction model is minimal. The id column too is not very useful as the name assigned for the space objects by NASA suits the data. Therefore, we remove these features from the dataset! 

# In[50]:


df.drop(["id", "orbiting_body", "sentry_object"],axis=1,inplace=True)


# In[51]:


df.head()


# In[129]:


mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, annot=True)


# Now we have input features pleasing to the senses!

# ### Visualizing the Dataset

# Plotting a pie chart to visualize the percentage hazardous status of the space objects

# In[82]:


hazard_size = [df.hazardous.value_counts()[1], df.hazardous.value_counts()[0]]
hazard_size


# In[83]:


explode = [0.1, 0]
labels = ["Dangerous", "Not Dangerous"]
labels


# In[128]:


fig, ax = plt.subplots()
ax.pie(hazard_size, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True,startangle=45)
ax.axis('equal')
ax.set_title("Danger Percentage for Nearest Earth Objects")
plt.show()


# Separating the Numeric Data for better visulaization

# In[131]:


numeric_col = ["est_diameter_min","est_diameter_max","relative_velocity","miss_distance","absolute_magnitude"]


# Creating pairplot to plot multiple pairwise bivariate distributions in the dataset will give us more visual details

# In[132]:


sns.pairplot(df[numeric_col])


# In[137]:


sns.pairplot(df[numeric_col+['hazardous']], hue='hazardous')


# Plotting every numeric feature against the hazardous column gives us more insight and could be beneficial. 

# In[ ]:


for feature in numeric_col:
    plt.figure(figsize=(10,4))
    sns.violinplot(data=df, x=feature, y='hazardous',orient='h')
    title = 'Hazardous Level due to ' + feature
    plt.title(title)
    plt.grid()


# We can further visualize the three most important features in relation to the classification outputs

# In[ ]:


sns.displot(df, x="absolute_magnitude", hue="hazardous")


# In[ ]:


sns.boxplot(data=df, x="hazardous", y="relative_velocity")


# In[ ]:


sns.displot(df, x="miss_distance", hue="hazardous")


# ### Training and Testing Sets Preparation

# Binarizing the output feature

# In[149]:


labelencoder = LabelEncoder()
df["hazardous"] = labelencoder.fit_transform(df["hazardous"])
df.head()


# In[150]:


X = df.drop(["name","hazardous"], axis=1)
X.head()


# In[155]:


Y = df["hazardous"].astype("int")
Y.head()


# In[157]:


print(X.shape, Y.shape)


# In[159]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)


# Applying standardization to the data so it resembles more like a standard normally distributed data (resize the distribution of values so that the mean of the observed values is 0 and the standard deviation is 1) proves to be helpful for various ML models.

# In[160]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# [A good read for why we use fit_transform() on our training set and just transform() on our testing set](https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe)

# ### Fitting the data to Models

# Defining object instances for the classification models

# In[179]:


xgb = XGBClassifier()
knn = KNeighborsClassifier(n_neighbors=7)
dtr = DecisionTreeClassifier(random_state=32)
mlp = MLPClassifier()
svc = SVC(kernel='linear', C=1.0)
rfr = RandomForestClassifier(random_state=40)

rand_dict = {'n_estimators': randint(low=1, high=200),
             'max_features': randint(low=1, high=8)}
rnd_searchCV_rfr = RandomizedSearchCV(rfr, rand_dict, cv=5)


# The holdout set (testing set) will be used on the best model after score evaluation on cross_val_score

# In[185]:


score_xgb = cross_val_score(xgb, x_train_scaled, y_train, cv=5).mean()
score_knn = cross_val_score(knn, x_train_scaled, y_train, cv=5).mean()
score_dtr = cross_val_score(dtr, x_train_scaled, y_train, cv=5).mean()
score_mlp = cross_val_score(mlp, x_train_scaled, y_train, cv=5).mean()
score_svc = cross_val_score(svc, x_train_scaled, y_train, cv=5).mean()
score_rfr = cross_val_score(rfr, x_train_scaled, y_train, cv=5).mean()
score_rnd_searchCV_knn = cross_val_score(rnd_searchCV_knn, x_train_scaled, y_train, cv=5).mean()
score_rnd_searchCV_dtr = cross_val_score(rnd_searchCV_dtr, x_train_scaled, y_train, cv=5).mean()
score_rnd_searchCV_rfr = cross_val_score(rnd_searchCV_rfr, x_train_scaled, y_train, cv=5).mean()


# In[194]:


scores = [score_xgb, score_knn, score_dtr, score_mlp, score_svc, score_rfr, score_rnd_searchCV_rfr]
for i in range(0, len(scores)):
    scores[i] = round(scores[i] * 100, 2)


# In[195]:


scores


# In[200]:


models = pd.DataFrame({
    "Model" : ["XG Boost", "KNeighborsClassifier", "DecisionTreeClassifier", "MLPClassifier", "Support Vector Classification", "RandomForestClassifier", "Random_SearchCV_RandomForest"],
    "Score" : [score_xgb, score_knn, score_dtr, score_mlp, score_svc, score_rfr, score_rnd_searchCV_rfr]
}).sort_values(by="Score", ascending=False).reset_index(drop=True)


# In[217]:


models


# In[224]:


best_model_name = models.loc[0][0]


# ### Save the best performing model

# In[223]:


xgb.fit(x_train,y_train)


# In[226]:


joblib.dump(xgb, best_model_name+'_saved_model.pkl')


# #### Now we can load the saved model and apply the test set on it

# In[228]:


saved_model = joblib.load(best_model_name+'_saved_model.pkl')


# In[229]:


test_pred = saved_model.predict(x_test_scaled)


# In[231]:


test_acc_score = round(accuracy_score(test_pred, y_test) * 100, 2)
test_acc_score


# This will be it for now.
# *NASA Rocks!*
