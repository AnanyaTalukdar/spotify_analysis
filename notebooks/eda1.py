#!/usr/bin/env python
# coding: utf-8

# #### Imports: pandas, numpy

# In[117]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score


# ##### Reading the data from an excel file and getting the first five rows as a measure of check

# In[118]:


df = pd.read_excel("data/raw/Spotify_data.xlsx")
df.head()


# In[119]:


df.info()
df.describe()
df.isnull().sum()


# #### understanding the type of responses

# In[120]:


for c in df.columns:
    print(str(df[c].value_counts(dropna=False))+"\n")


# ##### to understand if a user might be willing to switch to premium : switch to premium? yes/no -> premium preference/skip. 
# ##### users who chose no for willingnes, mostly were people who skipped the preference question(203). some still gave their preference. 
# ##### users who chose yes for willingness, mostly were people on the premium plan or willing to switch. very few(5) skipped preference choice

# In[121]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["preffered_premium_plan"],
    dropna=False
)


# ##### willingness to switch as a function of age

# In[122]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["Age"],
    dropna=False
)


# ##### willingness to switch as a function of gender

# In[123]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["Gender"],
    dropna=False
)


# ##### willingness to switch as a function of spotify usage period

# In[124]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["spotify_usage_period"],
    dropna=False
)


# ##### willingness to switch as a function of preferred_listening_content

# In[125]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["preferred_listening_content"],
    dropna=False
)


# ##### willingness to switch as a function of spotify_listening_device

# In[126]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["spotify_listening_device"],
    dropna=False
)


# ##### willingness to switch as a function of music_time_slot

# In[127]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["music_time_slot"],
    dropna=False
)


# ##### willingness to switch as a function of music_lis_frequency 

# In[128]:


pd.crosstab(
    df["premium_sub_willingness"],
    df["music_lis_frequency"],
    dropna=False
)


# ##### get a list of all columns we want as features and a target. create a copy of df in model_df with the features and target so as to not disturb the original dataframe
# ##### Now X contains all information inside of input features and y contains information from target column
# ##### since the model can not understand english, we map strings like yes and no in target column to numbers. we also do one hot encoding for the input features. NOTE: the point of drop_first=True is so that we can reduce noise from overlabelling. eg: if user is listening on laptop automatically means he is not listening on phone. so instead of writing both, we write just the second and drop the first

# In[129]:


features = [
    "Age",
    "Gender",
    "spotify_usage_period",
    "spotify_listening_device",
    "preferred_listening_content",
    "music_time_slot",
    "music_lis_frequency"
]
target = "premium_sub_willingness"
model_df = df[features + [target]].copy()


# x has input features for the model, y has what we want the model to predict   
X = model_df[features]
y = model_df[target]
# for every target in model_df, map yes to 1 and no to 0
model_df[target] = model_df[target].map({"Yes": 1,"No": 0})
# one hot encoding for machine ready data
X_encoded = pd.get_dummies(X, drop_first=True)


# ##### split the input and output into train and test data using train_test_split from sklearn where input comes from X encoded, output comes from y, test:train = 25:75, random_state=42 for reproducability of results and stratify y so that the ratio of yes to no in train is same as yes to no in test.
# ##### we use logistic regression to predict whether a user will switch to premium and then fit it on training data i.e. learn the pattern

# In[130]:


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ##### using the X_test data and using the model we trained, predict y and store it in y_pred
# ##### using predicted results and actual result, calculate accuracy and confusion matrix

# In[131]:


y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()


# ##### Shows which user behaviors increase or decrease the chance of switching to Premium, and by how much
# ##### create a new table coef_df, put name of every feature in the table and next to it put how strongly it influences the model’s decision. because we hot encoded the input, every category in features becomes a column. display results in desceding order that is from most influential to least

# In[132]:


coef_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

coef_df


# In[135]:


df_numeric = df.copy()

if df_numeric['Age'].dtype == 'object':
    def convert_age(x):
        if '-' in str(x):
            low, high = x.split('-')
            return (int(low) + int(high)) / 2
        elif '+' in str(x):  # handles "46+" type cases
            return int(x.replace('+',''))
        else:
            return pd.to_numeric(x, errors='coerce')
    
    df_numeric['Age'] = df_numeric['Age'].apply(convert_age)


categorical_cols = df_numeric.select_dtypes(include=['object']).columns


df_numeric = pd.get_dummies(df_numeric, columns=categorical_cols, drop_first=True)

print("All columns numeric now:")
print(df_numeric.dtypes.head())


# In[136]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,10))

corr_matrix = df_numeric.corr()

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    square=False
)

plt.title("Correlation Heatmap (All Numeric Features)")
plt.show()


# In[137]:


target_corr = corr_matrix[['premium_sub_willingness']].sort_values(
    by='premium_sub_willingness',
    ascending=False
)

plt.figure(figsize=(6,12))

sns.heatmap(
    target_corr,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation with Premium Willingness")
plt.show()

