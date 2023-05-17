#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing libraries necessary


# In[2]:


dataset = pd.read_csv('combined_data_1.txt',header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
#reading dataset


# In[3]:


dataset.head()


# In[4]:


dataset['Rating'] = dataset['Rating'].astype(float)


# In[5]:


dataset.dtypes


# In[6]:


dataset.shape


# In[7]:


dataset.head()


# In[8]:


p = dataset.groupby('Rating')['Rating'].count()


# In[9]:


p


# In[10]:


p=pd.DataFrame(p)
p


# In[11]:


p=p.rename(columns = {'Rating':'Count'})


# In[12]:


p


# In[13]:


p.sort_values(by='Count', ascending=False)


# In[14]:


dataset.isnull().sum()


# In[15]:


movie_count = dataset.isnull().sum()[1]


# In[16]:


movie_count


# In[17]:


dataset['Cust_Id'].nunique()


# In[18]:


dataset['Cust_Id'].unique()


# In[19]:


cust_count = dataset['Cust_Id'].nunique()-movie_count
cust_count


# In[20]:


rating_count = dataset['Cust_Id'].count() - movie_count

rating_count


# In[21]:


p


# In[22]:


p.reset_index(inplace=True)


# In[23]:


p


# In[24]:


df_nan = pd.DataFrame(pd.isnull(dataset.Rating) )
df_nan.head()


# In[25]:


df_nan = df_nan[df_nan['Rating'] == True]
df_nan.shape


# In[26]:


df_nan


# In[27]:


df_nan=df_nan.reset_index()
df_nan.head()


# In[28]:


df_nan.head(5)


# In[29]:


df_nan.shape


# In[30]:


df_nan['index'][:-1]


# In[31]:


df_nan['index'][1:]


# In[32]:


movie_np = []
movie_id = 1
for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    temp=np.full((1,i-j-1),movie_id)
    movie_np=np.append(movie_np,temp)
    movie_id+=1
    
last_record = np.full((1,len(dataset) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

print(f'Movie numpy: {movie_np}')
print(f'Length: {len(movie_np)}')


# In[33]:


r=zip(df_nan['index'][1:],df_nan['index'][:-1])


# In[34]:


tuple(r)


# In[35]:


dataset[pd.notnull(dataset['Rating'])]


# In[36]:


#To append the above created array to the datset after removing the 'nan' rows
dataset = dataset[pd.notnull(dataset['Rating'])]

dataset['Movie_Id'] = movie_np.astype(int)
dataset['Cust_Id'] =dataset['Cust_Id'].astype(int)
print('-Dataset examples-')
dataset.head()


# In[37]:


dataset.shape


# # data cleaning

# In[38]:


f = ['count','mean']


# In[39]:


dataset.groupby('Movie_Id').agg(f)


# In[40]:


f


# In[41]:


dataset.groupby('Movie_Id')['Rating'].agg(f)


# In[42]:


#To create a list of all the movies rated less often(only include top 30% rated movies)
dataset_movie_summary = dataset.groupby('Movie_Id')['Rating'].agg(f)

dataset_movie_summary.index = dataset_movie_summary.index.map(int)

movie_benchmark = round(dataset_movie_summary['count'].quantile(0.75),2)

drop_movie_list = dataset_movie_summary[dataset_movie_summary['count'] <= movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))


# In[43]:


#calculate the thershold value of each customer,
dataset_cust_summary = dataset.groupby('Cust_Id')['Rating'].agg(f)#f= count(), mean()
dataset_cust_summary.index = dataset_cust_summary.index.map(int)
cust_benchmark = round(dataset_cust_summary['count'].quantile(0.75),0)
drop_cust_list = dataset_cust_summary[dataset_cust_summary['count'] < cust_benchmark].index

print(f'Customer minimum times of review: {cust_benchmark}')


# In[44]:


print(f'Original Shape: {dataset.shape}')


# In[45]:


dataset = dataset[~dataset['Movie_Id'].isin(drop_movie_list)]#~ symbol will not include the true values coming from the isin()
dataset = dataset[~dataset['Cust_Id'].isin(drop_cust_list)]
print('After Trim Shape: {}'.format(dataset.shape))


# In[46]:


print('-Data Examples-')
dataset.head()


# # Create ratings matrix for 'ratings' matrix with Rows = userId, Columns = movieId

# In[47]:


# sparce matrix


# In[48]:


df_p = pd.pivot_table(dataset,values='Rating',index='Cust_Id',columns='Movie_Id')
print(df_p.shape)


# In[49]:


df_p.head(10)


# In[50]:


import pandas as pd
df_title = pd.read_csv(r"movie_titles.csv",encoding="ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
#unique code escape as encoding

df_title.set_index('Movie_Id', inplace = True)

print (df_title.head(10))


# # To install the scikit-surprise library for implementing SVD

# In[51]:


get_ipython().system('pip install scikit-surprise')


# In[52]:


import math
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
# prevent data from overfitting
#fixed number of folds and we will be validating on each folds here 3 folds almost equal


# In[53]:


# Load Reader library
reader = Reader()

# get just top 100K rows for faster run time
data = Dataset.load_from_df(dataset[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)

# Use the SVD algorithm.
svd = SVD()

# Compute the RMSE of the SVD algorithm
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)


# In[54]:


data


# In[55]:


dataset.head()


# In[56]:


df_title


# # To find all the movies rated as 5 stars by user with userId = 712664

# In[57]:


dataset_712664 = dataset[(dataset['Cust_Id'] == 712664) & (dataset['Rating'] == 5)] # rating is 5
dataset_712664 = dataset_712664.set_index('Movie_Id') # setting movie_id as index
dataset_712664 = dataset_712664.join(df_title)['Name'] # joined to df title in order to get names of the movie acc to movie id
dataset_712664.head(10)


# In[58]:


user_712664=df_title


# In[59]:


df_title


# # Train an SVD to predict ratings for user with userId = 1
# 

# In[60]:


# Create a shallow copy for the movies dataset
user_712664 = df_title.copy() # so that there is no change in parent variable
user_712664 = user_712664.reset_index() # coz we need movie id as normal column


# In[61]:


#To remove all the movies rated less often 
user_712664 = user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)] # removing movies rated very less 


# In[62]:


# getting full dataset
data = Dataset.load_from_df(dataset[['Cust_Id', 'Movie_Id', 'Rating']], reader)


# In[63]:


#create a training set for svd
trainset = data.build_full_trainset()  # we cant do 70 30 split for svd here we have to give full data for trainset by build full trainset it is from surprise package
svd.fit(trainset)  # fitting train data set in algo 


# In[64]:


#Predict the ratings for user_712664
user_712664['Estimate_Score'] = user_712664['Movie_Id'].apply(lambda x: svd.predict(712664, x).est) # est eastimating rating score it eestimates the rating customer can give to other movies
#apply function is used when u have to apply diff functions to the data x here indicates the movie id or on ehich function is being applied


# In[65]:


#Drop extra columns from the user_712664 data frame
user_712664 = user_712664.drop('Movie_Id', axis = 1) # now we just need the movie name and no the id so dropping this table


# In[66]:


# Sort predicted ratings for user_712664 in descending order
user_712664 = user_712664.sort_values('Estimate_Score', ascending=False) # sorting acc to est in decending order


# In[67]:


#Print top 10 recommendations
print(user_712664.head(10))


# In[68]:


# here we can see all the top values are rated 5 or around 5


# In[ ]:




