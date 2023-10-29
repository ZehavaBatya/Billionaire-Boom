#!/usr/bin/env python
# coding: utf-8

# # Billionaire Boom: Unraveling the Wealth Enigma
# Source: https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset/

# ## The issues related to billionaires' impact on society
# 1. Corporate Practices: Some billionaires may be associated with corporations that engage in exploitative labor practices, environmental harm, or other unethical behaviors to maximize profits.
# 2. Wealth Inequality: Billionaires often represent extreme disparities in wealth, which can exacerbate income inequality and lead to social and economic imbalances.
# 3. Tax Avoidance: Wealthy individuals may employ legal loopholes and offshore tax shelters to minimize their tax contributions, which can place a burden on the general public and public services.

# ## My objectives of studying billionaires
# 1. Understanding Wealth Dynamics: Analyze how billionaires amass and manage their wealth, including investments, businesses, and financial strategies.
# 2. Innovation and Technology: Analyze the role of billionaires in driving innovation and technological advancements in various industries.
# 3. Entrepreneurial Success: Investigate the entrepreneurial journeys of billionaires to identify common traits, strategies, and lessons for aspiring business leaders.

# ## How the solutions studied will be used
# ### 1. Understanding Wealth Dynamics:
# Wealth Management Strategies: Identify and adopt effective wealth management strategies based on the practices of billionaires.
# Investment Insights: Gain insights into successful investment approaches and asset allocation.
# Business Models: Learn from billionaires' business models to enhance the profitability of your own ventures.
# Tax Planning: Develop tax-efficient strategies for wealth preservation and growth.
# ### 2. Innovation and Technology:
# Innovation Inspiration: Use insights from billionaire-led innovations to inspire new technological advancements.
# Competitive Advantage: Apply lessons from technological pioneers to gain a competitive edge in your business or field.
# R&D Strategies: Learn how billionaires fund and manage research and development to drive innovation in various industries.
# Startup Guidance: Leverage knowledge of how billionaires support startups and emerging technologies.
# ### 3. Entrepreneurial Success: 
# Entrepreneurial Skills: Identify and cultivate the key traits and skills that successful entrepreneurs, including billionaires, possess.
# Business Strategy: Develop or refine business strategies based on the experiences and strategies of billionaire entrepreneurs.
# Mentorship and Networking: Seek mentorship from successful business leaders and build a network of contacts for guidance and support.
# Case Studies: Utilize billionaire case studies as practical learning experiences for aspiring entrepreneurs.

# ## Import Dataset

# In[248]:


import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression


# In[249]:


# Import data frame
blockOfData = pd.read_csv('Billionaires Statistics Dataset.csv')


# In[250]:


# Inspect data frame
print(blockOfData.shape)
blockOfData.head(100)


# ## EDA

# In[251]:


# Reference data frame information
blockOfData.info()


# In[252]:


## Check null values
null = blockOfData.isna().mean()*100
print(null[null > 0])


# In[253]:


## Inspect null values
blockOfData.isna().any()


# ### Search for duplicates

# In[254]:


# Find duplicates
blockOfData.duplicated().value_counts()


# ### Descriptive Analysis

# In[255]:


# Initialize the descriptive analysis
blockOfData.describe(percentiles=[.0, .25, .5, .75, .9, .95, .99, .1]).T


# In[256]:


# Calculate the median
np.median(blockOfData['finalWorth'])


# In[257]:


# Histogram
feat = plt.figure(figsize=(20,20))
ax = plt.gca()
blockOfData.hist(bins=50, ax=ax, layout=(4, 2), column=["cpi_country", "cpi_change_country", 
                                                        "tax_revenue_country_country", "total_tax_rate_country"])

plt.tight_layout()
plt.show()                                                        


# ### Word Cloud

# In[258]:


import os
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image


# In[259]:


print(blockOfData.head())
print(blockOfData.columns)
text = blockOfData['country'].iloc[0]


# In[260]:


txt = blockOfData['industries'].str.cat(sep=' ')

wc = WordCloud(max_font_size=50, max_words=100,
               background_color="white").generate(txt)

# Display the generated image:
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Validation framework

# In[261]:


no = len(blockOfData)

no_val = int(no * 0.2)
no_test = int(no * 0.2)
no_train = no - no_val - no_test


# In[262]:


no


# In[263]:


no_val, no_test, no_train


# In[264]:


blockOfData.iloc[[100, 50, 25, 5]]


# In[265]:


blockOfData_instruct = blockOfData.iloc[:no_train]
blockOfData_val = blockOfData.iloc[no_train:no_train+no_val]
blockOfData_test = blockOfData.iloc[no_train+no_val:]


# In[266]:


# no = 0 # Temporary variable
ind = np.arange(no)


# In[267]:


np.random.seed(4)
np.random.shuffle(ind)


# In[268]:


blockOfData_instruct = blockOfData.iloc[ind[:no_train]]
blockOfData_val = blockOfData.iloc[ind[no_train:no_train+no_val]]
blockOfData_test = blockOfData.iloc[ind[no_train+no_val:]]


# In[269]:


blockOfData_instruct = blockOfData_instruct.reset_index(drop=True)
blockOfData_val = blockOfData_val.reset_index(drop=True)
blockOfData_test = blockOfData_test.reset_index(drop=True)


# In[270]:


# y_instruct = np.log1p(blockOfData_instruct.gdp_country.values)
# y_val = np.log1p(blockOfData_val.gdp_country.values)
# y_test = np.log1p(blockOfData_test.gdp_country.values)


# In[271]:


del blockOfData_instruct['gdp_country']
del blockOfData_val['gdp_country']
del blockOfData_test['gdp_country']


# ### Simple Linear regression model

# In[272]:


# Simple regression test
def t_lin_reg(X, y):
    multiple_ones = np.multiple_ones(X.shape[0])
    X = np.column_stack([multiple_ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


# ### More linear regression

# In[273]:


print(len(blockOfData_instruct))


# In[274]:


xi = [400, 10, 80]
w0 = 9.2
w = [0.01, 0.04, 0.002]


# In[275]:


def lin_reg(xi):
    n = len(xi)

    pred = w0

    for j in range(n):
        pred = pred + w[j] * xi[j]

    return pred


# In[276]:


xi = [400, 10, 80]
w0 = 9.2
w = [0.01, 0.04, 0.002]


# In[277]:


lin_reg(xi)


# In[278]:


np.expm1(13.76)


# ### Linear regression (vector version)

# In[279]:


def period(xi, w):
    no = len(xi)

    res = 0.0

    for j in range(no):
        res = res + xi[j] * w[j]

    return res


# In[280]:


def lin_reg(xi):
    return w0 + period(xi, w)


# In[281]:


w_latest = [w0] + w


# In[282]:


w_latest


# In[283]:


def lin_reg(xi):
    xi = [1] + xi
    return period(xi, w_latest)


# In[284]:


lin_reg(xi)


# In[285]:


w0 = 9.2
w = [0.01, 0.04, 0.002]
w_latest = [w0] + w


# In[286]:


x1 = [1, 140, 20, 1380]
x2 = [1, 130, 25, 2030]
x10 = [1, 450, 10, 85]

X = [x1, x2, x10]
X = np.array(X)
X


# In[287]:


def lin_reg(X):
    return X.period(w_latest)


# ### Train linear regression model

# In[288]:


def t_lin_reg(X, y):
    pass


# In[289]:


X = [
    [140, 20, 1380],
    [130, 20, 2001],
    [455, 11, 86],
    [158, 24, 185],
    [172, 20, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 20, 431],
    [455, 31, 86],
]

X = np.array(X)
X


# In[300]:


y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]


# In[303]:


XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
w_full = XTX_inv.dot(X.T).dot(y)


# In[304]:


w0 = w_full[0]
w = w_full[1:]


# In[305]:


w0, w


# In[309]:


def t_lin_reg(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


# In[310]:


t_lin_reg(X, y)


# ### Tree-based model

# ### Clean and prepare for tree-based model

# In[311]:


blockOfData.columns = blockOfData.columns.str.lower()


# In[313]:


blockOfData.status.value_counts()


# In[315]:


stat_vals = {
    1: 'alright', 
    2: 'standard',
    0: 'unk'
}

blockOfData.status = blockOfData.status.map(stat_vals)


# In[322]:


category_values = {
    1: 'Fashion & Retail',
    2: 'Automotive',
    3: 'Technology',
    4: 'Finance & Investments',
    5: 'Media & Entertainment',
    6: 'Telecom',
    7: 'Diversified',
    8: 'Food & Beverage',
    9: 'Logistics',
    10: 'Gambling & Casinos',
    11: 'Manufacturing',
    12: 'Real Estate',
    13: 'Metals & Mining',
    14: 'Energy',
    15: 'Healthcare',
    16: 'Sports'
}

blockOfData.category = blockOfData.category.map(category_values)

gender_values = {
    1: 'Male',
    2: 'Female',
}

blockOfData.gender = blockOfData.gender.map(gender_values)

status_values = {
    1: 'U',
    2: 'D'
}

blockOfData.status = blockOfData.status.map(status_values)

title_values = {
    1: 'Chairman and CEO',
    2: 'CEO',
    3: 'Chairman and Founder',
    4: 'CTO and Founder',
    5: 'Cochair',
    6: 'Owner',
    7: 'Director',
    8: 'ConnectionError',
    9: 'Philanthropist',
    10: 'CEO & President'
}

blockOfData.title = blockOfData.title.map(title_values)



# In[324]:


blockOfData.head()


# In[325]:


blockOfData.describe().round()


# In[353]:


y_train = (blockOfData_instruct.status == 'default').astype('int').values
y_val = (blockOfData_val.status == 'default').astype('int').values
y_test = (blockOfData_test.status == 'default').astype('int').values


# ### Decision-making trees. Disclaimer: The setup

# In[335]:


def profile(billionaire):
    if billionaire['category'] == 'yep':
        if billionaire['Technology'] == 'tech':
            return 'standard'
        else:
            return 'okay'
    else:
        if billionaire['finalWorth'] > 100000:
            return 'okay'
        else:
            return 'standard'


# In[336]:


xi = blockOfData_instruct.iloc[0].to_dict()


# In[337]:


profile(xi)


# In[338]:


from sklearn.tree import export_text
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer



# In[342]:


t_dicts = blockOfData_instruct.fillna(0).to_dict(orient='records')


# In[345]:


dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(t_dicts)


# In[354]:


decision_t = DecisionTreeClassifier()
decision_t.fit(X_train, y_train) 


# In[356]:


value_dict = blockOfData_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(value_dict)


# In[367]:


# Note-to-self: To be worked on in future iterations
# y_guess = decision_t.predict_proba(X_val)[:, 1]
# roc_auc_score(y_val, y_pred)

# IndexError: index 1 is out of bounds for axis 1 with size 1


# In[368]:


print(export_text(decision_t, feature_names=list(dv.get_feature_names_out()))) 

