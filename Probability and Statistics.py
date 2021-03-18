#!/usr/bin/env python
# coding: utf-8

# ### Pearson correlation coefficient vs Spearman correlation coefficient

# In[91]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import random


# In[75]:


x = np.linspace(-1, 1, 21)
x


# In[82]:


y = x ** 5 + 0.1 * np.random.randn(len(x))
y


# In[83]:


plt.plot(x, y)


# In[84]:


stats.pearsonr(x, y), stats.spearmanr(x, y)


# ### Central limit theorem

# In[114]:


n = 1000
e = np.random.exponential(size = n)
sns.displot(e)


# In[119]:


m = 30
sample_means = []

for i in range(1000):
    sample = random.sample(list(e), m)
    sample_means.append(np.mean(sample))

sns.distplot(sample_means)


# In[116]:


stats.probplot(sample_means, dist = "norm", plot = plt)


# ### Confidence Interval (CI) of a random variable

# In[154]:


n = 10000
mean = 2
sigma = 5
d = np.random.normal(loc = mean, scale = sigma, size = n)
sns.distplot(d)


# In[159]:


M = [2, 10, 30, 50, 100, 500]
print("95% confidence interval")
for m in M:
    sample = random.sample(list(d), m)
    sample_mean = np.mean(sample)
    lower_bound = sample_mean - ((2 * sigma)/np.sqrt(m))
    upper_bound = sample_mean + ((2 * sigma)/np.sqrt(m))
    print("sample size = {}, lower bound = {}, upper bound = {}, population mean = {}".format(m, lower_bound, upper_bound, mean))


# ### CI using empirical bootstrap

# In[183]:


pop_size = 100000
pop_mean = 20
pop_sigma = 10
#population = np.random.normal(loc = pop_mean, scale = pop_sigma, size = pop_size)
population = np.random.exponential(scale = 3, size = pop_size)
pop_mean = np.mean(population)
print(pop_mean)
sns.distplot(population)


# In[184]:


n = 200
sample = random.sample(list(population), n)
sns.distplot(sample)


# In[185]:


M = [2, 10, 30, 50]
print("95% CI")
for m in M:
    samples_mean = []
    for i in range(1000):
        samples_mean.append(np.mean(random.sample(list(sample), m)))

    lower_bound = np.percentile(samples_mean, 2.5)
    upper_bound = np.percentile(samples_mean, 97.5)
    print("sample size = {}, lower bound = {}, upper bound = {}, population mean = {}".format(m, lower_bound, upper_bound, pop_mean))


# ### Hypothesis testing

# In[ ]:




