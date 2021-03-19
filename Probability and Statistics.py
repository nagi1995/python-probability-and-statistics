#!/usr/bin/env python
# coding: utf-8

# ### Pearson correlation coefficient vs Spearman correlation coefficient

# In[310]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import random


# In[311]:


x = np.linspace(-1, 1, 21)
x


# In[312]:


y = x ** 5 + 0.1 * np.random.randn(len(x))
y


# In[313]:


plt.plot(x, y)


# In[314]:


stats.pearsonr(x, y), stats.spearmanr(x, y)


# ### Central limit theorem

# In[315]:


n = 1000
e = np.random.exponential(size = n)
sns.displot(e)


# In[316]:


m = 30
sample_means = []

for i in range(1000):
    sample = random.sample(list(e), m)
    sample_means.append(np.mean(sample))

sns.distplot(sample_means)


# In[317]:


stats.probplot(sample_means, dist = "norm", plot = plt)


# ### Confidence Interval (CI) of a random variable

# In[318]:


n = 10000
mean = 2
sigma = 5
d = np.random.normal(loc = mean, scale = sigma, size = n)
sns.distplot(d)


# In[319]:


M = [2, 10, 30, 50, 100, 500]
print("95% confidence interval")
for m in M:
    sample = random.sample(list(d), m)
    sample_mean = np.mean(sample)
    lower_bound = sample_mean - ((2 * sigma)/np.sqrt(m))
    upper_bound = sample_mean + ((2 * sigma)/np.sqrt(m))
    print("sample size = {}, lower bound = {}, upper bound = {}, population mean = {}".format(m, lower_bound, upper_bound, mean))


# ### CI using empirical bootstrap

# In[320]:


pop_size = 100000
pop_mean = 20
pop_sigma = 10
#population = np.random.normal(loc = pop_mean, scale = pop_sigma, size = pop_size)
population = np.random.exponential(scale = 3, size = pop_size)
pop_mean = np.mean(population)
print(pop_mean)
sns.distplot(population)


# In[321]:


n = 200
sample = random.sample(list(population), n)
sns.distplot(sample)


# In[322]:


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

# In[323]:


n = 100
s1 = np.random.uniform(low = 3, high = 5.65, size = n)
mu1 = np.mean(s1)
print(mu1)
s2 = np.random.normal(loc = 4, size = n)
mu2 = np.mean(s2)
print(mu2)


# In[324]:



# H0: both s1 and s2 have same mean
# H1: s1 and s2 does not have same mean
# test statistic: diff of means

obs = abs(mu1 - mu2)

S = np.concatenate((s1, s2))
test_stats = []
k = 1000
for i in range(k):
    np.random.shuffle(S)
    temp_s1, temp_s2 = S[:n], S[n:]
    test_stats.append(abs(np.mean(temp_s1) - np.mean(temp_s2)))

kwargs = {'cumulative': True}
sns.distplot(test_stats, hist_kws=kwargs, kde_kws=kwargs)
plt.grid()


# In[325]:


# prob = P(test_stats >= obs | H0)
print(obs)
prob = np.sum(np.array(test_stats) >= obs)/k
print(prob)


# In[326]:


# K-S test
stats.ks_2samp(s1, s2)


# In[ ]:




