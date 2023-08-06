#!/usr/bin/env python
# coding: utf-8

# In[138]:


import matplotlib.pyplot as plt
import seaborn as sns
import math
import sklearn
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import scikitplot as skplt
import statsmodels.api as sm
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

# example of grid searching key hyperparametres for logistic regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import warnings
import itertools
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve, roc_auc_score

warnings.filterwarnings('ignore')


# In[4]:


file = r'C:\Users\Pichau\Downloads\treino.xlsx'
df_train = pd.read_excel(file)
df_train=df_train.iloc[:,1:]


# In[80]:


train_size = int(df_train.shape[0] * 0.70)
train_df, test_df = df_train.iloc[:train_size, :], df_train.iloc[train_size:, :]

X_train = train_df[train_df.iloc[:, 0:21].columns]
y_train = train_df[train_df.iloc[:, 21:].columns]

X_test = test_df[test_df.iloc[:, 0:21].columns]
y_test = test_df[test_df.iloc[:, 21:].columns]


# In[81]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled,columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21'])
X_test_scaled = pd.DataFrame(X_test_scaled,columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21'])


# In[82]:


smote = SMOTE()
X_train_scaled_smote, y_train = smote.fit_resample(X_train_scaled, y_train)


# In[66]:


def RunModel(lr, X_train, y_train, X_test, y_test):
    lr.fit(X_train, y_train.values.ravel())
    pred = lr.predict(X_test)
    matrix = confusion_matrix(y_test, pred)
    return matrix, pred


# In[65]:


def PrintStats(cmat, y_test, pred):
    tpos = cmat[0][0]
    fneg = cmat[1][1]
    fpos = cmat[0][1]
    tneg = cmat[1][0]


# In[155]:


lr = LogisticRegression(C= 1.0, penalty= 'l2', solver= 'newton-cg', random_state=0)
cmat, pred = RunModel(lr, X_train_scaled_smote, y_train, X_test_scaled, y_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred, pos_label=1)

random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

auc_score1 = roc_auc_score(y_test, pred)

PrintStats(cmat, y_test, pred)
skplt.metrics.plot_confusion_matrix(y_test, pred)


# In[156]:


print (classification_report(y_test, pred))
roc_auc_score(y_test, pred)


# In[101]:


plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
#plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[ ]:





# In[159]:


file = r'C:\Users\Pichau\Downloads\teste.xlsx'
df_teste = pd.read_excel(file)
df_teste=df_teste.iloc[:,1:]


# In[164]:


df_teste=pd.DataFrame(scaler.fit_transform(df_teste),columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21'])


# In[165]:


df_teste


# In[175]:


final_df = pd.concat([df_teste, pd.DataFrame(lr.predict(df_teste))], axis=1)


# In[174]:





# In[176]:


dat1.to_excel(r'C:\Users\Pichau\Downloads\df_teste_final.xlsx', index=False)


# In[177]:


dat1


# In[ ]:




