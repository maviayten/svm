#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

data_path = "C:\\Users\\User\\Downloads\\banana_quality.csv"
data = pd.read_csv(data_path)

data.head()


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

basic_stats = data.describe()
print(basic_stats)


# In[4]:


data.info()


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

basic_stats = data.describe()

selected_features = data.columns.drop('Quality')

plt.figure(figsize=(18, 12))
for i, feature in enumerate(selected_features, start=1):
    plt.subplot(3, 3, i)
    sns.histplot(data[feature], kde=True, bins=20)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.show()


# In[6]:


corr_matrix = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Banana Quality Dataset Korelasyon Matrisi')
plt.show()

corr_matrix


# In[7]:


plt.figure(figsize=(5, 3))
sns.scatterplot(x=data['Size'], y=data['HarvestTime'])
plt.title('Boyut ve Hasat Zamanı İlişkisi')
plt.xlabel('Boyut')
plt.ylabel('Hasat Zamanı')
plt.show()

plt.figure(figsize=(5, 3))
sns.scatterplot(x=data['Weight'], y=data['Acidity'])
plt.title('Ağırlık ve Asitlik İlişkisi')
plt.xlabel('Ağırlık')
plt.ylabel('Asitlik')
plt.show()

plt.figure(figsize=(5, 3))
sns.scatterplot(x=data['Weight'], y=data['Sweetness'])
plt.title('Tatlılık ve Ağırlık İlişkisi')
plt.xlabel('Ağırlık')
plt.ylabel('Tatlılık')
plt.show()

plt.figure(figsize=(5, 3))
sns.scatterplot(x=data['Ripeness'], y=data['Acidity'])
plt.title('Olgunluk ve Asitlik İlişkisi')
plt.xlabel('Olgunluk')
plt.ylabel('Asitlik')
plt.show()


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

numeric_data = data.select_dtypes(include=[np.number])

melted_numeric_data = pd.melt(numeric_data)

plt.figure(figsize=(12, 8))
sns.boxplot(x='variable', y='value', data=melted_numeric_data)
plt.xticks(rotation=90)
plt.title('Sayısal Değişkenler İçin Kutu Grafiği')
plt.xlabel('Değişkenler')
plt.ylabel('Değerler')
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Quality', data=data)
plt.title('Muz Kalite Dağılımı')
plt.xlabel('Kalite Sınıfı')
plt.ylabel('Frekans')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[11]:


from sklearn.preprocessing import LabelEncoder, StandardScaler


missing_values = data.isnull().sum()
print(missing_values)


# In[12]:


label_encoder = LabelEncoder()
data['Quality'] = label_encoder.fit_transform(data['Quality'])


# In[13]:


Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1

outliers = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))

outlier_counts = outliers.sum()
outlier_counts = outlier_counts[outlier_counts > 0]

outlier_counts


# In[14]:


from scipy import stats

print("Aykırı değerlerden arındırılmadan önce veri setinin boyutu:", data.shape)


# In[15]:


numeric_columns = data.select_dtypes(include=[np.number]).columns

z_scores = stats.zscore(data[numeric_columns])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_filtered = data[filtered_entries]

print("Aykırı değerlerden arındırılmış veri setinin boyutu:", data_filtered.shape)

scaler = StandardScaler()
data_filtered[numeric_columns] = scaler.fit_transform(data_filtered[numeric_columns])


# In[17]:


numeric_features = data.columns.drop('Quality')

scaler = StandardScaler()

scaled_features = scaler.fit_transform(data[numeric_features])

scaled_data = pd.DataFrame(scaled_features, columns=numeric_features)

scaled_data.head()


# In[18]:


plt.figure(figsize=(18, 12))
for i, feature in enumerate(scaled_data, start=1):
    plt.subplot(3, 3, i)
    sns.histplot(scaled_data[feature], kde=True, bins=20)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.show()


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X = scaled_data
y = data['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Doğruluk: ", accuracy)
print(classification_rep)


# In[20]:


kernel_functions = ['linear', 'rbf', 'poly', 'sigmoid']
accuracy_scores = {}

for kernel in kernel_functions:
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[kernel] = accuracy

accuracy_scores


# In[21]:


from sklearn.model_selection import train_test_split, GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 2, 3, 10, 100]}

grid_search_linear = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5, scoring='accuracy', verbose=3)

grid_search_linear.fit(X_train, y_train)

best_params_linear = grid_search_linear.best_params_['C']
best_score_linear = grid_search_linear.best_score_

best_params_linear, best_score_linear


# In[22]:


best_linear_model = SVC(kernel='linear', C=best_params_linear)
best_linear_model.fit(X_train, y_train)

y_pred_linear = best_linear_model.predict(X_test)

classification_report_linear = classification_report(y_test, y_pred_linear)

print("En iyi Lineer Çekirdek Parametreleri:", best_params_linear)
print("En İyi Doğruluk Skoru:", best_score_linear)
print("\nSınıflandırma Raporu:\n", classification_report_linear)


# In[23]:


param_grid_poly = {
    'C': [0.1, 1, 10],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

grid_search_poly = GridSearchCV(SVC(kernel='poly'), param_grid_poly, refit=True, verbose=3, cv=5)

grid_search_poly.fit(X_train, y_train)

best_params_poly = grid_search_poly.best_params_
best_score_poly = grid_search_poly.best_score_

best_params_poly, best_score_poly


# In[24]:


best_poly_model = SVC(kernel='poly', C=best_params_poly['C'], degree=best_params_poly['degree'], gamma=best_params_poly['gamma'])
best_poly_model.fit(X_train, y_train)

y_pred_poly = best_poly_model.predict(X_test)

classification_report_poly = classification_report(y_test, y_pred_poly)

print("En iyi Polinom Çekirdek Parametreleri:", best_params_poly)
print("En İyi Doğruluk Skoru:", best_score_poly)
print("\nSınıflandırma Raporu:\n", classification_report_poly)


# In[25]:


param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
}

grid_search_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid_rbf, refit=True, verbose=3, cv=5)

grid_search_rbf.fit(X_train, y_train)

best_params_rbf = grid_search_rbf.best_params_
best_score_rbf = grid_search_rbf.best_score_

best_params_rbf, best_score_rbf


# In[26]:


best_rbf_model = SVC(kernel='rbf', C=best_params_rbf['C'], gamma=best_params_rbf['gamma'])
best_rbf_model.fit(X_train, y_train)

y_pred_rbf = best_rbf_model.predict(X_test)

classification_report_rbf = classification_report(y_test, y_pred_rbf)

print("En iyi RBF Çekirdek Parametreleri:", best_params_rbf)
print("En İyi Doğruluk Skoru:", best_score_rbf)
print("\nSınıflandırma Raporu:\n", classification_report_rbf)


# In[27]:


param_grid_sigmoid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
}

grid_search_sigmoid = GridSearchCV(SVC(kernel='sigmoid'), param_grid_sigmoid, refit=True, verbose=3, cv=5)

grid_search_sigmoid.fit(X_train, y_train)

best_params_sigmoid = grid_search_sigmoid.best_params_
best_score_sigmoid = grid_search_sigmoid.best_score_

best_params_sigmoid, best_score_sigmoid


# In[28]:


best_sigmoid_model = SVC(kernel='sigmoid', C=best_params_sigmoid['C'], gamma=best_params_sigmoid['gamma'])
best_sigmoid_model.fit(X_train, y_train)

y_pred_sigmoid = best_sigmoid_model.predict(X_test)

classification_report_sigmoid = classification_report(y_test, y_pred_sigmoid)

print("En iyi Sigmoid Çekirdek Parametreleri:", best_params_sigmoid)
print("En İyi Doğruluk Skoru:", best_score_sigmoid)
print("\nSınıflandırma Raporu:\n", classification_report_sigmoid)


# In[ ]:




