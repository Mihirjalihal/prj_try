#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas scikit-learn matplotlib seaborn


# In[4]:


import pandas as pd

# Load the dataset
file_path = 'glass.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())


# In[ ]:





# In[5]:


from sklearn.preprocessing import StandardScaler

# Check for missing values
print(data.isnull().sum())

# Separate features and target
X = data.drop(columns=['Type'])
y = data['Type']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)


# In[8]:


from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")


# In[9]:


import matplotlib.pyplot as plt

# Test different values of n_neighbors
neighbors = range(1, 21)
accuracies = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the results
plt.plot(neighbors, accuracies)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Varying number of neighbors')
plt.show()


# In[ ]:




