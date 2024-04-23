import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris  


df = pd.read_csv('csv_files/breast_cancer_diagnosis.csv')

label_encoder = LabelEncoder()
df['diagnosiss'] = label_encoder.fit_transform(df['diagnosis'])
df.drop(columns=['diagnosis'], inplace=True)

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
null_data = df.isnull().sum()


print("Null Data:")
print(null_data)
df_modified = df.drop(columns=['Unnamed: 32'])
print(df_modified.head())


X = df_modified.drop(columns=['diagnosiss'])  
y = df_modified['diagnosiss']  


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# Verify the shapes of the subsets
print("Training set shape (X, y):", X_train.shape, y_train.shape)
print("Validation set shape (X, y):", X_valid.shape, y_valid.shape)



# hyperparameters
epochs = 20
batch_size = 32
learning_rate = 0.001 
no_of_neighbors=5


nb_model = GaussianNB()
print("Epoch's for Naive Bayes")

for epoch in range(epochs):
    # Shuffle the training data (optional)
    shuffled_indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train.iloc[shuffled_indices]
    y_train_shuffled = y_train.iloc[shuffled_indices]
    
    # Mini-batch training
    for batch_start in range(0, len(X_train), batch_size):
        batch_end = batch_start + batch_size
        X_batch = X_train_shuffled.iloc[batch_start:batch_end]
        y_batch = y_train_shuffled.iloc[batch_start:batch_end]
        
        # Fit the model on the current mini-batch
        nb_model.partial_fit(X_batch, y_batch, classes=np.unique(y))
        
    # Evaluate the model on the validation set
    y_valid_pred = nb_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    
    print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {accuracy:.4f}")
    
for _ in range(5):
    print("-" * 20)  
    

# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=no_of_neighbors)
print ("Epochs for KNN")


for epoch in range(epochs):
    # Shuffle the training data (optional)
    shuffled_indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train.iloc[shuffled_indices]
    y_train_shuffled = y_train.iloc[shuffled_indices]
    
    # Fit the model on the entire training data
    knn_model.fit(X_train_shuffled, y_train_shuffled)
        
    # Evaluate the model on the validation set
    y_valid_pred = knn_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    

    
    print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {accuracy:.4f}")
for _ in range(5):
    print("-" * 20)

iris = load_iris()
X, y = iris.data, iris.target

nb_model = GaussianNB()

cv_scores = cross_val_score(nb_model, X, y, cv=5)  
print("Cross-validation scores for GaussianNB:", cv_scores)
print("Mean CV score of GausianNB:", cv_scores.mean())
print("Standard deviation of CV scores of GaussianNB:", cv_scores.std())
for _ in range(5):
    print("-" * 20)  
    


no_of_neighbors=5
knn_model = KNeighborsClassifier(n_neighbors=no_of_neighbors)
cv_scores_knn = cross_val_score(knn_model, X, y, cv=no_of_neighbors)


print("Cross-validation scores for KNN:", cv_scores_knn)
print("Mean CV score for KNN:", cv_scores_knn.mean())
print("Standard deviation of CV scores for KNN:", cv_scores_knn.std())
