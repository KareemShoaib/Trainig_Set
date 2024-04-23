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


df_iphone = pd.read_csv('csv_files/Iphone_purchase.csv')


label_encoder = LabelEncoder()
df_iphone['gender_encoded'] = label_encoder.fit_transform(df_iphone['Gender'])


df_iphone.drop(columns=['Gender'], inplace=True)


scaler = MinMaxScaler()
numerical_columns = df_iphone.select_dtypes(include=['float64', 'int64']).columns
df_iphone[numerical_columns] = scaler.fit_transform(df_iphone[numerical_columns])


X_iphone = df_iphone.drop(columns=['Purchased'])  
y_iphone = df_iphone['Purchased'] 

# Split the data into training and validation sets with a 70-30 ratio
X_train_iphone, X_valid_iphone, y_train_iphone, y_valid_iphone = train_test_split(X_iphone, y_iphone, test_size=0.3, random_state=42)

#Hyperparameters
epochs_iphone = 20
batch_size_iphone = 32
learning_rate_iphone = 0.001  
no_of_neighbors_iphone = 5

nb_model_iphone = GaussianNB()

print("Epochs for Naive Bayes on Iphone Purchase Dataset")

# Train the Naive Bayes model
for epoch in range(epochs_iphone):
    # Shuffle the training data (optional)
    shuffled_indices = np.random.permutation(len(X_train_iphone))
    X_train_shuffled = X_train_iphone.iloc[shuffled_indices]
    y_train_shuffled = y_train_iphone.iloc[shuffled_indices]
    
    # Mini-batch training (not applicable for Naive Bayes, but keeping the structure consistent)
    for batch_start in range(0, len(X_train_iphone), batch_size_iphone):
        batch_end = batch_start + batch_size_iphone
        X_batch = X_train_shuffled.iloc[batch_start:batch_end]
        y_batch = y_train_shuffled.iloc[batch_start:batch_end]
        
        # Fit the model on the current mini-batch (not applicable for Naive Bayes)
        nb_model_iphone.partial_fit(X_batch, y_batch, classes=np.unique(y_iphone))
        
    # Evaluate the model on the validation set
    y_valid_pred = nb_model_iphone.predict(X_valid_iphone)
    accuracy = accuracy_score(y_valid_iphone, y_valid_pred)
    
   
    print(f"Epoch {epoch+1}/{epochs_iphone} - Validation Accuracy: {accuracy:.4f}")


for _ in range(5):
    print("-" * 20)


knn_model_iphone = KNeighborsClassifier(n_neighbors=no_of_neighbors_iphone)
print ("Epochs for KNN on Iphone Purchase Dataset")

# Train the KNN model
for epoch in range(epochs_iphone):
    # Shuffle the training data (optional)
    shuffled_indices = np.random.permutation(len(X_train_iphone))
    X_train_shuffled = X_train_iphone.iloc[shuffled_indices]
    y_train_shuffled = y_train_iphone.iloc[shuffled_indices]
    
    # Fit the model on the entire training data
    knn_model_iphone.fit(X_train_shuffled, y_train_shuffled)
        
    # Evaluate the model on the validation set
    y_valid_pred = knn_model_iphone.predict(X_valid_iphone)
    accuracy = accuracy_score(y_valid_iphone, y_valid_pred)
    
   
    print(f"Epoch {epoch+1}/{epochs_iphone} - Validation Accuracy: {accuracy:.4f}")


for _ in range(5):
    print("-" * 20)


print("Cross-validation scores for GaussianNB on Iphone Purchase Dataset:")
cv_scores_nb_iphone = cross_val_score(nb_model_iphone, X_iphone, y_iphone, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", cv_scores_nb_iphone)
print("Mean CV score:", cv_scores_nb_iphone.mean())
print("Standard deviation of CV scores:", cv_scores_nb_iphone.std())


for _ in range(5):
    print("-" * 20)


print("Cross-validation scores for KNN on Iphone Purchase Dataset:")
cv_scores_knn_iphone = cross_val_score(knn_model_iphone, X_iphone, y_iphone, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", cv_scores_knn_iphone)
print("Mean CV score:", cv_scores_knn_iphone.mean())
print("Standard deviation of CV scores:", cv_scores_knn_iphone.std())
