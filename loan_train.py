import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from datetime import datetime

# Load the "loan_train" dataset
df_loan = pd.read_csv('Notebooks/loan_train.csv')

# Encode the "gender" column if needed
label_encoder_gender = LabelEncoder()
df_loan['Gender'] = label_encoder_gender.fit_transform(df_loan['Gender'])



# Function to encode date column using Unix timestamp
def encode_unix_timestamp(date_string):
    timestamp = datetime.strptime(date_string, "%m/%d/%Y").timestamp()
    return int(timestamp)

date_columns = ["effective_date", "due_date"] 

for col in date_columns:
    # Encode using Unix timestamp
    df_loan[col + '_UnixTimestamp'] = df_loan[col].apply(encode_unix_timestamp)

df_loan.drop(columns=['effective_date'], inplace=True)
df_loan.drop(columns=['due_date'], inplace=True) 

# Encode the target variable 'loan_status' to numerical values
label_encoder_loan_status = LabelEncoder()
df_loan['loan_status_encoded'] = label_encoder_loan_status.fit_transform(df_loan['loan_status'])
df_loan.drop(columns=['loan_status'], inplace=True) 

# Encode the 'education' column
label_encoder_education = LabelEncoder()
df_loan['education_encoded'] = label_encoder_education.fit_transform(df_loan['education'])
df_loan.drop(columns=['education'], inplace=True) 


# Encode the target variable 'loan_status' to numerical values


# Normalize the numerical columns
scaler = MinMaxScaler()
numerical_columns_loan = df_loan.select_dtypes(include=['float64', 'int64']).columns
df_loan[numerical_columns_loan] = scaler.fit_transform(df_loan[numerical_columns_loan])

# Encode the target variable 'loan_status' to numerical values


# Split the data into features (X_loan) and target variable (y_loan)
X_loan = df_loan.drop(columns=['loan_status_encoded'])  # Features
y_loan = df_loan['loan_status_encoded']  # Target variable


# Split the data into training and validation sets with a 70-30 ratio
X_train_loan, X_valid_loan, y_train_loan, y_valid_loan = train_test_split(X_loan, y_loan, test_size=0.3, random_state=42)

# Define hyperparameters
epochs_loan = 20
batch_size_loan = 32
learning_rate_loan = 0.001  # Not applicable for Naive Bayes
no_of_neighbors_loan = 2

# Initialize Naive Bayes model
nb_model_loan = GaussianNB()

print("Epochs for Naive Bayes on Loan Dataset")

# Train the Naive Bayes model
for epoch in range(epochs_loan):
    # Shuffle the training data (optional)
    shuffled_indices = np.random.permutation(len(X_train_loan))
    X_train_shuffled = X_train_loan.iloc[shuffled_indices]
    y_train_shuffled = y_train_loan.iloc[shuffled_indices]
    
    # Mini-batch training (not applicable for Naive Bayes, but keeping the structure consistent)
    for batch_start in range(0, len(X_train_loan), batch_size_loan):
        batch_end = batch_start + batch_size_loan
        X_batch = X_train_shuffled.iloc[batch_start:batch_end]
        y_batch = y_train_shuffled[batch_start:batch_end]
        
        # Fit the model on the current mini-batch
        nb_model_loan.partial_fit(X_batch, y_batch, classes=np.unique(y_loan))
        
    # Evaluate the model on the validation set
    y_valid_pred = nb_model_loan.predict(X_valid_loan[X_train_loan.columns])  # Use only the columns present during training
    accuracy = accuracy_score(y_valid_loan, y_valid_pred)
    
    # Print progress
    print(f"Epoch {epoch+1}/{epochs_loan} - Validation Accuracy: {accuracy:.4f}")

# Print dashes for separation
for _ in range(5):
    print("-" * 20)
    
    
knn_model_loan = KNeighborsClassifier(n_neighbors=no_of_neighbors_loan)

print("Epochs for KNN on Loan Dataset")

# Train the KNN model
for epoch in range(epochs_loan):
    # Shuffle the training data (optional)
    shuffled_indices = np.random.permutation(len(X_train_loan))
    X_train_shuffled = X_train_loan.iloc[shuffled_indices]
    y_train_shuffled = y_train_loan.iloc[shuffled_indices]

    # Fit the model on the entire training data
    knn_model_loan.fit(X_train_shuffled, y_train_shuffled)

    # Evaluate the model on the validation set
    y_valid_pred = knn_model_loan.predict(X_valid_loan[X_train_loan.columns])  # Use only the columns present during training
    accuracy = accuracy_score(y_valid_loan, y_valid_pred)

    # Print progress
    print(f"Epoch {epoch+1}/{epochs_loan} - Validation Accuracy: {accuracy:.4f}")

# Perform cross-validation for Naive Bayes
print("Cross-validation scores for GaussianNB on Loan Dataset:")
cv_scores_nb_loan = cross_val_score(nb_model_loan, X_loan, y_loan, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", cv_scores_nb_loan)
print("Mean CV score:", cv_scores_nb_loan.mean())
print("Standard deviation of CV scores:", cv_scores_nb_loan.std())

for _ in range(5):
    print("-" * 20)

# Perform cross-validation for KNN
print("Cross-validation scores for KNN on Loan Dataset:")
cv_scores_knn_loan = cross_val_score(knn_model_loan, X_loan, y_loan, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", cv_scores_knn_loan)
print("Mean CV score:", cv_scores_knn_loan.mean())
print("Standard deviation of CV scores:", cv_scores_knn_loan.std())
