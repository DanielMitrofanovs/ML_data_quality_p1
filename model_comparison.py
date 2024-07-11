import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

# Load the loan data
loan_data = pd.read_csv('loan.csv')

# Ensure correct data types
loan_data['applicationDate'] = pd.to_datetime(loan_data['applicationDate'], errors='coerce')
loan_data['originatedDate'] = pd.to_datetime(loan_data['originatedDate'], errors='coerce')
loan_data['originated'] = loan_data['originated'].astype('bool', errors='ignore')
loan_data['approved'] = loan_data['approved'].astype('bool', errors='ignore')
loan_data['isFunded'] = loan_data['isFunded'].astype('bool', errors='ignore')
loan_data['loanAmount'] = loan_data['loanAmount'].astype('float', errors='ignore')
loan_data['originallyScheduledPaymentAmount'] = loan_data['originallyScheduledPaymentAmount'].astype('float', errors='ignore')
loan_data['leadCost'] = loan_data['leadCost'].astype('float', errors='ignore')


# Function to generate random dates
def random_date(start, end):
    return start + timedelta(days=random.randint(0, int((end - start).days)))


# Create a list to hold the inconsistent rows
inconsistent_data = []

# Define the date range for generating dates
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)

# Generate 1000 inconsistent rows
for _ in range(100000):
    row = {
        'loanId': f'LN{random.randint(100000, 999999)}',
        'anon_ssn': f'{random.randint(100000000, 999999999)}',
        'payFrequency': random.choice(['X', 'Y', 'Z', None]),  # Invalid values
        'apr': random.uniform(-100, -1),  # Negative APR
        'applicationDate': random_date(start_date, end_date),
        'originated': random.choice([True, False]),
        'originatedDate': random_date(start_date, end_date),
        'nPaidOff': random.randint(0, 10),
        'approved': random.choice([True, False]),
        'isFunded': random.choice([True, False]),
        'loanStatus': random.choice(['Paid Off Loan', 'Rejected', 'New Loan', 'Internal Collection']),
        'loanAmount': random.uniform(-5000, -1),  # Negative loan amount
        'originallyScheduledPaymentAmount': random.uniform(-5000, -1),  # Negative payment amount
        'state': random.choice(['CA', 'TX', 'NY']),
        'leadType': random.choice(['bvMandatory', 'lead', 'california', 'organic', 'rc_returning']),
        'leadCost': random.uniform(-5000, -1),  # Negative lead cost
        'fpStatus': random.choice(['Checked', 'Rejected', 'Cancelled', 'No Payments/No Schedule']),
        'clarityFraudId': f'{random.randint(100000000, 999999999)}',
        'hasCF': random.choice([True, False])
    }
    # Ensure some logical inconsistencies
    if row['loanStatus'] == 'Paid Off Loan' and row['nPaidOff'] == 0:
        row['nPaidOff'] = 0  # Inconsistent: Paid off loan but no paid off loans
    if row['approved'] and not row['isFunded']:
        row['isFunded'] = False  # Inconsistent: Approved but not funded
    if row['applicationDate'] > row['originatedDate']:
        row['originatedDate'] = (row['applicationDate'] - timedelta(days=1))  # Inconsistent: applicationDate after originatedDate
    row['target'] = -1
    inconsistent_data.append(row)

# Convert to DataFrame
inconsistent_df = pd.DataFrame(inconsistent_data)

# Define the target variable (1 for normal data, -1 for inconsistent data)
# Sample 20% of normal data
normal_sample = loan_data.sample(frac=0.5, random_state=42)
normal_sample['target'] = 1

# Combine the sampled normal data and all inconsistent data
sampled_data = pd.concat([normal_sample, inconsistent_df])

# Encode categorical columns
categorical_columns = ['payFrequency', 'loanStatus', 'state', 'leadType', 'fpStatus', 'clarityFraudId']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    sampled_data[col] = sampled_data[col].astype(str)
    sampled_data[col] = le.fit_transform(sampled_data[col])
    label_encoders[col] = le

# Drop non-numeric columns
sampled_data.drop(columns=['loanId', 'anon_ssn', 'applicationDate', 'originatedDate'], inplace=True)

# Fill missing values with a placeholder
sampled_data.fillna(-999, inplace=True)

# Split features and target
X = sampled_data.drop(columns=['target'])
y = sampled_data['target']

# First split: separate clean data
X_clean = X[y == 1]
y_clean = y[y == 1]

# Second split: separate corrupted data
X_corrupted = X[y == -1]
y_corrupted = y[y == -1]

# Split clean data into training and testing sets
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Determine the number of corrupted samples to add to the test set (10% of the test set size)
num_test_samples = len(X_test_clean)
num_corrupted_samples = int(0.1 * num_test_samples)

# Sample corrupted data to add to the test set
X_test_corrupted = X_corrupted.sample(n=num_corrupted_samples, random_state=42)
y_test_corrupted = y_corrupted.sample(n=num_corrupted_samples, random_state=42)

# Combine clean and corrupted test sets
X_test = pd.concat([X_test_clean, X_test_corrupted])
y_test = pd.concat([y_test_clean, y_test_corrupted])

# Verify the distribution of labels in the training and test sets
print("Distribution of target labels in the training set:")
print(pd.Series(y_train_clean).value_counts())
print("Distribution of target labels in the test set:")
print(pd.Series(y_test).value_counts())

# Train Isolation Forest
print("Forest")
start_time = time.time()
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(X_train_clean)
train_time_iso_forest = time.time() - start_time

start_time = time.time()
y_pred_iso_forest = iso_forest.predict(X_test)
test_time_iso_forest = time.time() - start_time

# Save the results of spotted corrupted data by Isolation Forest
iso_forest_spotted = X_test[y_pred_iso_forest == -1]
iso_forest_spotted.to_csv('iso_forest_spotted.csv', index=False)

# Train One-Class SVM
print("SVM")
start_time = time.time()
oc_svm = OneClassSVM(nu=0.1)
oc_svm.fit(X_train_clean)
train_time_oc_svm = time.time() - start_time

start_time = time.time()
y_pred_oc_svm = oc_svm.predict(X_test)
test_time_oc_svm = time.time() - start_time

# Save the results of spotted corrupted data by One-Class SVM
oc_svm_spotted = X_test[y_pred_oc_svm == -1]
oc_svm_spotted.to_csv('oc_svm_spotted.csv', index=False)

# Train Autoencoder
input_dim = X_train_clean.shape[1]
encoding_dim = 14

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh")(input_layer)
encoder = tf.keras.layers.Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = tf.keras.layers.Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = tf.keras.layers.Dense(input_dim, activation='relu')(decoder)
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

start_time = time.time()
autoencoder.fit(X_train_clean, X_train_clean, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)
train_time_autoencoder = time.time() - start_time

start_time = time.time()
reconstructions = autoencoder.predict(X_test)
mse = ((X_test - reconstructions) ** 2).mean(axis=1)
threshold = mse.mean() + mse.std()
y_pred_autoencoder = (mse > threshold).astype(int)
y_pred_autoencoder = [1 if y == 0 else -1 for y in y_pred_autoencoder]
test_time_autoencoder = time.time() - start_time

# Save the results of spotted corrupted data by Autoencoder
autoencoder_spotted = X_test[[y == -1 for y in y_pred_autoencoder]]
autoencoder_spotted.to_csv('autoencoder_spotted.csv', index=False)

# Evaluate the models
print("Isolation Forest Classification Report:")
print(classification_report(y_test, y_pred_iso_forest))

print("One-Class SVM Classification Report:")
print(classification_report(y_test, y_pred_oc_svm))

print("Autoencoder Classification Report:")
print(classification_report(y_test, y_pred_autoencoder))

# Calculate confusion matrix for each model
conf_matrix_iso_forest = confusion_matrix(y_test, y_pred_iso_forest, labels=[1, -1])
conf_matrix_oc_svm = confusion_matrix(y_test, y_pred_oc_svm, labels=[1, -1])
conf_matrix_autoencoder = confusion_matrix(y_test, y_pred_autoencoder, labels=[1, -1])


def get_metrics_from_conf_matrix(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    return tn, fp, fn, tp


tn_iso_forest, fp_iso_forest, fn_iso_forest, tp_iso_forest = get_metrics_from_conf_matrix(conf_matrix_iso_forest)
tn_oc_svm, fp_oc_svm, fn_oc_svm, tp_oc_svm = get_metrics_from_conf_matrix(conf_matrix_oc_svm)
tn_autoencoder, fp_autoencoder, fn_autoencoder, tp_autoencoder = get_metrics_from_conf_matrix(conf_matrix_autoencoder)

# Print confusion matrix results
print(f"Isolation Forest - True Negative: {tn_iso_forest}, False Positive: {fp_iso_forest}, False Negative: {fn_iso_forest}, True Positive: {tp_iso_forest}")
print(f"One-Class SVM - True Negative: {tn_oc_svm}, False Positive: {fp_oc_svm}, False Negative: {fn_oc_svm}, True Positive: {tp_oc_svm}")
print(f"Autoencoder - True Negative: {tn_autoencoder}, False Positive: {fp_autoencoder}, False Negative: {fn_autoencoder}, True Positive: {tp_autoencoder}")

# Print timing results
print(f"Isolation Forest training time: {train_time_iso_forest:.2f} seconds")
print(f"Isolation Forest testing time: {test_time_iso_forest:.2f} seconds")

print(f"One-Class SVM training time: {train_time_oc_svm:.2f} seconds")
print(f"One-Class SVM testing time: {test_time_oc_svm:.2f} seconds")

print(f"Autoencoder training time: {train_time_autoencoder:.2f} seconds")
print(f"Autoencoder testing time: {test_time_autoencoder:.2f} seconds")
