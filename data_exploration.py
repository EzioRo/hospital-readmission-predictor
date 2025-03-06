########################################
# Step 1: Load the Dataset and Explore It
########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset using the specified file path
data = pd.read_csv(r'C:\Users\ROHIT D VIBHUTI\OneDrive\Desktop\hackathon project\Code\train.csv')

# Display the dataset's shape (rows, columns)
print(f"Dataset shape: {data.shape}")

# Preview the first 30 rows of the dataset
print("\nFirst 30 rows:")
print(data.head(30))

# Check for missing values in each column
print("\nMissing values per column:")
print(data.isnull().sum())

# Display basic statistics for numeric columns
print("\nBasic statistics:")
print(data.describe())

########################################
# Step 2: Understand the Target Variable
########################################

# Display the distribution of the target variable 'readmitted'
print("\nReadmission distribution:")
print(data['readmitted'].value_counts())

# Visualize the distribution using a count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='readmitted', data=data)
plt.title('Distribution of Hospital Readmissions')
plt.xlabel('Readmission Status')
plt.ylabel('Count')
plt.savefig('readmission_distribution.png')
plt.show()

########################################
# Step 3: Initial Data Cleaning
########################################

# Specify columns to drop based on analysis of the dataset's column names
columns_to_drop = [
    'payer_code_?', 
    'medical_specialty_?', 
    'acetohexamide_No', 
    'troglitazone_No', 
    'examide_No', 
    'citoglipton_No'
]

# Drop the specified columns (ignore errors if they don't exist)
data = data.drop(columns=columns_to_drop, errors='ignore')

# Handle missing values:
# - For numeric columns, fill missing values with the median.
# - For categorical columns, fill missing values with the mode.
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])
    else:
        data[column] = data[column].fillna(data[column].median())

# Rename the target column for clarity (from 'readmitted' to 'readmitted_binary')
data.rename(columns={'readmitted': 'readmitted_binary'}, inplace=True)

# Confirm cleaning results by printing the first few rows and column list
print("\nData after initial cleaning (first 5 rows):")
print(data.head())
print("\nColumns after cleaning:")
print(data.columns)
print("\nShape of data after cleaning:")
print(data.shape)

########################################
# Step 4: Feature Engineering and Preprocessing
########################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Create New Features ----

# 1. Total Visits: Sum of outpatient, emergency, and inpatient visits.
data['total_visits'] = data['number_outpatient'] + data['number_emergency'] + data['number_inpatient']

# 2. Polypharmacy Indicator: 1 if the number of medications is greater than 10, else 0.
data['polypharmacy'] = (data['num_medications'] > 10).astype(int)

# 3. Age Grouping: Create an ordinal 'age_group' based on one-hot encoded age columns.
def get_age_group(row):
    if row['age_[40-50)'] == 1:
        return 1
    elif row['age_[50-60)'] == 1:
        return 2
    elif row['age_[60-70)'] == 1:
        return 3
    elif row['age_[70-80)'] == 1:
        return 4
    elif row['age_[80-90)'] == 1:
        return 5
    else:
        return np.nan

data['age_group'] = data.apply(get_age_group, axis=1)

# Impute missing age_group values with the mode
data['age_group'].fillna(data['age_group'].mode()[0], inplace=True)

# Drop the original one-hot age columns
age_columns = ['age_[70-80)', 'age_[60-70)', 'age_[50-60)', 'age_[80-90)', 'age_[40-50)']
data.drop(columns=age_columns, inplace=True)

# 4. Chronic Condition Indicator: 1 if any of diag_1_428, diag_2_250, or diag_3_401 equals 1, else 0.
data['chronic_condition'] = data[['diag_1_428', 'diag_2_250', 'diag_3_401']].apply(
    lambda row: 1 if (row == 1).any() else 0, axis=1
)

# ---- Feature Selection ----
# Select the features for modeling (including new derived features)
selected_features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'total_visits',         # New feature: total visits
    'number_diagnoses',
    'gender_Female',
    'polypharmacy',         # New binary feature for polypharmacy
    'age_group',            # New ordinal age feature
    'chronic_condition'     # New binary feature for chronic condition indicator
]

# Define X (features) and y (target)
X = data[selected_features]
y = data['readmitted_binary']

print("\nSelected features:")
print(X.columns.tolist())
print("\nTarget variable distribution:")
print(y.value_counts())

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Ensures balanced class proportions in both splits
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# ---- Scaling Numeric Features ----
# Identify continuous numeric columns that need scaling.
numeric_cols = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'total_visits',
    'number_diagnoses'
]

scaler = StandardScaler()

# Fit the scaler on training data and transform both training and test data
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\nData is now preprocessed and ready for modeling!")


########################################
# Step 5: Model Development
########################################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Train a Logistic Regression Model ----
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Make predictions with Logistic Regression
y_pred_log = log_model.predict(X_test)

# Calculate performance metrics for Logistic Regression
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

print("\nLogistic Regression Model Performance:")
print(f"Accuracy: {accuracy_log:.4f}")
print(f"Precision: {precision_log:.4f}")
print(f"Recall: {recall_log:.4f}")
print(f"F1 Score: {f1_log:.4f}")

# Plot confusion matrix for Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_logistic.png')
plt.show()

# ---- Train a Random Forest Classifier ----
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with Random Forest
y_pred_rf = rf_model.predict(X_test)

# Calculate performance metrics for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\nRandom Forest Model Performance:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")

# Plot confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_rf.png')
plt.show()

# ---- Compare Models and Select the Best One ----
if f1_rf > f1_log:
    final_model = rf_model
    model_name = "Random Forest"
    final_f1 = f1_rf
else:
    final_model = log_model
    model_name = "Logistic Regression"
    final_f1 = f1_log

print(f"\nSelected {model_name} as the final model with F1 Score: {final_f1:.4f}")

# ---- Feature Importance / Coefficient Analysis ----
if model_name == "Random Forest":
    # Feature importance for Random Forest
    importances = final_model.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Important Features (Random Forest):")
    print(feature_importances.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
    plt.title('Top 10 Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.show()
else:
    # Coefficient analysis for Logistic Regression
    coefficients = log_model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)
    
    print("\nTop Positive Coefficients (Logistic Regression):")
    print(coef_df.head(5))
    print("\nTop Negative Coefficients (Logistic Regression):")
    print(coef_df.tail(5))
    
    # Visualize coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=pd.concat([coef_df.head(5), coef_df.tail(5)]))
    plt.title('Top Coefficients (Logistic Regression)')
    plt.tight_layout()
    plt.savefig('coefficients_lr.png')
    plt.show()

########################################
# Step 6: Model Evaluation and Optimization
########################################
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

# ---- Cross-Validation ----
cv_scores = cross_val_score(final_model, X, y, cv=5, scoring='f1')
print("\nCross-Validation Results:")
print(f"F1 Scores for each fold: {cv_scores}")
print(f"Mean F1 Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# ---- ROC Curve Analysis (if applicable) ----
if hasattr(final_model, "predict_proba"):
    # Obtain probability predictions on the test set
    y_prob = final_model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.show()
    
    print(f"\nAUC Score: {auc:.4f}")
else:
    print("\nThe final model does not support probability predictions for ROC curve analysis.")

# ---- Save the Final Model and Preprocessing Objects ----
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the selected features list for future reference
with open('selected_features.txt', 'w') as f:
    for feature in selected_features:
        f.write(feature + "\n")

print("\nFinal model, scaler, and selected features have been saved successfully!")
