import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

# Load dataset
df = pd.read_csv('Student_performance_data _.csv')

# Drop unnecessary columns
df.drop(['StudentID', 'GPA'], axis=1, inplace=True)

# Separate features and target
X = df.loc[:, df.columns != "GradeClass"]
y = df['GradeClass']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=101, stratify=y)

# Define logistic regression model
model = LogisticRegression(max_iter=1000)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2'],
    'multi_class': ['ovr', 'multinomial']
}

grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           verbose=1)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and scaler to .pkl files
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model and scaler have been saved to 'model.pkl' and 'scaler.pkl'.")
