import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy import stats

sns.set_theme(context='notebook', palette='pastel', style='whitegrid')

df = pd.read_csv('Student_performance_data _.csv')
df.head()

df.info()

df.describe()

# Hitung baris duplikat di DataFrame
sum(df.duplicated())

# Hapus kolom yang tidak diperlukan dari DataFrame
df.drop(['StudentID', 'GPA'], axis=1, inplace=True)

df.shape

# dentifikasi kolom numerik: kolom dengan lebih dari 5 nilai unik dianggap numerik
numerical_columns = [col for col in df.columns if df[col].nunique() > 5]

# Identifikasi kolom kategoris: kolom yang bukan numerik dan bukan 'GradeClass'
categorical_columns = df.columns.difference(numerical_columns).difference(['GradeClass']).to_list()

# Label khusus untuk Kolom Kategori
custom_labels = {
    'Ethnicity': ['Caucasian', 'African American', 'Asian', 'Other'],
    'Age': [15, 16, 17, 18],
    'ParentalEducation': ['None', 'High School', 'Some College', 'Bachelor\'s', 'Higher'],
    'Tutoring': ['No', 'Yes'],
    'ParentalSupport': ['No', 'Low', 'Moderate', 'High', 'Very High'],
    'Extracurricular': ['No', 'Yes'],
    'Sports': ['No', 'Yes'],
    'Music': ['No', 'Yes'],
    'Volunteering': ['No', 'Yes'],
    'Gender': ['Male', 'Female']
}

# Countplots untuk setiap kolom kategoris
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column)
    plt.title(f'Countplot of {column}')

    # Mengatur label khusus
    labels = custom_labels[column]
    ticks = range(len(labels))
    plt.xticks(ticks=ticks, labels=labels)

    plt.tight_layout()
    plt.show()

# Plot histogram untuk setiap numerical column
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.show()


# Hitung koefisien korelasi Pearson
correlations = df.corr(numeric_only=True)['GradeClass'][:-1].sort_values()

# Mengatur ukuran gambar
plt.figure(figsize=(20, 7))

# Buat diagram batang koefisien korelasi Pearson
ax = correlations.plot(kind='bar', width=0.7)

# Atur batas dan label y-axis
ax.set(ylim=[-1, 1], ylabel='Pearson Correlation', xlabel='Features',
       title='Pearson Correlation with Grade Class')

# Putar label sumbu x untuk keterbacaan yang lebih baik
ax.set_xticklabels(correlations.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Membuat mask untuk upper triangle
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

# Choose a diverging color scheme
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Plot heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(),cmap=cmap, cbar_kws={"shrink": .5}, vmin=-0.7, mask=mask)
plt.title('Correlation Matrix Heatmap', fontsize=16)

plt.show()


# Hitung korelasi dengan GradeClass dan temukan korelasi terkuat
grade_class_corr = df.corr(numeric_only=True)['GradeClass'].drop('GradeClass')
strongest_correlation = grade_class_corr.idxmax()
strongest_correlation_value = grade_class_corr.max()

print(f"Korelasi terkuat dengan GradeClass adalah {strongest_correlation} dengan value {strongest_correlation_value:.2f}")

# Membuat 1x2 subplot dengan ukuran figure 20x5 inci
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

# Loop through the first two numerical columns in the DataFrame
for idx in range(2):
    # Membuat boxplot untuk setiap kolom
    sns.boxplot(ax=ax[idx], x=df[numerical_columns[idx]])
    ax[idx].set_title(numerical_columns[idx])

# Atur label kostumnya
labels = ["A", "B", "C", "D", "E"]
ticks = range(len(labels))

# Buat figur dengan dua subplot yang berdampingan
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot untuk count plot di subplot pertama
sns.countplot(data=df, x='GradeClass', ax=axes[0])
axes[0].set_title('Distribution of Grade Class')
axes[0].set_xticks(ticks)
axes[0].set_xticklabels(labels)

# Kalkulasi counts untuk pie chartnya
grade_counts = df['GradeClass'].value_counts().sort_index()

# Plot  pie chart di subplot kedua
axes[1].pie(grade_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
axes[1].set_title('Grade Class Distribution (Pie Chart)')

# Sesuaikan layout untuk menghindari overlapping
plt.tight_layout()
plt.show()

# Pisahkan features (X) and target variable (y)
X = df.loc[:, df.columns != "GradeClass"]
y = df['GradeClass']

# Standardize features menggunakan StandardScaler
scaler = StandardScaler()

# Fit scaler ke data
scaler.fit(X)

# Transform  data
X_scaled = scaler.transform(X)

# Split data ke dalam training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=101, stratify=y)

# Membuat Model Klasfikasi dengan default parameters
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
}

model_names = []
accuracies = []

# Train dan evaluasi tiap model
for name, clf in models.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    model_names.append(name)
    accuracies.append(score)
    print(f"{name} accuracy: {score:.2f}")

# membuat DataFrame untuk akurasi model
df_models = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})

# Plot model akurasi menggunakan Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=df_models, palette='pastel')
plt.title('Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Definisikan model yang dipilih
model = LogisticRegression(max_iter=1000)

# Definisikan Hyperparameter Tuning (GridSearchCV)
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['lbfgs', 'liblinear'],  # Solver for optimization
    'penalty': ['l2'],  # Regularization penalty type
    'multi_class': ['ovr', 'multinomial']  # Multiclass strategy
}

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Train GridSearchCV
grid_search.fit(X_train, y_train)

# Tampilkan parameter terbaik dan skor terbaik
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

# Gunakan model terbaik yang ditemukan oleh GridSearchCV untuk prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Kalkulasi dan tampilkan akurasi dan F1-score nya
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy = {accuracy:.4f}, F1-score = {f1:.4f}")

# Menampilkan classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Membuat Plot confusion matrix dengan kustomisasi colormap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()