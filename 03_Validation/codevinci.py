from matplotlib import pyplot as plt
import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import seaborn as sns

numerical_cols = [
    'length', 'diameter', 'height', 
    'whole_weight', 'shucked_weight', 'viscera_weight', 
    'shell_weight', 'height_times_shell_weight', 'length_plus_shell_weight', 
    'diameter_times_shell_weight'
]

def load_data(file_path='abalone_dataset.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['height_times_shell_weight'] = (df['height'] * df['shell_weight'])
    df['length_plus_shell_weight'] = df['length'] + df['shell_weight']
    df['diameter_times_shell_weight'] = df['diameter'] + df['shell_weight']

    # Use OneHotEncoder for categorical columns
    df = pd.get_dummies(df, columns=['sex'], prefix=['sex'])
    
    # Normalize numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Generate new rows mean with noise
    new_rows = df.sample(frac=0.1, random_state=42, weights=(df['sex_I']))

    # Get average values of numerical columns
    average_values = df[numerical_cols].mean()
    unitRandomValues = np.random.uniform(low=-0.1, high=0.1, size=new_rows[numerical_cols].shape)

    new_rows[numerical_cols] +=  unitRandomValues
    df = pd.concat([df, new_rows])

    return df

def split_data(X, y):
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    return X_train, X_test, y_train, y_test

def train_isolation_forest(X_train):
    isolation_forest = IsolationForest(contamination=0.1)
    outliers = isolation_forest.fit_predict(X_train)
    return isolation_forest, outliers

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia: {accuracy}")
    return y_pred

def train_knn_after_forest(X_train, y_train, outliers):
    # Remove outliers from training data
    X_train_cleaned = X_train
    y_train_cleaned = y_train

    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=17)
    knn_classifier.fit(X_train_cleaned, y_train_cleaned)
    return knn_classifier

def train_svc_after_forest(X_train, y_train, outliers):
    # Remove outliers from training data
    X_train_cleaned = X_train
    y_train_cleaned = y_train

    # Train SVM classifier
    svm_classifier = SVC(random_state=42, kernel='rbf', C=100, gamma='auto', probability=True, break_ties=True, class_weight='balanced')
    svm_classifier.fit(X_train_cleaned, y_train_cleaned)
    return svm_classifier

def train_rf_classifier(X_train, y_train, outliers):

    # Remove outliers from training data
    X_train_cleaned = X_train
    y_train_cleaned = y_train

    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=120, random_state=2048)
    rf_classifier.fit(X_train_cleaned, y_train_cleaned)
    return rf_classifier

# Load data
df = load_data()

# Preprocess data
preProcess_df = preprocess_data(df)

# Separate features and target variable
X = preProcess_df.drop('type', axis=1)
y = preProcess_df['type']

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

isolation_forest, outliers = train_isolation_forest(X_train)
svm_classifier = train_svc_after_forest(X_train, y_train, outliers)
"""
knn_classifier = train_knn_after_forest(X_train, y_train, outliers)
rf_classifier = train_rf_classifier(X_train, y_train, outliers)
"""

# Train SVM classifier
svm_scores = cross_val_score(svm_classifier, X_train, y_train, cv=10)
svm_accuracy = accuracy_score(y_test, svm_classifier.predict(X_test))
svm_best_accuracy = max(svm_scores)

""" # Train KNN classifier
knn_scores = cross_val_score(knn_classifier, X_train, y_train, cv=10)
knn_accuracy = accuracy_score(y_train, knn_classifier.predict(X_train))
knn_best_accuracy = max(knn_scores)

# Train Random Forest classifier
rf_scores = cross_val_score(rf_classifier, X_train, y_train, cv=10)
rf_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train))
rf_best_accuracy = max(rf_scores) """

# Find the best model
models = [
    ('SVM', svm_best_accuracy, svm_classifier),
"""     ('KNN', knn_best_accuracy, knn_classifier), """
"""     ('RandomForest', rf_best_accuracy, rf_classifier) """
]

best_model = max(models, key=lambda x: x[0])
best_classifier_name, best_accuracy, best_classifier = best_model


y_pred = evaluate_model(best_classifier, X_test, y_test)
cm = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print(classification_report)
""" # Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_classifier.classes_, yticklabels=best_classifier.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show() """
""" print(f"Best model: {best_classifier} with accuracy: {best_accuracy}") """
  
  # Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"


#TODO Substituir pela sua chave aqui
DEV_KEY = "Codevinci"

data_app = load_data('abalone_app.csv')
data_app = preprocess_data(data_app)
predicted_data = best_classifier.predict(data_app)

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(predicted_data).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n") 