import pandas as pd
import requests
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler



import seaborn as sns
import matplotlib.pyplot as plt

numerical_cols = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']

def load_data(file_path='abalone_dataset.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['test'] = -(df['height'] - df['shell_weight'])
    df['test3'] = df['length'] + df['shell_weight']
    df['test2'] = df['diameter'] + df['shell_weight']
    df["size"] = 0.707 * df["height"] + 0.707 * df["diameter"]

    # Use OneHotEncoder for categorical columns
    df = pd.get_dummies(df, columns=['sex'], prefix=['sex'])

    # Drop viscera_weight and height column
    columns_to_drop = ['height', 'sex_I', 'sex_M']
    df.drop(columns=columns_to_drop, inplace=True)

    return df

def split_data(X, y):
    # Use StratifiedShuffleSplit for splitting data
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train = X
    y_train = y
    
    return X_train, X_test, y_train, y_test

def train_isolation_forest(X_train):
    isolation_forest = IsolationForest(contamination=0.1)
    outliers = isolation_forest.fit_predict(X_train)
    return isolation_forest, outliers

def evaluate_forest_model(model, X, y):
    anomaly_scores = model.decision_function(X)
    y_pred = model.predict(X)

    # For simplicity, let's assume que qualquer instância com pontuação de anomalia menor que 0 é considerada anomalia (outlier)
    binary_labels = [1 if score < 0 else 0 for score in anomaly_scores]

    accuracy = accuracy_score(y, binary_labels)
    print(f"Acurácia: {accuracy}")

    # Inclua a avaliação da previsão do tipo de abalone
    print(f"Predições de tipo de abalone: {y_pred}")

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia: {accuracy}")
    return y_pred

def train_knn_after_forest(X_train, y_train, outliers):
    # Remove outliers from training data
    X_train_cleaned = X_train[outliers == 1]
    y_train_cleaned = y_train[outliers == 1]

    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=15)
    knn_classifier.fit(X_train_cleaned, y_train_cleaned)
    return knn_classifier

def train_svm_after_forest(X_train, y_train, outliers):
    # Remove outliers from training data
    X_train_cleaned = X_train[outliers == 1]
    y_train_cleaned = y_train[outliers == 1]

    # Train SVM classifier
    svm_classifier = SVC(random_state=90, kernel='rbf', C=100, gamma='auto')
    svm_classifier.fit(X_train_cleaned, y_train_cleaned)
    return svm_classifier

# Load data
df = load_data()
print(df.head())

# Preprocess data
df = preprocess_data(df)

# Separate features and target variable
X = df.drop('type', axis=1)
y = df['type']

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

isolation_forest, outliers = train_isolation_forest(X_train)

svm_classifier = train_svm_after_forest(X_train, y_train, outliers)
evaluate_model(svm_classifier, X_test, y_test)

knn_classifier = train_knn_after_forest(X_train, y_train, outliers)
knn_predictions= evaluate_model(knn_classifier, X_test, y_test)

rf_classifier = RandomForestClassifier(n_estimators=2, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_test=evaluate_model(rf_classifier, X_test, y_test)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': rf_y_test})
# Plot a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=results_df.index, y='Actual', data=results_df, label='Actual', marker='o')
sns.scatterplot(x=results_df.index, y='Predicted', data=results_df, label='Predicted', marker='x')
plt.title('Actual vs. Predicted - KNN Classifier')
plt.xlabel('Sample Index')
plt.ylabel('Abalone Type')
plt.legend()
plt.savefig('knn_classifier.png')

abalone_data = load_data('abalone_app.csv')
abalone_data = preprocess_data(abalone_data)


""" from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance


# Assuming X_train, y_train are your training data
perm_importance = permutation_importance(knn_classifier, X_test, y_test, n_repeats=30, random_state=42)
feature_importances = perm_importance.importances_mean

# Print the ranking of features
for feature, importance in zip(X_test.columns, feature_importances):
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# Plot the results

# Plot data

plt.scatter(X['length'], X['diameter'], s=40, c=y)
plt.savefig('load_data.png') """

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing the features
correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_matrix.png')


 # Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Codevinci"

data_app = load_data('abalone_app.csv')
data_app = preprocess_data(data_app)
predicted_data = rf_classifier.predict(data_app)

#plot predictions scatter
plt.figure(figsize=(10, 6))
plt.scatter(data_app['length'], data_app['diameter'], s=40, c=predicted_data)
plt.savefig('predicted_data_svm.png')

"""  # json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(predicted_data).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")   """