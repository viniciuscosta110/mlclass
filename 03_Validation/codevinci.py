import pandas as pd
import requests

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

numerical_cols = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']

def load_data(file_path='abalone_dataset.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['length_times_diameter'] = df['length'] * df['diameter']

    # Use SimpleImputer with constant value 0 for numerical columns
    df[numerical_cols] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[numerical_cols])

    # Use OneHotEncoder for categorical columns
    df = pd.get_dummies(df, columns=['sex'], prefix=['sex'])
    
    # Drop viscera_weight and height column
    columns_to_drop = ['height', 'viscera_weight']
    df.drop(columns=columns_to_drop, inplace=True) 
    
    return df

def split_data(X, y):
    # Use StratifiedShuffleSplit for splitting data
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    return X_train, X_test, y_train, y_test

def train_knn(X_train, y_train, n_neighbors=10):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia: {accuracy}")

def hyperparameter_tuning(X_train, y_train):
    # Define the parameter grid
    param_grid = {'n_neighbors': [5, 10, 15, 20]}
    
    # Create KNN classifier
    knn_classifier = KNeighborsClassifier()
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    
    # Train the model with the best parameters
    best_knn_classifier = grid_search.best_estimator_
    best_knn_classifier.fit(X_train, y_train)
    
    return best_knn_classifier

# Load data
df = load_data()

# Preprocess data
df = preprocess_data(df)

# Separate features and target variable
X = df.drop('type', axis=1)
y = df['type']

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Train KNN classifier and evaluate the model
knn_classifier = train_knn(X_train, y_train)
evaluate_model(knn_classifier, X_train, y_train)

# Hyperparameter tuning and evaluate the model with the best parameters
best_knn_classifier = hyperparameter_tuning(X_train, y_train)
evaluate_model(best_knn_classifier, X_train, y_train)


# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Codevinci"

data_app = load_data('abalone_app.csv')
data_app = preprocess_data(data_app)
predicted_data = knn_classifier.predict(data_app)

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(predicted_data).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")