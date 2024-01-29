from matplotlib import pyplot as plt
import requests
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

numerical_cols = [
    'length', 'diameter', 'height', 
    'whole_weight', 'shucked_weight', 'viscera_weight', 
    'shell_weight', 'height_t_shell', 'length_p_shell', 
    'diameter_t_shell'
]

def load_data(file_path='abalone_dataset.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['height_t_shell'] = (df['height'] * df['shell_weight'])
    df['length_p_shell'] = df['length'] + df['shell_weight']
    df['diameter_t_shell'] = df['diameter'] * df['shell_weight']

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

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia: {accuracy}")
    return y_pred

def train_svc_after_forest(X_train, y_train):
    # Train SVM classifier
    svm_classifier = SVC(random_state=42, kernel='rbf', C=100, gamma='auto', probability=True, break_ties=True, class_weight='balanced')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

# Load data
df = load_data()

# Preprocess data
preprocessed_df = preprocess_data(df)

# Separate features and target variable
X = preprocessed_df.drop('type', axis=1)
y = preprocessed_df['type']

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

svm_classifier = train_svc_after_forest(X_train, y_train)

y_pred = evaluate_model(svm_classifier, X_test, y_test)

classification_report = classification_report(y_test, y_pred)
print(classification_report)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"


#TODO Substituir pela sua chave aqui
DEV_KEY = "Codevinci"

data_app = load_data('abalone_app.csv')
data_app = preprocess_data(data_app)
predicted_data = svm_classifier.predict(data_app)

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(predicted_data).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")