# modules/preprocessing.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data(filepath):

    data = pd.read_csv(filepath)

    return data

def clean_data(data):
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Replace underscores with spaces in column names
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # Drop Duplicates
    data = data.drop_duplicates(keep='first')

    # Convert numeric strings to int where applicable
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].astype(int)
            except:
                pass

    # Replace all 2→1, 1→0
    data = data.replace({2: 1, 1: 0})

    # GENDER: M→1, F→0
    data['GENDER'] = data['GENDER'].astype('category')

    # LUNG_CANCER: Yes→1, No→0
    if 'LUNG CANCER' in data.columns:
        data['LUNG CANCER'] = data['LUNG CANCER'].str.strip().str.lower().map({'yes': 1, 'no': 0}).astype(int)

    return data

def one_hot_encoder(df):

    df = pd.get_dummies(df, columns=['GENDER'])

    for col in ['GENDER_F', 'GENDER_M']:
        if col not in df.columns:
            df[col] = 0

    return df

def select_features_for_training(df):

    selected_features = ['GENDER_M', 'GENDER_F', 'AGE', 
                         'SMOKING', 'PEER PRESSURE',
                         'ALCOHOL CONSUMING', 'COUGHING', 'WHEEZING',
                         'YELLOW FINGERS', 'ALLERGY', 'FATIGUE',
                         'SWALLOWING DIFFICULTY', 'CHEST PAIN',
                         'SHORTNESS OF BREATH', 'CHRONIC DISEASE', 'LUNG CANCER']
    
    return df[selected_features]

def select_features_for_prediction(df):

    selected_features = ['GENDER_M','GENDER_F', 'AGE', 
                         'SMOKING', 'PEER PRESSURE',
                         'ALCOHOL CONSUMING', 'COUGHING', 'WHEEZING',
                         'YELLOW FINGERS', 'ALLERGY', 'FATIGUE',
                         'SWALLOWING DIFFICULTY', 'CHEST PAIN',
                         'SHORTNESS OF BREATH', 'CHRONIC DISEASE']
    
    return df[selected_features]

def split_features_target(df, target_col='LUNG CANCER'):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y

def split_data(x, y, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    print('Shape of training data : ', X_train.shape, y_train.shape)
    print('Shape of testing data : ', X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def preprocessing(filepath):
    
    # Load dataset
    patient_data = load_data(filepath=filepath)

    # Clean dataset
    cleaned_data = clean_data(patient_data)

    # One-Hot Encoding
    encoded_data = one_hot_encoder(cleaned_data)

    # Select features based on the results of chi-square test and Pearson correlation analysis
    selected_features_data = select_features_for_training(encoded_data)

    # Split features and target
    X, y = split_features_target(selected_features_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale features
    X_train_scaled, X_test_scaled, scaler  = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler