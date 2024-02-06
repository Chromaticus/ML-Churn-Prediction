import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load a CSV file into a pandas DataFrame
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Function to preprocess the data, including cleaning and encoding
def preprocess_data(df):
    # Convert 'TotalCharges' column to numeric, setting any errors as NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill any NaN values in 'TotalCharges' column with the column's mean value
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # Drop 'Churn' and 'customerID' columns from the DataFrame to create feature set X
    X = df.drop(['Churn', 'customerID'], axis=1)
    # Assign 'Churn' column to target variable y
    y = df['Churn']

    # Convert categorical variables in X to numerical variables using one-hot encoding
    X = pd.get_dummies(X)

    # Split X and y into training and testing sets, with 30% of the data reserved for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


