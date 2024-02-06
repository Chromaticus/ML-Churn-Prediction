import pandas as pd
from preprocessing import load_data, preprocess_data
from model import train_model, evaluate_model
from visualizations import plot_churn_distribution, plot_tenure_distribution, plot_monthly_charges_vs_tenure

# Load and preprocess data
df = load_data('C:\\Users\\cdryden\\Downloads\\capstoneData\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train a model using the training data and then evaluate it using the test data
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

# Generate and display visualizations of the data
plot_churn_distribution(df)
plot_tenure_distribution(df)
plot_monthly_charges_vs_tenure(df)

# Function to ask user for data about a new customer
def ask_user_data(X_train):
    single_vars = {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': ['No', 'Yes']
    }
    # Define the possible answers for multi-choice variables
    multi_vars = {
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }

    new_customer = {}# Initialize an empty dictionary to store the new customer's data

    # Ask for single variable inputs:
    for var, answers in single_vars.items():
        question = f"What is the new customer's {var}? ({', '.join(answers)}): "
        answer = input(question)
        while answer not in answers:
            print(f"Invalid input. Please enter one of the following: {', '.join(answers)}")
            answer = input(question)
        new_customer[var + "_" + answer] = [1]

    # Ask for multiple variable inputs:
    for var, answers in multi_vars.items():
        question = f"Does the new customer have {var}? ({', '.join(answers)}): "
        answer = input(question)
        while answer not in answers:
            print(f"Invalid input. Please enter one of the following: {', '.join(answers)}")
            answer = input(question)
        # Add the prefix to the answer
        new_customer[var + "_" + answer] = [1]

    # Ask for numerical inputs
    for var in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        answer = input(f"What is the new customer's {var}? ")
        new_customer[var] = [float(answer)] # Convert the user's answer to a float and add it to the new_customer dictionary
    return new_customer

# Function to interact with the user
def user_interface(dtc, X_train):
    print("Please enter the following customer data:")
    new_customer = ask_user_data(X_train)

    # Convert the dictionary of new customer data to a DataFrame
    new_customer_df = pd.DataFrame(new_customer)

    # Make sure the new customer DataFrame has the same columns as the training data
    for column in X_train.columns:
        if column not in new_customer_df.columns:
            new_customer_df[column] = 0  # If a column in the training data doesn't exist in the new customer data, add it and set to 0

    # Order the columns in the same order as in the training data
    new_customer_df = new_customer_df[X_train.columns]

    # Use the model to make a prediction about the new customer
    prediction = dtc.predict(new_customer_df)

    print(f"The predicted churn status for this new customer is: {prediction[0]}")

# Call the user_interface function in the main.py
user_interface(model, X_train)
