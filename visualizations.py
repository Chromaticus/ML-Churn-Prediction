import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot the distribution of customer churn
def plot_churn_distribution(df):
    sns.countplot(x='Churn', data=df)  # Create a count plot for the 'Churn' column
    plt.title('Distribution of Customer Churn')
    plt.show()

# Function to plot the distribution of customer tenure
def plot_tenure_distribution(df):
    sns.histplot(df['tenure'], kde=False, bins=30)  # Create a histogram for the 'tenure' column
    plt.title('Distribution of Tenure')
    plt.show()

# Function to plot monthly charges against tenure, categorized by customer churn
def plot_monthly_charges_vs_tenure(df):
    sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df)  # Create a scatter plot with 'tenure' on the x-axis, 'MonthlyCharges' on the y-axis, and different colors for 'Churn' values
    plt.title('Monthly Charges vs Tenure')
    plt.show()

