from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to train a Decision Tree Classifier using the training data
def train_model(X_train, y_train):
    dtc = DecisionTreeClassifier()  # Create an instance of DecisionTreeClassifier
    dtc.fit(X_train, y_train)  # Fit the model on the training data
    return dtc

# Function to evaluate the trained model using testing data
def evaluate_model(dtc, X_test, y_test):
    # Make predictions on the testing data
    y_pred = dtc.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Create a confusion matrix to evaluate the model's performance
    cm = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dtc.classes_)
    disp.plot()
    plt.show()
