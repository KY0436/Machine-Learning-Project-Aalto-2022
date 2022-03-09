import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def open_file(name):
    PizzaData = pd.read_csv(name)
    print("The original size of the dataset:")
    print(PizzaData.shape)
    print("The column of the dataset:")
    print(PizzaData.columns)

    # Visualize dataset
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))  # Create the plot with 3 subplots to show the relationship between the label and the feature
    axes[0].scatter(PizzaData['Price'],
                    PizzaData['Company'])
    axes[0].set_xlabel("Price", size=15)
    axes[0].set_ylabel("Company", size=15)
    axes[0].set_title("Company vs Price", size=16)

    axes[1].scatter(PizzaData['Size'],
                    PizzaData['Company'])
    axes[1].set_xlabel("Size", size=15)
    axes[1].set_ylabel("Company", size=15)
    axes[1].set_title('Company vs Size', size=16)

    axes[2].scatter(PizzaData['Size'],
                    PizzaData['Type'])
    axes[2].set_xlabel("Type", size=15)
    axes[2].set_ylabel("Company", size=15)
    axes[2].set_title('Company vs Type', size=16)
    plt.show()

    X = PizzaData['Price'].to_numpy().reshape(-1, 1)
    y = PizzaData['Company'].to_numpy()

    print(X.shape)
    print(y.shape)

    X_clf = np.copy(X)
    y_clf = y
    np.random.seed(600)
    idx = np.random.choice(np.arange(371), 140)  # choose 140 datapoints from the entire dataset

    X_train, X_val, y_train, y_val = train_test_split(X_clf[idx, :], y_clf[idx], test_size=0.2,
                                                                      random_state=42)
    c = 100000

    clf_1 = SVC(C=c, decision_function_shape="ovo")

    clf_1.fit(X_train, y_train)
    y_pred_train = clf_1.predict(X_train)
    tr_accs = accuracy_score(y_train, y_pred_train)  # calculate the training accuracy

    y_pred_val = clf_1.predict(X_val)
    val_accs = accuracy_score(y_val, y_pred_val)  # calculate the validation accuracy

    print("\nTraining accuracies:\n", tr_accs)
    print("Validation accuracies:\n", val_accs)

def main():
    open_file('processed_data.csv')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

