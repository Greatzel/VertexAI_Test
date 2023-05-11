import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_uri', type=str)
    parser.add_argument('--validation_data_uri', type=str)
    args = parser.parse_args()

    # Load the training data from a CSV file
    training_data = pd.read_csv(args.training_data_uri)

    # Split the training data into features and target
    X_train = training_data.drop('target', axis=1)
    y_train = training_data['target']

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model to a file
    model_filename = 'model.joblib'
    joblib.dump(model, model_filename)

    print(f'Model saved to {model_filename}')
