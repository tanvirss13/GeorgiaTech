import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import time
from absl import flags, app
import os
import joblib

FLAGS = flags.FLAGS
flags.DEFINE_string("ticker", '^GSPC', "Enter a ticker symbol: ")
flags.DEFINE_string("location", os.getcwd(), "Enter your preferred output filepath: ")


class NeuralNetworks:
    def __init__(self, csv_path):
        self.ticker = FLAGS.ticker 
        self.df = pd.read_csv(csv_path)
        self.df = pd.DataFrame(self.df)
        self.df['DailyChange'] = self.df['Close'] - self.df['Open']
        self.df['PriceMovement'] = np.where(self.df['DailyChange'] > 0, 1, 0)
        self.X = self.df[['Open', 'High', 'Low', 'Volume']]
        self.y = self.df['PriceMovement']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.mlp_clf = None
        self.best_model = None

    def train_neural_network(self):
        self.mlp_clf = MLPClassifier(activation='relu', solver='adam', max_iter=500, random_state=42)
        t0 = time.time()
        self.mlp_clf.fit(self.X_train, self.y_train)
        self.training_time = time.time() - t0

    def calculate_training_time(self):
        print(f"Training time: {self.training_time} seconds")

    def predict_neural_network(self):
        y_pred = self.mlp_clf.predict(self.X_test)
        return y_pred

    def evaluate_neural_network(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'F1-Score: {f1}')
        print(f'Recall score: {recall}')
        print(f'Precision score: {precision}')

    def visualize_neural_network(self, y_pred):
        plt.figure(figsize=(16, 8))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.title(self.ticker + " Close Price and Predicted Movement")
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.scatter(self.X_test.index, self.X_test['Open'], c=y_pred, cmap='viridis', label='Predicted Movement')
        plt.legend()
        plt.show()

    def validate_neural_network(self):
        param_range = [(100,), (200,), (300,), (400,), (500,)]
        train_scores, test_scores = validation_curve(
            MLPClassifier(activation='relu', solver='adam', random_state=42),
            self.X_train, self.y_train, param_name="hidden_layer_sizes", param_range=param_range, scoring="accuracy", n_jobs=-1)
        # Plot validation curve
        plt.figure(figsize=(10, 6))
        plt.plot([100, 200, 300, 400, 500], np.mean(train_scores, axis=1), color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.plot([100, 200, 300, 400, 500], np.mean(test_scores, axis=1), color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.title('Model Complexity Graph (Neural Network)')
        plt.xlabel('Hidden Layer Sizes')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    def learn_neural_network(self):
        # Define range of training set sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, test_scores = learning_curve(
            self.mlp_clf, self.X_train, self.y_train, train_sizes=train_sizes, cv=5, scoring="accuracy")
        
        # Calculate mean and standard deviation of training and test scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.title('Learning Curve (Neural Network)')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    def hyperparameter_tuning_neural_network(self):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (250,), (100, 50), (150, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'max_iter': [100, 200, 300, 400, 500],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }

        # Initialize the MLPClassifier
        mlp_clf = MLPClassifier(random_state=42)

        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(mlp_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Get the best model
        best_mlp_clf = grid_search.best_estimator_

        # Print the best parameters
        print("Best Parameters:", grid_search.best_params_)

        # Make predictions with the best model
        y_pred = best_mlp_clf.predict(self.X_test)

        # Evaluate the best model
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy of the best model:", accuracy)


def main(argv):
    ticker = FLAGS.ticker
    location = FLAGS.location
    model = NeuralNetworks(location + "/" + ticker + ".csv")
    model.train_neural_network()
    y_pred = model.predict_neural_network()
    model.evaluate_neural_network(y_pred)
    model.visualize_neural_network(y_pred)
    model.validate_neural_network()
    model.learn_neural_network()
    model.hyperparameter_tuning_neural_network()


if __name__ == '__main__':
    app.run(main)
