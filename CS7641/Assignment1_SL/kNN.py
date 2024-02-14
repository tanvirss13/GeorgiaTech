import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from absl import flags, app
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("ticker", '^GSPC', "Enter a ticker symbol: ")
flags.DEFINE_string("location", os.getcwd(), "Enter your preferred output filepath: ")

class kNearestNeighbors:
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
        self.knn_clf = None
        self.best_model = None

    def train_knn(self, n_neighbors=5):
        self.knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        t0 = time.time()
        self.knn_clf.fit(self.X_train, self.y_train)
        self.training_time = time.time() - t0

    def calculate_training_time(self):
        print(f"Training time: {self.training_time} seconds")

    def predict_knn(self):
        y_pred = self.knn_clf.predict(self.X_test)
        return y_pred

    def evaluate_knn(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

    def visualize_knn(self, y_pred):
        plt.figure(figsize=(16, 8))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.title(self.ticker + " Close Price and Predicted Movement (kNN)")
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.scatter(self.X_test.index, self.X_test['Open'], c=y_pred, cmap='viridis', label='Predicted Movement')
        plt.legend()
        plt.show()

    def visualize_model_complexity(self):
        param_range = range(1, 21)
        train_scores, test_scores = validation_curve(
            KNeighborsClassifier(), self.X_train, self.y_train, param_name='n_neighbors', param_range=param_range, cv=5)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, label='Training accuracy', color='blue', marker='o')
        plt.plot(param_range, test_mean, label='Validation accuracy', color='green', marker='s')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Accuracy')
        plt.title('Model Complexity Graph (kNN)')
        plt.legend()
        plt.grid()
        plt.show()

    def visualize_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            KNeighborsClassifier(n_neighbors=5), self.X_train, self.y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.title('Learning Curve (kNN)')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    def hyperparameter_tuning_knn(self):
        param_grid = {
            'n_neighbors': range(1, 21),
            'weights': ['uniform', 'distance']
        }

        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Get the best model
        best_knn_clf = grid_search.best_estimator_

        # Print the best parameters
        print("Best Parameters:", grid_search.best_params_)

        # Make predictions with the best model
        y_pred = best_knn_clf.predict(self.X_test)

        # Evaluate the best model
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy of the best model:", accuracy)


def main(argv):
    ticker = FLAGS.ticker
    location = FLAGS.location
    model = kNearestNeighbors(location + "/" + ticker + ".csv")
    model.train_knn()
    y_pred = model.predict_knn()
    model.evaluate_knn(y_pred)
    model.visualize_knn(y_pred)
    model.visualize_model_complexity()
    model.visualize_learning_curve()
    model.hyperparameter_tuning_knn()


if __name__ == '__main__':
    app.run(main)
