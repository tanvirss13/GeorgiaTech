import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
from absl import flags, app
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("ticker", '^GSPC', "Enter a ticker symbol: ")
flags.DEFINE_string("location", os.getcwd(), "Enter your preferred output filepath: ")

class StockPredictionModel:
    def __init__(self, csv_path):
        self.ticker = FLAGS.ticker
        self.df = pd.read_csv(csv_path)
        self.df = pd.DataFrame(self.df)
        self.df['DailyChange'] = self.df['Close'] - self.df['Open']
        self.df['PriceMovement'] = np.where(self.df['DailyChange'] > 0, 1, 0)
        self.X = self.df[['Open', 'High', 'Low', 'Volume']]
        self.y = self.df['PriceMovement']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_decision_tree(self):
        self.tree_clf = DecisionTreeClassifier(max_depth=9, min_samples_split=7, min_samples_leaf=5)
        t0 = time.time()
        self.tree_clf.fit(self.X_train, self.y_train)
        self.training_time = time.time() - t0

    def calculate_training_time(self):
        print(f"Training time: {self.training_time} seconds")

    def evaluate_model(self):
        self.y_pred = self.tree_clf.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)

    def plot_data(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.title(self.ticker + " Close Price and Predicted Movement")
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.scatter(self.X_test.index, self.X_test['Open'], c=self.y_pred, cmap='viridis', label='Predicted Movement')
        plt.legend()
        plt.show()

    def plot_model_complexity(self):
        param_range = [1, 5, 10, 15, 20]
        train_scores, test_scores = validation_curve(
            DecisionTreeClassifier(max_depth=9, min_samples_split=7, min_samples_leaf=5), self.X_train, self.y_train,
            param_name="max_depth", param_range=param_range, cv=35, scoring="f1_macro")
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.title('Model Complexity Graph (Decision Tree)')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.xticks(param_range)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.tree_clf, X=self.X, y=self.y, train_sizes=np.linspace(0.1, 1.0, 10), cv=35)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.title('Learning Curve (Decision Tree)')
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    def hyperparameter_tuning(self):
        param_grid = {
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 11),
            'min_samples_leaf': range(1, 11)
        }
        tree_clf = DecisionTreeClassifier()
        grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        print("Best hyperparameters:", best_params)
        best_model = grid_search.best_estimator_
        accuracy = best_model.score(self.X_test, self.y_test)
        print("Accuracy of the best model:", accuracy)


def main(argv):
    ticker = FLAGS.ticker
    location = FLAGS.location
    model = StockPredictionModel(location + "/" + ticker + ".csv")
    model.train_decision_tree()
    model.calculate_training_time()
    model.evaluate_model()
    print(f'Accuracy: {model.accuracy}')
    print(f'F1 Score: {model.f1}')
    print(f'Recall: {model.recall}')
    print(f'Precision: {model.precision}')
    model.plot_data()
    model.plot_model_complexity()
    model.plot_learning_curve()
    model.hyperparameter_tuning()


if __name__ == '__main__':
    app.run(main)
