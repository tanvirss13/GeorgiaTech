import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from absl import flags, app
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("ticker", '^GSPC', "Enter a ticker symbol: ")
flags.DEFINE_string("location", os.getcwd(), "Enter your preferred output filepath: ")

class SupportVectorMachines:
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
        self.svm_clf = None
        self.best_model = None

    def train_svm(self):
        self.svm_clf = SVC(kernel='rbf')
        t0 = time.time()
        self.svm_clf.fit(self.X_train, self.y_train)
        self.training_time = time.time() - t0

    def calculate_training_time(self):
        print(f"Training time: {self.training_time} seconds")

    def predict_svm(self):
        y_pred = self.svm_clf.predict(self.X_test)
        return y_pred

    def evaluate_svm(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

    def visualize_svm(self, y_pred):
        plt.figure(figsize=(16, 8))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.title(self.ticker + " Close Price and Predicted Movement (SVM)")
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.scatter(self.X_test.index, self.X_test['Open'], c=y_pred, cmap='viridis', label='Predicted Movement')
        plt.legend()
        plt.show()

    def visualize_model_complexity(self):
        param_range = np.logspace(-3, 3, 7)
        train_scores, test_scores = validation_curve(
            SVC(kernel='rbf'), self.X_train, self.y_train, param_name='C', param_range=param_range, cv=5)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, label='Training accuracy', color='blue', marker='o')
        plt.plot(param_range, test_mean, label='Validation accuracy', color='green', marker='s')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
        plt.xscale('log')
        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')
        plt.title('Model Complexity Graph (SVM)')
        plt.legend()
        plt.grid()
        plt.show()

    def visualize_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            SVC(kernel='rbf', C=1), self.X_train, self.y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.title('Learning Curve (SVM)')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    def hyperparameter_tuning_svm(self):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf']
        }

        # Perform Grid Search with cross-validation
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Get the best model
        best_svm_clf = grid_search.best_estimator_

        # Print the best parameters
        print("Best Parameters:", grid_search.best_params_)

        # Make predictions with the best model
        y_pred = best_svm_clf.predict(self.X_test)

        # Evaluate the best model
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy of the best model:", accuracy)

def main(argv):
    ticker = FLAGS.ticker
    location = FLAGS.location
    model = SupportVectorMachines(location + "/" + ticker + ".csv")
    model.train_svm()
    y_pred = model.predict_svm()
    model.evaluate_svm(y_pred)
    model.visualize_svm(y_pred)
    model.calculate_training_time()
    model.hyperparameter_tuning_svm()
    model.visualize_model_complexity()
    model.visualize_learning_curve()

if __name__ == '__main__':
    app.run(main)
