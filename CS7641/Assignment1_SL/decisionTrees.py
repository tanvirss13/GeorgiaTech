import pandas as pd
import numpy as np
from absl import app
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time

class DecisionTree:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = pd.DataFrame(self.df)
        self.df['DailyChange'] = self.df['Close'] - self.df['Open']
        self.df['PriceMovement'] = np.where(self.df['DailyChange'] > 0, 1, 0)
        self.X = self.df[['Open', 'High', 'Low', 'Volume']]
        self.y = self.df['PriceMovement']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.tree_clf = None
        self.best_model = None

    def train_decision_tree(self):
        self.tree_clf = DecisionTreeClassifier(max_depth=9, min_samples_split=7, min_samples_leaf=5)
        self.tree_clf.fit(self.X_train, self.y_train)

    def predict_decision_tree(self):
        y_pred = self.tree_clf.predict(self.X_test)
        return y_pred

    def evaluate_decision_tree(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'F1-Score: {f1}')
        print(f'Recall score: {recall}')
        print(f'Precision score: {precision}')

    def visualize_decision_tree(self, y_pred):
        plt.figure(figsize=(16, 8))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.title('S&P 500 Close Price and Predicted Movement')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.scatter(self.X_test.index, self.X_test['Open'], c=y_pred, cmap='viridis', label='Predicted Movement')
        plt.legend()
        plt.show()

    def validate_decision_tree(self):
        param_range = [1, 5, 10, 15, 20]
        train_scores, test_scores = validation_curve(
            DecisionTreeClassifier(max_depth=9, min_samples_split=7, min_samples_leaf=5),
            self.X_train, self.y_train, param_name="max_depth", param_range=param_range, cv=35, scoring="f1_macro")
        # Plot validation curve
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, np.mean(train_scores, axis=1), color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.plot(param_range, np.mean(test_scores, axis=1), color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.title('Model Complexity Graph (Decision Tree)')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.xticks(param_range)
        plt.legend()
        plt.grid()
        plt.show()

    def learn_decision_tree(self):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.tree_clf, X=self.X, y=self.y, train_sizes=np.linspace(0.1, 1.0, 10), cv=35)
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), color='blue', marker='o', markersize=5, label='Training accuracy')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
        plt.title('Learning Curve (Decision Tree)')
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    def hyperparameter_tuning_decision_tree(self):
        param_grid = {
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 11),
            'min_samples_leaf': range(1, 11)
        }
        tree_clf = DecisionTreeClassifier()
        grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print("Best hyperparameters:", best_params)
        accuracy = self.best_model.score(self.X_test, self.y_test)
        print("Accuracy of the best model:", accuracy)


if __name__ == '__main__':
    app = DecisionTree('/Users/tanvirsethi/Desktop/Almost all Docs/Georgia Institute of Technology-MSCS/GaTech_Codework/GeorgiaTech/CS7641/Assignment1_SL/AAPL.csv')
    app.train_decision_tree()
    y_pred = app.predict_decision_tree()
    app.evaluate_decision_tree(y_pred)
    app.visualize_decision_tree(y_pred)
    app.validate_decision_tree()
    app.learn_decision_tree()
    app.hyperparameter_tuning_decision_tree()
