import numpy as np
import pandas as pd
import time
import copy

from ml_from_scratch.ensemble import AdaBoostClassifier
from ml_from_scratch.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifiersk


def split_train_test(X, y, test_size, random_state=42):
    return train_test_split(X, y,
                            test_size = test_size,
                            stratify = y,
                            random_state = random_state)

if __name__ == "__main__":
    # PREPARE THE DATA
    # -----------------
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    y[y==0] = -1

    # Split data into 3 (train, test, valid)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = split_train_test(X_train, y_train, test_size=0.5)

    n_estimators = np.arange(5, 26, 5)

    # ADABOOST MODELING
    # -------------
    print("Adaboost Classifier")
    print("-------------")

    best_val_acc = 0
    best_n_estimators = 0
    training_time = []
    for i in n_estimators:
        # Create AdaBoost Classifier
        start_time = time.time()
        clf_boost = AdaBoostClassifier(n_estimators=i)
        clf_boost.fit(X_train, y_train)
        total_time = time.time() - start_time

        # Predict the tree
        y_pred_train_boost = clf_boost.predict(X_train)
        y_pred_valid_boost = clf_boost.predict(X_valid)

        acc_train_boost = accuracy_score(y_train, y_pred_train_boost)
        acc_valid_boost = accuracy_score(y_valid, y_pred_valid_boost)
        print(f"n_estimators: {i}; val score: {acc_valid_boost*100:.2f}%")
        print(f"training time: {total_time:.2f}s")

        if acc_valid_boost > best_val_acc:
            best_n_estimators = i
            best_val_acc = acc_valid_boost
            best_clf = copy.deepcopy(clf_boost)

    print('Best n_estimators:', best_n_estimators)
    print(f"Best n_estimators acc. validation  : {best_val_acc*100:.2f}%")
    y_pred_test_boost = clf_boost.predict(X_test)
    acc_test_boost = accuracy_score(y_test, y_pred_test_boost)

    print(f"acc. train  : {acc_train_boost*100:.2f}%")
    print(f"acc. test  : {acc_test_boost*100:.2f}%")

    print()

    # ADABOOST (SKLEARN)
    # -------------
    print("Adaboost Classifier (SKLEARN)")
    print("-------------")

    best_val_acc = 0
    best_n_estimators = 0
    for i in n_estimators:
        # Create AdaBoost Classifier
        start_time = time.time()
        clf_boost = AdaBoostClassifiersk(n_estimators=i)
        clf_boost.fit(X_train, y_train)
        total_time = time.time() - start_time

        # Predict the tree
        y_pred_train_boost = clf_boost.predict(X_train)
        y_pred_valid_boost = clf_boost.predict(X_valid)

        acc_train_boost = accuracy_score(y_train, y_pred_train_boost)
        acc_valid_boost = accuracy_score(y_valid, y_pred_valid_boost)
        print(f"n_estimators: {i}; val score: {acc_valid_boost*100:.2f}%")
        print(f"training time: {total_time:.2f}s")

        if acc_valid_boost > best_val_acc:
            best_n_estimators = i
            best_val_acc = acc_valid_boost
            best_clf = copy.deepcopy(clf_boost)
    
    print('Best n_estimators:', best_n_estimators)
    print(f"Best n_estimators acc. validation  : {best_val_acc*100:.2f}%")
    y_pred_test_boost = clf_boost.predict(X_test)
    acc_test_boost = accuracy_score(y_test, y_pred_test_boost)

    print(f"acc. train  : {acc_train_boost*100:.2f}%")
    print(f"acc. test  : {acc_test_boost*100:.2f}%")
