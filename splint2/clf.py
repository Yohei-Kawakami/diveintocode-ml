# import library and module
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

# initialize argument value
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='argparse sample')
parser.add_argument('--model', default=LogisticRegression(),
                    help='method')
parser.add_argument('--normalization', default=True, type=bool,
                    help='normalization')


def main(args):
    """
    Parameter
    ---------------
    model : the model of method
    target_value :train data
    feature_value : objective valuable
    normalization :name, True= standardize　False= no standardize

    """
    X = pd.read_csv('classifier_X.csv')
    y = pd.read_csv('classifier_y.csv')

    # split train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # standalized
    if args.normalization == True:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    # pick
    y_train, y_test = y_train.iloc[:, 0], y_test.iloc[:, 0]

    # learnig and predicting
    if args.model == 'LogisticRegression()':
        clf = LogisticRegression()

    elif args.model == 'DecisionTreeClassifier()':
        clf = DecisionTreeClassifier()

    elif args.model == 'SVC()':
        clf = SVC(gamma='auto')
        # trick sklearn:
        clf.probability = True


    clf.fit(X_train, y_train)

    result = clf.predict(X_test)

    return print(result)


if __name__ == '__main__':
    # running first when pyfile start

    # read a argument of command line
    args = parser.parse_args()
    main(args)