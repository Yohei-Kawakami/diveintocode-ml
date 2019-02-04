# import library and module
import argparse

# import library module
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# initialize argument value
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='argparse sample')
parser.add_argument('--model', default=LinearRegression(), type=str,
                    help='model')
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
    X = pd.read_csv('linear_X.csv')
    y = pd.read_csv('linear_y.csv')

    # split train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # standalized
    if args.normalization == True:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

    # pick
    y_train, y_test = y_train.iloc[:, 0], y_test.iloc[:, 0]

    # learnig and predicting
    if args.model == 'LinearRegression()':
        clf = LinearRegression()
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)

    return print(result)


if __name__ == '__main__':
    # running first when pyfile start

    # read a argument of command line
    args = parser.parse_args()
    main(args)