# split data
def train_test_split(X, y, train_size=0.8):
    """
    split the train data and test data

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
      learning data
    y : ndarray, shape (n_samples, )
      correct value
    train_size : float (0<train_size<1)
      % of train data in whole data

    Returns
    ----------
    X_train : ndarray, shape (n_samples, n_features)
      learning data
    X_test : ndarray, shape (n_samples, n_features)
      test data
    y_train : ndarray, shape (n_samples, )
      correct vale of train data
    y_test : ndarray, shape (n_samples, )
      correct vale of test data
    """
    train_len = int(len(X)*train_size)
    X_train, X_test = X [:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    return X_train, X_test, y_train, y_test