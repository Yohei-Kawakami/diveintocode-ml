import numpy as np
import math
from sklearn.metrics import accuracy_score
import copy
import matplotlib.pyplot as plt
% matplotlib
inline


class ScratchSVMClassifier():
    """
    SVM分類のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    lambda : float
      正則化パラメーター

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,1)
      パラメータ
    """

    def __init__(self, num_iter=10000, lr=0.000000005, threshold=0.1 ** 5):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.lam = 0
        self.lam_X = 0
        self.lam_y = 0
        self.theta = 0
        self.threshold = threshold

    def fit(self, X, y):

        # ラベル０をマイナス１に変換
        y[y == 0] = -1
        # リシェイプ
        y = y.reshape(len(y), 1)
        # ラムダの更新
        self._compute_lambda(X, y)

    def predict(self, X):
        pred = np.dot(self._kernel_linear(X, self.lam_X.T), self.lam * self.lam_y)
        print(pred)
        pred = (pred >= 0).astype(int)
        return pred

    def _compute_lambda(self, X, y):
        # lambdaの初期化
        # lam = np.random.rand(len(X), 1)
        lam = np.ones(len(X)).reshape(len(X), 1)
        lam = lam * -10

        for num in range(self.iter):
            lam = lam + self.lr * (1 - np.dot(np.dot(y, y.T) * self._kernel_linear(X, X.T), lam))
            lam = np.where(lam < 0, 0, lam)

            if sum((lam > self.threshold).astype(int)) >= 2:
                print("ラムダ更新回数")
                print(num)
                break

        # 閾値以上のサポートベクトルを抜き出す
        self.lam = lam[np.any(lam > self.threshold, axis=1), :]
        self.lam_X = X[np.any(lam > self.threshold, axis=1), :]
        self.lam_y = y[np.any(lam > self.threshold, axis=1), :]
        # thetaの計算n*1のベクトル
        self.theta = np.dot(self.lam_X.T, self.lam * self.lam_y)
        if len(self.lam) == 0:
            print("Error：can not find any support vectors")

    def _kernel_linear(self, x1, x2):

        return np.dot(x1, x2)

    def accuracy(self, y_test, y_pred):
        # accuracyを計算
        return accuracy_score(y_test, y_pred)