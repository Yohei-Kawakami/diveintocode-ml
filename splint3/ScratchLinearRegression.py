import numpy as np


class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter=500, lr=0.01, bias=True, verbose=True):
        # ハイパーパラメータを属性として記録
        self.num_iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.n = 0

        # 損失を記録する配列を用意
        self.loss = np.zeros(self.num_iter)
        self.use_loss = np.zeros(self.num_iter)
        self.val_loss = np.zeros(self.num_iter)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        m = len(X)
        # 基本統計量
        mu = X.mean(axis=0)  # 平均の算出 （axis=0を指定して列毎に標準化）
        sigma = X.std(axis=0)  # 標準偏差の算出 （axis=0を指定して列毎に標準化）

        # 標準化処理
        X = (X - mu) / sigma

        # バイアス項の挿入
        if self.bias:
            X = np.hstack((np.ones(m).reshape(m, 1), X))
        # 特徴量の数取得
        self.n = X.shape[1]

        # X_vaにもバイアスの追加
        if type(X_val) == np.ndarray:
            m_val = len(X_val)
            # 基本統計量
            mu_val = X_val.mean(axis=0)  # 平均の算出 （axis=0を指定して列毎に標準化）
            sigma_val = X_val.std(axis=0)  # 標準偏差の算出 （axis=0を指定して列毎に標準化）

            # 標準化処理
            X_val = (X_val - mu_val) / sigma_val

            # バイアス項の挿入
            X_val = np.hstack((np.ones(m_val).reshape(m_val, 1), X_val))

        # シータの初期化
        self._init_theta()

        # 最急降下法
        if type(X_val) == np.ndarray:
            self._gradient_descent(X_val, y_val)
            self.val_loss = self.loss

        # シータの初期化
        self._init_theta()

        self._gradient_descent(X, y)
        self.use_loss = self.loss

    def _init_theta(self):
        """
        self.coef_: 次の形のndarray, shape(n_features)
        パラメータをランダムに初期化します。
        """
        self.coef_ = np.random.rand(self.n, 1)

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          線形の仮定関数による推定結果
        """

        hx = np.dot(X, self.coef_)
        return hx

    def _gradient_descent(self, X, y):
        """
        Parameters
         ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """

        alpha = self.lr
        m = len(X)
        y = y.reshape(m, 1)
        hx = self._linear_hypothesis(X)
        count = 0

        for i in range(self.num_iter):
            self.coef_ = self.coef_ - alpha * (1 / m) * (np.dot(X.T, (hx - y)))
            hx = self._linear_hypothesis(X)

            if self.verbose:
                log = 'COUNT:{}, MSE:{}'
                # verboseをTrueにした際は学習過程を出力
                print(log.format(count + 1, self.MSE(hx, y)))

            # trainデータのlossのリザルトを出す
            if self.verbose:
                self.loss[i] = self._compute_cost(X, y)

            #             # X_val入力ある場合MSEのリザルトを出す
            #             if type(X_val) == np.ndarray:
            #                 val_pred = self._linear_hypothesis(X_val)
            #                 self.val_loss[i] = self._compute_cost(val_pred, y_val)

            count += 1

    def predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        # 標準化処理
        mu = X.mean(axis=0)  # 平均の算出 （axis=0を指定して列毎に標準化）
        sigma = X.std(axis=0)  # 標準偏差の算出 （axis=0を指定して列毎に標準化）

        # 標準化処理
        X = (X - mu) / sigma

        if self.bias:
            m = len(X)
            X = np.hstack((np.ones(m).reshape(m, 1), X))
        hx = np.dot(X, self.coef_)

        return hx

    def MSE(self, y_pred, y):
        """
        平均二乗誤差の計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
          推定した値
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        mse : numpy.float
          平均二乗誤差
        """
        # リシェイプ
        y = y.reshape(len(y), 1)
        m = len(y)
        mse = (0.5 * (1 / m) * np.sum(((y_pred - y) ** 2)))

        return mse

    def _compute_cost(self, X, y):
        """
        平均二乗誤差を計算する。MSEは共通の関数を作っておき呼び出す

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値
        Returns
        -------
          次の形のndarray, shape (1,)
          平均二乗誤差
        """
        J = self.MSE(self._linear_hypothesis(X), y)

        return J