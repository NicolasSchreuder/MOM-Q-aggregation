import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from Qagg import Q_aggregation

def make_gaussian_regression(n_samples, d, noise_sd):
    """
    n_samples : int, number of samples
    d : int, number of features
    noise_sd : float, standard deviation of the noise
    """
    # Generate independent Gaussian features
    X = np.random.multivariate_normal(np.zeros(d), np.eye(d), n_samples)

    # Generate random regression parameter
    t = np.random.poisson(lam=0.1, size=d)
    # The smaller the mean of the Poisson the more t is sparse

    y = np.dot(X, t) + np.random.multivariate_normal(np.zeros(n_samples),
                                                     noise_sd**2*np.eye(n_samples))

    return X, y

def corrupt_dataset(X_agg, y_agg, prop_outliers, corruption_type='constant'):
    """
    Corrupts a dataset by randomly changing the value of the target
    """
    n, d = X_agg.shape

    assert prop_outliers >= 0 and prop_outliers <= 1

    n_outliers = int(prop_outliers*n)

    outliers_idx = np.random.choice(n, n_outliers, replace=False)

    if corruption_type == 'constant':
        y_agg[outliers_idx] = 50

    return X_agg, y_agg

def compute_fast_residual_term(n_predictors, n_samples):
    """
    Compute the theoretical minimax optimal aggregation rate
    """
    return np.log(n_predictors)/n_samples

def make_function_class_regression():
    """
    Returns a list of regression predictors :
    Ridge with different parameters, Linear, Elastic Net and Lasso with different parameters
    """
    # Ridge
    predictors = [linear_model.Ridge (alpha = alpha) for alpha in np.linspace(0.5, 30, 10)]

    # Linear Regression
    predictors.append(linear_model.LinearRegression())

    # Elastic Net
    predictors.append(linear_model.ElasticNet())

    # Lasso
    for alpha in np.linspace(0.5, 20, 20):
        predictors.append(linear_model.Lasso(alpha=alpha))

    return predictors

def train_agg_test_split(X, y, train_size, agg_size, random_state=None):
    """
    Split dataset in 3 non overlapping datasets
    X : 2D numpy array, feature matrix of shape n_samples*d
    y : 1D numpy array, targer vector of shape n_sampless
    train_size : proportion of obs from original dataset in training dataset
    agg_size : proportion of obs from original dataset in aggregation dataset
    random_state : int, numpy seed
    """
    if random_state != None:
        np.random.seed(random_state)

    n, d = X.shape

    # Shuffle observations
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]

    assert train_size + agg_size < 1
    train_idx = int(n*train_size)
    X_train, y_train = X[:train_idx], y[:train_idx]

    agg_idx = train_idx + int(n*agg_size)
    X_agg, y_agg = X[train_idx:agg_idx], y[train_idx:agg_idx]

    X_test, y_test = X[agg_idx:], y[agg_idx:]

    return X_train, X_agg, X_test, y_train, y_agg, y_test

def test_gaussian_regression_outliers(n_samples, d, n_replications,
                                      prop_outliers, agg_method='GMA1', noise_sd=0.5):
    """
    Test performance of aggregation method on gaussian regression example with outliers
    """
    regrets = []

    predictors = make_function_class_regression()

    for i in tqdm(range(n_replications)):
        X, y = make_gaussian_regression(n_samples, d, noise_sd)

        X_train, X_agg, X_test, y_train, y_agg, y_test = train_agg_test_split(X, y, 0.4, 0.3)

        q = Q_aggregation(predictors)

        q.train_predictors(X_train, y_train)

        X_agg, y_agg = corrupt_dataset(X_agg, y_agg, prop_outliers, corruption_type='constant')

        if agg_method == 'GMA_1':
            q.GMA_1(X_agg, y_agg, 10)

        elif agg_method == 'GMA_1_MOM':
            q.GMA_1_MOM(X_agg, y_agg, 10, 50)

        elif agg_method == 'IP_MOM':
            q.IP_MOM(X_agg, y_agg, n_blocks=50, t=1, mu=5, tol=1e-1)

        else:
            raise "Unrecognized aggregation method"

        MSE_agg = mean_squared_error(q.predict(X_test), y_test)

        oracle_agg = np.min([mean_squared_error(q.predictors[i].predict(X_test), y_test) for i in range(len(predictors))])

        regrets.append(MSE_agg - oracle_agg)

    return regrets
