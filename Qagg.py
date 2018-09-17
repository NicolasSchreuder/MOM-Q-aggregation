import numpy as np

from sklearn.metrics import mean_squared_error

class Q_aggregation():

    def __init__(self, predictors, nu=0.5, beta=0, trained_predictors=False):
        """
        predictors : list of predictors ()
        """
        self.predictors = predictors # list of predictors
        self.M = len(predictors) # number of predictors
        self.nu = nu # interpolation coefficient
        self.beta = beta # temperature parameter for prior weights
        self.trained_predictors = False # True for already trained predictors

    def set_prior_weights(self, prior_weights):
        """
        Set prior weights (see the article for meaning of such quantities)
        """
        self.pi = prior_weights

    def train_predictors(self, X, y):
        """
        Train each predictor
        """
        self.predictors = [predictor.fit(X, y) for predictor in self.predictors] # trained_predictors
        self.trained_predictors = True

    def Q(self, theta, X, y):
        """
        Evaluate Q functional


        Parameters
        ----------
        theta : float, aggregation parameter
        X : 2D array, feature matrix
        y : 1D array, target vector

        Returns
        -------
        Q : float, value of the Q function at point theta
        """
        n, p = X.shape
        predictions = np.zeros((n, self.M))

        for m, predictor in enumerate(self.predictors):
            predictions[:, m] = predictor.predict(X)

        if theta.any()>0:
            MSE_theta = mean_squared_error(np.average(predictions, axis=1, weights=theta), y)
        else:
            MSE_theta = mean_squared_error(np.zeros(y.shape), y)

        Q = (1-self.nu)*MSE_theta + self.nu*np.sum([theta[m]*mean_squared_error(predictions[:, m], y)
                                                    for m in range(self.M)])
        return Q


    def compute_individual_predictions(self, X):
        """
        Compute predictions on sample for each predictor

        Input: feature matrix of shape (n_samples, d_samples)

        Return:
        predictions: 2D array of shape (n_samples, n_predictors)
        """
        predictions = np.zeros((X.shape[0], self.M))

        for m, predictor in enumerate(self.predictors):
                predictions[:, m] = predictor.predict(X)

        return predictions

    def NablaQ(self, theta, X, y, predictions):
        """
        Gradient of Q functional

        Parameters
        ----------
        theta : float, aggregation parameter
        X : 2D array, feature matrix
        y : 1D array, target vector

        Returns
        -------
        grad : 1D array, gradient of the Q functional

        """
        n, p = X.shape
        grad = np.zeros(self.M)

        for j in range(self.M):
            for i in range(n):
                if theta.any()>0:
                    grad[j] += (-2*(1-self.nu)*predictions[i, j]*(y[i] - np.average(predictions[i, :], weights=theta))
                                      + self.nu * (y[i] - predictions[i, j])**2)/n

                else:
                    grad[j] += (-2*(1-self.nu)*predictions[i, j]*y[i]
                                      + self.nu * (y[i] - predictions[i, j])**2)/n
        #grad = np.zeros(self.M)
        #aggregate_error = y - np.dot(predictions, theta)

        #for l in range(self.M):

        #    quadratic_term = np.sum(np.multiply(predictions[:, l], aggregate_error))
        #    linear_term = np.linalg.norm(y - predictions[:, l], ord=2)**2

        #    grad[l] = -2*(1-self.nu)/n*quadratic_term + self.nu/n*linear_term

        #print(grad)
        #print(derivatives)

        #print('------')

        return grad

    def GMA_0(self, X, y, K):

        theta = np.zeros(self.M)

        for k in range(1, K):

            alpha = 2/(k+1)

            J = np.argmin([self.Q(theta, X, y) + alpha*(np.eye(self.M)[:, m] - theta) for m in range(self.M)])

            # option-0
            theta = theta + alpha*(np.eye(self.M)[:, J] - theta)

        self.theta = theta

    def GMA_1(self, X, y, K):
        """
        Greedy Frank-Wolfe algorithm

        Parameters
        ----------
        X : 2D numpy array, design matrix
        y : 1D numpy array, target vector
        K : integer, number of iterations

        Returns
        -------
        True
        """
        # initialize theta to null vector
        theta = np.zeros(self.M)

        predictions = self.compute_individual_predictions(X)

        for k in range(K):
            # compute stepsize
            alpha = 2/(k+2)

            # find index of the smallest gradient coordinate
            J = np.argmin(self.NablaQ(theta, X, y, predictions))
            # option-1
            theta += alpha*(np.eye(self.M)[:, J] - theta)
            # np.eye(self.M)[:, J] corresponds ot the J-th vector of the canonical basis

        self.theta = theta

        return theta

    ###############
    # MOM version #
    ###############

    def NablaQ_MOM(self, theta, predictions, y, aggregate_median_block_idx, individual_median_block_idx):
        """
        Gradient of MOM Q-functional
        """
        derivatives = np.zeros(self.M)

        for j in range(self.M):
            for i in aggregate_median_block_idx:
                if theta.any()>0:
                    derivatives[j] += (-2*(1-self.nu)*predictions[i, j]*
                                       (y[i] - np.average(predictions[i, :], weights=theta)))/30
                else:
                    derivatives[j] += (-2*(1-self.nu)*predictions[i, j]*y[i])/30

            for i in individual_median_block_idx[j, :]:
                derivatives[j] += (self.nu * (y[i] - predictions[i, j])**2)/30

        return derivatives

    def split_into_blocks(self, y, predictions, theta, n_blocks):
        """
        Split data into blocks and returns indexes of median blocks
        """
        n = predictions.shape[0]

        #print(n)

        if n%n_blocks != 0:
            n = (n//n_blocks)*n_blocks

        # MOM risk for aggregate estimator

        #print(n)
        #print(n_blocks)

        # split data into n_blocks
        blocks_idx = np.split(np.random.permutation(n), n_blocks)

        if theta.any()>0:
            aggregate_pred = np.average(predictions, axis=1, weights = theta)[:n]
        else:
            aggregate_pred = np.zeros(n)

        # compute empirical risk on each block
        block_risks = [mean_squared_error(aggregate_pred[block], y[block])
                       for block in blocks_idx]
        # Index of median block
        aggregate_median_block_idx = blocks_idx[np.argsort(block_risks)[n_blocks//2]]

        individual_median_block_idx = np.zeros((self.M, len(blocks_idx[0])))

        # MOM risk for each individual estimator
        for m, predictor in enumerate(self.predictors):
            # split data into n_blocks
            blocks_idx = np.split(np.random.permutation(n), n_blocks)

            block_risks = [mean_squared_error(predictions[block, m], y[block])
                       for block in blocks_idx]

            individual_median_block_idx[m,:] = blocks_idx[np.argsort(block_risks)[n_blocks//2]]

        individual_median_block_idx = individual_median_block_idx.astype(int)

        return aggregate_median_block_idx, individual_median_block_idx

    def GMA_1_MOM(self, X, y, n_steps, n_blocks):
        """
        MOM version of GMA-1 algorithm
        """
        theta = np.zeros(self.M)

        n, d = X.shape

        predictions = self.compute_individual_predictions(X)

        for k in range(1, n_steps):
            alpha = 2/(k+1)

            aggregate_median_block_idx, individual_median_block_idx = self.split_into_blocks(y, predictions, theta, n_blocks)

            J = np.argmin(self.NablaQ_MOM(theta, predictions, y, aggregate_median_block_idx, individual_median_block_idx))

            theta = theta + alpha*(np.eye(self.M)[:, J] - theta)

        self.theta = theta

    def predict(self, X):
        """
        X : 2D numpy array, feature matrix of shape (n_samples, d)
        """
        n, p = X.shape
        predictions = np.zeros((n, self.M))

        for m, predictor in enumerate(self.predictors):
            predictions[:, m] = predictor.predict(X)

        return np.average(predictions, axis=1, weights=self.theta)
