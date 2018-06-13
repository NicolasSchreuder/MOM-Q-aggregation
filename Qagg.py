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
        self.predictors = [predictor.fit(X, y) for predictor in self.predictors] # trained_predictors
        self.trained_predictors = True

    def Q(self, theta, X, y):
        """
        Evaluate Q functional
        theta : aggregation parameter
        X : feature matrix
        y : target vector
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

    def NablaQ(self, theta, X, y):
        """
        Gradient of Q functional
        """
        n, p = X.shape

        predictions = np.zeros((n, self.M))

        for m, predictor in enumerate(self.predictors):
            predictions[:, m] = predictor.predict(X)

        derivatives = np.zeros(self.M)
        for j in range(self.M):
            for i in range(n):
                if theta.any()>0:
                    derivatives[j] += (-2*(1-self.nu)*predictions[i, j]*(y[i] - np.average(predictions[i, :], weights=theta))
                                      + self.nu * (y[i] - predictions[i, j])**2)
                else:
                    derivatives[j] += (-2*(1-self.nu)*predictions[i, j]*y[i]
                                      + self.nu * (y[i] - predictions[i, j])**2)
        return derivatives

    def GMA_0(self, X, y, K):

        theta = np.zeros(self.M)

        for k in range(1, K):

            alpha = 2/(k+1)

            J = np.argmin([self.Q(theta, X, y) + alpha*(np.eye(self.M)[:, m] - theta) for m in range(self.M)])

            # option-1
            theta = theta + alpha*(np.eye(self.M)[:, J] - theta)

        self.theta = theta

    def GMA_1(self, X, y, K):
        """
        Greedy Frank-Wolfe algorithm
        X : 2D numpy array, design matrix
        y : 1D numpy array, target vector
        K : integer, number of iterations
        """
        theta = np.zeros(self.M)

        for k in range(1, K):
            alpha = 2/(k+1)

            J = np.argmin(self.NablaQ(theta, X, y))

            # option-1
            theta = theta + alpha*(np.eye(self.M)[:, J] - theta)

        self.theta = theta

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

        # MOM risk for aggregate estimator

        # split data into n_blocks
        blocks_idx = np.split(np.random.permutation(n), n_blocks)

        if theta.any()>0:
            aggregate_pred = np.average(predictions, axis=1, weights = theta)
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

    # Interior point method

    def gradient_Q(self, theta, y, F, B_star):
        """
        Compute gradient of the Q-functional

        Inputs :
        theta: 1D array, current probability vector
        y: 1D array, target vector
        F: 2D array of shape (n_samples, n_predictors), prediction matrix
        B_star: 2D array of shape (n_obs_per_block, n_predictors+1), indexes of obs in median blocks
        """

        n, M = F.shape
        grad = np.zeros(M)

        # indexes of aggregate risk median block
        B_0_star = B_star[:, 0]

        n_samples_per_block = len(B_0_star) # =30 OK

        aggregate_error = y[B_0_star] - np.dot(F[B_0_star, :], theta)

        for l in range(M):
            B_l_star = B_star[:, l+1]

            # to account for the fact that the first column correponds to indexes of obs from median aggregate block

            quadratic_term = np.sum(np.multiply(F[B_0_star, l], aggregate_error))
            linear_term = np.linalg.norm(y[B_l_star] - F[B_l_star, l], ord=2)**2

            grad[l] = -2*(1-self.nu)/n_samples_per_block*quadratic_term + self.nu/n_samples_per_block*linear_term

        return grad

    def hessian_Q(self, F, B_star):
        """
        Compute Hessian of the Q-functional

        Inputs :
        F: 2D array of shape (n, M), prediction matrix
        B_star: 2D array of shape (n_obs_per_block, n_predictors+1), indexes of obs in median blocks
        """
        B_0_star = B_star[:, 0]

        H = 2*(1-self.nu)/len(B_0_star)*np.dot(F[B_0_star, :].T, F[B_0_star, :])

        return H

    def gradient_phi(self, theta):
        """
        Compute gradient of the barrier function
        """
        return -np.reciprocal(theta) # np.reciprocal returns coordinate-wise inverse

    def hessian_phi(self, theta):
        """
        Compute Hessian of the barrier function
        """
        return np.diag(np.reciprocal(theta**2))

    def KKT_system(self, grad_theta, hessian_theta):
        """
        Compute KKT matrix and KKT "bias"

        Inputs :
        theta: 1D array, aggregation parameter
        grad_theta: gradient of the full objective (i.e. including barrier function) w.r.t. theta
        hessian_theta: Hessian of the full objective (i.e. including barrier function) w.r.t. theta
        """
        # upper part of KKT matrix ()
        upper_left = hessian_theta # shape OK : (32, 32)
        upper_right = np.ones(self.M).T # shape OK : (32, )
        upper = np.column_stack((upper_left, upper_right)) # shape OK : (32, 33)

        # lower part of KKT matrix (Lagrange multiplier eq)
        lower_left = np.ones(self.M)
        lower = np.concatenate((lower_left, np.zeros(1)))

        # form full KKT matrix
        KKT_matrix = np.row_stack((upper, lower)) # shape OK : (33, 33)

        # form KKT bias
        KKT_bias = np.concatenate((grad_theta, np.zeros(1))) # shape OK : (33,)

        return KKT_matrix, KKT_bias

    def squaredNewtonDecrement(self, grad_theta, hessian_theta):
        """
        Compute squared Newton decrement
        Input : gradient of full objective in theta, Hessian of full objective in theta
        """
        return np.dot(grad_theta.T, np.linalg.inv(hessian_theta).dot(grad_theta))

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

    def newton_method(self, X, y, F, theta0, t, B_star, tol=1e-1):
        """
        y: target vector
        F: prediction matrix
        theta0: initial theta for Newton method
        t: float (>0), IP method parameter
        B_star: 2D array, indexes
        """

        theta = theta0

        # compute gradient of full objective
        grad_theta = t*self.gradient_Q(theta, y, F, B_star) + self.gradient_phi(theta)

        # compute Hessian of full objective
        hessian_theta = t*self.hessian_Q(F, B_star) + self.hessian_phi(theta)

        # compute squared decrement
        squared_decrement = self.squaredNewtonDecrement(grad_theta, hessian_theta)

        while 0.5*squared_decrement > tol:

            # compute Newton Step
            newton_step = -self.newton_step(grad_theta, hessian_theta)

            # compute damped coefficient
            damped_coeff = 1/(1 + np.sqrt(squared_decrement))

            # update theta
            theta += damped_coeff*newton_step

            # update gradient and Hessian in new theta
            grad_theta = t*self.gradient_Q(theta, y, F, B_star) + self.gradient_phi(theta)

            hessian_theta = t*self.hessian_Q(F, B_star) + self.hessian_phi(theta)

            # update squared decrement
            squared_decrement = self.squaredNewtonDecrement(grad_theta, hessian_theta)

        return theta

    def newton_step(self, grad_theta , hessian_theta):
        """
        Solve KKT system and return delta Newton step

        """
        KKT_matrix, KKT_bias = self.KKT_system(grad_theta, hessian_theta)
        KKT_sol = np.linalg.solve(KKT_matrix, KKT_bias)

        delta_theta = KKT_sol[:-1]

        return delta_theta

    def IP_MOM(self, X, y, n_blocks, t=0, mu=5, tol=1e-1):
        """
        MOM version of interior point method for Q-aggregation

        Inputs :
        X:
        y:
        n_blocks:
        t:
        mu:
        tol:
        """

        # initialization with a strictly feasible point
        theta = np.ones(self.M)/self.M

        # compute individual predictions
        predictions = self.compute_individual_predictions(X)
        # shape = (n_samples, n_predictors) OK

        # get median block indexes
        aggregate_median_block_idx, individual_median_block_idx = self.split_into_blocks(y, predictions, theta, n_blocks)
        B_star = np.column_stack((aggregate_median_block_idx, individual_median_block_idx.T))
        # shape = (n_samples_per_block, n_predictors+1)

        # first step of newton_method
        theta = self.newton_method(X, y, predictions, theta, t, B_star)

        while 1/t > tol :
            # get median block indexes
            aggregate_median_block_idx, individual_median_block_idx = self.split_into_blocks(y, predictions, theta, n_blocks)
            B_star = np.column_stack((aggregate_median_block_idx, individual_median_block_idx.T))

            theta = self.newton_method(X, y, predictions, theta, t, B_star)

            t = mu*t

        self.theta = theta

    def predict(self, X):
        """
        X : 2D numpy array, feature vectors
        """
        n, p = X.shape
        predictions = np.zeros((n, self.M))

        for m, predictor in enumerate(self.predictors):
            predictions[:, m] = predictor.predict(X)

        return(np.average(predictions, axis=1, weights=self.theta))
