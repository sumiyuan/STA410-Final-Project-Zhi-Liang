import statsmodels.api as sm
import numpy as np
import scipy.stats as st


class BayesianModel:
    """
    Bayesian Model Class,
    created so that it would be easier to store and call upon attribute of the model from the selector
    """
    def __init__(self, X, y):
        """
        constructor of bayesian model class

        :param X: X
        :param y: y
        """
        # Store the data in the object as well
        X = sm.add_constant(X, prepend=True)
        self.X = X
        self.y = y
        self.n, self.p = X.shape

        # Fit a model using OLS
        self.model = sm.OLS(y, X)
        self.result = self.model.fit()

        # Extract MLE estimates
        self.beta_hat = self.result.params
        self.cov_beta_hat = self.result.cov_params()
        # Use unbiased estimate of variance or from fit
        self.sigma2_hat = self.result.mse_resid
        self.sigma_hat = np.sqrt(self.sigma2_hat)
        # compute diagonal hat matrix
        XtX_inv = np.linalg.inv(X.T @ X)
        H = X @ XtX_inv @ X.T
        h_diag = np.diagonal(H)
        self.h_diag = h_diag

    def draw_posterior_samples(self, n, seed=123123):
        """
        draw_posterior_samples, approximate the posterior draws multivariate normal, this allows us to avoid complicated
        posterior functions, and it should work well enough for the scope of this course project

        :param n: number of draws
        :param seed: the seed of the draw
        :return: returns the posterior samples
        """
        # create a rng variable so we can control the seed of the draw
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(self.beta_hat, self.cov_beta_hat, size=n)
        return draws


class BayesianModelSelector:
    """
    Bayesian Model Selector Class
    This class will act as the main class that allows the user to add/remove, and calculate the different metrics that
    measure the quality of a model
    class variables:
        models : dict(name: string) -> model (BayesianModel)
    Methods:
        add_model: adds model to the models dict
        remove_model: remove model from the models dict
        calc_loo : calculates the LOO
        calc_waic : calculates waic
        calc_aic : calculates aic
        calculate_model_selector : calculates the selected methods
    """

    def __init__(self):
        """
        constructor of the class, create a private models variable
        """
        self.models = {}

    def add_model(self, name, model):
        """
        add_model, adds a BayesianModel model object into the models dictionary
        :param name: name of the model (String)
        :param model: bayesian model (BayesianModel)
        :return: None
        """
        if name not in self.models:
            self.models[name] = model
        else:
            print(f"Model '{name}' is already in the model, if you wish to update please delete existing model first")

    def remove_model(self, name):
        """
        remove_model, removes the bayesian model object with 'name' from the models dictionary
        :param name: name of the model (String)
        :return: None
        """
        if name in self.models:
            self.models.pop(name)
        else:
            print(f"'{name}' is not a model in the dictionary")

    def get_all_models(self):
        """
        get all the models' names
        :return: the name of all the stored models
        """
        return [*self.models]

    def calc_aic(self, name):
        """
        calculate the aic of the model using the formula AIC = -2 log p(y | \theta) + 2*k
        :param name: name (String) of the model that we want to calculate the aic for
        :return: calculated aic value (float)
        """
        # check if the model is in the model
        if name not in self.models:
            print(f"'{name}' is not a model in the dictionary")
            return None

        # calculate the log_likelihood of the model
        curr_model = self.models[name]
        log_likelihood = curr_model.result.llf

        # get the number of parameters of the model, +1 for the intercept that we include
        k = curr_model.result.df_model + 1

        # aic is calculated as -2 * log_likelihood + 2*k
        aic = -2 * log_likelihood + 2 * k
        return aic

    def calc_bic(self, name):
        """
        calculate the bic based on the formula, BIC = -2 log p(y | \theta) + log(n)*k
        :param name: name (String) of the model
        :return: calculated BIC (float)
        """
        # check if the model is in the model
        if name not in self.models:
            print(f"'{name}' is not a model in the dictionary")
            return None

        # calculate the log_likelihood of the model
        curr_model = self.models[name]
        log_likelihood = curr_model.result.llf

        # get the number of parameters of the model, +1 for the intercept that we include
        k = curr_model.result.df_model + 1

        # number of observations/samples
        n = curr_model.result.nobs

        # calculate the BIC based on the formula BIC = -2 log p(y | \theta) + log(n)*k
        bic = -2 * log_likelihood + np.log(n) * k
        return bic

    def calc_waic(self, name, num=1000, seed=123123):
        """
        calculate the waic for this model, based on the formula waic = -2(lppd - p_waic)
        :param model: name of the model (String)
        :param num: number of posterior draws (int)
        :param seed: seed of the random draw (int)
        :return: the calculated waic (float)
        """
        # for waic we want to find the point log likelihood based on posterior draws, therefore we would need to
        # calculate the likelihood for each point, thus,

        # check if the model is in the model
        if name not in self.models:
            print(f"'{name}' is not a model in the dictionary")
            return None

        # calculate the log_likelihood of the model
        model = self.models[name]
        # posterior draws based on the model
        draws = model.draw_posterior_samples(num, seed)

        # create an empty log likelihood array for storing
        log_likelihood = np.zeros((model.n, num))

        # loop over the draws to calculate the log likelihood for each point
        for x in range(num):
            curr_draw = draws[x]

            # need to calculate the residual
            mu = model.X @ curr_draw
            curr_resid = model.y.ravel() - mu

            # we can now calculate the log likelihood for each posterior draw
            log_likelihood[:, x] = (
                    -0.5 * np.log(2.0 * np.pi * model.sigma2_hat)
                    - 0.5 * (curr_resid ** 2) / model.sigma2_hat
            )

        max_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
        # we want to calculate the llpd for each draw
        likelihood_i = np.exp(log_likelihood - max_likelihood)
        # get the average likelihood across the draw
        avg_likelihood_i = np.mean(likelihood_i, axis=1)
        # now we can calcualte the lppd for this draw
        lppd_i = np.log(avg_likelihood_i) + np.squeeze(max_likelihood, axis=1)

        # Summation across all draws to get the lppd
        lppd = np.sum(lppd_i)

        # we now want to calculate the p_waic
        var_log_lik = np.var(log_likelihood, axis=1, ddof=1)  # sample var across draws
        p_waic = np.sum(var_log_lik)

        elppd_waic = lppd - p_waic
        waic_value = -2.0 * elppd_waic

        return waic_value

    def calc_loocv(self, name):
        """
        calculate the loocv
        :param name: name (String) of the model
        :return: calculated BIC (float)
        """
        # check if the model is in the model
        if name not in self.models:
            print(f"'{name}' is not a model in the dictionary")
            return None

        # calculate the log_likelihood of the model
        curr_model = self.models[name]
        y_hat = curr_model.result.fittedvalues
        resid = curr_model.result.resid

        # to calculate the predicted value at teach point
        y_hat_loo = y_hat - resid / curr_model.h_diag

        # calculate the residual
        e_loo = curr_model.y.ravel() - y_hat_loo

        # calculate the pointwise likelihood
        likelihood_loo_i = -0.5 * np.log(2.0 * np.pi * curr_model.sigma2_hat) \
                           - 0.5 * (e_loo ** 2) / curr_model.sigma2_hat

        return np.sum(likelihood_loo_i)

    def calc_metrics(self, name, metric, num=1000, seed=123123):
        # check if the model is in the model
        if name not in self.models:
            print(f"'{name}' is not a model in the dictionary")
            return None

        # determine which metric needs to be calculated
        if metric == "aic":
            curr_aic = self.calc_aic(name)
            print(f"The AIC metric for model '{name}' is '{curr_aic}', the lower the AIC the better")
            return curr_aic
        elif metric == "bic":
            curr_bic = self.calc_bic(name)
            print(f"The BIC metric for model '{name}' is '{curr_bic}', the lower the BIC the better")
            return curr_bic
        elif metric == "waic":
            curr_waic = self.calc_waic(name, num, seed)
            print(
                f"The WAIC metric for model '{name}' is '{curr_aic}' on '{num}' posterior draws, the lower the WAIC the better")
            return curr_waic
        elif metric == "loocv":
            curr_loocv = self.calc_loocv(name)
            print(f"The LOOCV metric for model '{name}' is '{curr_loocv}', the higher the LOOCV the better")
            return curr_loocv
        elif metric == "all":
            curr_aic = self.calc_aic(name)
            curr_bic = self.calc_bic(name)
            curr_waic = self.calc_waic(name, num, seed)
            curr_loocv = self.calc_loocv(name)
            print(f"The AIC metric for model '{name}' is '{curr_aic}', the lower the AIC the better \n" +
                  f"The BIC metric for model '{name}' is '{curr_bic}', the lower the BIC the better \n" +
                  f"The WAIC metric for model '{name}' is '{curr_aic}' on '{num}' posterior draws, the lower the WAIC the better \n" +
                  f"The LOOCV metric for model '{name}' is '{curr_loocv}', the higher the LOOCV the better")
            return [curr_aic, curr_bic, curr_waic, curr_loocv]
        else:
            print("There is an error with the metric you have entered, please try again")


if __name__ == "__main__":
    mtcars = sm.datasets.get_rdataset("mtcars")
    y = mtcars.data[['mpg']].values
    y = y - y.mean()
    X = mtcars.data[['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']].values
    X = (X - X.mean(axis=0))
    X = X / (X ** 2).mean(axis=0) ** 0.5
    b = BayesianModel(X=X, y=y)