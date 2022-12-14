from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import statistics
import math

def gaussian_kernal(x):
    return (1 / math.sqrt((2 * math.pi)) * math.exp(-0.5 * x ** 2))

def gaussian(x, mu, sig, h):
    return np.multiply(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))), h)

def polymodeling(xVals, yVals, degree = 2):
    return np.polyfit(xVals, yVals, degree)
    

class BAZUOD(BaseDetector):
    def __init__(self, contamination=0.1):
        super(BAZUOD, self).__init__(contamination=contamination)
        # self.skewness_eval(self.skewness_map, -1.5, 1.5)
        # self.skewness_eval(self.skewness_gaussian_map, -1.5, 1.5)

    def ecdf(self, X):
        ecdf = ECDF(X)
        return ecdf(X)

    def fit(self, X, y=None):
        self.X_train = X

    def skewness_map(self, x):
        L_por = [1, 0.9, 0.8, 0.5] # highly skewed, moderately skewed, slightly skewed, balanced
        if x < -1:
            return L_por[0]
        elif x < -0.5:
            return L_por[1]
        elif x < 0:
            return L_por[2]
        elif x == 0:
            return L_por[3]
        elif x < 0.5:
            return 1 - L_por[2]
        elif x < 1:
            return 1 - L_por[1]
        else:
            return 1 - L_por[0]

    def skewness_gaussian_map(self, x):
        if x == 0:
            return 0.5
        elif x < 0:
            return 1 - gaussian(x, 0, 0.2, 0.2)
        else:
            return gaussian(x, 0, 0.2, 0.2)

    def skewness_eval(self, func, lb = -3, ub = 3, prec = 1000):
        x = [i / prec for i in range(int(lb * prec), int(ub * prec), 1)]
        y = [func(i / prec) for i in range(int(lb * prec), int(ub * prec), 1)]
        plt.plot(x, y, '-', label='Left tail proportion')
        plt.plot(x, [1 - i for i in y], '-', label='Right tail proportion')
        plt.xlabel('Skewness coefficient')
        plt.ylabel('Proportion')
        plt.legend()
        plt.show()

    def get_GKernal_pdf(self):
        # calculating mean vector
        u = (0, ) * len(self.X_train[0])
        for x in self.X_train:
            u = np.add(u, x)
        u = u / len(self.X_train)
        # calculating averaged pdf
        pdf = 0
        for x in self.X_train:
            pdf += gaussian_kernal(np.linalg.norm(x - u))
        pdf = pdf / len(self.X_train)
        return pdf

    def estimate_best_contamination(self):
        deg = len(self.X_train[0])
        # retrieve the best a and b params for our linear model
        df = pd.read_csv('dim_eval.csv')
        rec = df.iloc[deg - 1]
        a = rec['a']
        b = rec['b']
        # predict best contamination rate
        pdf = self.get_GKernal_pdf()
        self.contamination = (pdf - b) / a  # x = (y - b) / a
        return self.contamination
        try:
            a = 0
        except:
            return 0.1

    def decision_function(self, X):
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)
            # self.estimate_best_contamination()
        size = X.shape[0]
        dim = X.shape[1]
        self.U_l = pd.DataFrame(-1*np.log(np.apply_along_axis(self.ecdf, 0, X)))
        self.U_r = pd.DataFrame(-1*np.log(np.apply_along_axis(self.ecdf, 0, -X)))
        # skewness = np.sign(np.apply_along_axis(skew, 0, X))
        skewness = np.apply_along_axis(skew, 0, X)
        kurtosis_list = np.apply_along_axis(kurtosis, 0, X)
        self.l_skewness = [self.skewness_map(x) for x in skewness]
        # self.l_skewness = [self.skewness_gaussian_map(x) for x in skewness]
        self.r_skewness = [(1 - x) for x in self.l_skewness]
        self.U_skew = self.U_l * self.l_skewness + self.U_r * self.r_skewness
        self.O = np.maximum(self.U_skew, np.add(self.U_l, self.U_r)/2)
        if hasattr(self, 'X_train'):
            self.decision_scores_ = self.O.sum(axis=1).to_numpy()[-original_size:]
        else:
            self.decision_scores_ = self.O.sum(axis=1).to_numpy()
        self.threshold_ = np.percentile(self.decision_scores_, (1-self.contamination)*100)
        self.labels_ = np.zeros(len(self.decision_scores_))
        for i in range(len(self.decision_scores_)):
            self.labels_[i] = 1 if self.decision_scores_[i] >= self.threshold_ else 0
        return self.decision_scores_

    def explain_outlier(self, ind, cutoffs=None):
        cutoffs = [1-self.contamination, 0.99] if cutoffs is None else cutoffs
        plt.plot(range(1, self.O.shape[1] + 1), self.O.iloc[ind], label='Outlier Score')
        for i in cutoffs:
            plt.plot(range(1, self.O.shape[1] + 1), self.O.quantile(q=i, axis=0), '-', label=f'{i} Cutoff Band')
        plt.xlim([1, self.O.shape[1] + 1])
        plt.ylim([0, int(self.O.max().max()) + 1])
        plt.ylabel('Dimensional Outlier Score')
        plt.xlabel('Dimension')
        plt.xticks(range(1, self.O.shape[1] + 1, 10))
        plt.yticks(range(0, int(self.O.max().max()) + 1))
        label = 'Outlier' if self.labels_[ind] == 1 else 'Inlier'
        plt.title(f'Outlier Score Breakdown for Data #{ind+1} ({label})')
        plt.legend()
        plt.show()
        return self.O.iloc[ind], self.O.quantile(q=cutoffs[0], axis=0), self.O.quantile(q=cutoffs[1], axis=0)