from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# OD models
from models.cod import COD
from models.bazuod import BAZUOD
from pyod.models.copod import COPOD
# For data generation and visualization
from scipy.stats import skewnorm, lognorm
import plotly.express as px
import random
import math
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Generate synthetic dataset
def jiggle(xVals, appendant = 1):
    vals = (xVals,)
    for _ in range(appendant):
        vals = (*vals, [x + random.gauss(0, 0.4) for x in xVals])
    return list(zip(*vals))

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def appendLabel(data, label):
    return list([(*x, label) for x in data])

def skew_norm_pdf(lo = 30, ro = 5, dsize = 1350, dimension = 2, labelled = True):
    mu = 1 
    sigma = 0.1
    normal_distributed = np.random.normal(mu, sigma, size = (dsize, dimension))
    outliers_l = jiggle(np.random.rand(lo) - 0.2, dimension - 1)
    outliers_r = jiggle(np.random.rand(ro) * 0.4 + 1.3, dimension - 1)
    if labelled:
        normal_distributed = appendLabel(normal_distributed, 0)
        outliers_l = appendLabel(outliers_l, 1)
        outliers_r = appendLabel(outliers_r, 1)

    normal_distributed = np.concatenate((normal_distributed, outliers_l, outliers_r), axis=0)
    return normal_distributed

def detached_data(data):
    if len(data[0]) == 2:
        return [w[0] for w in data], [w[1] for w in data]
    elif len(data[0]) >= 3:
        return [w[0] for w in data], [w[1] for w in data], [w[2] for w in data]
    return None

def detached_label(data):
    return np.array(list([w[:-1] for w in data])), [w[-1] for w in data]

def plot_gaussian_kernel_curve(d = 2):
    pdf_list = []
    baz = BAZUOD()
    lo = 30
    ro = 5
    to = lo + ro
    for dsize in range(35, 3001, 5):
        data = skew_norm_pdf(lo, ro, dsize, d, False)
        baz.fit(data)
        pdf_list.append((to / (dsize + to) * 100, baz.get_GKernal_pdf()))
    df = pd.DataFrame(pdf_list)
    plt.plot(df[0], df[1], '-')
    plt.xlabel('Outlier proportion (in percentage)')
    plt.ylabel('Gaussian kernal')
    plt.show()
    return df

def plot_gaussian_kernel_curve_per():
    pdf_list = []
    baz = BAZUOD()
    lo = 30
    ro = 5
    to = lo + ro
    for per in range(1, 600, 1):
        dsize = int(to / (per / 1000) - to)
        data = skew_norm_pdf(lo, ro, dsize, 2, False)
        baz.fit(data)
        pdf_list.append((per / 10, baz.get_GKernal_pdf()))
    df = pd.DataFrame(pdf_list)
    plt.plot(df[0], df[1], '-')
    plt.xlabel('Outlier proportion (in percentage)')
    plt.ylabel('Gaussian kernal')
    plt.show()
    return df

def mirror_list(list):
    minVal = min(list)
    maxVal = max(list)
    return [ minVal + (maxVal - x) for x in list]

def compactness_test():
    attr1, attr2, labels = detached_data(skew_norm_pdf(30, 5, 2000))
    data = list(zip(attr1, attr2))
    baz = BAZUOD()
    baz.fit(data)
    best_c = baz.estimate_best_contamination()

    # ----- Symmetricity Test -----
    mattr1, mattr2 = mirror_list(attr1), mirror_list(attr2)
    mdata = list(zip(mattr1, mattr2))
    # Calculate Contamination Estimate
    baz.fit(mdata)
    mbest_c = baz.estimate_best_contamination()
    # plot both graphs
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(attr1, attr2, s=None, c = labels)
    axs[0].set_title(f'Original data (Est. {best_c:.3f})')
    axs[1].scatter(mattr1, mattr2, s=None, c = labels)
    axs[1].set_title(f'Mirrored data (Est. {mbest_c:.3f})')
    plt.show()

    # ----- Integrity Test on Scability -----
    sattr1 = [*attr1, *attr1, *attr1]
    sattr2 = [*attr2, *attr2, *attr2]
    slabels = [*labels, *labels, *labels]
    
    sdata = list(zip(sattr1, sattr2))
    # Calculate Contamination Estimate
    baz.fit(sdata)
    sbest_c = baz.estimate_best_contamination()
    # plot distribution of both graphs
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(attr1, 30)
    axs[0].set_title(f'Original data (Est. {best_c:.3f})')
    axs[1].hist(sattr1, 30)
    axs[1].set_title(f'3X quantity data (Est. {sbest_c:.3f})')
    plt.show()

# 2D only
def compare():
    to = 35
    per_col, prec_c, prec_b  = [], [], []
    total_win, total_trial = 0, 0
    clf = COD()
    baz = BAZUOD()
    for per in range(10, 501, 5):
        per_col.append(per / 10)
        dsize = int(to / (per / 1000) - to)
        data = skew_norm_pdf(30, 5, dsize)
        attr1, attr2, labels = detached_data(data)
        data = np.array([list([x[0], x[1]]) for x in data])
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=48)
        clf.fit(X_train)
        clf.decision_function(X_train)
        pred_clf = clf.predict(X_test)
        prec_c.append(metrics.f1_score(pred_clf, y_test))

        baz.fit(X_train)
        print(baz.estimate_best_contamination())
        baz.decision_function(X_train)
        pred_baz = baz.predict(X_test)
        prec_b.append(metrics.f1_score(pred_baz, y_test))

        total_trial += 1
        if prec_b[-1] > prec_c[-1]:
            total_win += 1

    print(f'BAZUOD outperforms COPOD {(total_win * 100 / total_trial)}% of the times')
    plt.plot(per_col, prec_c, '-', label = 'COPOD')
    plt.plot(per_col, prec_b, '-', label = 'BAZUOD')
    plt.xlabel('Outlier proportion (in percentage)')
    plt.ylabel('F1-score')
    plt.legend()
    plt.show()

# Test for varing dimensionality
def linear_model_dimensionality_eval(plot = True):
    lo, ro, to = 30, 5, 35
    res_list = []
    baz = BAZUOD()
    for d in range(1, 1001, 1):
        pdf_list = []
        for per in range(10, 501, 5):
            dsize = int(to / (per / 1000) - to)
            data = skew_norm_pdf(lo, ro, dsize, d)
            data, labels = detached_label(data)
            
            baz.fit(data)
            pdf_list.append((to / (dsize + to) * 100, baz.get_GKernal_pdf()))

        df = pd.DataFrame(pdf_list)
        b = linear_modeling(df[0], df[1], False)
        res_list.append((d, b[0], b[1]))
    df = pd.DataFrame(res_list)
    if plot:
        plt.plot(df[0], df[1], '-')
        plt.show()

        plt.plot(df[0], df[2], '-')
        plt.show()    

        plt.plot(df[0], df[1], '-', label = 'a')
        plt.plot(df[0], df[2], '-', label = 'b')
        plt.legend()
        plt.show()
    return df

# Test contamination estimate against dimensionality
def contamination_examination(contamination = 2): # contamination = (0, 100)
    baz = BAZUOD()
    x = []
    y = []
    g = []
    prec = 0
    dsize = 3500 / contamination - 35
    for d in range(1, 1001, 10):
        data, labels = detached_label(skew_norm_pdf(30, 5, int(dsize), d))
        baz.fit(data)
        est = baz.estimate_best_contamination()
        error = abs(contamination - est)
        x.append(d)
        y.append(est)
        g.append(contamination)
        prec += (1 - error / contamination) * 100
    
    print(f'Accuracy: {(prec / len(x)):.2f}%')
    plt.ylim(top = 100)
    plt.ylim(bottom = 0)
    plt.plot(x, y, label = 'Est. contamination')
    plt.plot(x, g, label = 'Ground Truth')
    plt.xlabel('Dimension')
    plt.ylabel('Contamination')
    plt.legend()
    plt.show()
    
    

# Linear regression
def linear_modeling(x, y, plot = True, x_title = 'Outlier proportion (in percentage)', y_title = 'Gaussian kernal'):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    if(plot):
        plt.scatter(x, y, color = "m", marker = "o", s = 30)
        y_pred = b_0 + b_1 * x
        plt.plot(x, y_pred, color = "g")
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
    return (b_0, b_1)

# fit power 2 model
def parabolic_modeling(xVals, yVals, plot = True, x_title = 'Outlier proportion (in percentage)', y_title = 'Gaussian kernal'):
    par = np.polyfit(xVals, yVals, 2)
    if(plot):
        x = np.linspace(min(xVals), max(xVals), 1000)
        y = par[0] * x ** 2 + par[1] * x + par[2]  
        fig, ax = plt.subplots()
        ax.plot(xVals, yVals, 'o')
        ax.plot(x, y)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
    return par

# fit polynomial model
def polymodeling(xVals, yVals, degree = 6, plot = True, x_title = 'Outlier proportion (in percentage)', y_title = 'Gaussian kernal'):
    par = np.polyfit(xVals, yVals, degree)
    if(plot):
        x = np.linspace(min(xVals), max(xVals), 1000)
        y = []
        for _x in x:
            sum = 0
            for i in range(degree + 1):
                sum = sum + par[i] * _x ** (degree - i)
            y.append(sum)
        fig, ax = plt.subplots()
        ax.plot(xVals, yVals, 'o')
        ax.plot(x, y)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
    return par

# fit logarithmic model
def logarithmic_modeling(xVals, yVals, plot = True, x_title = 'Outlier proportion (in percentage)', y_title = 'Gaussian kernal'):
    par = np.polyfit(np.log(xVals), yVals, 1)
    if(plot):
        x = np.linspace(min(xVals), max(xVals), 1000)
        y = par[0] + par[1] * np.log(x)
        fig, ax = plt.subplots()
        ax.plot(xVals, yVals, 'o')
        ax.plot(x, y)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
    return par

# fit log normal distribution model
def log_norm_modeling(xVals, yVals, plot = True, x_title = 'Outlier proportion (in percentage)', y_title = 'Gaussian kernal'):
    s = np.std(yVals)
    par = lognorm.stats(s, moments='mvsk')
    
    if(plot):
        fig, ax = plt.subplots()
        x = np.linspace(lognorm.ppf(0.01, s), lognorm.ppf(0.99, s), 1000)
        y = lognorm.pdf(x, s)
        ax.plot(x, y) # , 'r-', lw=5, alpha=0.6, label='lognorm pdf'
        ax.plot(xVals, yVals, 'o')
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
    return par

if __name__ == "__main__":
    np.random.seed(48)
    random.seed(48)

    contamination_examination(36)

    # df = plot_gaussian_kernel_curve(2000)
    # b = linear_modeling(df[0], df[1])
    # print('Linear model: ', b)
    # exit()

    # df = linear_model_dimensionality_eval(plot=False)
    # df.columns = ['deg', 'b', 'a']
    # df.to_csv('dim_eval.csv', index = False)

    df = pd.read_csv('dim_eval.csv')

    # Polynomial fitting
    for d in range(1, 13):
        a_par = polymodeling(df['deg'], df['a'], d, True, x_title='Dimensionality', y_title='Param a in GK linear model')
        print('A: ', a_par)
    
    for d in range(1, 7):
        b_par = polymodeling(df['deg'], df['b'], d, True, x_title='Dimensionality', y_title='Param b in GK linear model')
        print('B: ', b_par)
    exit()

    # df = plot_gaussian_kernel_curve(4)
    # b = linear_modeling(df[0], df[1])
    # print('Linear model: ', b)
    # plot_regression_line(df[0], df[1], b)

    baz = BAZUOD()
    data, labels = detached_label(skew_norm_pdf(30, 5, 35, 10000))
    baz.fit(data)
    print(baz.estimate_best_contamination())
    #baz.decision_function(data)
    exit()

    for dsize in range(35, 10000, 35):
        baz = BAZUOD()
        data, labels = detached_label(skew_norm_pdf(30, 5, 1000))
        baz.fit(data)
        baz.decision_function(data)
    exit()

    data = skew_norm_pdf(20, 10, 1000)

    # mat = loadmat(os.path.abspath(os.path.join(os.path.dirname(__file__), '..') + '/data/breastw.mat'))
    # data = mat['X']

    attr1, attr2, labels = detached_data(data)
    data = np.array([list([x[0], x[1]]) for x in data])

    plt.scatter(attr1, attr2, s=None, c = labels)
    plt.show()
    
    #############################

    # Drop labels (answer)
    data = np.array([list([x[0], x[1]]) for x in data])
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=48)

    # fig = px.scatter(df, x="attr1", y="attr2", marginal_x="histogram", marginal_y="rug")
    # fig.show()
        
    clf = COD()
    clf.fit(X_train)
    clf.decision_function(X_train)
    pred_clf_train = clf.labels_
    pred_clf = clf.predict(X_test)

    baz = BAZUOD()
    # baz.fit(data)
    # print('Best contimination: ', baz.get_best_contamincation())

    baz.fit(X_train)
    baz.decision_function(X_train)
    pred_baz_train = baz.labels_
    pred_baz = baz.predict(X_test)

    print('--------------------COPOD (Training)--------------------')
    print('precision: ', metrics.precision_score(pred_clf_train, y_train))
    print('   recall: ', metrics.recall_score(pred_clf_train, y_train))
    print(' f1-score: ', metrics.f1_score(pred_clf_train, y_train))
    print(confusion_matrix(pred_clf_train, y_train))

    print('--------------------BAZUOD (Training)-------------------')
    print('precision: ', metrics.precision_score(pred_baz_train, y_train))
    print('   recall: ', metrics.recall_score(pred_baz_train, y_train))
    print(' f1-score: ', metrics.f1_score(pred_baz_train, y_train))
    print(confusion_matrix(pred_baz_train, y_train))

    print('--------------------COPOD (Testing)--------------------')
    print('precision: ', metrics.precision_score(pred_clf, y_test))
    print('   recall: ', metrics.recall_score(pred_clf, y_test))
    print(' f1-score: ', metrics.f1_score(pred_clf, y_test))
    print(confusion_matrix(pred_clf, y_test))

    print('--------------------BAZUOD (Testing)-------------------')
    print('precision: ', metrics.precision_score(pred_baz, y_test))
    print('   recall: ', metrics.recall_score(pred_baz, y_test))
    print(' f1-score: ', metrics.f1_score(pred_baz, y_test))
    print(confusion_matrix(pred_baz, y_test))

    print('----------------------Evaluation---------------------')
    print('[Train] COPOD  (F1-score): ', metrics.f1_score(pred_clf_train, y_train))
    print('[Train] BAZUOD (F1-score): ', metrics.f1_score(pred_baz_train, y_train))
    print('[Test]  COPOD  (F1-score): ', metrics.f1_score(pred_clf, y_test))
    print('[Test]  BAZUOD (F1-score): ', metrics.f1_score(pred_baz, y_test))

    attr1_train, attr2_train = detached_data(X_train)
    attr1_test, attr2_test = detached_data(X_test)

    # Plot the graph of classification result
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    axs[0, 0].scatter(attr1_train, attr2_train, s=None, c=y_train)
    axs[0, 0].set_title('Ground Truth (Training)')
    axs[1, 0].scatter(attr1_test, attr2_test, s=None, c=y_test)
    axs[1, 0].set_title('Ground Truth (Testing)')
    axs[0, 1].scatter(attr1_train, attr2_train, s=None, c=pred_clf_train)
    axs[0, 1].set_title('COPOD (Training)')
    axs[1, 1].scatter(attr1_test, attr2_test, s=None, c=pred_clf)
    axs[1, 1].set_title('COPOD (Testing)')
    axs[0, 2].scatter(attr1_train, attr2_train, s=None, c=pred_baz_train)
    axs[0, 2].set_title('BAZUOD (Training)')
    axs[1, 2].scatter(attr1_test, attr2_test, s=None, c=pred_baz)
    axs[1, 2].set_title('BAZUOD (Testing)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()
