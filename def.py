import csv
import esmvalcore.preprocessor as eprep
import iris
from iris.util import equalise_attributes
import iris.plot as iplt
import cf_units
import cftime 
import cartopy.crs as ccrs 
import climextremes as cex
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import genextreme as gev
from scipy import signal
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import random
import statsmodels.api as sm # import statsmodels

def imperfect_model_test(model_his,model_fu):
    pre_series=[]
    obs_series=[]
    width_series=[]
    n=0
   
    for i in range(len(model_his)):
        obs_trend=model_his[i]
        truth=model_fu[i]
        obs_series=np.append(obs_series,truth)
        
        train_his=np.delete(model_his,i)
        train_fu=np.delete(model_fu,i)
        train_his = train_his.reshape(-1, 1)
        train_fu = train_fu.reshape(-1, 1)
        obs_trend=obs_trend.reshape(-1, 1)
        
   
        model = LinearRegression()
        model.fit(train_his, train_fu)
        best_est = model.predict(obs_trend)
        pre_series=np.append(pre_series,best_est)

        y_pred = model.predict(train_his)  # X is your predictor data
        residuals = train_fu - y_pred  # y_true is your actual response data
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        predicted_values_distribution = norm(loc=residual_mean, scale=residual_std)

        confidence_level = 0.90  # 90% confidence level (5% to 95%)
        alpha = 1 - confidence_level

        lower_quantile = predicted_values_distribution.ppf(alpha / 2)
        upper_quantile = predicted_values_distribution.ppf(1 - alpha / 2)
        CI5 = best_est + lower_quantile
        CI95 = best_est + upper_quantile
        
        width=CI95-CI5
        width_series=np.append(width_series,width)
        
        if CI5 <= truth <= CI95:
            n=n+1

    # dif mean
    cor_best=stats.pearsonr(obs_series,pre_series)[0]
    rmse_best = rmse(obs_series, pre_series)
    width_mean=np.mean(width_series)
    ratio=(n/len(model_fu))*100
    param_dic = {'cor_best': cor_best, 'rmse_best': rmse_best, 'width_mean': width_mean, 'ratio':ratio}
    return param_dic


def emergent_relationship(model_his_raw,model_his_no_etp, model_fu):
    corr_raw=stats.pearsonr(model_his_raw,model_fu)[0]
    corr_no_etp=stats.pearsonr(model_his_no_etp,model_fu)[0]
    param_dic = {'raw_corr': corr_raw, 'no_etp_corr': corr_no_etp}

    return param_dic

def obs_cons(obs_raw,obs_no_etp,model_his_raw,model_his_no_etp, model_fu,ssp):
    model_raw = LinearRegression()
    model_his_raw=model_his_raw.reshape(-1, 1)
    model_fu=model_fu.reshape(-1, 1)
    Xobs=np.array([1, obs_raw])
    obs_raw=obs_raw.reshape(-1, 1)
    model_raw.fit(model_his_raw, model_fu)
    best_est_raw = model_raw.predict(obs_raw)

    y_pred = model_raw.predict(model_his_raw)  # X is your predictor data
    residuals = model_fu- y_pred  # y_true is your actual response data
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    mod_his_raw=[float(item) for sublist in model_his_raw for item in sublist]
    list2_r = [list(a) for a in zip(mod_his_raw)]
    list2_r = sm.add_constant(list2_r)
    
    aa=np.dot(list2_r.transpose(),list2_r)
    aa_re=np.linalg.inv(aa)
    bb=np.dot(Xobs,aa_re)
    cc=np.dot(bb,Xobs.transpose())
    S=np.sqrt(np.sum(np.square(residuals)/(len(model_his_raw)-2)))
    scale=S*np.sqrt(cc+1)
    samp = norm.rvs(loc=best_est_raw, scale=scale, size=150)
    param = norm.fit(samp)

    predicted_values_distribution = norm(loc=best_est_raw, scale=residual_std)
    
    if ssp=='ssp126':
        x_values = np.linspace(-0.5, 3, 1000)
    if ssp=='ssp245':
        x_values = np.linspace(0, 4, 1000)
    if ssp=='ssp585':
        x_values = np.linspace(1, 8, 1000)
    if ssp=='ssp585 canada':
        x_values = np.linspace(2, 18, 1000) 
    pdf_values_raw = norm.pdf(x_values,loc=param[0],scale=param[1])
    cdf_values_raw = norm.cdf(x_values,loc=param[0],scale=param[1])

    model_no_etp = LinearRegression()
    model_his_no_etp=model_his_no_etp.reshape(-1, 1)
    model_no_etp.fit(model_his_no_etp, model_fu)
    Xobs=np.array([1, obs_no_etp])
    obs_no_etp=obs_no_etp.reshape(-1, 1)
    best_est_no_etp = model_no_etp.predict(obs_no_etp)

    y_pred = model_no_etp.predict(model_his_no_etp)  # X is your predictor data
    residuals = model_fu- y_pred  # y_true is your actual response data
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    predicted_values_distribution = norm(loc=best_est_no_etp, scale=residual_std)
    
    mod_his_no_etp = [float(item) for sublist in model_his_no_etp for item in sublist]
    list2_r = [list(a) for a in zip(mod_his_no_etp)]
    list2_r = sm.add_constant(list2_r)
    # Xobs=np.array([1, obs_no_etp])
    aa=np.dot(list2_r.transpose(),list2_r)
    aa_re=np.linalg.inv(aa)
    bb=np.dot(Xobs,aa_re)
    cc=np.dot(bb,Xobs.transpose())
    S=np.sqrt(np.sum(np.square(residuals)/(len(model_his_no_etp)-2)))
    scale=S*np.sqrt(cc+1)
    samp = norm.rvs(loc=best_est_no_etp, scale=scale, size=150)
    param = norm.fit(samp)

    pdf_values_no_etp = norm.pdf(x_values,loc=param[0],scale=param[1]) 
    cdf_values_no_etp = norm.cdf(x_values,loc=param[0],scale=param[1])

    uncons_mean = np.mean(model_fu)
    uncons_std = np.std(model_fu,ddof=2)
    predicted_values_distribution = norm(loc=uncons_mean, scale=uncons_std)
    pdf_values_uncons = predicted_values_distribution.pdf(x_values) 
    cdf_values_uncons = predicted_values_distribution.cdf(x_values)  
    # uncons_percentile = [np.percentile(cdf_values_uncons, 5),uncons_mean, np.percentile(cdf_values_uncons, 95)]
    param_dic = {'pdf_values_raw': pdf_values_raw, 'pdf_values_no_etp': pdf_values_no_etp,'pdf_values_uncons': pdf_values_uncons, 'cdf_values_raw': cdf_values_raw, 'cdf_values_no_etp': cdf_values_no_etp,'cdf_values_uncons': cdf_values_uncons}
    return param_dic


def plot_obs_cons_pdf(x,tas,tas_remove,un_mean,cd_raw_mean,cd_no_etp_mean,cd_uncons_mean,ssp):
    fig = plt.figure(figsize=(8, 4)) 
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.plot(x, un_mean,color="k", alpha=0.8,linewidth=2,label='Unconstrained')
    ##ax1.fill_between(x, un_pb_min.flatten(), un_pb_max.flatten() ,color='k',alpha=.2)
    ax1.plot(x, tas,color="b", alpha=0.8,linewidth=2,label='Constrained with raw GSAT trend metric')
    ##ax1.fill_between(x, tas_min.flatten(), tas_max.flatten() ,color='b',alpha=.2)
    ax1.fill_between(x, tas.flatten(), color='b',alpha=.2)
    ax1.plot(x, tas_remove,color="r", alpha=0.8,linewidth=2,label='Constrained with GSAT trend metric \n with ETP-congruent variability regressed out')
    ##ax1.fill_between(x, tas_remove_min.flatten(), tas_remove_max.flatten() ,color='r',alpha=.2)
    ax1.fill_between(x, tas_remove.flatten() ,color='r',alpha=.2)
    ##ax1.plot(x, zs_mean,color="green", alpha=0.8, linestyle='--',label='Constrained with cloud metrics')
    
    raw_precentile = [x[np.argmin(np.abs(cd_raw_mean - 0.05))],x[np.argmin(np.abs(cd_raw_mean - 0.5))],x[np.argmin(np.abs(cd_raw_mean - 0.95))]]
    noetp_precentile = [x[np.argmin(np.abs(cd_no_etp_mean - 0.05))],x[np.argmin(np.abs(cd_no_etp_mean - 0.5))],x[np.argmin(np.abs(cd_no_etp_mean - 0.95))]]
    un_precentile = [x[np.argmin(np.abs(cd_uncons_mean - 0.05))],x[np.argmin(np.abs(cd_uncons_mean - 0.5))],x[np.argmin(np.abs(cd_uncons_mean - 0.95))]]

    ax1.set_ylabel('Probability density')
    ax1.set_ylim(0,1.65)
    
    plt.tight_layout()
    plt.savefig('constrained_projefction.png', dpi=300)
    return

def scatter_plot_constrained(modens_fu, modens_gsatslope, modens_gsatslope_noetp,obs_no_etp,obs_gsat,tas,tas_remove,x_values,ssp):
    plt.figure(figsize=(8, 9))  # Set the figure size
    gsatslope=[]
    fu=[]
    gsatslope_noetp=[]
    x=np.linspace(0.01, 0.038, 100)
    for model in modens_fu.keys():
        gsatslope =np.append(gsatslope,modens_gsatslope[model]['mean'])
        fu =np.append(fu,modens_fu[model]['mean'])
        gsatslope_noetp =np.append(gsatslope_noetp,modens_gsatslope_noetp[model]['mean'])

        plt.errorbar(modens_gsatslope[model]['mean'],modens_fu[model]['mean'], xerr=modens_gsatslope[model]['std'], yerr=modens_fu[model]['std'], fmt='o', markersize=5, color='b', ecolor='b')
        plt.errorbar(modens_gsatslope_noetp[model]['mean'],modens_fu[model]['mean'], xerr=modens_gsatslope_noetp[model]['std'], yerr=modens_fu[model]['std'], fmt='o', markersize=5, color='r', ecolor='r')
    
    slope, intercept = np.polyfit(gsatslope, fu, 1)
    plt.plot(x, slope * x + intercept, color='b',label='Raw GSAT trend as predictor')
    
    slope, intercept = np.polyfit(gsatslope_noetp, fu, 1)
    plt.plot(x, slope *x + intercept, color='r',label='GSAT trend with ETP-congruent variability removed')
    
    plt.axvline(x=obs_no_etp, color='r', linestyle='--', label=f'GSAT trend with ETP-congruent variability removed (obs)')
    plt.axvline(x=obs_gsat, color='b', linestyle='--', label=f'Raw GSAT trend (obs)')

    plt.xlabel('GSAT trend (K/y)')
    plt.ylabel('GSAT changes (K)')

    plt.xlim((0,0.05))
    if ssp=='ssp245':
        plt.ylim((0,4))
        plt.title('SSP 2-4.5')
    
    if ssp=='ssp585':
        plt.ylim((2.5,6.5)) 
        plt.title('SSP 5-8.5') 

    if ssp=='ssp126':
        plt.ylim((0,3)) 
        plt.title('SSP 1-2.6')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('scatter_constrained.png', dpi=300)
    
    return
