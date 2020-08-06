# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:03:01 2020

@author: youss
"""
import os
from typing import Text, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor as gbm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from scipy import cluster
from sklearn import preprocessing
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import random
import umap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm


def drug_eff(studies):
    study = studies[0].study
    name = "drugg_effect_model_summary_study_"+ studies[0].name
    plot = "study_"+ studies[0].name
    if(len(studies) > 1):
        for st in studies[1:]:
              study = pd.concat([study, st.study])
        name = "drugg_effect_model_summary_all_studies.txt"
        plot = "all studies"

    passed = study[study["LeadStatus"] == 'Passed']
    #Testing the drug effect doing some statistical analysis:
    model = ols(formula = "PANSS_Total ~ VisitDay + TxGroup + VisitDay:TxGroup", data = passed).fit()
    ols_summary = model.summary() 
    with open(name + ".txt", "w") as f:
        f.write(ols_summary.as_text())
    print(ols_summary)

    #Testing quadratic relation:
    polynomial_features = PolynomialFeatures(degree=2)
    x = passed[["VisitDay", "TxGroup"]]
    xp = polynomial_features.fit_transform(x)
    model_quad = sm.OLS(passed["PANSS_Total"], xp).fit()
    ols_summary = model_quad.summary() 
    with open(name + "_quad.txt", "w") as f:
        f.write(ols_summary.as_text())
    print(ols_summary)
    
    #only choosing the passed assessments
    grp = passed.pivot_table(index='VisitDay',columns='TxGroup',aggfunc=np.mean)
    
    panss = grp["PANSS_Total"]
    panss.fillna(method='bfill',inplace=True)
    win = 10
    MA = panss.rolling(window=win).mean()
    fig, ax = plt.subplots(1,2,figsize=(16,7))
    
    ax[0].plot(panss.index, panss[0], '--m', label = 'Control')
    ax[0].plot(panss.index, panss[1], '--y', label = 'Treatment')
    ax[0].set_xlabel("VisitDay")
    ax[0].set_ylabel("Mean PANSS score")
    ax[0].set_title("Evolution of mean PANSS score with VisitDay for both groups " + plot)
    ax[0].legend()
    
    ax[1].plot(panss.index, MA[0], '--m', label = 'Control')
    ax[1].plot(panss.index, MA[1], '--y', label = 'Treatment')
    ax[1].set_xlabel("VisitDay")
    ax[1].set_ylabel("Moving average (w="+str(win)+") PANSS score")
    ax[1].set_title("Evolution of MA (w="+str(win)+") PANSS score with VisitDay for both groups " + plot)
    ax[1].legend()
    
    plt.show() 
    

class Study(object):

  def __init__(self, name, figsize: Tuple[int, int]=(10, 10)):
    """Constructor.

    Attributes:
      data_path: A str indicating the location of the data.         
      figsize: A tuple specifying the default size for figures.
    """
    self.name = name
    self.data_path = "Study_"+name+".csv"
    # assert os.path.exists(self.data_path), "Study " + name + " dataset not found."
    self.study = pd.read_csv(self.data_path)
    self.figsize = figsize

  def preprocess(self):
    """
    Preprocess the study data.
    """
    #Describing the data:
    print(self.study.describe(include = 'all'))
    with open("study_"+ self.name + "_description.txt", "w") as text_file:
        self.study.describe().to_string(text_file)
    
    #Assign 0 to Control and 1 to Treatment group:
    self.study.TxGroup = self.study.TxGroup.apply(
      lambda x: 1 if x=="Treatment" else 0)
    #We start by removing the abnormal data, like multiple assessment,...
    subsets = [['PatientID', 'VisitDay'], ['AssessmentID']]
    for subset in subsets:
        duplicate = self.study.duplicated(keep='last', subset=subset)  
        self.study = self.study[duplicate == False]
        
    #Patients that came only once
    # duplicate = self.study.duplicated(keep=False, subset=['PatientID', 'TxGroup'])  
    # self.study = self.study[duplicate == True]
    
    #Check if there is any NaN:
    NaN_col = (self.study.isnull() == True).sum()
    if NaN_col.sum() != 0:
        self.study.dropna(how = 'any', inplace = True)
    
    num_visit = self.study.pivot_table(index=['PatientID'], aggfunc='size').values
    num_visit = np.repeat(num_visit, num_visit)
    self.study['numVisit'] = num_visit

    #Histograms of the data:
    self.study.hist(figsize=(20,20))
    
    #Describing the data:
    print(self.study.describe(include = 'all'))
    with open("study_"+ self.name + "_description_after_processing.txt", "w") as text_file:
        self.study.describe().to_string(text_file)
    
    #Check columns variance:
    std = self.study.std()
    print("Columns variance:")
    print(std)   
    std.to_csv("std_"+self.name+".csv")
    
    #scatter_matrix:
    # excluded_col = ['Study', 'Country', 'PatientID', 'SiteID', 'RaterID', 'AssessmentID',
    #                 'P1','P2','P3','P4','P5','P6','P7','N1','N2','N3','N4','N5','N6','N7',
    #                 'G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12','G13',
    #                 'G14','G15','G16']
    # excluded_col = ['Study', 'Country', 'PatientID', 'SiteID', 'RaterID', 'AssessmentID',
    #                 'P1','P2','P3','P4','P5','P6','P7']
    # excluded_col = ['Study', 'Country', 'PatientID', 'SiteID', 'RaterID', 'AssessmentID',
    #                 'N1','N2','N3','N4','N5','N6','N7']
    # excluded_col = ['Study', 'Country', 'PatientID', 'SiteID', 'RaterID', 'AssessmentID',
    #                 'G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12','G13','G14',
    #                 'G15','G16']

    # scatter_matrix(self.study.loc[:, ~self.study.columns.isin(excluded_col)], figsize = (20,20))
    
    #corr matrix:
    corr_matrix = self.study.corr()
    
    corr = corr_matrix['PANSS_Total']
    ind = corr.index
    corr = corr[ind != 'PANSS_Total']
    corr = corr.sort_values(ascending = True)
    xs = corr.plot.barh()
    fig, ax = plt.subplots(figsize=(10,30))
    ax = corr.plot.barh(color = 'gold')
    ax.set_xlabel("Correlation coefficient")
    ax.set_title("Correlation coefficient of the variables")
    plt.show()
    # corr_matrix.to_csv("corr_matrix_study_"+self.name+".csv")
    
    #Heatmap for important variables:
    corr_limit = 0.6
    ind_imp = corr.index[(corr >= corr_limit) | (corr<= -corr_limit)]
    fig, ax = plt.subplots(figsize=(17,17))   
    corr_imp = corr_matrix.loc[ind_imp, ind_imp]
    if corr_imp.size != 0:
        sns.heatmap(corr_imp, vmax = 1)
        plt.title("Important variables correlation map", fontsize=15)
        plt.show()
    
    
        
    #important parameters using randomForest, boosting,...:
    #encoding the LeadStatus variable:
    ohe_df = pd.get_dummies(self.study.LeadStatus, prefix='LeadStatus')
    ohe_df.reset_index(drop=True, inplace=True)
    self.study.reset_index(drop=True, inplace=True)
    data = pd.concat([self.study, ohe_df], axis=1).drop(['LeadStatus'], axis=1)
    self.study_ohe = data
    # data = self.study
    
    xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
    }
    train_df = data.loc[:, ~data.columns.isin(['Study', 'Country', 'PatientID', 'AssessmentID',
                                                    'PANSS_Total', 'LeadStatus'])]
    train_y = data.PANSS_Total    
    
    dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
    # plot the important features #
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(12,18))
    reg = gbm(random_state=0)
    reg.fit(train_df, train_y)
    imp_feat = reg.feature_importances_
    indices_sorted = imp_feat.argsort()
    imp_feat = imp_feat[indices_sorted]
    feat = train_df.columns[indices_sorted]
    ax.barh(feat, imp_feat)
    plt.show()

  def drug_effect(self):
    """
    Verifying the drug effect.

    Returns
    -------
    None.

    """  
    #using only the current study:
    drug_eff([self])
   
    
    
  def classify_patients(self, k = 5, t = 'l'):
      """
      Classify the patient during the first visit to k different groups.

      Parameters
      ----------
      k : TYPE, optional
          DESCRIPTION. The default is 5.

      Returns
      -------
      None.

      """
      vis_0 = self.study_ohe[self.study_ohe["VisitDay"] == 0]
      pos = vis_0.loc[:, vis_0.columns.isin(['P1','P2','P3','P4','P5','P6','P7'])].sum(axis=1)
      neg = vis_0.loc[:, vis_0.columns.isin(['N1','N2','N3','N4','N5','N6','N7'])].sum(axis=1)
      gen = vis_0.loc[:, vis_0.columns.isin(['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12','G13','G14',
                                             'G15','G16'])].sum(axis=1)
      
      new_feat = pd.DataFrame({'pos':pos, 'neg':neg, 'gen':gen})
      
      # data = self.study_ohe.drop(['Study', 'Country'], axis=1)
      # scaled_var = preprocessing.scale(data)
      # figure,_ = plt.subplots(figsize=(10, 10))
      # clusters = cluster.hierarchy.linkage(scaled_var, 'complete')
      # dendo = cluster.hierarchy.dendrogram(clusters, labels = data.index)
      
      #pos sum vs. neg sum
      X = new_feat.loc[:, ['pos', 'neg']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.pos, X.neg, c=y_pred)
      plt.xlabel('Positive score')
      plt.ylabel('Negative score')
      plt.title("Clustering based on positive and negative rates")
      plt.show()
      
      #pos sum vs. gen sum
      X = new_feat.loc[:, ['pos', 'gen']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.pos, X.gen, c=y_pred)
      plt.xlabel('Positive score')
      plt.ylabel('General score')
      plt.title("Clustering based on positive and general rates")
      plt.show()
      
      #neg sum vs. gen sum
      X = new_feat.loc[:, ['neg', 'gen']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.neg, X.gen, c=y_pred)
      plt.xlabel('Negative score')
      plt.ylabel('General score')
      plt.title("Clustering based on negative and general rates")  
      plt.show()
      
      #P2 vs PANSS_Total
      X = vis_0.loc[:, ['P2', 'PANSS_Total']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.P2, X.PANSS_Total, c=y_pred)
      plt.xlabel('P2')
      plt.ylabel('PANSS_Total')
      plt.title("Clustering based on P2 score and Total score")  
      plt.show()   
      
      #P6 vs PANSS_Total
      X = vis_0.loc[:, ['P6', 'PANSS_Total']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.P6, X.PANSS_Total, c=y_pred)
      plt.xlabel('P6')
      plt.ylabel('PANSS_Total')
      plt.title("Clustering based on P6 score and Total score")  
      plt.show()  
      
      
      #P2 vs P6
      X = vis_0.loc[:, ['P2', 'P6']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.P2, X.P6, c=y_pred)
      plt.xlabel('P2')
      plt.ylabel('P6')
      plt.title("Clustering based on P2 score and P6 score")  
      plt.show()   
      
      #P2 vs. P6 vs. PANSS_Total:
      fig = plt.figure(1)
      fig.clf()
      ax = Axes3D(fig)
      X = vis_0.loc[:, ['P2', 'P6', 'PANSS_Total']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      ax.scatter(X.P2, X.P6, X.PANSS_Total, c=y_pred)
      ax.set_xlabel('P2')
      ax.set_ylabel('P6')
      ax.set_zlabel('PANSS_Total')
      ax.set_title("Clustering based on P2 score, P6 score and the total score")  
      plt.show()   
      
      #P2 vs. N2 vs. PANSS_Total:
      fig = plt.figure(1)
      fig.clf()
      ax = Axes3D(fig)
      X = vis_0.loc[:, ['P2', 'N2', 'PANSS_Total']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      ax.scatter(X.P2, X.N2, X.PANSS_Total, c=y_pred)
      ax.set_xlabel('P2')
      ax.set_ylabel('N2')
      ax.set_zlabel('PANSS_Total')
      ax.set_title("Clustering based on P2 score, N2 score and the total score")  
      plt.show()  
      
      
      #P2 vs. G9 vs. PANSS_Total:
      fig = plt.figure(1)
      fig.clf()
      ax = Axes3D(fig)
      X = vis_0.loc[:, ['P2', 'G9', 'PANSS_Total']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      ax.scatter(X.P2, X.G9, X.PANSS_Total, c=y_pred)
      ax.set_xlabel('P2')
      ax.set_ylabel('G9')
      ax.set_zlabel('PANSS_Total')
      ax.set_title("Clustering based on P2 score, G9 score and the total score")  
      plt.show()  
      
      
      #RaterID effect on total score:
      X = vis_0.loc[:, ['RaterID', 'PANSS_Total']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.RaterID, X.PANSS_Total, c=y_pred)
      plt.xlabel('RaterID')
      plt.ylabel('PANSS_Total')
      plt.title("Clustering based on RaterID and Total score")  
      plt.show()      
  
    
      #SiteID effect on total score:
      X = vis_0.loc[:, ['SiteID', 'PANSS_Total']]
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(X)
      plt.scatter(X.SiteID, X.PANSS_Total, c=y_pred)
      plt.xlabel('SiteID')
      plt.ylabel('PANSS_Total')
      plt.title("Clustering based on SiteID and Total score")  
      plt.show()      
    
  def umap(self):
      """
      Clustering the data using UMAP.

      Returns
      -------
      None.

      """
      study = self.study.loc[:, ~self.study.columns.isin(['Study', 'Country', 'PatientID', 'AssessmentID',
                                                    'LeadStatus'])].values
      study_scaled = StandardScaler().fit_transform(study)
      reducer = umap.UMAP()
      embedding = reducer.fit_transform(study_scaled)
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(study_scaled)
      plt.scatter(embedding[:,0], embedding[:,1], c=y_pred, cmap='Spectral')
      plt.gca().set_aspect('equal', 'datalim')
      plt.title('UMAP projection of the Patients dataset')
      plt.show()
      
      study = self.study.loc[:, (['P2', 'P6', 'N2', 'G9', 'PANSS_Total'])].values
      study_scaled = StandardScaler().fit_transform(study)
      reducer = umap.UMAP()
      embedding = reducer.fit_transform(study_scaled)
      y_pred = KMeans(n_clusters=4, random_state=170).fit_predict(study_scaled)
      plt.scatter(embedding[:,0], embedding[:,1], c=y_pred, cmap='Spectral')
      plt.gca().set_aspect('equal', 'datalim')
      plt.title('UMAP projection of the Patients dataset')  
      plt.show()
      
      
  def forecast(self, studies):
      """
      Predict the Total score at the 18th week.

      Returns
      -------
      None.

      """
      study = studies[0].study
      for st in studies[1:]:
          study = pd.concat([study, st.study])
      study = studies[-1].study
      sorted_data = study.sort_values(by = ['PatientID', 'VisitDay'])
      train_X = sorted_data.loc[:,
                   ~study.columns.isin(['Study', 'Country', 'AssessmentID',
                                                    'LeadStatus', 'PANSS_Total'])]
      duplicate = train_X.duplicated(keep='last', subset=['PatientID'])  
      train_X = train_X[duplicate == True]
      
      train_y = sorted_data[['PANSS_Total', 'PatientID']]
      duplicate = train_y.duplicated(keep='first', subset=['PatientID'])  
      train_y = train_y[duplicate == True]
      
      duplicate = train_X.duplicated(keep='last', subset=['PatientID'])  
      test_X = train_X[duplicate == False]
      train_X = train_X[duplicate == True]
      
      duplicate = train_y.duplicated(keep='first', subset=['PatientID'])  
      test_y = train_y[duplicate == False]
      train_y = train_y[duplicate == True]
      
      train_X.drop('PatientID', axis=1, inplace = True)
      train_y.drop('PatientID', axis=1, inplace = True)
      test_X.drop('PatientID', axis=1, inplace = True)
      test_y.drop('PatientID', axis=1, inplace = True)
      
      xgbr = xgb.XGBRegressor() 
      xgbr.fit(train_X, train_y)
      
      #testing on Study_E:
      studyE = studies[-1].study
      sorted_data = studyE.sort_values(by = ['PatientID', 'VisitDay'])
      test_X_E = sorted_data.loc[:,
                   ~studyE.columns.isin(['Study', 'Country', 'AssessmentID',
                                                    'LeadStatus', 'PANSS_Total'])]
      test_X_E.drop_duplicates(keep='last', subset=['PatientID'], inplace= True)  
      patientsID_E = test_X_E["PatientID"]
      test_X_E.drop('PatientID', axis=1, inplace = True)
      
      
      
      pred_y_ts = xgbr.predict(test_X)
      pred_y_tr = xgbr.predict(train_X)
      score_ts = xgbr.score(test_X, test_y.PANSS_Total)
      score_tr = xgbr.score(train_X, train_y)
      
      print("Training results:")
      print(train_y)
      print(pred_y_tr)
      print(np.mean((train_y.values - pred_y_tr)**2))
      
      
      print("Test results:")
      print(test_y)
      print(pred_y_ts)
      print(np.mean((test_y.values - pred_y_ts)**2))
      
      pred_y_E = xgbr.predict(test_X_E)
      results = pd.DataFrame({"PatientID": patientsID_E, "PANSS_Total": pred_y_E})
      results.to_csv("submission_PANSS_2.csv")
      
      
      
      
      
      
      
    
    

study_A = Study(name = 'A')
study_A.preprocess()
study_A.drug_effect()
# study_A.classify_patients()
# study_A.umap()


study_B = Study(name = 'B')
study_B.preprocess()
study_B.drug_effect()

study_C = Study(name = 'C')
study_C.preprocess()
study_C.drug_effect()

study_D = Study(name = 'D')
study_D.preprocess()
study_D.drug_effect()

# study_E = Study(name = 'E')
# study_E.preprocess()

# studies = [study_A, study_B, study_C, study_D, study_E]
studies = [study_A, study_B, study_C, study_D]
drug_eff(studies)

# study_A.forecast(studies)
