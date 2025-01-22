import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

pd.set_option('display.max_rows', None)

data_filename = 'data/correctUASDatabase-HGNSPredictorsOfSucc_DATA_LABELS_2025-01-18_1334.csv'

# Columns used in this analysis
usecols=[
    # AHI variables
    'Preop sleep study: AHI',
    'Preop PSG Medicare AHI',
    'Preop PSG AASMI AHI',
    'postop sleep study: AHI',
    'postop PSG Medicare AHI',
    'postop PSG AASMI AHI',
    # Predictor variables
    'Age at time of surgery',
    'Gender',
    'BMI - preop',
    'DISE PAP opening pressure',
    'Oropharynx'
]

def main(df):

  # ==================== BEGIN PREPROCESSING ====================
  # Shorten some column names to be easier to use
  df = df.rename(columns={
          'Age at time of surgery': 'Age',
          'BMI - preop': 'BMI',
          'DISE PAP opening pressure': 'DISE',
  })

  # When sleep study AHI is missing, use Medicare or AASMI AHI instead
  df['preop_AHI'] = (df['Preop sleep study: AHI']
                      .fillna(df['Preop PSG Medicare AHI'])
                      .fillna(df['Preop PSG AASMI AHI']))
  df['postop_AHI'] = (df['postop sleep study: AHI']
                      .fillna(df['postop PSG Medicare AHI'])
                      .fillna(df['postop PSG AASMI AHI']))
  df = df.dropna(subset=['postop_AHI', 'preop_AHI'])
  
  # Compute other success measures
  df['change_AHI'] = df['postop_AHI'] - df['preop_AHI']
  df['Sher15'] = (((df['postop_AHI'] <= 0.5*df['preop_AHI'])
                  & (df['postop_AHI'] < 15))).astype('float')
  
  # After this, we no longer need the original distinct AHIs
  df = df.drop(columns=[
      'Preop sleep study: AHI',
      'Preop PSG Medicare AHI',
      'Preop PSG AASMI AHI',
      'postop sleep study: AHI',
      'postop PSG Medicare AHI',
      'postop PSG AASMI AHI',
  ])
  
  # There's one datapoint with preop_AHI = 178 which seems like an error (too large);
  # I've dropped it for this analysis.
  df = df[df['preop_AHI'] < 175]
  
  # Since DISE PAP opening pressure is >90% missing, we need some way to handle this;
  # for regression, replacing with the mean value is reasonable if we assume the values
  # are "missing completely at random" (MCAR); if not, we should impute by regression.
  df['DISE'] = df['DISE'].fillna(df['DISE'].mean())

  # Drop any rows with remaining missing values
  df = df.dropna(how='any')
  
  # ===================== END PREPROCESSING =====================

  # Print a summary of the data after preprocessing
  print(df.describe())
  print(df['Gender'].value_counts())
  print(df['Oropharynx'].value_counts())
  print(df['Sher15'].value_counts())

  # ================= BEGIN REGRESSION ANALYSES =================
  # Since postop_AHI is a count, use Poisson regression
  postop_AHI_model = smf.poisson(
      formula=('postop_AHI ~ Age + C(Gender, Treatment(reference="Male")) + BMI + DISE'
               ' + C(Oropharynx, Treatment(reference=0)) + preop_AHI'),
      data=df
  ).fit()
  print(postop_AHI_model.summary())
  
  # Since change_AHI is continuous, use linear regression
  change_AHI_model = smf.ols(
      formula=('change_AHI ~ Age + C(Gender, Treatment(reference="Male")) + BMI + DISE'
               ' + C(Oropharynx, Treatment(reference=0)) + preop_AHI'),
      data=df
  ).fit()
  print(change_AHI_model.summary())
  
  # Since Sher15 is boolean, use logistic regression
  Sher15_model = smf.logit(
      formula=('Sher15 ~ Age + C(Gender, Treatment(reference="Male")) + BMI + DISE'
               ' + C(Oropharynx, Treatment(reference=0)) + preop_AHI'),
      data=df
  ).fit()
  print(Sher15_model.summary())
  # ================== END REGRESSION ANALYSES ==================
  
  sns.pairplot(df)
  plt.gcf().subplots_adjust(bottom=0.05)
  pylab.show()


if __name__ == "__main__":
  # Read in just the columns we need from the data table
  df = pd.read_csv(data_filename, usecols=usecols)
  main(df)
