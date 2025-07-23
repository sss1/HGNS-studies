import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 4)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:

  # Remove rows without treatment assignment (mostly some empty rows)
  df = df.dropna(subset='CPAP or UAS')

  # Standardize coding of 'CPAP'
  df.loc[df['CPAP or UAS'] != 'UAS', 'CPAP or UAS'] = 'CPAP'

  # Fix an inconsistency in race coding
  df.loc[df['Race'] == 'White or Caucasian', 'Race'] = 'White/Caucasian'

  return df

def propensity_score_match(demographics_filename: str) -> pd.DataFrame:
  
  # Read in just the data we need from the Excel file
  usecols=[
      # Patient ID
      'Patient #',
      # Demographic matching variables
      'Age at Intervention Start', 'Male (0)/ Female (1)', 'Race',
      # Baseline symptom matching variables
      # 'Avg RDI', 'AVG Nadir', 
      'AVG AHI',
      # Treatment variable
      'CPAP or UAS',  
  ]
  df = pd.read_excel(demographics_filename, usecols=usecols, index_col='Patient #')
  
  df = preprocess_df(df)

  # Mean imputation for a small number of missing baseline AHIs
  df['AVG AHI'] = df['AVG AHI'].fillna(value=df['AVG AHI'].mean())

  df = df.rename(columns={
      'Age at Intervention Start': 'Age',
      'Male (0)/ Female (1)': 'IsFemale',
      'AVG AHI': 'Baseline AHI',
      'CPAP or UAS' : 'Treatment',
  })

  # Compute propensity scores
  matching_variables = ['Age', 'IsFemale', 'Baseline AHI']
  X = df[matching_variables]
  y = (df['Treatment'] == 'UAS')
  log_reg = LogisticRegression()
  log_reg.fit(X, y)
  df['Propensity Score'] = log_reg.predict_proba(X)[:, 1]

  # Split control and treatment groups
  CPAP_df = df[df['Treatment'] == 'CPAP']
  UAS_df = df[df['Treatment'] == 'UAS']

  # Nearest neighbor matching
  nn = NearestNeighbors(n_neighbors=1)
  nn.fit(CPAP_df[['Propensity Score']])
  _, indices = nn.kneighbors(UAS_df[['Propensity Score']])
  CPAP_df = CPAP_df.iloc[indices.flatten()]

  # Combine matched control and treatment groups
  matched_df = pd.concat([CPAP_df, UAS_df])

  return matched_df


if __name__ == "__main__":
  print(propensity_score_match('data/watchpat_with_AGE.xlsx'))
