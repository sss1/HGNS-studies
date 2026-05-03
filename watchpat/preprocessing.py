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

_DATA_FILENAME = 'data/WatchPATStudyData.xlsx'
_DEMOGRAPHICS_FILENAME = 'data/watchpat_with_AGE.xlsx'
_MATCHING_VARIABLES = ['IsFemale', 'Age', 'BaselineAHI']
_DEPENDENT_VARIABLES = [
    'ESS Total', 'General Productivity', 'Social Outcome', 'Activity Level',
    'Vigilance', 'Intimate Relationship/Sexual Activity', 'FOSQ Score',
    'Avg RDI', 'AVG Nadir', 'AVG AHI', 'Adjusted Adherence',
    'Treatment Efficacy', 'MDA',
]


def preprocess_df(df: pd.DataFrame, is_demographics_df: bool = False) -> pd.DataFrame:

  # Remove rows without treatment assignment (mostly some empty rows)
  df = df.dropna(subset='CPAP or UAS')

  # Standardize coding of 'CPAP'
  df.loc[df['CPAP or UAS'] != 'UAS', 'CPAP or UAS'] = 'CPAP'

  # Fix an inconsistency in race coding
  df = df.replace(to_replace={'White or Caucasian': 'White/Caucasian'})

  if is_demographics_df:
    return df

  # Since patient age was not included in the main WatchPAT dataset, read it
  # from the demographics dataset and join to main dataset.
  df = add_age(df)

  # Drop a lot of the columns we don't need for these analyses
  columns_to_drop = [
      'Ethnicity',
      'Six Month Data ', 'Twelve Mo. Data', '18 months', '24 months',
      'AHI', 'AHI.1', 'AHI.2', 'AHI.3',
  ]
  for column_name in df.columns:
    for substring in ['Unnamed', 'Date', 'Time', 'Night', 'usage', 'Used', 'TST', 'Start', 'Adherent']:
      if substring in column_name:
        columns_to_drop.append(column_name)
  df = df.drop(columns=columns_to_drop)

  df = df.rename(columns={
      # Append a '.0' identifier to the end of each first measurement column name
      # Also, fix several typos in the column names.
      'ESS Total'                             : 'ESS Total.0',
      'General Productivity'                  : 'General Productivity.0',
      'Social Outcome'                        : 'Social Outcome.0',
      'Activity Level'                        : 'Activity Level.0',
      'Vigilance'                             : 'Vigilance.0',
      'Intimate Relationship/Sexual Activity' : 'Intimate Relationship/Sexual Activity.0',
      'FOSQ Score'                            : 'FOSQ Score.0',
      'Avg RDI'                               : 'Avg RDI.0',
      'AVG Nadir'                             : 'AVG Nadir.0',
      'AVG AHI'                               : 'AVG AHI.0',
      # Some variables were only measured after treatment start; index these 1-4 instead of 0-4
      'Adjusted compliance '                  : 'Adjusted Adherence.1',
      'Adjusted compliance .1'                : 'Adjusted Adherence.2',
      'Adjusted compliance .2'                : 'Adjusted Adherence.3',
      'Adjusted compliance .3'                : 'Adjusted Adherence.4',
      'Treatment Efficacy'                    : 'Treatment Efficacy.1',
      'Treatment Effiacy'                     : 'Treatment Efficacy.2',
      'Treatment Effiacy.1'                   : 'Treatment Efficacy.3',
      'Treatment Efficacy.1'                  : 'Treatment Efficacy.4',
      'MDA'                                   : 'MDA.1',
      'MDA.1'                                 : 'MDA.2',
      'MDA.2'                                 : 'MDA.3',
      'MDA.3'                                 : 'MDA.4',
      'CPAP or UAS'                           : 'Treatment',
      'Male (0)/ Female (1)'                  : 'IsFemale',
  })

  # Baseline AHI is used both as a DV and a matching variable
  df['BaselineAHI'] = df['AVG AHI.0']

  # Since 18 and 24 month timepoints are missing a lot of data,
  # Pool 18 and 24 month values into '18+' months; specifically:
  # 1) If 18 is missing, use 24
  # 2) Else if 24 is missing use 18
  # 3) Else (both are present) use average
  for dv in _DEPENDENT_VARIABLES:
    df[f'{dv}.3'] = df[f'{dv}.3'].fillna(df[f'{dv}.4'])
    df[f'{dv}.4'] = df[f'{dv}.4'].fillna(df[f'{dv}.3'])
    df[f'{dv}.3'] = (df[f'{dv}.3'] + df[f'{dv}.4'])/2
    df = df.drop(columns=f'{dv}.4')

  return df, _MATCHING_VARIABLES, _DEPENDENT_VARIABLES

def add_age(base_df: pd.DataFrame) -> pd.DataFrame:
  """Since patient age was not included in the main WatchPAT dataset, read it
     from the demographics dataset and join to main dataset.
  """
  age_df = pd.read_excel(
      _DEMOGRAPHICS_FILENAME,
      usecols=['Patient #', 'Age at Intervention Start'],
  ).dropna().rename(columns={'Age at Intervention Start': 'Age'})
  return pd.merge(
      left=base_df,
      right=age_df,
      how='inner',
      on='Patient #',
      validate='1:1'
  )

def propensity_score_match(show_plots: bool = False) -> pd.DataFrame:
  """Returns a subset of the demographics dataset with CPAP and UAS groups
     matched according to propensity scores.
  """

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
  df = pd.read_excel(_DEMOGRAPHICS_FILENAME, usecols=usecols, index_col='Patient #')
  
  df = preprocess_df(df, is_demographics_df=True)

  # Mean imputation for a small number of missing baseline AHIs
  df['AVG AHI'] = df['AVG AHI'].fillna(value=df['AVG AHI'].mean())

  df = df.rename(columns={
      'Age at Intervention Start': 'Age',
      'Male (0)/ Female (1)': 'IsFemale',
      'AVG AHI': 'Baseline AHI',
      'CPAP or UAS' : 'Treatment',
  })
  df['Sex'] = pd.Categorical(
      df['IsFemale'].replace({0: 'Male', 1: 'Female'}),
      ['Male', 'Female']
  )

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

  # sns.set(font_scale = 1.5)
  # sns.set_style("whitegrid", {'axes.grid' : False})
  if show_plots:
    plt.subplot(2,4,1)
    ax = sns.histplot(data=df, x='Sex', hue='Treatment')
    ax.set_title('Sex', fontsize=20)
    plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
    plt.xlabel('')
    plt.ylabel('Before Matching', fontsize=20)
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.subplot(2,4,2)
    ax = sns.histplot(data=df, x='Age', hue='Treatment', binrange=(30, 80), binwidth=10, legend=False)
    ax.set_title('Age', fontsize=20)
    plt.xlabel('')
    plt.ylabel('')
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.subplot(2,4,3)
    ax = sns.histplot(data=df, x='Baseline AHI', hue='Treatment', binrange=(0, 100), binwidth=10, legend=False)
    ax.set_title('Baseline AHI', fontsize=20)
    plt.xlabel('')
    plt.ylabel('')
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.subplot(2,4,4)
    ax = sns.histplot(data=df, x='Propensity Score', hue='Treatment', binrange=(0, 1), binwidth=0.1, legend=False)
    ax.set_title('Propensity Score', fontsize=20)
    plt.xlabel('')
    plt.ylabel('')
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.subplot(2,4,5)
    ax = sns.histplot(data=matched_df, x='Sex', hue='Treatment', legend=False)
    plt.xlabel('')
    plt.ylabel('After Matching', fontsize=20)
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.subplot(2,4,6)
    ax = sns.histplot(data=matched_df, x='Age', hue='Treatment', binrange=(30, 80), binwidth=10, legend=False)
    plt.xlabel('')
    plt.ylabel('')
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.subplot(2,4,7)
    ax = sns.histplot(data=matched_df, x='Baseline AHI', hue='Treatment', binrange=(0, 100), binwidth=10, legend=False)
    plt.xlabel('')
    plt.ylabel('')
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.subplot(2,4,8)
    ax = sns.histplot(data=matched_df, x='Propensity Score', hue='Treatment', binrange=(0, 1), binwidth=0.1, legend=False)
    plt.xlabel('')
    plt.ylabel('')
    ax.xaxis.set_tick_params(labelsize = 15)

    plt.show()

  return matched_df


if __name__ == "__main__":
  sheet_name = 'All'  # or 'Active'
  df = pd.read_excel(_DATA_FILENAME, sheet_name=sheet_name)
  df, _, _ = preprocess_df(df)

  # Propensity Score Matching
  matching_indices = propensity_score_match(show_plots=True).index
  print(matching_indices)
  df = df.set_index('Patient #').loc[matching_indices].reset_index()
  print(df)
