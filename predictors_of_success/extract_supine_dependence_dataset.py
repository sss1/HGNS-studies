import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

pd.set_option('display.max_rows', None)

input_data_filename = 'data/correctUASDatabase-HGNSPredictorsOfSucc_DATA_LABELS_2025-01-18_1334.csv'
output_data_filename = 'data/supine_depenence_subset.xlsx'

# Columns used in this analysis
usecols=[
    'Record ID',
    # Demographic variables
    'Age at time of surgery',
    'Gender',
    'BMI - preop',
    'Race/Ethnicity',
    # AHI variables
    'Preop sleep study: AHI',
    'Preop PSG Medicare AHI',
    'Preop PSG AASMI AHI',
    'postop sleep study: AHI',
    'postop PSG Medicare AHI',
    'postop PSG AASMI AHI',
    # Variables used to identify supine-dependent apnea
    'Supine AHI',
    'Preop PSG AASM Supine AHI',
    'Preop PSG Medicare Supine AHI',
    'Non-supine AHI',
    # Variables needed for changes in supine-dependent apnea
    'Supine AHI.1',
    'Non-supine AHI.1',
]

def main(df: pd.DataFrame) -> None:

  # ==================== BEGIN PREPROCESSING ====================
  # Shorten some column names to be easier to use
  df = df.rename(columns={
          'Age at time of surgery': 'Age',
          'BMI - preop': 'BMI',
          'Race/Ethnicity': 'Race',
  })
  
  # Fix a few values that were incorrectly recorded in the dataset
  df['Preop sleep study: AHI'] = df['Preop sleep study: AHI'].replace(to_replace={178: 34.8})
  df['Non-supine AHI.1'] = df['Non-supine AHI.1'].replace(to_replace={238.8: 59.9, 232: 50.0})

  # When sleep study AHI is missing, use Medicare or AASMI AHI instead
  df['preop_AHI'] = (df['Preop sleep study: AHI']
                      .fillna(df['Preop PSG Medicare AHI'])
                      .fillna(df['Preop PSG AASMI AHI']))
  df['postop_AHI'] = (df['postop sleep study: AHI']
                      .fillna(df['postop PSG Medicare AHI'])
                      .fillna(df['postop PSG AASMI AHI']))
  
  # Compute other success measures
  df['change_AHI'] = df['postop_AHI'] - df['preop_AHI']
  df['Sher15'] = (((df['postop_AHI'] <= 0.5*df['preop_AHI'])
                 & (df['postop_AHI'] < 15))).astype('float')
  df['Sher20'] = (((df['postop_AHI'] <= 0.5*df['preop_AHI'])
                 & (df['postop_AHI'] < 20))).astype('float')
  df['change_supine_AHI'] = df['Supine AHI.1'] - df['Supine AHI']
  df['change_nonsupine_AHI'] = df['Non-supine AHI.1'] - df['Non-supine AHI']

  # Identify subgroup of patients with supine-dependent apnea for secondary analysis
  df['supine_AHI'] = (df['Supine AHI']
                      .fillna(df['Preop PSG Medicare Supine AHI'])
                      .fillna(df['Preop PSG AASM Supine AHI']))
  df['is_supine_dependent'] = (df['supine_AHI'] >= 2*df['Non-supine AHI'])
  df['group'] = df['is_supine_dependent'].replace({True: 'sOSA', False: 'nOSA'})

  df = df.dropna(subset=['postop_AHI', 'preop_AHI', 'supine_AHI', 'Non-supine AHI'])

  df.to_excel(output_data_filename)
  print(df.groupby('group')['Sher15'].value_counts(dropna=False))
  print(df.groupby('group')['Sher20'].value_counts(dropna=False))

if __name__ == "__main__":
  # Read in just the columns we need from the data table
  df = pd.read_csv(input_data_filename, usecols=usecols, index_col='Record ID')
  main(df)
