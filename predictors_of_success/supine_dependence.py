import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    # Variables used to identify supine-dependent apnea
    'Supine AHI',
    'Preop PSG AASM Supine AHI',
    'Preop PSG Medicare Supine AHI',
    'Non-supine AHI',
    # Variables needed for changes in supine-dependent apnea
    'Supine AHI.1',
    'Non-supine AHI.1',
    # Demographic variables
    'Age at time of surgery',
    'Gender',
    'BMI - preop',
    'Race/Ethnicity',
]

def print_demographics(df: pd.DataFrame, header: str) -> None:
  print(f'\n{header}:')
  print(df['Gender'].value_counts(dropna=False))
  print(df['Race'].value_counts(dropna=False))
  print(df[['Age', 'BMI']].describe())
  if 'Sher15' in df.columns:
    print(df['Sher15'].value_counts(dropna=False))
  if 'is_supine_dependent' in df.columns:
    print(df['is_supine_dependent'].value_counts(dropna=False))

  ahi_cols = list(set(df.columns).intersection({'preop_AHI', 'Supine AHI', 'Non-supine AHI'}))
  print(df[ahi_cols].describe())
  print("\n\n")

def main(df: pd.DataFrame) -> None:

  # ==================== BEGIN PREPROCESSING ====================
  # Shorten some column names to be easier to use
  df = df.rename(columns={
          'Age at time of surgery': 'Age',
          'BMI - preop': 'BMI',
          'Race/Ethnicity': 'Race',
  })

  print_demographics(df, header='Summary statistics before preprocessing')
  
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
  df = df.dropna(subset=['postop_AHI', 'preop_AHI'])
  
  # Compute other success measures
  df['change_AHI'] = df['postop_AHI'] - df['preop_AHI']
  df['Sher15'] = (((df['postop_AHI'] <= 0.5*df['preop_AHI'])
                  & (df['postop_AHI'] < 15))).astype('float')
  df['change_supine_AHI'] = df['Supine AHI.1'] - df['Supine AHI']
  df['change_nonsupine_AHI'] = df['Non-supine AHI.1'] - df['Non-supine AHI']

  # Identify subgroup of patients with supine-dependent apnea for secondary analysis
  df['supine_AHI'] = (df['Supine AHI']
                      .fillna(df['Preop PSG Medicare Supine AHI'])
                      .fillna(df['Preop PSG AASM Supine AHI']))
  df['is_supine_dependent'] = (df['supine_AHI'] >= 2*df['Non-supine AHI'])

  df = df.dropna(subset=['supine_AHI', 'Non-supine AHI'])

  df['Group'] = df['is_supine_dependent'].replace({True: 'sOSA', False: 'nOSA'})
  df_long = (
      pd.melt(df, id_vars='Group', value_vars=['preop_AHI', 'postop_AHI', 'Supine AHI', 'Supine AHI.1', 'Non-supine AHI', 'Non-supine AHI.1'], var_name='AHI_type', value_name='AHI')
      .replace({
          'preop_AHI': 'Preoperative\nOverall',
          'postop_AHI': 'Postoperative\nOverall',
          'Supine AHI': 'Preoperative\nSupine',
          'Supine AHI.1': 'Postoperative\nSupine',
          'Non-supine AHI': 'Preoperative\nNon-Supine',
          'Non-supine AHI.1': 'Postoperative\nNon-Supine',
      })
  )
  # print(df_long)
  # df = pd.wide_to_long(df=df, stubnames=dependent_variables, i='Patient #',
  #                      j='Month', sep='.',).reset_index()
  # plt.subplot(1, 3, 1)
  # sns.violinplot(data=df, x='is_supine_dependent', y='postop_AHI')
  sns.barplot(data=df_long, x='AHI_type', y='AHI', hue='Group')
  plt.xlabel('AHI Type')
  plt.ylabel('AHI (events/hour)')
  plt.grid(axis='y')
  plt.gca().set_axisbelow(True)
  # plt.subplot(1, 3, 2)
  # # sns.violinplot(data=df, x='is_supine_dependent', y='postop_AHI')
  # sns.barplot(data=df, x='group', y='postop_AHI', legend=False)
  # plt.subplot(1, 3, 3)
  # sns.barplot(data=df, x='group', y='change_AHI', legend=False)
  # # sns.violinplot(data=df, x='is_supine_dependent', y='change_AHI')
  df_long = (
      pd.melt(df, id_vars='Group', value_vars=['change_AHI', 'change_supine_AHI', 'change_nonsupine_AHI'], var_name='AHI_type', value_name='AHI')
      .replace({
      })
  )
  plt.figure()
  sns.barplot(data=df_long, x='AHI_type', y='AHI', hue='Group')
  plt.xlabel('AHI Type')
  plt.ylabel('AHI Change (events/hour)')
  plt.grid(axis='y')
  plt.gca().set_axisbelow(True)
  
  # After this, we no longer need the original distinct AHIs
  df = df.drop(columns=[
      'Preop sleep study: AHI',
      'Preop PSG Medicare AHI',
      'Preop PSG AASMI AHI',
      'postop sleep study: AHI',
      'postop PSG Medicare AHI',
      'postop PSG AASMI AHI',
      # 'Supine AHI',
      'Preop PSG AASM Supine AHI',
      'Preop PSG Medicare Supine AHI',
      # 'Non-supine AHI',
      'supine_AHI',
  ])

  print_demographics(df, header='Summary statistics after preprocessing')
  print_demographics(df[df['is_supine_dependent']], header='sOSA group statistics after preprocessing')
  print_demographics(df[~df['is_supine_dependent']], header='nOSA group statistics after preprocessing')

  # ===================== END PREPROCESSING =====================
  # ============ BEGIN SUPINE DEPENDENCE ANALYSES ===============

  # Since is_supine_dependent is binary and postop_AHI is continuous, we use a t-test
  result = stats.mannwhitneyu(
    df['postop_AHI'][df['is_supine_dependent']],
    df['postop_AHI'][~df['is_supine_dependent']],
    alternative='greater',
  )
  print('Mann-Whitney U-test for independence of is_supine_dependent and postop_AHI:')
  print(f'U: {result.statistic}  p: {result.pvalue}')

  # Since is_supine_dependent is binary and change_AHI is continuous, we use a t-test
  result = stats.mannwhitneyu(
    df['change_AHI'][df['is_supine_dependent']],
    df['change_AHI'][~df['is_supine_dependent']],
    alternative='greater',
  )
  print('\nMann-Whitney U-test for independence of is_supine_dependent and change_AHI:')
  print(f'U: {result.statistic}  p: {result.pvalue}\n\n')

  # Since both is_supine_dependent and Sher15 are binary, we use a Chi^2 test
  contingency_table = pd.crosstab(df['is_supine_dependent'], df['Sher15'])
  chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
  print('\nChi^2 test for independence of supine dependence and Sher15:')
  print('Observed contingency table:')
  print(contingency_table)
  print('Expected contingency table:')
  print(expected)
  print(f'Test tesults: chi2: {chi2}  p: {p}  dof: {dof}\n\n')

  # Compare change_supine_AHI and change_nonsupine_AHI to see if HGNS has a
  # differential effect.
  supine_ci = stats.bootstrap(data=(df['change_supine_AHI'],), method='percentile', statistic=np.nanmean, confidence_level=0.95, rng=0).confidence_interval
  print(f'Supine AHI had mean change {df["change_supine_AHI"].mean():.3f} (95% bootstrap CI ({supine_ci.low}, {supine_ci.high})).')
  nonsupine_ci = stats.bootstrap(data=(df['change_nonsupine_AHI'],), method='percentile', statistic=np.nanmean, confidence_level=0.95, rng=0).confidence_interval
  print(f'Non-Supine AHI had mean change {df["change_nonsupine_AHI"].mean():.3f} (95% bootstrap CI ({nonsupine_ci.low}, {nonsupine_ci.high})).')
  result = stats.ttest_rel(df['change_supine_AHI'], df['change_nonsupine_AHI'], nan_policy='omit')
  print(f'Difference in means between changes in supine and nonsupine AHIs:')
  print((df['change_supine_AHI'] - df['change_nonsupine_AHI']).mean())
  print(f't: {result.statistic}  p: {result.pvalue}  df: {result.df}\n\n')

  # plt.plot([df['change_supine_AHI'], df['change_nonsupine_AHI']])

  # ============= END SUPINE DEPENDENCE ANALYSES ================
  
  plt.show()


if __name__ == "__main__":
  # Read in just the columns we need from the data table
  df = pd.read_csv(data_filename, usecols=usecols)
  main(df)
