import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

pd.set_option('display.max_rows', None)

dataset = 'subset'

if dataset == 'subset':
  data_filename = 'data/HGNS_supine_datasheet-CH.xlsx'  # Subset for supine-dependence
  loading_function = pd.read_excel
  extra_usecols = [
    'Race',
    'Age',
    'BMI',
  ]
else:
  data_filename = 'data/correctUASDatabase-HGNSPredictorsOfSucc_DATA_LABELS_2025-01-18_1334.csv'  # Original dataset
  loading_function = pd.read_csv
  extra_usecols = [
    'Race/Ethnicity',
    'Age at time of surgery',
    'BMI - preop',
  ]


# Columns used in this analysis
usecols=extra_usecols + [
    # Demographic variables
    'Gender',
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

def print_demographics(df: pd.DataFrame, header: str) -> None:
  print(f'\n{header}:')
  print(df['Gender'].value_counts(dropna=False))
  print(df['Race'].value_counts(dropna=False))
  print(df[['Age', 'BMI']].describe())
  if 'Sher15' in df.columns:
    print(df['Sher15'].value_counts(dropna=False))
  if 'Sher20' in df.columns:
    print(df['Sher20'].value_counts(dropna=False))
  if 'is_supine_dependent' in df.columns:
    print(df['is_supine_dependent'].value_counts(dropna=False))

  ahi_cols = list(set(df.columns).intersection({'preop_AHI', 'Supine AHI', 'Non-supine AHI', 'postop_AHI', 'change_AHI'}))
  print(df[ahi_cols].describe())
  print("\n\n")

def main(df: pd.DataFrame) -> None:

  # ==================== BEGIN PREPROCESSING ====================
  # print_demographics(df, header='Summary statistics before preprocessing')
  
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
  df.dropna(subset=['postop_AHI', 'preop_AHI'], inplace=True)
  
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

  df.dropna(subset=['supine_AHI', 'Non-supine AHI'], inplace=True)

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
  sns.barplot(data=df_long, x='AHI_type', y='AHI', hue='Group')
  plt.xlabel('AHI Type')
  plt.ylabel('AHI (events/hour)')
  plt.grid(axis='y')
  plt.gca().set_axisbelow(True)
  df_long = (
      pd.melt(df, id_vars='Group', value_vars=['change_AHI', 'change_supine_AHI', 'change_nonsupine_AHI'], var_name='AHI_type', value_name='AHI')
      .replace({
          'change_AHI': 'Total AHI',
          'change_supine_AHI': 'Supine AHI',
          'change_nonsupine_AHI': 'Non-Supine AHI',
      })
  )
  plt.figure()
  sns.barplot(data=df_long, x='AHI_type', y='AHI', hue='Group')
  plt.xlabel('AHI Type')
  plt.ylabel('AHI Change (events/hour)')
  plt.grid(axis='y')
  plt.gca().set_axisbelow(True)
  
  # After this, we no longer need the original distinct AHIs
  df.drop(columns=[
      'Preop sleep study: AHI',
      'Preop PSG Medicare AHI',
      'Preop PSG AASMI AHI',
      'postop sleep study: AHI',
      'postop PSG Medicare AHI',
      'postop PSG AASMI AHI',
      'Preop PSG AASM Supine AHI',
      'Preop PSG Medicare Supine AHI',
      'supine_AHI',
  ], inplace=True)

  # print_demographics(df, header='Summary statistics after preprocessing')
  print_demographics(df[df['is_supine_dependent']], header='sOSA group statistics after preprocessing')
  print_demographics(df[~df['is_supine_dependent']], header='nOSA group statistics after preprocessing')

  # ===================== END PREPROCESSING =====================
  # ============ BEGIN SUPINE DEPENDENCE ANALYSES ===============

  # ------------- Compare AHIs before vs. after treatment -------------
  result = stats.wilcoxon(
    df['change_AHI'][df['is_supine_dependent']],
    alternative='less',
    nan_policy='omit',
  )
  print('\n\nWilcoxon signed-rank test for change in total AHI, in sOSA group:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  result = stats.wilcoxon(
    df['change_supine_AHI'][df['is_supine_dependent']],
    alternative='less',
    nan_policy='omit',
  )
  print('Wilcoxon signed-rank test for change in supine AHI, in sOSA group:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  result = stats.wilcoxon(
    df['change_nonsupine_AHI'][df['is_supine_dependent']],
    alternative='less',
    nan_policy='omit',
  )
  print('Wilcoxon signed-rank test for change in nonsupine AHI, in sOSA group:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  result = stats.wilcoxon(
    df['change_AHI'][~df['is_supine_dependent']],
    alternative='less',
    nan_policy='omit',
  )
  print('Wilcoxon signed-rank test for change in total AHI, in nOSA group:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  result = stats.wilcoxon(
    df['change_supine_AHI'][~df['is_supine_dependent']],
    alternative='less',
    nan_policy='omit',
  )
  print('Wilcoxon signed-rank test for change in supine AHI, in nOSA group:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  result = stats.wilcoxon(
    df['change_nonsupine_AHI'][~df['is_supine_dependent']],
    alternative='less',
    nan_policy='omit',
  )
  print('Wilcoxon signed-rank test for change in nonsupine AHI, in nOSA group:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}\n\n')

  # ------------- Compare AHIs between sOSA & nOSA groups -------------
  # Since is_supine_dependent is binary and postop_AHI is continuous, we use a t-test
  result = stats.mannwhitneyu(
    df['postop_AHI'][df['is_supine_dependent']],
    df['postop_AHI'][~df['is_supine_dependent']],
    alternative='greater',
    nan_policy='raise',
  )
  print('\n\nMann-Whitney U-test for independence of is_supine_dependent and postop_AHI:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  # Since is_supine_dependent is binary and AHI changes are continuous, we use a Mann-Whitney
  result = stats.mannwhitneyu(
    df['change_AHI'][df['is_supine_dependent']],
    df['change_AHI'][~df['is_supine_dependent']],
    alternative='greater',
    nan_policy='raise',
  )
  print('Mann-Whitney U-test for independence of is_supine_dependent and change_AHI:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  result = stats.mannwhitneyu(
    df['change_supine_AHI'][df['is_supine_dependent']],
    df['change_supine_AHI'][~df['is_supine_dependent']],
    alternative='less',
    nan_policy='omit',
  )
  print('Mann-Whitney U-test for independence of is_supine_dependent and change_supine_AHI:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}')

  result = stats.mannwhitneyu(
    df['change_nonsupine_AHI'][df['is_supine_dependent']],
    df['change_nonsupine_AHI'][~df['is_supine_dependent']],
    alternative='greater',
    nan_policy='omit',
  )
  print('Mann-Whitney U-test for independence of is_supine_dependent and change_nonsupine_AHI:')
  print(f'U: {result.statistic:.03f}  p: {result.pvalue:.03f}\n\n')

  # Since both is_supine_dependent and Sher15 are binary, we use a Chi^2 test
  contingency_table = pd.crosstab(df['is_supine_dependent'], df['Sher15'])
  chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
  print('Chi^2 test for independence of supine dependence and Sher15:')
  print('Observed contingency table:')
  print(contingency_table)
  print('Expected contingency table:')
  print(expected)
  print(f'Test tesults: chi2: {chi2:.03f}  p: {p:.03f}  dof: {dof}\n\n')

  # Since both is_supine_dependent and Sher20 are binary, we use a Chi^2 test
  contingency_table = pd.crosstab(df['is_supine_dependent'], df['Sher20'])
  chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
  print('\nChi^2 test for independence of supine dependence and Sher20:')
  print('Observed contingency table:')
  print(contingency_table)
  print('Expected contingency table:')
  print(expected)
  print(f'Test tesults: chi2: {chi2:.03f}  p: {p:.03f}  dof: {dof}\n\n')

  # Compare change_supine_AHI and change_nonsupine_AHI to see if HGNS has a
  # differential effect.
  supine_ci = stats.bootstrap(data=(df['change_supine_AHI'],), method='percentile', statistic=np.nanmean, confidence_level=0.95, rng=0).confidence_interval
  print(f'Supine AHI had mean change {df["change_supine_AHI"].mean():.3f} (95% bootstrap CI ({supine_ci.low:.03f}, {supine_ci.high:.03f})).')
  nonsupine_ci = stats.bootstrap(data=(df['change_nonsupine_AHI'],), method='percentile', statistic=np.nanmean, confidence_level=0.95, rng=0).confidence_interval
  print(f'Non-Supine AHI had mean change {df["change_nonsupine_AHI"].mean():.3f} (95% bootstrap CI ({nonsupine_ci.low:.03f}, {nonsupine_ci.high:.03f})).')
  result = stats.wilcoxon(df['change_supine_AHI'], df['change_nonsupine_AHI'], nan_policy='omit')
  print(f'Difference in means between changes in supine and nonsupine AHIs:')
  print(f'Mean: {(df['change_supine_AHI'] - df['change_nonsupine_AHI']).mean():.02f}')
  print(f'p: {result.pvalue:.03f}\n\n')

  # ============= END SUPINE DEPENDENCE ANALYSES ================
  
  plt.show()


if __name__ == "__main__":
  # Read in just the columns we need from the data table
  df = loading_function(data_filename, usecols=usecols).rename(columns={
      'Race/Ethnicity': 'Race',
      'Age at time of surgery': 'Age',
      'BMI - preop': 'BMI',
  })
  main(df)
