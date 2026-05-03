import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

from preprocessing import preprocess_df, propensity_score_match

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 4)

_DATA_FILENAME = 'data/WatchPATStudyData.xlsx'

def print_demographics(df: pd.DataFrame, header: str) -> None:
  print(f'\n{header}:')
  print(df['IsFemale'].value_counts(dropna=False))
  print(df['Race'].value_counts(dropna=False))
  print(df[['Age', 'AVG AHI.0']].describe())

def main(df: pd.DataFrame) -> None:
  
  # =========================== BEGIN PREPROCESSING ===========================
  df, matching_variables, dependent_variables = preprocess_df(df)

  # Print demographics overall and by group
  print_demographics(df, header='Overall demographic statistics')
  print_demographics(df.groupby('Treatment'), header='Group demographic statistics')
  print('\n\n\n\n')

  plt.figure()
  sns.histplot(data=df, x='BaselineAHI', hue='Treatment')

  # Propensity Score Matching
  matching_indices = propensity_score_match().index
  print(df.index)
  df = df.set_index('Patient #').loc[matching_indices].reset_index()
  print(df.index)

  # Report a few timepoint-to-timepoint differences before converting the data
  # from wide to long format
  # Numbers of patients with normal pretreatment ESS or FOSQ scores
  df['ESS_baseline_normal'] = (df['ESS Total.0'] <= 10).astype(float)
  df.loc[df['ESS Total.0'].isnull(), 'ESS_baseline_normal'] = np.nan
  print(df.groupby(by=['Treatment'])['ESS_baseline_normal'].value_counts(dropna=False))
  df['FOSQ_baseline_normal'] = (df['FOSQ Score.0'] > 17.9).astype(float)
  df.loc[df['FOSQ Score.0'].isnull(), 'FOSQ_baseline_normal'] = np.nan
  print(df.groupby(by=['Treatment'])['FOSQ_baseline_normal'].value_counts(dropna=False))

  # After matching, reprint demographics overall and by group
  print_demographics(df, header='Overall demographic statistics after matching')
  print_demographics(df.groupby('Treatment'), header='Group demographic statistics after matching')
  print('\n\n\n\n')

  # Instead of having separate columns for each measurement period, combine
  # measurements of each dependent variable and add a 'Month' column indicating
  # the measurement period
  df['Patient #'] = range(df.shape[0])
  df = pd.wide_to_long(
      df=df,
      stubnames=dependent_variables,
      i=['Patient #'],
      j='Month',
      sep='.',
  ).reset_index()
  df['Month'] *= 6  # Convert 6-month measurement periods to months

  # Plot all DVs over time
  plt.figure()
  for idx, dv in enumerate(dependent_variables, start=1):
    plt.subplot(4, 4, idx)
    legend = 'auto' if dv == 'MDA' else False # Only show legend on the last plot
    sns.lineplot(data=df, x='Month', y=dv, hue='Treatment', err_style='bars',
                 err_kws={'capsize': 5}, legend=legend)
    if dv == 'Intimate Relationship/Sexual Activity':
      plt.ylabel('Intimate Relationship/\nSexual Activity', size=11)
    else:
      plt.ylabel(dv, size=14)
    if idx < 10:
      plt.xlabel('')
    else:
      plt.xlabel('Month', size=14)
    if idx == 2 or idx == 4:
      plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.xticks([0, 6, 12, 18])

  # ========================== END PREPROCESSING ==============================
  # ================== BEGIN PAIRWISE COMPARISONS =====================

  # Calculate comparisons separately for each DV and timepoint, but accumulate
  # the results into a single table to print
  results = []
  for dv in dependent_variables:
    row = []
    for T in [0, 6, 12, 18]:

      # Skip variables not measured at time 0
      if (T == 0) and (dv in ['Adjusted Adherence', 'Treatment Efficacy', 'MDA']):
        row.extend([0, 0, float('nan'), ''])
        continue

      cpap_group = df[dv][(df['Treatment'] == 'CPAP') & (df['Month'] == T)].dropna()
      uas_group = df[dv][(df['Treatment'] == 'UAS') & (df['Month'] == T)].dropna()
      U, p = stats.mannwhitneyu(cpap_group, uas_group)

      sig = '*' if p < 0.05 else ''  # Indicate significant results with a '*'
      row.extend([len(cpap_group), len(uas_group), U, f'{sig}{p:.3f}'])

    results.append(row)

  column_names = [
      'nCPAP_0',  'nUAS_0',  'U_0',  'p_0',
      'nCPAP_6',  'nUAS_6',  'U_6',  'p_6',
      'nCPAP_12', 'nUAS_12', 'U_12', 'p_12',
      'nCPAP_18', 'nUAS_18', 'U_18', 'p_18',
  ]
  print(pd.DataFrame(results, columns=column_names, index=dependent_variables))

  # =================== END PAIRWISE COMPARISONS ======================

  df['MDA_gt_0.4'] = (df['MDA'] >= 0.4).astype(float)
  df.loc[df['MDA'].isnull(), 'MDA_gt_0.4'] = np.nan
  print(df.groupby(by=['Month', 'Treatment'])['MDA'].mean())
  print(df.groupby(by=['Month', 'Treatment'])['MDA_gt_0.4'].value_counts(dropna=False))
  plt.show()

if __name__ == "__main__":
  # Read in just the sheet we need from the Excel file
  sheet_name = 'All'  # or 'Active'
  main(pd.read_excel(_DATA_FILENAME, sheet_name=sheet_name))
