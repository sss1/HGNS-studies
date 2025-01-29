import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 4)

data_filename = 'data/WatchPATStudyData.xlsx'

def main(df: pd.DataFrame) -> None:
  
  # =========================== BEGIN PREPROCESSING ===========================
  # Remove rows without treatment assignment (mostly some empty rows)
  df = df.dropna(subset='CPAP or UAS')

  # Standardize coding of 'CPAP'
  df.loc[df['CPAP or UAS'] != 'UAS', 'CPAP or UAS'] = 'CPAP'

  # Drop a lot of the columns we don't need for these analyses
  columns_to_drop = [
      'Male (0)/ Female (1)', 'Race', 'Ethnicity',
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
      'Adjusted compliance '                  : 'Adjusted compliance.1',
      'Adjusted compliance .1'                : 'Adjusted compliance.2',
      'Adjusted compliance .2'                : 'Adjusted compliance.3',
      'Adjusted compliance .3'                : 'Adjusted compliance.4',
      'Treatment Efficacy'                    : 'Treatment Efficacy.1',
      'Treatment Effiacy'                     : 'Treatment Efficacy.2',
      'Treatment Effiacy.1'                   : 'Treatment Efficacy.3',
      'Treatment Efficacy.1'                  : 'Treatment Efficacy.4',
      'MDA'                                   : 'MDA.1',
      'MDA.1'                                 : 'MDA.2',
      'MDA.2'                                 : 'MDA.3',
      'MDA.3'                                 : 'MDA.4',
      'CPAP or UAS'                           : 'Treatment',
  })

  dependent_variables = [
      'ESS Total', 'General Productivity', 'Social Outcome', 'Activity Level',
      'Vigilance', 'Intimate Relationship/Sexual Activity', 'FOSQ Score',
      'Avg RDI', 'AVG Nadir', 'AVG AHI', 'Adjusted compliance',
      'Treatment Efficacy', 'MDA',
  ]
  # Since 18 and 24 month timepoints are missing a lot of data,
  # Pool 18 and 24 month values into '18+' months; specifically:
  # 1) If 18 is missing, use 24
  # 2) Else if 24 is missing use 18
  # 3) Else (both are present) use average
  for dv in dependent_variables:
    df[f'{dv}.3'] = df[f'{dv}.3'].fillna(df[f'{dv}.4'])
    df[f'{dv}.4'] = df[f'{dv}.4'].fillna(df[f'{dv}.3'])
    df[f'{dv}.3'] = (df[f'{dv}.3'] + df[f'{dv}.4'])/2
    df = df.drop(columns=f'{dv}.4')

  # Instead of having separate columns for each measurement period, combine
  # measurements of each dependent variable and add a 'Month' column indicating
  # the measurement period
  df = pd.wide_to_long(df=df, stubnames=dependent_variables, i='Patient #',
                       j='Month', sep='.',).reset_index()
  df['Month'] *= 6  # Convert 6-month measurement periods to months

  # Make a pairs plot just to sanity check everything
  sns.pairplot(df.drop(columns=['Patient #', 'Month']))
  plt.gcf().subplots_adjust(bottom=0.05)

  # Plot all DVs over time
  plt.figure()
  for idx, dv in enumerate(dependent_variables, start=1):
    plt.subplot(4, 4, idx)
    legend = False if idx > 1 else  'auto'  # Only show legend on the first plot
    sns.lineplot(data=df, x='Month', y=dv, hue='Treatment', err_style='bars',
                 err_kws={'capsize': 5}, legend=legend)
    plt.xticks([0, 6, 12, 18])

  # ========================== END PREPROCESSING ==============================
  # ===================== BEGIN MIXED EFFECTS ANALYSIS ========================

  # Fit a separate model for each DV, but accumulate the results into a single
  # table and print them together.
  results = []
  for dv in dependent_variables:
    df_dv = df.dropna(subset=dv)
    mixed_effects_model = smf.mixedlm(
        f"Q('{dv}') ~ Month * Treatment",
        data=df_dv,
        groups=df_dv["Patient #"]
    ).fit()

    # Interleave coeffs and ps for ease of reading table by eye
    coeffs = mixed_effects_model.fe_params
    ps = mixed_effects_model.pvalues[:-1]  # Omit p-value for Group Var
    beta_month_plus_interaction = coeffs[2] + coeffs[3]
    interleaved = [x for pair in zip(coeffs, ps) for x in pair]
    results.append(
        [mixed_effects_model.nobs] + interleaved + [beta_month_plus_interaction]
    )

  column_names = [
      'n',
      'beta_Intercept', 'p_Intercept',
      'beta_Treatment[UAS]', 'p_Treatment[UAS]',
      'beta_Month', 'p_Month',
      'beta_Interaction', 'p_Interaction',
      # Although this isn't an explicit model parameter, this corresponds to the
      # rate of change per month for the UAS group.
      'beta_Month+Interaction',
  ]
  print(pd.DataFrame(results, columns=column_names, index=dependent_variables))

  # TODO: Figure out a good way to visualize the mixed effects model
  # for idx, dv in enumerate(dependent_variables, start=1):
  #   plt.subplot(4, 4, idx)
  #   legend = False if idx > 1 else  'auto'  # Only show legend on the first plot
  #   # sns.scatterplot(data=df, x='Month', y=dv, hue='Treatment', legend=legend)
  #   plot_df = df.pivot(index='Patient #', columns='Month', values=dependent_variables)
  #   print(plot_df)
  #   plt.plot(np.tile([0, 6, 12, 18], reps=(50, 1)), plot_df[dv], 'o-')
  #   # plt.xticks([0, 6, 12, 18])
  # plt.show()

  # ====================== END MIXED EFFECTS ANALYSIS =========================
  # ================== BEGIN PAIRWISE COMPARISONS =====================

  # Calculate comparisons separately for each DV and timepoint, but accumulate
  # the results into a single table to print
  results = []
  for dv in dependent_variables:
    row = []
    for T in [0, 6, 12, 18]:

      # Skip variables not measured at time 0
      if (T == 0) and (dv in ['Adjusted compliance', 'Treatment Efficacy', 'MDA']):
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

  plt.show()

if __name__ == "__main__":
  # Read in just the sheet we need from the Excel file
  sheet_name = 'Active'  # or 'All'
  main(pd.read_excel(data_filename, sheet_name=sheet_name))

