import matplotlib.pyplot as plt
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
    # Predictor variables
    'Age at time of surgery',
    'Gender',
    'BMI - preop',
    'DISE PAP opening pressure',
    'Oropharynx',
    # Variables used to identify supine-dependent apnea
    'Supine AHI',
    'Preop PSG AASM Supine AHI',
    'Preop PSG Medicare Supine AHI',
    'Non-supine AHI',
    # Variables needed for changes supine-dependent apnea
    'Supine AHI.1',
    'Non-supine AHI.1',
]

def main(df: pd.DataFrame) -> None:

  # ==================== BEGIN PREPROCESSING ====================
  # Shorten some column names to be easier to use
  df = df.rename(columns={
          'Age at time of surgery': 'Age',
          'BMI - preop': 'BMI',
          'DISE PAP opening pressure': 'DISE',
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
  df = df.dropna(subset=['postop_AHI', 'preop_AHI'])
  
  # Compute other success measures
  df['change_AHI'] = df['postop_AHI'] - df['preop_AHI']
  df['Sher15'] = (((df['postop_AHI'] <= 0.5*df['preop_AHI'])
                  & (df['postop_AHI'] < 15))).astype('float')
  df['change_supine_AHI'] = df['Supine AHI.1'] - df['Supine AHI']
  df['change_nonsupine_AHI'] = df['Non-supine AHI.1'] - df['Non-supine AHI']
  print(df)

  # Identify subgroup of patients with supine-dependent apnea for secondary analysis
  df['supine_AHI'] = (df['Supine AHI']
                      .fillna(df['Preop PSG Medicare Supine AHI'])
                      .fillna(df['Preop PSG AASM Supine AHI']))
  df['is_supine_dependent'] = (df['supine_AHI'] >= 2*df['Non-supine AHI'])
  
  # After this, we no longer need the original distinct AHIs
  df = df.drop(columns=[
      'Preop sleep study: AHI',
      'Preop PSG Medicare AHI',
      'Preop PSG AASMI AHI',
      'postop sleep study: AHI',
      'postop PSG Medicare AHI',
      'postop PSG AASMI AHI',
      'Supine AHI',
      'Preop PSG AASM Supine AHI',
      'Preop PSG Medicare Supine AHI',
      'Non-supine AHI',
      'supine_AHI',
  ])
  
  # Since DISE PAP opening pressure is >90% missing, we need some way to handle this;
  # for regression, replacing with the mean value is reasonable if we assume the values
  # are "missing completely at random" (MCAR); if not, we should impute by regression.
  df['DISE'] = df['DISE'].fillna(df['DISE'].mean())

  # Since very few patients have group Oropharynx 2, group Oropharynx values 1
  # and 2 into a single category
  df['Oropharynx'] = (df['Oropharynx'] > 0)
  

  # Print a summary of the data after preprocessing
  print(df.describe())
  print(df['Gender'].value_counts())
  print(df['Oropharynx'].value_counts())
  print(df['Sher15'].value_counts())

  # Make a pairs plot just to sanity check everything
  sns.pairplot(df)
  plt.gcf().subplots_adjust(bottom=0.05)

  # ===================== END PREPROCESSING =====================
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
  # ============ BEGIN SUPINE DEPENDENCE ANALYSES ===============

  # Since is_supine_dependent is binary and postop_AHI is continuous, we use a t-test
  result = stats.mannwhitneyu(
    df['postop_AHI'][df['is_supine_dependent']],
    df['postop_AHI'][~df['is_supine_dependent']],
    alternative='greater',
  )
  print('\nMann-Whitney U-test for independence of is_supine_dependent and postop_AHI:')
  print(f'U: {result.statistic}  p: {result.pvalue}')
  sns.violinplot(data=df, x='is_supine_dependent', y='postop_AHI')

  # Since is_supine_dependent is binary and change_AHI is continuous, we use a t-test
  result = stats.mannwhitneyu(
    df['change_AHI'][df['is_supine_dependent']],
    df['change_AHI'][~df['is_supine_dependent']],
    alternative='greater',
  )
  print('\nMann-Whitney U-test for independence of is_supine_dependent and change_AHI:')
  print(f'U: {result.statistic}  p: {result.pvalue}')
  plt.figure()
  sns.violinplot(data=df, x='is_supine_dependent', y='change_AHI')

  # Since both is_supine_dependent and Sher15 are binary, we use a Chi^2 test
  contingency_table = pd.crosstab(df['is_supine_dependent'], df['Sher15'])
  chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
  print('\nChi^2 test for independence of supine dependence and Sher15:')
  print('Observed contingency table:')
  print(contingency_table)
  print('Expected contingency table:')
  print(expected)
  print(f'Test tesults: chi2: {chi2}  p: {p}  dof: {dof}\n')

  # Calculate change_supine_AHI and change_nonsupine_AHI to see if HGNS has a
  # differential effect.
  result = stats.ttest_rel(df['change_supine_AHI'], df['change_nonsupine_AHI'], nan_policy='omit')
  print(f'\n\nDifference in means between changes in supine and nonsupine AHIs:')
  print((df['change_supine_AHI'] - df['change_nonsupine_AHI']).mean())
  print(f't: {result.statistic}  p: {result.pvalue}  df: {result.df}\n\n')

  # ============= END SUPINE DEPENDENCE ANALYSES ================
  # ================== BEGIN SCORE ANALYSES =====================

  # Calculate score and see if it is significantly predictive of outcomes
  df['score'] = (5 * (df['DISE'] <= 8)
                 + 5 * (df['Gender'] == 'Female')
                 + (df['BMI'] <= 35) + (df['BMI'] <= 32.5) + (df['BMI'] <= 30) + (df['BMI'] <= 27.5) + (df['BMI'] <= 25)
                 + 5 * (~df['Oropharynx'])  # False = no collapse = 5pts, True = partial or complete collapse = 0 pts
                 + (df['preop_AHI'] >= 65) + (df['preop_AHI'] >= 55) + (df['preop_AHI'] >= 45) + (df['preop_AHI'] >= 35) + (df['preop_AHI'] >= 25)
                 + (df['Age'] < 75) + (df['Age'] < 65) + (df['Age'] < 55) + (df['Age'] < 45) + (df['Age'] < 35))
  # Since postop_AHI is a count, use Poisson regression
  print(smf.poisson(formula=('postop_AHI ~ score'), data=df).fit().summary())
  # Since change_AHI is continuous, use linear regression
  print(smf.ols(formula=('change_AHI ~ score'), data=df).fit().summary())
  # Since Sher15 is boolean, use logistic regression
  print(smf.logit(formula=('Sher15 ~ score'), data=df).fit().summary())

  # =================== END SCORE ANALYSES ======================
  
  plt.show()


if __name__ == "__main__":
  # Read in just the columns we need from the data table
  df = pd.read_csv(data_filename, usecols=usecols)
  main(df)
