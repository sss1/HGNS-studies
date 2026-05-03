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
    'Race/Ethnicity',
]

def print_demographics(df: pd.DataFrame, header: str) -> None:
  print(f'\n{header}:')
  print(df['Gender'].value_counts(dropna=False))
  print(df['Race'].value_counts(dropna=False))
  print(df['Oropharynx'].value_counts(dropna=False))
  print(df[['Age', 'BMI']].describe())
  if 'Sher15' in df.columns:
    print(df['Sher15'].value_counts(dropna=False))

def main(df: pd.DataFrame) -> None:

  # ==================== BEGIN PREPROCESSING ====================
  # Shorten some column names to be easier to use
  df = df.rename(columns={
          'Age at time of surgery': 'Age',
          'BMI - preop': 'BMI',
          'DISE PAP opening pressure': 'DISE',
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

  # Make pairs plots and histograms
  plot_df = (
      df.copy(deep=True)
        .drop(columns=[
            'Supine AHI.1',
            'Non-supine AHI.1',
            'change_supine_AHI',
            'change_nonsupine_AHI',
            'is_supine_dependent',
            # 'Race',  # Race has too many categorical values for this plot
        ]).rename(columns={
            'Oropharynx': 'LWC',
            'Gender': 'Sex',
            'DISE': 'PhOP',
            'preop_AHI': 'Baseline\nAHI',
            'postop_AHI': 'Post-op\nAHI',
            'change_AHI': 'Change in AHI\n(post - pre)',
            'Sher15': 'Sher15\nSuccess',
        })
  )
  sns.set_theme(font_scale=1.5)  # Increase font size

  # Pair plot for continuous variables
  sns.pairplot(plot_df, vars=['Age', 'BMI', 'PhOP', 'Baseline\nAHI', 'Post-op\nAHI', 'Change in AHI\n(post - pre)'])
  plt.gcf().subplots_adjust(bottom=0.1)
  
  # Histograms for categorical variables
  plot_df['Race'] = plot_df['Race'].replace({
      'White/Caucasian': 'White',
      'Black/African American': 'Black',
      'Hispanic/Latino/a': 'Hispanic',
      'NaN': 'Not Reported',
      'White/Caucasian,Hispanic/Latino/a,Other': 'Hispanic',
      'Other': 'Not Reported',
      'White/Caucasian,Asian': 'Asian',
      'White/Caucasian,Hispanic/Latino/a': 'Hispanic',
      'Black/African American,Other': 'Black',
  })
  plot_df['LWC'] = plot_df['LWC'].replace({
      0.0: 'None',
      1.0: 'Partial',
      2.0: 'Complete',
  })
  plot_df['Sher15\nSuccess'] = plot_df['Sher15\nSuccess'].astype('bool').astype('str')
  plt.figure()
  plt.subplot(2, 2, 1)
  sns.histplot(data=plot_df[plot_df['Race'] != 'Not Reported'], x='Race')
  plt.subplot(2, 2, 2)
  sns.histplot(data=plot_df, x='Sex')
  plt.subplot(2, 2, 3)
  sns.histplot(data=plot_df, x='LWC')
  plt.subplot(2, 2, 4)
  sns.histplot(data=plot_df, x='Sher15\nSuccess')
  
  # Since DISE PAP opening pressure is >90% missing, we need some way to handle this;
  # for regression, replacing with the mean value is reasonable if we assume the values
  # are "missing completely at random" (MCAR); since missing data was mostly due to a
  # change in the data collection procudure, this seems reasonable.
  df['DISE'] = df['DISE'].fillna(df['DISE'].mean())

  # Since very few patients have group Oropharynx 2, group Oropharynx values 1
  # and 2 into a single category
  print(df['Oropharynx'].value_counts(dropna=False))
  df['Oropharynx'] = (df['Oropharynx'] > 0)
  
  print(df['Gender'].value_counts(dropna=False))
  print(df['Oropharynx'].value_counts(dropna=False))
  # Drop patients still missing any of the predictor variables
  df = df.dropna(subset=['Age', 'Gender', 'BMI', 'DISE', 'Oropharynx'])

  print_demographics(df, header='Summary statistics after preprocessing')
  print("\n\n\n\n")

  # ===================== END PREPROCESSING =====================
  # ================= BEGIN REGRESSION ANALYSES =================

  # Since postop_AHI is a count, use Poisson regression
  postop_AHI_model = smf.poisson(
      formula=('postop_AHI ~ Age + C(Gender, Treatment(reference="Male")) + BMI + DISE'
               ' + C(Oropharynx, Treatment(reference=0)) + preop_AHI'),
      data=df
  ).fit()
  print("Poisson Regression of Post-Op AHI:\n")
  print(postop_AHI_model.summary())
  print("\n\n\n\n")
  
  # Since change_AHI is continuous, use linear regression
  change_AHI_model = smf.ols(
      formula=('change_AHI ~ Age + C(Gender, Treatment(reference="Male")) + BMI + DISE'
               ' + C(Oropharynx, Treatment(reference=0)) + preop_AHI'),
      data=df
  ).fit()
  print("Linear Regression of Change in AHI:\n")
  print(change_AHI_model.summary())
  print("\n\n\n\n")
  
  # Since Sher15 is boolean, use logistic regression
  Sher15_model = smf.logit(
      formula=('Sher15 ~ Age + C(Gender, Treatment(reference="Male")) + BMI + DISE'
               ' + C(Oropharynx, Treatment(reference=0)) + preop_AHI'),
      data=df
  ).fit()
  print("Logistic Regression of Sher15:\n")
  print(Sher15_model.summary())
  print("\n\n\n\n")

  # ================== END REGRESSION ANALYSES ==================
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
