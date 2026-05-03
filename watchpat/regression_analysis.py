import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from preprocessing import add_age, preprocess_df

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 4)

data_filename = 'data/WatchPATStudyData.xlsx'
demographics_filename = 'data/watchpat_with_AGE.xlsx'  # Used for matching

def main(df: pd.DataFrame) -> None:
  
  # =========================== BEGIN PREPROCESSING ===========================
  df, matching_variables, dependent_variables = preprocess_df(df)

  # Instead of having separate columns for each measurement period, combine
  # measurements of each dependent variable and add a 'Month' column indicating
  # the measurement period
  df = pd.wide_to_long(
      df=df,
      stubnames=dependent_variables,
      i=['Patient #'],
      j='Month',
      sep='.',
  ).reset_index()
  df['Month'] *= 6  # Convert 6-month measurement periods to months

  # ========================== END PREPROCESSING ==============================
  # ===================== BEGIN MIXED EFFECTS ANALYSIS ========================

  # Fit a separate model for each DV, but accumulate the results into a single
  # table and print them together.
  results = []
  covariates = ' + '.join(matching_variables) + " + Month * Treatment"
  for dv in dependent_variables:
    variables_needed = matching_variables + [dv, 'Patient #', 'Month', 'Treatment']
    df_dv = df[variables_needed].dropna()
    formula = f"Q('{dv}') ~ {covariates}"
    mixed_effects_model = smf.mixedlm(
        formula,
        data=df_dv,
        groups=df_dv["Patient #"],
    ).fit()

    # Interleave coeffs and ps for ease of reading table by eye
    coeffs = mixed_effects_model.fe_params
    ps = mixed_effects_model.pvalues[:-1]  # Omit p-value for Group Var
    beta_month_plus_interaction = coeffs.iloc[2] + coeffs.iloc[3]
    interleaved = [x for pair in zip(coeffs, ps) for x in pair]
    results.append(
        [mixed_effects_model.nobs] + interleaved + [beta_month_plus_interaction]
    )

  matching_variable_columns = [f'{prefix}_{variable}' for variable in matching_variables for prefix in ('beta', 'p')]
  column_names = (
      ['n', 'beta_Intercept', 'p_Intercept']
      + matching_variable_columns
      + [
          'beta_Treatment[UAS]', 'p_Treatment[UAS]',
          'beta_Month', 'p_Month',
          'beta_Interaction', 'p_Interaction',
          # Although this isn't an explicit model parameter, this corresponds to the
          # rate of change per month for the UAS group.
          'beta_Month+Interaction',
      ]
  )
  print(pd.DataFrame(results, columns=column_names, index=dependent_variables))#.drop(columns=['beta_Intercept', 'p_Intercept']))

  # ====================== END MIXED EFFECTS ANALYSIS =========================

  df['MDA_gt_0.4'] = (df['MDA'] >= 0.4).astype(float)
  df.loc[df['MDA'].isnull(), 'MDA_gt_0.4'] = np.nan
  print(df.groupby(by=['Month', 'Treatment'])['MDA'].mean())
  print(df.groupby(by=['Month', 'Treatment'])['MDA_gt_0.4'].value_counts(dropna=False))

if __name__ == "__main__":
  # Read in just the sheet we need from the Excel file
  sheet_name = 'All'  # or 'Active'
  main(pd.read_excel(data_filename, sheet_name=sheet_name))
