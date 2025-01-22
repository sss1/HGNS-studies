import pandas as pd

import HGNS_predictors_of_success as hgns_pos

pd.set_option('display.max_rows', None)

def get_supine_dependent_group(df):
  # When supine AHI is missing, use Medicare or AASMI supine AHI instead
  df['supine_AHI'] = (df['Supine AHI']
                      .fillna(df['Preop PSG Medicare Supine AHI'])
                      .fillna(df['Preop PSG AASM Supine AHI']))
  df['is_supine_dependent'] = (df['supine_AHI'] >= 2*df['Non-supine AHI'])
  return df[df['is_supine_dependent']]


if __name__ == "__main__":
  # Additional columns needed to identify supine-dependent subgroup
  supine_dependent_usecols = [
      'Supine AHI',
      'Preop PSG AASM Supine AHI',
      'Preop PSG Medicare Supine AHI',
      'Non-supine AHI'
  ]
  df = pd.read_csv(
      hgns_pos.data_filename,
      usecols=hgns_pos.usecols + supine_dependent_usecols
  )
  df = get_supine_dependent_group(df)
  hgns_pos.main(df[hgns_pos.usecols])
