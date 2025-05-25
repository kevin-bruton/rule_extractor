import time

from versionk.functions import evaluate_rules_numpy, transform_df, map_creator, generate_all_rules
from versionk.utils import create_directory

def entry_finder(csv_file, expose_days, threshold, short, year_start, year_end):
  date_column = 'DateTime'
  df = transform_df(csv_file, expose_days, threshold, short)
  dir_results_name = f'results_{expose_days}_{threshold}'
  create_directory(dir_results_name)
  full_df = df.copy()
  df = df.query(f'{year_start} < year < {year_end}')

  df = df.reset_index(drop=True)
  column_map = map_creator(df)
  print('Number of columns:', len(df.columns))

  return_columns = [col for col in df.columns if 'Return_' in col]
  df[return_columns+['Return']].tail(25)

  print('Num data on day of week 0:', len(df[df['day_of_week'] == 0]))
  print('Num data on day of week 1:', len(df[df['day_of_week'] == 1]))
  print('Num data on day of week 2:', len(df[df['day_of_week'] == 2]))
  print('Num data on day of week 3:', len(df[df['day_of_week'] == 3]))
  print('Num data on day of week 4:', len(df[df['day_of_week'] == 4]))
  print('Num data on day of week 5:', len(df[df['day_of_week'] == 5]))
  print('Num data on day of week 6:', len(df[df['day_of_week'] == 6]))
  print('Num data on day of week 7:', len(df[df['day_of_week'] == 7]))
  
  all_rules = generate_all_rules(column_map, df)

  print('Num rules:', len(all_rules))
  
  start_time = time.time()
  df_rules, sorted_stats = evaluate_rules_numpy(df, all_rules)

  print(f"The process took {time.time() - start_time} seconds.")

  print('Num rules:', len(df_rules))

  df_rules

  sorted_stats

  df_sorted = df_rules.sort_values(by='pf_10', ascending=False)
  df_sorted
