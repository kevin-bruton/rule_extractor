import pandas as pd
import hashlib
import talib as ta
import numpy as np
import time
from datetime import datetime
import joblib
from scipy.stats import pointbiserialr
import re
from itertools import groupby

def transform_df(csv_name, exposicion_dias=3, threshold=25, date_column='DateTime', short=False):
    df = pd.read_csv(csv_name)

    df = add_indicators(df)

    #duplicates = df.columns[df.columns.duplicated()]
    #df = df.loc[:, ~df.columns.duplicated()]


    pips = 0
    if "JPY" in csv_name:
        pips = 100
    else:
        pips = 10000


    for i in range (2,31, 2):
        ret = []
        new_cols = []
        if(short == True):
            ret = ((df["Close"].shift(-1 * i) - df["Close"]) * pips) + 2
            new_cols = pd.DataFrame(np.array(ret) * -1, columns=[f"Return_{i}"])
        else:
            ret = ((df["Close"].shift(-1 * i) - df["Close"]) * pips) - 2
            new_cols = pd.DataFrame(np.array(ret), columns=[f"Return_{i}"])
			
        df = pd.concat([df, new_cols], axis=1)


    if(short == True):
        ret = ((df["Close"].shift(-1 * exposicion_dias) - df["Close"]) * pips) + 2
    else:
        ret = ((df["Close"].shift(-1 * exposicion_dias) - df["Close"]) * pips) - 2
		
    new_cols = pd.DataFrame(np.array(ret), columns=["Return"])
	
    if(short == True):
        new_cols["Return"] = new_cols["Return"] * -1
		
    df = pd.concat([df, new_cols], axis=1)
    
    target = (df["Return"] >= threshold).astype(int)
	

    target = df[f'Return_{exposicion_dias}'].copy()
    target = (df[f'Return_{exposicion_dias}'] >= threshold).astype(int)
    df = pd.concat([df, target.rename("Target")], axis=1) 

    for i in range (4,31, 2):
        target = (df[f'Return_{i}'] >= threshold).astype(int)
        df = pd.concat([df, target.rename(f'Target_{i}')], axis=1) 
	
    
    date_column = 'DateTime'
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], format='%Y%m%d %H:%M:%S.%f')
    #df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y %H:%M')

    day_of_month = df[date_column].apply(lambda x: x.day)
    month = df[date_column].apply(lambda x: x.month)
    day_of_week = df[date_column].apply(lambda x: x.weekday())
    year = df[date_column].apply(lambda x: x.year)

    df[date_column] = df[date_column].dt.strftime('%d/%m/%Y %H:%M')

    new_columns = pd.concat([day_of_month.rename('day_of_month'), month.rename('month'), day_of_week.rename('day_of_week'), year.rename('year')], axis=1)
    df = pd.concat([df, new_columns], axis=1)

    return df

def add_indicators(df):
    def add_indicator(indicator_calc_fn, df, indicator_column_names, start, end, step):
        new_columns = []
        for i in range(start, end, step):
            indicator_result = indicator_calc_fn(i)
            if type(indicator_column_names) is list:
                for result_idx in range(len(indicator_result)): 
                    new_columns.append(pd.DataFrame({f'{indicator_column_names[result_idx]}_{i}': indicator_result[result_idx]}))
            else:
                new_columns.append(pd.DataFrame({f'{indicator_column_names}_{i}': indicator_result}))
        return pd.concat([df] + new_columns, axis=1)
    
    calculate_ibs = lambda high, low, close: np.round((close - low) / (high - low), 2)
    df = add_indicator(lambda i: calculate_ibs(df['High'], df['Low'], df['Close']), df, 'ibs', 0, 1, 1)
    df = add_indicator(lambda i: ta.RSI(df['Close'], timeperiod=i), df, 'rsi', 2, 51, 2)
    df = add_indicator(lambda i: ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=i), df, 'adx', 2, 51, 2)
    df = add_indicator(lambda i: ta.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=i), df, 'plus_di', 2, 51, 2)
    df = add_indicator(lambda i: ta.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=i), df, 'minus_di', 2, 51, 2)
    df = add_indicator(lambda i: ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=i), df, 'willr', 2, 51, 2)
    df = add_indicator(lambda i: ta.MA(df['Close'], timeperiod=i, matype=0), df, 'sma', 2, 301, 2)
    df = add_indicator(lambda i: ta.EMA(df['Close'], timeperiod=i), df, 'ema', 2, 301, 2)
    df = add_indicator(lambda i: ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=i), df, 'atr', 2, 51, 2)
    df = add_indicator(lambda i: ta.STDDEV(df['Close'], timeperiod=i, nbdev=1), df, 'stdev', 2, 51, 2)
    df = add_indicator(lambda i: ta.BBANDS(df['Close'], timeperiod=i, nbdevup=2, nbdevdn=2, matype=0), df, ['bb_2_upper', 'bb_2_middle', 'bb_2_lower'], 2, 51, 2)
    df = add_indicator(lambda i: ta.BBANDS(df['Close'], timeperiod=i, nbdevup=3, nbdevdn=3, matype=0), df, ['bb_3_upper', 'bb_3_middle', 'bb_3_lower'], 2, 51, 2)
    df = add_indicator(lambda i: ta.BBANDS(df['Close'], timeperiod=i, nbdevup=4, nbdevdn=4, matype=0), df, ['bb_4_upper', 'bb_4_middle', 'bb_4_lower'], 2, 51, 2)
    df = add_indicator(lambda i: ta.MACD(df['Close'], fastperiod=7, slowperiod=13, signalperiod=i), df, ['macd_7_13', 'macdsig_7_13', 'macdh_7_13'], 3, 10, 3)
    df = add_indicator(lambda i: ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=i), df, ['macd_12_26', 'macdsig_12_26', 'macdh_12_26'], 3, 10, 3)
    df = add_indicator(lambda i: ta.MACD(df['Close'], fastperiod=26, slowperiod=52, signalperiod=i), df, ['macd_26_52', 'macdsig_26_52', 'macdh_26_52'], 3, 10, 3)
    df = add_indicator(lambda i: ta.MOM(df['Close'], timeperiod=i), df, 'mom', 2, 31, 2)
    df = add_indicator(lambda i: ta.AROONOSC(df['High'], df['Low'], timeperiod=i), df, 'aroonosc', 2, 51, 2)
    df = add_indicator(lambda i: ta.AROON(df['High'], df['Low'], timeperiod=i), df, ['aroon_up', 'aroon_down'], 2, 51, 2)
    df = add_indicator(lambda i: ta.ROC(df['Close'], timeperiod=i), df, 'roc', 2, 51, 2)

    return df

def add_shifted_columns(df, columns, shift=1):

    def shift_column(column, i):
        shifted = df[column].shift(i)
        if('ibs_' in column):
            column = 'ibs'
        return shifted.rename(f'{column}_sft_{i}')

    columns = df.columns
    lista_shift = ['rsi', 'adx', 'plus_di', 'minus_di', 'willr', 'bb', 'atr', 'stdev', 'Close', 'High', 'Low', 'aaro', 'mom']
    indicator_columns = {col for col in columns if any(name in col for name in lista_shift)}

    shifted_columns = []
    max_shift = 3

    for column in indicator_columns:
        for i in range(1, max_shift + 1):
            shifted_series = shift_column(column, i)
            shifted_columns.append(shifted_series)

    df = pd.concat([df] + shifted_columns, axis=1)
    return df

def split_data_validation(df, year_max_cut = '2022', year_min_cur= '2008'):
    data = df.query('year >= ' + year_max_cut).copy()
    df = df.query(year_min_cur+' < year <= 2022')
    df = df.reset_index(drop=True)
    return df, data

def map_creator(df):
    start_time = time.time()
    columns = df.columns
    no_sft_columns = [col for col in columns if 'sft' not in col]
    column_map = create_column_map(df, columns, no_sft_columns)
    end_time = time.time()
    seconds_taken = end_time - start_time
    print("Creation of the column map took", seconds_taken, "segundos")
    return column_map

def create_column_map(df, columns, no_sft_columns):
    column_map = {}
    rsi_columns = {col for col in columns if 'rsi_' in col}
    adx_columns = {col for col in columns if 'adx' in col}
    plus_di_columns = {col for col in columns if 'plus_di' in col}
    minus_di_columns = {col for col in columns if 'minus_di' in col}
    will_columns = {col for col in columns if 'willr' in col}
    sma_columns = {col for col in columns if 'sma' in col}
    mema_columns = {col for col in columns if 'mema' in col}
    ibs_columns = {col for col in columns if 'ibs_' in col}
    atr_columns = {col for col in columns if 'atr' in col}
    bbup_columns = {col for col in columns if 'bb_upper' in col}
    bbmid_columns = {col for col in columns if 'bb_middle' in col}
    bblow_columns = {col for col in columns if 'bb_lower' in col}
    macd_columns = {col for col in columns if 'macd' in col}
    macdsig_columns = {col for col in columns if 'macdsig' in col}
    macdh_columns = {col for col in columns if 'macdh' in col}
    ibsma_columns = {col for col in columns if 'ibma' in col}
    hh_columns = {col for col in columns if 'hh' in col}
    dayw_columns = {col for col in columns if 'day_of_week' in col}
    daym_columns = {col for col in columns if 'day_of_month' in col}
    ll_columns = {col for col in columns if 'll' in col}
    mom_columns = {col for col in columns if 'mom' in col}
    aaro_columns = {col for col in columns if 'aaro_' in col}
    roc_columns = {col for col in columns if 'roc' in col}
    stoch_columns = {col for col in columns if 'stoch' in col}
    stochk_columns = {col for col in columns if 'stochk' in col}
    stochd_columns = {col for col in columns if 'stochd' in col}
    stdev_columns = {col for col in columns if 'stdev' in col}
    aarod_columns = {col for col in columns if 'aarod_' in col}
    aarou_columns = {col for col in columns if 'aarou_' in col}
    
    for column in no_sft_columns:
        if 'rsi' in column:
            filtered_columns = rsi_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col and 'ibs' not in col}
            column_map[column] = [list(range(0, 101)), list(filtered_columns)]

        elif 'adx' in column:
            filtered_columns = adx_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(range(0, 101)), list(filtered_columns)]
        
        elif 'plus_di' in column:
            filtered_columns = plus_di_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(range(0, 101)), list(filtered_columns)]
                                                            
        elif 'minus_di' in column:
            filtered_columns = minus_di_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(range(0, 101)), list(filtered_columns)]
            
        elif 'willr' in column:
            filtered_columns = will_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(x for x in range(0, -101, -1)), list(filtered_columns)]
                                                            
        elif 'sma' in column:
            filtered_columns = sma_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(['Open', 'High', 'Low', 'Close', 'Close_sft_1', 'Close_sft_2', 'Close_sft_3', 'Low_sft_1', 'Low_sft_2', 'Low_sft_3', 'High_sft_1', 'High_sft_2', 'High_sft_3']), list(filtered_columns)]  

        elif 'mema' in column:
            filtered_columns = mema_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(['Open', 'High', 'Low', 'Close', 'Close_sft_1', 'Close_sft_2', 'Close_sft_3', 'Low_sft_1', 'Low_sft_2', 'Low_sft_3', 'High_sft_1', 'High_sft_2', 'High_sft_3']), list(filtered_columns)]			
            
        elif 'ibs_' in column:
            filtered_columns = ibs_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list([i / 100 for i in range(0, 101)]), list(filtered_columns)]
        
        elif 'atr' in column:
            filtered_columns = atr_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(filtered_columns)]
            
        elif 'bb_upper' in column:
            filtered_columns = bbup_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(['Open', 'High', 'Low', 'Close', 'Close_sft_1', 'Close_sft_2', 'Close_sft_3', 'Low_sft_1', 'Low_sft_2', 'Low_sft_3', 'High_sft_1', 'High_sft_2', 'High_sft_3']), list(filtered_columns)]
        
        elif 'bb_middle' in column:
            filtered_columns = bbmid_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(['Open', 'High', 'Low', 'Close', 'Close_sft_1', 'Close_sft_2', 'Close_sft_3', 'Low_sft_1', 'Low_sft_2', 'Low_sft_3', 'High_sft_1', 'High_sft_2', 'High_sft_3']), list(filtered_columns)]
            
        elif 'bb_lower' in column:
            filtered_columns = bblow_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(['Open', 'High', 'Low', 'Close', 'Close_sft_1', 'Close_sft_2', 'Close_sft_3', 'Low_sft_1', 'Low_sft_2', 'Low_sft_3', 'High_sft_1', 'High_sft_2', 'High_sft_3']), list(filtered_columns)]
            
        elif 'macd' in column:
            filtered_columns = macd_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(macdsig_columns)]
        
        elif 'macdsig' in column:
            filtered_columns = macdsig_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(macd_columns)]
                                  
        elif 'macdh' in column:
            filtered_columns = macdh_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(filtered_columns)]
        
        elif 'ibma' in column:
            filtered_columns = ibsma_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(ibs_columns)]
        
        elif 'hh' in column:
            filtered_columns = hh_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(['Open', 'High', 'Low', 'Close', 'Close_sft_1', 'Close_sft_2', 'Close_sft_3', 'Low_sft_1', 'Low_sft_2', 'Low_sft_3', 'High_sft_1', 'High_sft_2', 'High_sft_3'])]
            
        elif 'day_of_week' in column:
            filtered_columns = dayw_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list([i for i in range(0, 7)]), list([i for i in range(0, 7)])]
        
        elif 'day_of_week' in column:
            filtered_columns = dayw_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list([i for i in range(0, 7)]), list([i for i in range(0, 7)])]
        
        elif 'day_of_month' in column:
            filtered_columns = daym_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list([i for i in range(1, 32)]), list([i for i in range(1, 32)])]
        
        
        elif 'mom' in column:
            filtered_columns = mom_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(filtered_columns)]
            
        elif 'aaro_' in column:
            filtered_columns = aaro_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(x for x in range(-100, 101, 1)), list(filtered_columns)]
			
        elif 'aarod_' in column:
            filtered_columns = aarod_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(x for x in range(0, 101)), list(filtered_columns)]
			
        elif 'aarou_' in column:
            filtered_columns = aarou_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(x for x in range(0, 101)), list(filtered_columns)]
        
        elif 'roc' in column:
            filtered_columns = roc_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(filtered_columns)]
        
        elif 'stochd' in column:
            filtered_columns = stochd_columns - {column}
            comun_columns = stoch_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            filtered_comun_columns = {col for col in comun_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(filtered_comun_columns)]
            
        elif 'stochk' in column:
            filtered_columns = stochk_columns - {column}
            comun_columns = stoch_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            filtered_comun_columns = {col for col in comun_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(filtered_comun_columns)]
        
        elif 'stdev' in column:
            filtered_columns = stdev_columns - {column}
            filtered_columns = {col for col in filtered_columns if 'condition' not in col}
            column_map[column] = [list(filtered_columns), list(filtered_columns)]

    return column_map

def generate_all_rules(column_map, df):
    all_rules = [] 
    
    for column, (possible_values, related_columns) in column_map.items():
        if 'day' in column:
            operators = ['>', '<', '==', '>=', '<=']
        else:
            operators = ['>=', '<=']

        for value in possible_values:
            for operator in operators:
                condition = f"{column} {operator} {value}"
                all_rules.append(condition)

        for value in related_columns:
                for operator in operators:
                    related_condition = f"{column} {operator} {value}"
                    all_rules.append(related_condition)
    
    return all_rules

def generate_hash(s):
    try:
        first_part = re.findall(r'^\D+', s.split(' ')[0])[0]
        comparison_operator = re.findall(r'[<=>=]+', s)[0]
        try:
            second_part = re.findall(r'^\D+', s.split(' ')[2].split('_')[0])[0]
        except IndexError:
            second_part = 'num'
        combined = first_part + comparison_operator + second_part
        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception as e:
        return 12345678910


def process_rule_chunk(df, data, df_columns, rule_chunk, target_values, target, returns_columns):
    chunk_results = {}
    chunk_stats = {}
    for rule in rule_chunk:
        try:
            parts = rule.split()
            if len(parts) == 3:
                column1, operator, column2_or_value = parts
                idx1 = df_columns.get_loc(column1)
                idx_return = df_columns.get_loc('Return')
                
                try:
                    value = float(column2_or_value)
                    idx2 = None
                except ValueError:
                    idx2 = df_columns.get_loc(column2_or_value)
                    
                condition_eval_df = None
                if operator == '>=':
                    if idx2 is None:
                        condition_eval_df = (data[:, idx1] >= value)
                    else:
                        condition_eval_df = (data[:, idx1] >= data[:, idx2])
                elif operator == '<=':
                    if idx2 is None:
                        condition_eval_df = (data[:, idx1] <= value)
                    else:
                        condition_eval_df = (data[:, idx1] <= data[:, idx2])
                elif operator == '==':
                    if idx2 is None:
                        condition_eval_df = (data[:, idx1] == value)
                    else:
                        condition_eval_df = (data[:, idx1] == data[:, idx2])
                elif operator == '>':
                    if idx2 is None:
                        condition_eval_df = (data[:, idx1] > value)
                    else:
                        condition_eval_df = (data[:, idx1] > data[:, idx2])
                elif operator == '<':
                    if idx2 is None:
                        condition_eval_df = (data[:, idx1] < value)
                    else:
                        condition_eval_df = (data[:, idx1] < data[:, idx2])           
                        
                condition_eval = condition_eval_df.astype(np.int8)
                ones_count = np.sum(condition_eval)
                if ones_count < 100:
                    continue
                
                if np.all(condition_eval == condition_eval[0]) or np.all(target == target[0]):
                    continue  # Skip this iteration if either array is constant (all vaues are the same)

                correlation, _ = pointbiserialr(condition_eval, target)
                if np.isnan(correlation):
                    continue

                length = len(condition_eval)
                zeros_count = length - ones_count
                win_rate = np.sum(condition_eval & target_values) / ones_count if ones_count > 0 else 0

                
                sum_returns = np.sum(data[condition_eval_df, idx_return])
                
                sum_positive_returns = np.sum(data[condition_eval_df & (data[:, idx_return] > 0), idx_return])
                sum_negative_returns = np.sum(data[condition_eval_df & (data[:, idx_return] < 0), idx_return])
                
                profit_factor = 0
                if sum_negative_returns != 0:
                    profit_factor = sum_positive_returns / -sum_negative_returns
                else:
                    profit_factor = float('inf')
                    
                optimal_pf = 0
                optimal_exposition = str(4)
                profit_factors = {}
                for ret_col in returns_columns:
                    idx_return = df_columns.get_loc(ret_col)
                    sum_positive_returns = np.sum(data[condition_eval_df & (data[:, idx_return] > 0), idx_return])
                    sum_negative_returns = -np.sum(data[condition_eval_df & (data[:, idx_return] < 0), idx_return])
                    profit_factor = sum_positive_returns / sum_negative_returns if sum_negative_returns != 0 else float('inf')
                    match = re.search(r'Return_(\d+)', ret_col)
                    number = 4
                    if match:
                        number = match.group(1)
                        chunk_stats[f'pf_{str(number)}'] = profit_factor
                        if(profit_factor != np.inf and profit_factor > optimal_pf):
                            optimal_pf = profit_factor
                            optimal_exposition = str(number)
                        
                target_optimal = df[f'Target_{optimal_exposition}'].values
                target_values_optimal = df[f'Target_{optimal_exposition}'].apply(lambda x: 1 if x > 0 else 0).values
                correlation_optimal, _ = pointbiserialr(condition_eval, target_optimal)
                win_rate_optimal = np.sum(condition_eval & target_values_optimal) / ones_count if ones_count > 0 else 0
                
                
                chunk_results[rule] = condition_eval
                chunk_stats[rule] = {
                    'correlation': correlation,
                    'length': length,
                    'ones_count': ones_count,
                    'zeros_count': zeros_count,
                    'win_rate': round(win_rate * 100, 2),
                    'return': sum_returns,
                    'hash': generate_hash(rule),
                    'optimal_exposition': optimal_exposition,
                    'correlation_optimal': correlation_optimal,
                    'win_rate_optimal': round(win_rate_optimal * 100, 2),
                }
                chunk_stats[rule].update({f'pf_{i}': chunk_stats.pop(f'pf_{i}') for i in range(4, 31, 2)})
                
            else:
                print(f"Rule '{rule}' could not be parsed.")
        except Exception as e:
            print(f"Error processing rule '{rule}': {e}")

    return chunk_results, chunk_stats


def evaluate_rules_numpy(df, all_rules, target_column='Target', output_file='results2_h5.h5'):
    print('Starting process..', datetime.now().strftime('%d-%m-%Y %H:%M:%S'))
    data = df.to_numpy()
    target = df[target_column].values
    target_values = df[target_column].apply(lambda x: 1 if x > 0 else 0).values


    num_cores = joblib.cpu_count()

    # Dividimos las reglas en chunks
    chunk_size = len(all_rules) // num_cores 
    rule_chunks = [all_rules[i:i + chunk_size] for i in range(0, len(all_rules), chunk_size)]
    
    returns_columns = [f'Return_{i}' for i in range(4, 31, 2)]  

    # Utilizar Joblib para paralelizar la evaluación de las reglas
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_rule_chunk)(df, data, df.columns, rule_chunk, target_values, target, returns_columns) for rule_chunk in rule_chunks
    )
    
    all_results = {}
    all_stats = {}
    for chunk_results, chunk_stats in results:
        all_results.update(chunk_results)
        all_stats.update(chunk_stats)
        
    result_df = pd.DataFrame(all_results)
    result_df['Target'] = df['Target'].copy()
    result_df['Return'] = df['Return'].copy()
    
    returns_columns = []
    for i in range(4, 31, 2):
        returns_columns.append(f'Return_{i}')
        
        
    for column_ in returns_columns:
        result_df[column_] = df[column_].values
        
    
    print('Ending process.. saving', datetime.now().strftime('%d-%m-%Y %H:%M:%S'), len(result_df))
    result_df.to_hdf(output_file, key='result_df', mode='w')
    sorted_stats = sorted(all_stats.items(), key=lambda item: item[1]['correlation'], reverse=True)
    
    groups = groupby(sorted_stats, key=lambda item: item[1]['correlation'])
    unique_stats = [next(g) for _, g in groups]
    
    df = pd.DataFrame([item[1] for item in unique_stats])
    df['condition'] = [item[0] for item in unique_stats]
    df = df[['condition'] + [col for col in df.columns if col != 'condition']]
    
    print('Ended process',datetime.now().strftime('%d-%m-%Y %H:%M:%S'))
    return df, unique_stats