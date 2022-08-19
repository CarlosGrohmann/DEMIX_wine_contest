# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Functions for the DEMIX Wine Contest Jupyter notebook
# Carlos H. Grohmann
# version 2022-07-18

import sys,os
import pandas as pd
import numpy as np
from scipy.special import ndtri
from ipywidgets import Button
from tkinter import Tk, filedialog
from IPython.display import clear_output, display

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# from: https://edusecrets.com/lesson-02-creating-a-file-select-button-in-jupyter-notebook/
def select_files(b):
    clear_output()
    root = Tk()
    root.withdraw() # Hide the main window.
    root.call('wm', 'attributes', '.', '-topmost', True) # Raise the root to the top of all windows.
    b.files = filedialog.askopenfilename(multiple=True) # List of selected files will be set button's file attribute.
    # b.names = [f.split('/')[-1].split('.')[0] for f in b.files] # only names of files
    names = [f.split('/')[-1] for f in b.files]
    print(f'Selected file(s):{str(names)[1:-1]}') # Print the list of files selected.



# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
def make_criteria_df(csv_list):
    ''' open csv files with metrics (from MicroDEM), 
    already in a transposed format (one criterion per row)'''
    df_merged = pd.DataFrame()
    for f in csv_list:
        df = pd.read_csv(f, sep=',',engine='python',comment='#',quotechar='"')
        df_merged = pd.concat([df_merged,df],sort=False)
    return df_merged




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
def rank_with_tolerance(sr, tolerance, method):
    ''' thanks to https://stackoverflow.com/a/72957060/4984000'''
    vals = pd.Series(sr.unique()).sort_values()
    vals.index = vals
    vals = vals.mask(vals - vals.shift(1) < tolerance, vals.shift(1))
    return sr.map(vals).fillna(sr).rank(method=method)




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
def sort_with_tolerance_np(sr, tolerance):
    '''sort values in a row using numpy and a given tolerance '''
    arr = sr.to_numpy().astype(float)
    arr_sort = np.sort(arr)
    arr_argsort = arr.argsort().argsort()
    arr_shift = np.pad(arr_sort, (1,), 'constant', constant_values=np.nan)[:-2]
    arr_diff = arr_sort - arr_shift
    arr_tol = np.where(arr_diff < tolerance, arr_shift, arr_sort)
    sr_tol = pd.Series(arr_tol[arr_argsort])
    return sr_tol




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# calculate ranks for criteria (error metrics) in dataframes
def make_rank_df_oneTolerance(df,dem_list,tolerance,method):
    '''calculate ranks for metrics in dataframes - uses a single tolerance value'''
    # subset of df with only DEMs values
    df_for_ranking = df[dem_list]
    dem_cols_rank = [i+'_rank' for i in dem_list]
    # rank values in df
    if tolerance is not None:
        tolerance = tolerance + 1e-10
        print(f'Ranking using tolerance of: {tolerance:2.3}',end='\n\n')
        print()
        df_temp = df_for_ranking.apply(lambda row: sort_with_tolerance_np(row, tolerance=tolerance))
        df_temp = df_temp.rank(method=method, ascending=True, axis=1, numeric_only=True).add_suffix('_rank')
        df_ranks = pd.concat([df.reset_index(), df_temp.reset_index()], axis=1)
        df_ranks = df_ranks.drop(['index'], axis=1)
    else:
        print('Ranking without tolerance',end='\n\n')
        df_temp = df_for_ranking.rank(method=method,ascending=True,axis=1,numeric_only=True).add_suffix('_rank')
        df_ranks = pd.concat([df.reset_index(), df_temp.reset_index()], axis=1)
        df_ranks = df_ranks.drop(['index'], axis=1)
    # create cols for squared ranks
    for col in dem_list:
        df_ranks[col+'_rank_sq'] = df_ranks[col+'_rank']**2
    return df_ranks




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# calculate ranks for criteria (error metrics) in dataframes
def make_rank_df(df,dem_list,tolerance_dict,method):
    '''calculate ranks for metrics in dataframes - accepts a 
    dictionary of criterion/tolerance values '''
    dem_cols_rank = [i+'_rank' for i in dem_list]
    df_ranks = pd.DataFrame(columns=list(df.columns) + dem_cols_rank)
    if tolerance_dict is not None:
        print(f'Ranking with user-defined tolerances',end='\n\n')
        # iterate through dict of criterion/tolerance
        for key, value in tolerance_dict.items():
            criterion = key
            tolerance = value + 1e-10
            # subset of df - only rows of selected criterion 
            df_crit = df.loc[df['CRITERION'] == criterion]
            # subset of df_crit - only DEMs values
            df_for_ranking = df_crit[dem_list]
            # rank values in df
            df_temp = df_for_ranking.apply(lambda row: sort_with_tolerance_np(row, tolerance=tolerance))
            df_temp = df_temp.rank(method=method, ascending=True, axis=1, numeric_only=True).add_suffix('_rank')
            df_crit_rnk = pd.concat([df_crit.reset_index(), df_temp.reset_index()], axis=1)
            df_crit_rnk = df_crit_rnk.drop(['index'], axis=1)
            df_ranks = pd.concat([df_ranks, df_crit_rnk])
    else:
        print('Ranking without tolerance',end='\n\n')
        df_for_ranking = df[dem_list]
        df_temp = df_for_ranking.rank(method=method,ascending=True,axis=1,numeric_only=True).add_suffix('_rank')
        df_ranks = pd.concat([df.reset_index(), df_temp.reset_index()], axis=1)
        df_ranks = df_ranks.drop(['index'], axis=1)
    # df_ranks = df_ranks.drop(['index'], axis=1)  
    # create cols for squared ranks
    for col in dem_list:
        df_ranks[col+'_rank_sq'] = df_ranks[col+'_rank']**2
    return df_ranks




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# calculate friedman stats and check the critical values from tables
def friedman_stats(df,dem_list,tables_dir,cl):
    '''this func will calculate friedman stats and then check the critical values from tables'''
    dem_cols = dem_list
    dem_cols_rank = [i+'_rank' for i in dem_cols]
    dem_cols_rank_sq = [i+'_rank_sq' for i in dem_cols]
    #
    n = len(df) # number of CRITERIA 
    k = len(dem_cols) # number of DEMs being compared
    cf = 1/4*n*k*((k+1)**2)
    #
    ranks_vect = df[dem_cols_rank].sum() # excel Sheet1!J10:O10
    sum_ranks_vect = ranks_vect.sum() # excel SOMA(Sheet1!J10:O10)
    ranks_sq_vect = ranks_vect.pow(2) # excel Sheet1!J11:O11
    sum_ranks_sq_vect = ranks_sq_vect.sum() # excel SOMA(Sheet1!J11:O11)
    sum_squared_ranks = df[dem_cols_rank_sq].sum().sum() # excel SOMA(Sheet1!Q14:V322)

    chi_r =( (n * (k-1)) / (sum_squared_ranks - cf) * (sum_ranks_sq_vect/n - cf) )
    # =+E5*(E6-1)/(SOMA(Sheet1!Q14:V322)-E7)*(SOMA(Sheet1!J11:O11)/E5-E7)
    #
    print(f'N = {n} (number of "opinions")')
    print(f'k = {k} (number of DEMs)')
    print(f'CF = {cf}')
#     print(f'sum of ranks (vector) = {ranks_vect.tolist()}')  # excel Sheet1!J10:O10
#     print(f'sum of (ranks squared) = {ranks_sq_vect.tolist()}')  # excel Sheet1!J11:O11
#     print(f'sum of squared ranks = {sum_squared_ranks}')         # excel Sheet2!N4
#     print(f'sum of ranks squared (total) = {sum_ranks_sq_vect}') # excel Sheet2!N5
    print(f'chi_r = {chi_r:4.3f}')
    #
    #get values from tables
    CL = cl
    table_needed = f'k_{k}.txt'
    # print(f'Table needed: {table_needed}')
    df_critical = pd.read_csv(os.path.join(tables_dir,table_needed),sep=';')
    #df_critical = df_critical[: , :-1] # drop last column as it is empty
    # find chi_crit in table
    n_alpha = f'N={n}' 
    # try to get the value, if not possible, use last row
    try:
        idx = df_critical.loc[df_critical['alpha'] == n_alpha].index[0]
        col = f'{CL:05.3f}'
        chi_crit = df_critical.at[idx, col]
    except:
        idx = df_critical.index[-1]
        col = f'{CL:05.3f}'
        chi_crit = df_critical.at[idx, col]
    print(f'For k={k}, CL={CL}, and N={n}, the critical value to compare is chi_crit={chi_crit:4.3f}')
    # print(f'chi_r: {chi_r:04.3f}')
    #print(f'chi_crit: {chi_crit}')
    #
    if chi_r > chi_crit:
        print(f'And since chi_r ({chi_r:4.3f}) is greater than chi_crit ({chi_crit:4.3f})...')
        print(f'Yay!! We can reject the null hipothesis and go to the Post-Hoc analysis!!')
    else:
        print(f'But since chi_r ({chi_r:4.3f}) is less than chi_crit ({chi_crit:4.3f})...')
        print('Oh, no! We cannot disprove the null hipothesis at the given CL...')



# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# DEMs ranked
def print_dems_ranked(df,dem_list):
    '''print ranked DEMs'''
    dem_cols_rank = [i+'_rank' for i in dem_list]
    df_ranks = df
    n_opinions = len(df_ranks)
    pd_ranked = pd.DataFrame()
    dems_ranked = df_ranks[dem_cols_rank].sum()
    pd_ranked['rank_sum'] = dems_ranked
    # pd_ranked['rank'] = pd_ranked['rank_sum'].rank(ascending=1)
    pd_ranked['rnk_div_opn'] = pd_ranked['rank_sum'].div(n_opinions).round(3)
    pd_ranked.index = dem_list
    print(pd_ranked.sort_values(by='rank_sum'))




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
def show_filters(grid):
    '''func to show which filters are defined for each column'''
    state_dict_cols = grid.get_state()['_columns']
    nothing_select = True
    filters_dict = {}
    for key,val in state_dict_cols.items():
        for c_key,c_val in val.items():
            if c_key == 'filter_info':
                if val[c_key]['selected'] is not None:
                    nothing_select = False
                    cols_vals_lst = val['values']
                    filter_sel_idx = val[c_key]['selected']
                    filter_selection = [cols_vals_lst[i] for i in filter_sel_idx]
                    print(f'Filter settings for column {key}:{filter_selection}')
                    filters_dict[key] = filter_selection
    if nothing_select == True:
        print('No filters applied')
    print()
    return filters_dict



# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
#apply Bonferroni-Dunn test
def bonferroni_dunn_test(df,dems_list,alpha=0.95):
    '''apply Bonferroni-Dunn test'''
    
    dem_cols_rank = [i+'_rank' for i in dems_list]
    dems_ranked = df[dem_cols_rank].sum()

    k = len(dems_list)
    n = len(df) # number of CRITERIA 

    # alpha = 0.95 default value
    quant =  1-alpha/k/(k-1)
    zi = ndtri(quant)
    crit = zi*np.sqrt(n*k*(k+1)/6) # always divide by 6
    
    # create table
    cols = ['DEM'] + dems_list
    df_table = pd.DataFrame(columns=cols) # df and cols names
    df_table['DEM'] = dems_list # first column of df

    # get ranks values 
    ranks_vals = dems_ranked.to_frame().T

    # populate table
    for d1 in dems_list:
        r = dems_list.index(d1)
        for d2 in dems_list:
            rank_dem1 = ranks_vals[f'{d1}_rank'].values[0]
            rank_dem2 = ranks_vals[f'{d2}_rank'].values[0]
            # print(d1,d2,rank_dem1,rank_dem2)
            if np.abs(rank_dem1 - rank_dem2) > crit:
                df_table.at[r,d2] = 'Y'
            else:
                df_table.at[r,d2] = 'N'

    # use numpy to get only the upper triangle of the table 
    m = np.triu(df_table.values,k=2)
    df2 = pd.DataFrame(m,columns=cols)
    df2['DEM'] = dems_list
    # return df2
    return df2



# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
def tables_dems_ranked_bonferroni_dunn(df,dem_list,alpha=0.95):
    '''print ranked DEMs and apply Bonferroni-Dunn test'''
    
    dem_cols_rank = [i+'_rank' for i in dem_list]
    dems_ranked = df[dem_cols_rank].sum()
    k = len(dem_list)
    n = len(df) # number of opinions 
    # alpha = 0.95 # default value
    quant = 1-alpha/k/(k-1)
    zi = ndtri(quant)
    crit = zi*np.sqrt(n*k*(k+1)/6)
    tie_dict = {}
    
    # Y/N table - Bonferroni-Dunn test
    cols = ['DEM'] + dem_list
    df_table = pd.DataFrame(columns=cols) # df and cols names
    df_table['DEM'] = dem_list # first column of df
    # get ranks values 
    ranks_vals = dems_ranked.to_frame().T
    # populate table
    for d1 in dem_list:
        tie_dict[d1] = []
        r = dem_list.index(d1)
        for d2 in dem_list:
            rank_dem1 = ranks_vals[f'{d1}_rank'].values[0]
            rank_dem2 = ranks_vals[f'{d2}_rank'].values[0]
            # print(d1,d2,rank_dem1,rank_dem2)
            if np.abs(rank_dem1 - rank_dem2) > crit:
                df_table.at[r,d2] = 'Y'
            else:
                df_table.at[r,d2] = 'N'
                tie_dict[d1].append(d2)
                
    # use numpy to get only the upper triangle of the table 
    m = np.triu(df_table.values, k=2)
    df_yn = pd.DataFrame(m, columns=cols)
    df_yn['DEM'] = dem_list

    # table of ranked DEMs
    n_opinions = len(df)
    pd_ranked = pd.DataFrame()
    dems_rnk_sum = df[dem_cols_rank].sum()
    pd_ranked['sum_ranks'] = dems_rnk_sum
    pd_ranked['sum_ranks_div_opin'] = pd_ranked['sum_ranks'].div(n_opinions).round(3)
    pd_ranked.index = dem_list
    
    # check for ties in final ranking
    for k,v in tie_dict.items():
        v.remove(k)
        if v:
            tie_dict[k] = ','.join(v)
        else:
            tie_dict[k] = ''
    pd_ranked['not_stat_diff'] = pd.Series(tie_dict)

    cols_long = ['Sum of ranks',
                 'Sum of ranks divided \n by number of opinions',
                 'Not statistically \n different from']

    df_display = pd_ranked.sort_values(by='sum_ranks')
    df_display.columns = cols_long
    
    return df_display,df_yn,tie_dict




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# these 3 funcs are used to get a new dataframe with the ranks
# based on some conditions, like equality to DSM/DTM, the
# criteria, etc

# func to get ranks based on equality of criteria in df cols
def get_ranks_for_equal_criteria(df,crit_dict,dem_list):
    '''this func receives a dict of target_columns:value and
    returns the ranks for a subset of the dataframe where
    target_colum==value'''
    df_temp = pd.DataFrame(columns=dem_list)
    for key,val in crit_dict.items():
        df_select = df[df[val]==key]
        dem_cols_rank = [i+'_rank' for i in dem_list]
        dems_ranked = list(df_select[dem_cols_rank].sum())
        df_temp.loc[key] = list(dems_ranked)
    return df_temp

# func to get ranks based on GREATER THAN of criteria in df cols
def get_ranks_for_gt_criteria(df,crit_dict,dem_list):
    '''this func receives a dict of target_columns:value and
    returns the ranks for a subset of the dataframe where
    target_colum>=value'''
    df_temp = pd.DataFrame(columns=dem_list)
    for key,val in crit_dict.items():
        key_number = int(key.split(' > ')[1].split(' ')[0])
        df_select = df[df[val]>=key_number]
        dem_cols_rank = [i+'_rank' for i in dem_list]
        dems_ranked = list(df_select[dem_cols_rank].sum())
        df_temp.loc[key] = list(dems_ranked)
    return df_temp

# func to get ranks based on LESS THAN of criteria in df cols
def get_ranks_for_lt_criteria(df,crit_dict,dem_list):
    '''this func receives a dict of target_columns:value and
    returns the ranks for a subset of the dataframe where
    target_colum<=value'''
    df_temp = pd.DataFrame(columns=dem_list)
    for key,val in crit_dict.items():
        key_number = int(key.split(' < ')[1].split(' ')[0])
        df_select = df[df[val]<=key_number]
        dem_cols_rank = [i+'_rank' for i in dem_list]
        dems_ranked = list(df_select[dem_cols_rank].sum())
        df_temp.loc[key] = list(dems_ranked)
    return df_temp


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# func to get ranks based defined condition
def get_ranks_condition(df,cond_list,label_list,dem_list):
    '''returns a df based on a condition, passed as string'''
    df_temp = pd.DataFrame(columns=dem_list)
    for cond,label in zip(cond_list,label_list):
        df_select = df.query(cond)
        dem_cols_rank = [i+'_rank' for i in dem_list]
        dems_ranked = list(df_select[dem_cols_rank].sum().div(len(df_select)))
        df_temp.loc[f'{label}'] = list(dems_ranked)
    return df_temp



# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
