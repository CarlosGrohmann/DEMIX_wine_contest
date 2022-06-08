# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Functions for the DEMIX Wine Contest Jupyter notebook
# Carlos H. Grohmann - 2022

import sys,os
import pandas as pd
import numpy as np
from scipy.special import ndtri

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# make dataframes from csv files and append all to a single one
def make_criteria_df(csv_list,datadir):
    ''' open csv files with metrics (from PG), 
    get them in a format with one metric per row,
    join all into a single df at the end'''
    
    # final df
    df_merged = pd.DataFrame()
    
    # start with each csv file
    for f in csv_list:
        df_csv = pd.DataFrame()
        f = os.path.join(datadir,f)
        df = pd.read_csv(f, sep=',',engine='python',comment='#',quotechar='"')
        df_csv = pd.concat([df_csv,df],sort=False)
        
        # get lists from df
        dems_list = list(df_csv['DEM'].unique())
        tile_list = list(df_csv['DEMIX_TILE'].unique())
        ref_list = list(df_csv['REF_TYPE'].unique())
        
        # dfs for row-based format 
        df_left = pd.DataFrame(columns=(['AREA', 'DEMIX_TILE', 'REF_TYPE']))
        df_right = pd.DataFrame(columns=(['METRIC']+dems_list))     
        group_dem = df_csv.groupby(['DEMIX_TILE','REF_TYPE'])
        
        # convert from PG format to row-based
        for g_idx, group in group_dem:
            # ------------------------------
            # create metadata df (right one)
            area = group.T.iloc[0,1]
            tile = group.T.iloc[1,1]
            surf = group.T.iloc[4,1]
            dictemp = {'AREA':area, 'DEMIX_TILE':tile, 'REF_TYPE':surf}
            # ------------------------------
            # create metrics df (left one)
            metrics_T = group.T.drop(labels=group.T.index[[0,1,2,3,4]], axis=0).reset_index()
            # ISSUE: sometimes FABDEM has no values...
            met_cols = ['METRIC']+dems_list
            if len(metrics_T.columns) == len(met_cols):
                metrics_T.columns = met_cols
            else:
                metrics_T.columns = met_cols[:-1]
            # ------------------------------
            
            # add each row to df
            for (r_idx,row) in metrics_T.iterrows():
                df_right = df_right.append(row.to_dict(), ignore_index=True)
                df_left  = df_left.append(dictemp, ignore_index=True)
                
        # concat dfs left and right
        df_row_based = pd.concat([df_left,df_right], axis=1)
        
        # concat df
        df_merged = pd.concat([df_merged,df_row_based])
        
    # return
    df_merged.reset_index(inplace=True,drop=True)
    return df_merged




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# calculate ranks for criteria (error metrics) in dataframes
def make_rank_df(df,dem_list):
    '''calculate ranks for criteria (error metrics) in dataframes'''
    # rank values in df
    df_ranks = pd.concat([df, df.rank(method='min',ascending=True,axis=1,numeric_only=True).add_suffix('_rank')], axis=1)
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
    sum_ranks_dems = df[dem_cols_rank].sum()
    sum_squared_ranks_dems = df[dem_cols_rank_sq].sum().sum()
    sum_ranks = sum_ranks_dems.sum()
    sum_ranks_sq_dems = sum_ranks_dems.pow(2)
    sum_ranks_sq = sum_ranks_sq_dems.sum()
    #
    chi_r =( (n * (k-1)) / (sum_squared_ranks_dems - cf) * (sum_ranks_sq/n - cf) )
    # chi_r =( (n * (k-1)) * (sum_ranks_sq/n - cf) )/ (sum_ranks - cf)
    # =+E5*(E6-1)/(SUM($Sheet1.Q14:V322)-E7)*(SUM($Sheet1.J11:O11)/E5-E7)
    #
    print(f'n = {n} (number of criteria)')
    print(f'k = {k} (number of DEMs)')
    print(f'cf = {cf}')
    print(f'sum of ranks = {sum_ranks}')
    print(f'sum of (ranks squared) = {sum_ranks_sq}')
    print(f'sum of (squared ranks) = {sum_squared_ranks_dems}')
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
    print(f'For k={k}, CL={CL}, and N={n}, the critical value to compare is chi_crit={chi_crit}')
    # print(f'chi_r: {chi_r:04.3f}')
    #print(f'chi_crit: {chi_crit}')
    #
    if chi_r > chi_crit:
        print(f'Yay!! We can reject the null hipothesis and go to the Post-Hoc analysis!!')
    else:
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
    pd_ranked = pd.DataFrame()
    dems_ranked = df_ranks[dem_cols_rank].sum()
    pd_ranked['rank_sum'] = dems_ranked
    pd_ranked['rank'] = pd_ranked['rank_sum'].rank(ascending=1)
    # pd_ranked = pd_ranked.set_index('final_rank').sort_index()
    print(pd_ranked.sort_values(by='rank'))




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