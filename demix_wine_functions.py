# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Functions for the DEMIX Wine Contest Jupyter notebook
# Carlos H. Grohmann
# version 2022-06-15

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
# make dataframes from csv files and append all to a single one
def make_criteria_df_not_transposed(csv_list,datadir):
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
        #tile_list = list(df_csv['DEMIX_TILE'].unique())
        #ref_list = list(df_csv['REF_TYPE'].unique())
        #ref_slp_list = list(df_csv['REF_SLOPE'].unique())
        
        # dfs for row-based format 
        df_left = pd.DataFrame(columns=(['AREA','DEMIX_TILE','REF_TYPE','REF_SLOPE']))
        df_right = pd.DataFrame(columns=(['METRIC']+dems_list))     
        group_dem = df_csv.groupby(['DEMIX_TILE','REF_TYPE','REF_SLOPE'])
        
        # convert from PG format to row-based
        for g_idx, group in group_dem:
            # ------------------------------
            # create metadata df (right one)
            area = group.T.iloc[0,1]
            tile = group.T.iloc[1,1]
            surf = group.T.iloc[4,1]
            slop = group.T.iloc[5,1]
            dictemp = {'AREA':area, 'DEMIX_TILE':tile, 'REF_TYPE':surf,'REF_SLOPE':slop}
            # ------------------------------
            # create metrics df (left one)
            metrics_T = group.T.drop(labels=group.T.index[[0,1,2,3,4,5]], axis=0).reset_index()
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
    ranks_vect = df[dem_cols_rank].sum() # excel Sheet1!J10:O10
    sum_ranks_vect = ranks_vect.sum() # excel SOMA(Sheet1!J10:O10)
    ranks_sq_vect = ranks_vect.pow(2) # excel Sheet1!J11:O11
    sum_ranks_sq_vect = ranks_sq_vect.sum() # excel SOMA(Sheet1!J11:O11)
    sum_squared_ranks = df[dem_cols_rank_sq].sum().sum() # excel SOMA(Sheet1!Q14:V322)

    chi_r =( (n * (k-1)) / (sum_squared_ranks - cf) * (sum_ranks_sq_vect/n - cf) )
    # =+E5*(E6-1)/(SOMA(Sheet1!Q14:V322)-E7)*(SOMA(Sheet1!J11:O11)/E5-E7)
    #
    print(f'N = {n} (number of criteria)')
    print(f'k = {k} (number of DEMs)')
    print(f'CF = {cf}')
    print(f'sum of ranks (vector) = {ranks_vect.tolist()}')  # excel Sheet1!J10:O10
    print(f'sum of (ranks squared) = {ranks_sq_vect.tolist()}')  # excel Sheet1!J11:O11
    print(f'sum of squared ranks = {sum_squared_ranks}')         # excel Sheet2!N4
    print(f'sum of ranks squared (total) = {sum_ranks_sq_vect}') # excel Sheet2!N5
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

