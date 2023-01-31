# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Functions for the DEMIX Wine Contest Jupyter notebook
# Carlos H. Grohmann
# version 2022-09-29

import sys,os
import pandas as pd
import numpy as np
import math as m
# from decimal import *
# getcontext().prec = 3
from scipy.special import ndtri
from ipywidgets import Button
from tkinter import Tk, filedialog
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
def select_files(b):
    '''make a button for file selection in a jupyter notebook
    from: https://edusecrets.com/lesson-02-creating-a-file-select-button-in-jupyter-notebook/'''
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
    already in a format of one criterion per row'''
    df_merged = pd.DataFrame()
    for f in csv_list:
        df = pd.read_csv(f, sep=',',engine='python',comment='#',quotechar='"')
        df_merged = pd.concat([df_merged,df],sort=False)
    return df_merged





# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
def fix_vals_ranks_ties(sr,tolerance):
    '''  check for ties in a pandas Series (dataframe row) and make 
    those within a tolerance to be equal, allowing proper ranking'''
    tolerance = tolerance + 0.0001
    vals = np.array(sr)#.astype(np.float32)
    low = min(vals) # initial minimum
    vals_left = len(vals)
    fix_arr = np.empty(len(vals))#.astype(np.float32) # will hold the fixed values
    while vals_left > 0:
        # check if within tolerance
        lst_ties = np.array([low if abs(i - low) < tolerance else i for i in vals])
        lst_where = np.where(lst_ties == low)[0] # get indices of ties
        vals_left -= len(lst_where) # our counter
        fix_arr[lst_where] = low # values for result
        vals[lst_where] = np.nan # remove original values, so next round goes ok
        # with warnings.catch_warnings():
            # https://blog.finxter.com/numpy-runtimewarning-all-nan-slice-encountered/
            # warnings.simplefilter("ignore", category=RuntimeWarning)
            # warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        if np.all(np.isnan(vals)): # this happens when we finished the array
            low = np.nan
        else:
            low = np.nanmin(vals) # new minimum, after removing previous ties
    return pd.Series(fix_arr)






# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# calculate ranks for criteria (error metrics) in dataframes
def make_rank_df(df,dem_list,tolerance_dict,method):
    '''Calculate ranks for metrics in dataframes - accepts a 
         dictionary of criterion/tolerance values.
    Calls fix_vals_ranks_ties to adjust values before ranking '''
    dem_cols_rank = [i+'_rank' for i in dem_list]
    df_ranks = pd.DataFrame(columns=list(df.columns) + dem_cols_rank)
    if tolerance_dict is not None:
        print(f'Ranking with user-defined tolerances (might take a while...)',end='\n\n')
        # iterate through dict of criterion/tolerance
        for key, value in tolerance_dict.items():
            criterion = key
            tolerance = value # + 0.000001
            # subset of df - only rows of selected criterion 
            df_crit = df.loc[df['CRITERION'] == criterion]
            # subset of df_crit - only DEMs values
            df_for_ranking = df_crit[dem_list]
            # rank values in df
            df_temp = df_for_ranking.apply(lambda row: fix_vals_ranks_ties(row, tolerance=tolerance), axis=1)
            df_temp.columns = dem_list
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
# print the DEMs ranked
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





# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def friedman(df,dem_list,tables_dir,cl):
    '''this func will calculate friedman stats and then check the critical values from tables'''
    # preliminaries
    dem_cols = dem_list
    dem_cols_rank = [i+'_rank' for i in dem_cols]
    dem_cols_rank_sq = [i+'_rank_sq' for i in dem_cols]
    # Friedman's
    n = len(df) # number of opinions 
    k = len(dem_cols) # number of DEMs being compared
    cf = 1/4*n*k*((k+1)**2)
    #
    ranks_vect = df[dem_cols_rank].sum() 
    sum_ranks_vect = ranks_vect.sum() 
    ranks_sq_vect = ranks_vect.pow(2) 
    sum_ranks_sq_vect = ranks_sq_vect.sum() 
    sum_squared_ranks = df[dem_cols_rank_sq].sum().sum() 
    chi_r =( (n * (k-1)) / (sum_squared_ranks - cf) * (sum_ranks_sq_vect/n - cf) )
    #
    # get values from tables
    table_needed = f'k_{k}.txt'
    df_critical = pd.read_csv(os.path.join(tables_dir,table_needed),sep=';')
    # find chi_crit in table
    n_alpha = f'N={n}' 
    # try to get the value, if not possible, use last row
    try:
        idx = df_critical.loc[df_critical['alpha'] == n_alpha].index[0]
        col = f'{cl:05.3f}'
        chi_crit = df_critical.at[idx, col]
    except:
        idx = df_critical.index[-1]
        col = f'{cl:05.3f}'
        chi_crit = df_critical.at[idx, col]
    # return
    return n,k,cf,chi_r,chi_crit
       
    



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def bonferroni(df,dem_list,alpha=0.95):
    '''this func will calculate the bonferroni-dunn test, 
    and profuce the table of ranks and ties'''
    # alpha = 0.95 default value
    # preliminaries
    dem_cols = dem_list
    dem_cols_rank = [i+'_rank' for i in dem_cols]
    dem_cols_rank_sq = [i+'_rank_sq' for i in dem_cols]
    n = len(df) # number of opinions 
    k = len(dem_cols) # number of DEMs being compared
    # Bonferroni
    quant =  1-alpha/k/(k-1)
    zi = ndtri(quant)
    crit = zi*np.sqrt(n*k*(k+1)/6) 
    tie_dict = {}
    dems_ranked = df[dem_cols_rank].sum()
    #---------------------------------------------------------
    # Y/N table 
    cols = ['DEM'] + dem_cols
    df_table = pd.DataFrame(columns=cols) # df and cols names
    df_table['DEM'] = dem_list # first column of df
    # get ranks values 
    ranks_vals = dems_ranked.to_frame().T
    # populate Y/N table
    for d1 in dem_list:
        tie_dict[d1] = []
        r = dem_list.index(d1)
        for d2 in dem_list:
            rank_dem1 = ranks_vals[f'{d1}_rank'].values[0]
            rank_dem2 = ranks_vals[f'{d2}_rank'].values[0]
            if np.abs(rank_dem1 - rank_dem2) > crit:
                df_table.at[r,d2] = 'Y'
            else:
                df_table.at[r,d2] = 'N'
                tie_dict[d1].append(f'{d1}/{d2}')
    #----------------------------------------------------------
    # table of ranked DEMs (final result of wine contest)
    pd_ranked = pd.DataFrame()
    dems_rnk_sum = df[dem_cols_rank].sum()
    pd_ranked['sum_ranks'] = dems_rnk_sum
    # "normalize" ranks values)
    # divide by the number of opinions
    divider = n
    pd_ranked['sum_rnks_div_n'] = pd_ranked['sum_ranks'].div(n).round(3)
    pd_ranked.index = dem_list
    pd_ranked['rank'] = pd_ranked['sum_ranks'].rank(method='average', ascending=True, axis=0)
    # check for ties in final ranking
    pd_ranked_ties = rank_ties_bonf(pd_ranked,tie_dict)
    pd_ranked_ties['ties'] = pd_ranked_ties['not_stat_diff'].where(pd_ranked_ties['not_stat_diff']=='', pd_ranked_ties['rank'])
    # return
    return pd_ranked_ties 
    


    
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def rank_ties_bonf(pd_ranked,tie_dict):
    '''chek which DEMs are not statistically different'''
    for key,val in tie_dict.items():
        val.remove(f'{key}/{key}')
        if val:
            tie_dict[key] = ','.join(val)
        else:
            tie_dict[key] = ''
    pd_ranked['not_stat_diff'] = pd.Series(tie_dict)
    # return
    return pd_ranked




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def wine_contest(df,dem_list,tables_dir,cl,alpha=0.95,verbose=False):
    '''this func will calculate the wine contest, using Friedman and Bonferroni'''
    # Friedman stats
    n,k,cf,chi_r,chi_crit = friedman(df,dem_list,tables_dir,cl)
    if verbose:
        print('Results of the DEMIX Wine Contest')
        print()
        print(f'For k={k}, CL={cl}, and N={n}, the critical value to compare is chi_crit={chi_crit:4.3f}')
    # post-hoc (Bonferroni-Dunn)
    if chi_r > chi_crit:
        if verbose:
            print(f'And since chi_r ({chi_r:4.3f}) is greater than chi_crit ({chi_crit:4.3f})...')
            print(f'Yay!! We can reject the null hipothesis and go to the Post-Hoc analysis!!')
            print()
        # bonferroni 
        df_ranked = bonferroni(df,dem_list,alpha=0.95)
    else:
        if verbose:
            print(f'But since chi_r ({chi_r:4.3f}) is less than chi_crit ({chi_crit:4.3f})...')
            print('Oh, no! We cannot disprove the null hipothesis at the given CL...')
            print()
        df_ranked = None
    # return
    return df_ranked,n




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def get_winecontest_ranks_by_condition(df,cond_list,label_list,dem_list,tables_dir,rnk_col,cl):
    '''get ranks based on a given condition'''
    df_temp = pd.DataFrame(columns=dem_list)
    for cond,label in zip(cond_list,label_list):
        df_select = df.query(cond, engine='python').copy()
        df_ranked,n = wine_contest(df_select,dem_list,tables_dir,cl,alpha=0.95,verbose=False)
        if df_ranked is not None:
            dems_ranked = list(df_ranked[rnk_col])
            df_temp.loc[f'{label} (N={n})'] = list(dems_ranked)
        else:
            dems_ranked = [np.nan] * len(dem_list)
            df_temp.loc[f'{label} (N={n})'] = list(dems_ranked)
    return df_temp




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def get_winecontest_ranks_by_condition_for_mwc(df,cond_list,label_list,dem_list,tables_dir,crit_list,tiles_list,rnk_col,cl):
    '''get ranks based on a given condition.
    slightly different version of get_winecontest_ranks_by_condition
    due to some needs of the mini-wine-contest, namely the presence
    of crit_list and tiles_list '''
    df_temp = pd.DataFrame(columns=dem_list)
    for cond,label in zip(cond_list,label_list):
        df_select = df.query(cond, engine='python').copy()
        df_ranked= wine_contest(df_select,dem_list,tables_dir,cl,alpha=0.95,verbose=False)
        if df_ranked is not None:
            dems_ranked = list(df_ranked[rnk_col])
            df_temp.loc[f'{label} (N={n})'] = list(dems_ranked)
        else:
            dems_ranked = [np.nan] * len(dem_list)
            df_temp.loc[f'{label} (N={n})'] = list(dems_ranked)
    return df_temp



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def get_wc_ties_rects_by_condition(df,cond_list,dem_list,tables_dir,label_list,cl,rnk_col):
    '''get ties based on a given condition'''
    all_rects = []
    for cond,label in zip(cond_list,label_list):
        df_select = df.query(cond).copy()
        df_ranked,n = wine_contest(df_select,dem_list,tables_dir,cl,alpha=0.95,verbose=False)
        if df_ranked is not None:
            nsd_lst = list(df_ranked['not_stat_diff'])
            for l in range(len(nsd_lst)):
                if len(nsd_lst[l])>0: # not empty
                    ties_str = nsd_lst[l]
                    ties_lst = ties_str.split(',')
                    for tie in ties_lst:
                        d0 = tie.split('/')[0]
                        d1 = tie.split('/')[1]
                        d0_r = df_ranked.loc[f'{d0}',rnk_col]
                        d1_r = df_ranked.loc[f'{d1}',rnk_col]
                        x0 = min(d0_r,d1_r) - 0.25
                        x1 = max(d0_r,d1_r) + 0.25
                        wd = x1 - x0
                        rect = [f'{label} (N={n})',x0,0,wd,1]
                        all_rects.append(rect)
    return all_rects



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def ties_rectangles(rects,df):
    '''small funct to set the y-cords of the rectangles'''
    for r in rects:
        idx = df.index.get_loc(r[0])
        r[2] = idx-0.5
    df_rects = pd.DataFrame(rects, columns=['label','x0','y0','width','height'])
    errorboxes = [Rectangle((x,y),w,h) for x,y,w,h in zip(df_rects['x0'],df_rects['y0'],df_rects['width'],df_rects['height'])]
    pc = PatchCollection(errorboxes,fc='none',alpha=0.5,ec='black',lw=0.9)
    return pc




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def find_best_dem_row_cat(sr,dem_order):
    '''find the DEM with lowest rank in a row. Returns a string'''
    arr = sr.to_numpy().astype(float)
    amin = arr.min()
    idxs = np.where(np.isclose(arr, amin))[0]
    dems = [dem_order[i] for i in idxs]
    return ','.join(dems)




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def get_ties_best_dem_xycoords(df,jit):
    '''func to get all the ties and calculate x,y coords for each one,
    so they can be plotted all at once'''
    lst_ties = []
    cols = list(df.columns) + ['crit_num_tie']
    df_ties = pd.DataFrame(columns=cols)
    df_temp = df[df['best'].str.contains(',')]
    for row in df_temp.itertuples():
        num_ties = len(row.best.split(','))
        tile = row.DEMIX_TILE
        crit = row.CRITERION
        cnum = row.crit_num
        for i in range(num_ties):
            best = row.best.split(',')[i]
            cnum_i = cnum + (i * jit)
            sr = [tile, crit, best, cnum, cnum_i]
            df_ties.loc[len(df_ties)] = sr
    return df_ties




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def find_last(lst, elm):
    '''https://stackoverflow.com/a/23146126/4984000'''
    gen = (len(lst) - i for i, v in enumerate(reversed(lst)) if v == elm) # - 1
    return next(gen, None)




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def filter_by_cond_find_best_and_ties(df,dem_order,crit_order,ref,land,jit):
    dem_order_rank = [i+'_rank' for i in dem_order]
    cond = f"REF_TYPE=='{ref}' and LAND_TYPE=='{land}'"
    df_query = df.query(cond).copy()
    df_dropdup = df_query.drop_duplicates(subset=['DEMIX_TILE']).reset_index()[['DEMIX_TILE','AREA']].copy()
    df_query['best'] = df_query[dem_order_rank].apply(find_best_dem_row_cat, args=(dem_order,), axis=1)
    df_bests = df_query[['DEMIX_TILE','CRITERION','best']].copy()
    df_bests['crit_num'] = df_bests['CRITERION'].map(lambda x: crit_order.index(x))
    df_no_ties = df_bests[~df_bests['best'].str.contains(',')]
    df_ties = get_ties_best_dem_xycoords(df_bests,jit)
#     df_ties=df_bests
#     lst_ties=[]
    # find indexes of areas rows, for plotting
    n_tiles = len(df_bests['DEMIX_TILE'].unique())
    area_list = list(df_dropdup['AREA'])
    area_unique = list(df_dropdup['AREA'].unique())
    area_idxs = [find_last(area_list,i) for i in area_unique]
    return df_bests,df_no_ties,df_ties,n_tiles,area_unique,area_idxs





# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def plot_best_ranks_w_ties(dfnt1,dft1,dfnt2,dft2,rt1,rt2,no1,no2,au1,ai1,au2,ai2,
    land,mrks,size,pal,order,crit,figsize,txt_tol,suptitle):
    grid_kws = {'width_ratios': (0.5, 0.5), 'wspace': 0.1} # size of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw=grid_kws,figsize=figsize)
    # plot ax1
    sns.scatterplot(ax=ax1, data=dfnt1,x='crit_num', y='DEMIX_TILE', hue='best', style='best', 
                    markers=mrks, s=size, hue_order=order, palette=pal, legend=False)
    sns.scatterplot(ax=ax1, data=dft1,x='crit_num_tie', y='DEMIX_TILE', hue='best', style='best', 
                    markers=mrks, s=size, hue_order=order, palette=pal, legend=False)
    # adjust empty space at top and bottom
    miny, nexty, *_, maxy = ax1.get_yticks()
    eps = (nexty - miny) #/ 1.1  # <-- Your choice.
    ax1.set_ylim(maxy+eps, miny-eps)
    # plot ax2
    sns.scatterplot(ax=ax2, data=dfnt2,x='crit_num', y='DEMIX_TILE', hue='best', style='best', 
                    markers=mrks, s=size, hue_order=order, palette=pal, legend=False)
    sns.scatterplot(ax=ax2, data=dft2,x='crit_num_tie', y='DEMIX_TILE', hue='best', style='best', 
                    markers=mrks, s=size, hue_order=order, palette=pal, legend=True)
    # adjust empty space at top and bottom
    miny, nexty, *_, maxy = ax2.get_yticks()
    eps = (nexty - miny) #/ 1.1  # <-- Your choice.
    ax2.set_ylim(maxy+eps, miny-eps)
    # Customize tick marks and positions
    ax1.tick_params(labelsize=9)
    ax1.set_yticks([i-1 for i in ai1])
    ax1.set_yticklabels(au1)
    ax1.set_xticks(range(len(crit)))
    ax1.set_xticklabels(crit, rotation=90)
    ax1.set_ylabel('DEMIX TILE')
    ax1.set_xlabel('CRITERION')
    ax1.set_title(f'{rt1} - {land} - {no1} tiles')
    ax1.vlines([4.85,9.85], *ax1.get_ylim(), colors='grey')
    ax1.hlines([i-0.5 for i in ai1], *ax1.get_xlim(), colors='grey')
    ax1.margins(x=0)
    ax2.tick_params(labelsize=9)
    ax2.set_yticks([i-1 for i in ai2])
    ax2.set_yticklabels(au2)
    ax2.set_xticks(range(len(crit)))
    ax2.set_xticklabels(crit, rotation=90)
    ax2.set_ylabel('')
    ax2.set_xlabel('CRITERION')
    ax2.vlines([4.85,9.85], *ax2.get_ylim(), colors='grey')
    ax2.hlines([i-0.5 for i in ai2], *ax2.get_xlim(), colors='grey')
    ax2.set_title(f'{rt2} - {land} - {no2} tiles')
    ax2.legend(bbox_to_anchor=(1.0,0.99), prop={'size':16})
    ax2.text(1.05, 0.6,txt_tol, transform=ax2.transAxes)
    ax2.margins(x=0)
    fig.suptitle(suptitle, y=0.92)
    return fig, ax1, ax2




# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def filter_by_cond_pivot_corr_matrx(df,cond,crit_order,pvt_val):
    ''' aux func used to make the correlation matrix.
    query a df by a condition and returns pivoted criterion of a single DEM '''
    df_query = df.query(cond).copy()
    df_query_pvt = pd.pivot_table(df_query, index='DEMIX_TILE', columns='CRITERION', values=pvt_val, sort=False)
    df_query_pvt = df_query_pvt[crit_order]
    return df_query_pvt




# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# func to get ranks based on a defined condition
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



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def find_best_dem_row(sr,low):
    '''find the DEM with lowest rank in a row'''
    dem_order = ['ALOS','COP','NASA','FABDEM','SRTM','ASTER']
    dem_order_rank = [i+'_rank' for i in dem_order]
    arr = sr.to_numpy().astype(float)
    if low == 1.0: # no ties for first place
        try:
            i = np.where(np.isclose(arr, low))[0][0]
        except:
            i = -1 # will be transparent (ties)
        return i+0.75
    else: #if low == 1.5: # special case of ties for first place
        try:
            i = np.where(np.isclose(arr, low))[0][0]
        except:
            i = -1 # will be transparent (not ties)
        return i + 1.2



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def filter_by_cond_best_and_pivot(df,dem_order_rank,crit_order,ref,land,low):
    '''aux func to query a df by REF_TYPE and LAND_TYPE
    then finding the best ranked DEM, and returning a 
    pivoted df, to be plotted as a heatmap'''
    # run query and pivot
    cond = f"REF_TYPE=='{ref}' and LAND_TYPE=='{land}'"
    df_query = df.query(cond).copy()
    df_dropdup = df_query.drop_duplicates(subset=['DEMIX_TILE']).reset_index()[['DEMIX_TILE','AREA']].copy()
    df_query['best'] = df_query[dem_order_rank].apply(find_best_dem_row, args=(low,), axis=1)
    df_query_pvt = pd.pivot_table(df_query, index='DEMIX_TILE', columns='CRITERION', values='best', sort=False)
    df_query_pvt = df_query_pvt[crit_order]
    n_opn = len(df_query_pvt)
    # find indexes of areas rows, for plotting
    area_list = list(df_dropdup['AREA'])
    area_unique = list(df_dropdup['AREA'].unique())
    area_idxs = [find_last(area_list,i) for i in area_unique]
    return df_query_pvt,n_opn,area_unique,area_idxs



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

def plot_heatmaps(df1,df2,rt1,rt2,no1,no2,au1,ai1,au2,ai2,land,cmap,norm,hcbt,dem_labels):
    '''plot 2 heatmaps side by side'''
    # fig defs
    grid_kws = {'width_ratios': (0.4, 0.4, 0.015), 'wspace': 0.35} # size of subplots
    fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws,figsize=(20,15))
    # plot
    ax1 = sns.heatmap(df1, ax=ax1, cbar_ax=cbar_ax, cmap=cmap, norm=norm,
        cbar_kws={'orientation':'vertical', 'ticks':hcbt}, linewidths=.01, linecolor='lightgray',)
    ax2 = sns.heatmap(df2, ax=ax2, cmap=cmap, cbar=False, norm=norm,
        linewidths=.01, linecolor='lightgray')#,annot=True
    # Customize tick marks and positions
    ax1.set_yticks([i-0.5 for i in ai1])
    ax2.set_yticks([i-0.5 for i in ai2])
    ax1.set_yticklabels(au1)
    ax2.set_yticklabels(au2)
    ax2.tick_params(labelsize=9)
    cbar_ax.set_yticklabels(dem_labels)
    # X - Y axis labels
    ax1.set_ylabel('DEMIX TILE')
    ax1.set_xlabel('CRITERION')
    ax1.set_title(f'{rt1} - {land} - {no1} tiles')
    ax1.vlines([5,10], *ax1.get_ylim(), colors='grey')
    ax1.hlines(ai1, *ax1.get_xlim(), colors='grey')
    ax2.set_ylabel('')
    ax2.set_xlabel('CRITERION')
    ax2.vlines([5,10], *ax2.get_ylim(), colors='grey')
    ax2.hlines(ai2, *ax2.get_xlim(), colors='grey')
    ax2.set_title(f'{rt2} - {land} - {no2} tiles')
    return fig, ax1, ax2, cbar_ax



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def plot_heatmaps_ties(df1,df2,df1t,df2t,rt1,rt2,no1,no2,au1,ai1,au2,ai2,land,cmap,norm,hcbt,dem_labels):
    '''plot 2 heatmaps side by side - overlay heatmaps with and without ties'''
    # fig defs
    grid_kws = {'width_ratios': (0.4, 0.4, 0.015), 'wspace': 0.35} # size of subplots
    fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws,figsize=(20,15))
    # plot

    ax1 = sns.heatmap(df1t, ax=ax1, cbar=False, cmap=cmap, norm=norm, linewidths=0)
    ax1 = sns.heatmap(df1, ax=ax1, cbar_ax=cbar_ax, cmap=cmap, norm=norm,
        cbar_kws={'orientation':'vertical', 'ticks':hcbt}, linewidths=.01, linecolor='lightgray',)
    ax2 = sns.heatmap(df2t, ax=ax2, cbar=False, cmap=cmap, norm=norm, linewidths=0)
    ax2 = sns.heatmap(df2, ax=ax2, cmap=cmap, cbar=False, norm=norm,
        linewidths=.01, linecolor='lightgray')#,annot=True
    # Customize tick marks and positions
    ax1.set_yticks([i-0.5 for i in ai1])
    ax2.set_yticks([i-0.5 for i in ai2])
    ax1.set_yticklabels(au1)
    ax2.set_yticklabels(au2)
    ax2.tick_params(labelsize=9)
    cbar_ax.set_yticklabels(dem_labels)
    # X - Y axis labels
    ax1.set_ylabel('DEMIX TILE')
    ax1.set_xlabel('CRITERION')
    ax1.set_title(f'{rt1} - {land} - {no1} tiles')
    ax1.vlines([5,10], *ax1.get_ylim(), colors='black')
    ax1.hlines(ai1, *ax1.get_xlim(), colors='black')
    ax2.set_ylabel('')
    ax2.set_xlabel('CRITERION')
    ax2.vlines([5,10], *ax2.get_ylim(), colors='black')
    ax2.hlines(ai2, *ax2.get_xlim(), colors='black')
    ax2.set_title(f'{rt2} - {land} - {no2} tiles')
    return fig, ax1, ax2, cbar_ax





# dem_plt_order = ['ALOS','ALOS,COP','ALOS,NASA','COP','COP,NASA','COP,SRTM','FABDEM','FABDEM,SRTM','NASA','NASA,SRTM','NASA,FABDEM','NASA,FABDEM,SRTM','SRTM','ASTER']
# dem_plt_color = ['#D55E00','#8d4c6a','#be3460','#0072B2','#6866ad','#9b7f5c','#009E73','#a69725','#CC79A7','#cb6752','#737f77','#a17f4f','#56B4E9','#F0E442']

# m_alos = '$\u25cf$' # Black circle 
# m_copd = '$\u25b2$' # Black up-pointing triangle 
# m_fabd = '$\u25c6$' # Black diamond 
# m_srtm = '$\u25bc$' # Black down-pointing triangle 
# m_nasa = '$\u25a0$' # Black square
# m_astr = '$\u271a$' # heaby cross

# dem_plt_mrks = {'ALOS':m_alos,
# 'ALOS,COP':f'{m_alos}{m_copd}',
# 'ALOS,NASA':f'{m_alos}{m_nasa}',
# 'COP':f'{m_copd}',
# 'COP,NASA':f'{m_copd}{m_nasa}',
# 'COP,SRTM':f'{m_copd}{m_srtm}',
# 'FABDEM':f'{m_fabd}',
# 'FABDEM,SRTM':f'{m_fabd}{m_srtm}',
# 'NASA':f'{m_nasa}',
# 'NASA,SRTM':f'{m_nasa}{m_srtm}',
# 'NASA,FABDEM':f'{m_nasa}{m_fabd}',
# 'NASA,FABDEM,SRTM':f'{m_nasa}{m_fabd}{m_srtm}',
# 'SRTM':f'{m_srtm}',
# 'ASTER':f'{m_astr}',}

# sz1 = 20
# sz2 = sz1*3
# sz3 = sz1*5

# dem_plt_szs = {'ALOS':sz1,
# 'ALOS,COP':sz2,
# 'ALOS,NASA':sz2,
# 'COP':sz1,
# 'COP,NASA':sz2,
# 'COP,SRTM':sz2,
# 'FABDEM':sz1,
# 'FABDEM,SRTM':sz2,
# 'NASA':sz1,
# 'NASA,SRTM':sz2,
# 'NASA,FABDEM':sz2,
# 'NASA,FABDEM,SRTM':sz3,
# 'SRTM':sz1,
# 'ASTER':sz1,}


# #f'{}'
# palette = sns.color_palette(dem_plt_color, len(dem_plt_color))

# fig,ax = plt.subplots(figsize=(9,15))
# s =sns.scatterplot(data=df_plot,x='crit_num', y='DEMIX_TILE', hue='best', style='best',
#                    markers=dem_plt_mrks, sizes=dem_plt_szs, hue_order=dem_plt_order, palette=palette)#
# ax.legend(bbox_to_anchor=(1.0,0.99), prop={'size':16})
# # adjust empty space at top and bottom
# miny, nexty, *_, maxy = ax.get_yticks()
# eps = (nexty - miny) #/ 1.1  # <-- Your choice.
# ax.set_ylim(maxy+eps, miny-eps)
# #labels
# ax.tick_params(axis='x', labelrotation=90)
# fig.tight_layout()



# # simple example, heatmaps for landtype=ALL
# # the next cell has a loop that will make a plot for each landtype

# # get database
# df = df_ranks.sort_values(by='SORT_AREA',ascending=True)

# # query by land type
# land = 'ALL'

# # get DEMs where the best == 1.0
# low = 1.0
# # dsm
# df_heat_dsm,n_opn_dsm,area_unique_dsm,area_idxs_dsm = dw.filter_by_cond_best_and_pivot(df,dem_order_rank,crit_order,ref='DSM',land=land,low=low)
# # dtm
# df_heat_dtm,n_opn_dtm,area_unique_dtm,area_idxs_dtm = dw.filter_by_cond_best_and_pivot(df,dem_order_rank,crit_order,ref='DTM',land=land,low=low)

# # get DEMs where the best == 1.5 (that is, a tie for the first place)
# low = 1.5
# # dsm
# df_heat_dsm_t,n_opn_dsm,area_unique_dsm,area_idxs_dsm = dw.filter_by_cond_best_and_pivot(df,dem_order_rank,crit_order,ref='DSM',land=land,low=low)
# # dtm
# df_heat_dtm_t,n_opn_dtm,area_unique_dtm,area_idxs_dtm = dw.filter_by_cond_best_and_pivot(df,dem_order_rank,crit_order,ref='DTM',land=land,low=low)

# # plot heatmaps
# fig, ax1, ax2, cbar_ax = dw.plot_heatmaps_ties(df1=df_heat_dsm,df2=df_heat_dtm,
#                                        df1t=df_heat_dsm_t,df2t=df_heat_dtm_t,
#                                        rt1='DSM',rt2='DTM',
#                                        no1=n_opn_dsm,no2=n_opn_dtm,
#                                        au1=area_unique_dsm,ai1=area_idxs_dsm,
#                                        au2=area_unique_dtm,ai2=area_idxs_dtm,
#                                        land=land,cmap=cmap,norm=norm,hcbt=heat_cbar_ticks,dem_labels=dem_order)

# fig.savefig(f'heatmap_{land}_low_{low}.svg', dpi=300)






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
def sort_with_tolerance_np(sr,dem_list,tolerance):
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
def rank_with_tolerance(sr, tolerance, method):
    ''' thanks to https://stackoverflow.com/a/72957060/4984000'''
    vals = pd.Series(sr.unique()).sort_values()
    vals.index = vals
    vals = vals.mask(vals - vals.shift(1) < tolerance, vals.shift(1))
    return sr.map(vals).fillna(sr).rank(method=method)






