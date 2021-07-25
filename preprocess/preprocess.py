# -*- coding: utf-8 -*-


import argparse
import os
import numpy as np

import pandas as pd
from pandas import Series, DataFrame

from tqdm import tqdm

from bitwin_def.bitwin_test import *
from bitwin_def.bitwin_make_activity_features import *


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='test1', help="test csv number")
opt = parser.parse_args()
print(opt)


MODE = opt.mode

# make save folder
# SAVE_FOLDER = MODE # save location
# os.makedirs('./'+SAVE_FOLDER, exist_ok=True)




if __name__ == "__main__":
    print(f"--- START making {MODE} features !!! --- ")
    print('[ STEP 1 / STEP 3 ] Integral all DataFrame which served by NC_soft')
    # 1 Integral all DataFrame which served by NC_soft
    # 1-1 Make total_df with  [combat_df, activity_df, pledge_df]
    make_total_df(MODE)
    # 1-2 import totla_df
    total = pd.read_csv('./Preprocess/'+MODE+"_total.csv")


    print('[ STEP 2 / STEP 3 ] make activity, combat, pledge, trade, payment DataFrame')
    # 2 Make features
    # 2-1 make_activity_DataFrame
    activity_df = make_activity_features(total)

    # 2-2 make_combat_DataFrame
    combat_df = make_combat_features(total)

    # 2-3  make_pledge_DataFrame
    pledge_total = make_prepledge_df(total, MODE)
    pledge_df = make_pledge_features(pledge_total,total)

    # 2-4 make_trade_DataFrame # MUST BE FIX
    trade_df = make_trade_df(total, MODE)

    # 2-5 make_payment_DataFrame
    payment_df = make_payment_df(total, MODE)

    print('[ STEP 31 / STEP 3 ] Merge all DataFrame')
    # 3 Merge
    final_df_1 = pd.merge(activity_df, combat_df, how='left', on='acc_id')
    final_df_2 = pd.merge(final_df_1, pledge_df, how='left', on='acc_id')
    final_df_3 = pd.merge(final_df_2, trade_df, how='left', on='acc_id')
    final_df = pd.merge(final_df_3, payment_df, how='left', on='acc_id')

    print("Final_DataFrame's shape",final_df.shape)

    final_df.to_csv("./Preprocess/"+MODE+"_preprocess.csv",index=False)


print(f"--- {MODE}_preprocess.csv were created !!! --- ")

