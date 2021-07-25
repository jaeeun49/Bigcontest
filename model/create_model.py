# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMModel,LGBMClassifier
from lightgbm import LGBMModel,LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import  RandomForestRegressor,GradientBoostingRegressor,VotingRegressor







parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help="create final_model.sav")
opt = parser.parse_args()
print(opt)


MODE = opt.mode


# def for_category(df):
#     df['rich_monster2'] = df['rich_monster2'].astype('category')
#     df['npc_kill2'] = df['npc_kill2'].astype('category')
#     df['class'] = df['class'].astype('category')
#     df['level'] = df['level'].astype('category')
#     df['level_category'] = df['level_category'].astype('category')
#     df['first access day'] = df['first access day'].astype('category')
#
#     return df

def f1(k):
    if 1 <= k <= 7 :
        return 1
    elif 8 <= k <= 14 :
        return 8
    elif 15 <= k <= 21 :
        return 15
    elif 22 <= k <= 28 :
        return 22
    elif 29 <= k <= 35 :
        return 29
    elif 36 <= k <= 42 :
        return 36
    elif 43 <= k <= 49 :
        return 43
    elif 50 <= k <= 56 :
        return 50
    elif 57 <= k <= 63 :
        return 57
    elif k == 64 :
        return 64





if __name__ == "__main__":
    print(f"--- START create_model.py of {MODE} !!! --- ")

    if MODE == 'train':
        # data train load
        X_train = pd.read_csv("./preprocess/train_preprocess.csv")
        target = pd.read_csv("./raw/train_label.csv")
        X_train = pd.merge(X_train, target, on="acc_id")

    # else:
    #     print('please ')
    #     break

    # make categorical features
    X_train = pd.get_dummies(X_train, columns=['rich_monster2', 'npc_kill2', 'class', 'level', 'level_category',
                                               'first access day'])



    # copy data
    X_train_surv1 = X_train.copy()
    X_train_surv2 = X_train.copy()
    X_train_surv3 = X_train.copy()
    X_train_amount = X_train.copy()

    # convert shape of y
    X_train_surv1['survival_time'] = X_train_surv1['survival_time'].apply(f1)
    X_train_surv2['survival_time'] = np.where(
        X_train_surv2.survival_time == 64, 1, 0)
    X_train_surv3 = X_train_surv3[X_train_surv3['survival_time'] != 64]
    X_train_amount['amount_spent'] = X_train_amount['amount_spent'] * \
                                     X_train_amount['survival_time'] * 3.5

    # select y
    y_target_1 = X_train_surv1['survival_time']
    y_target_2 = X_train_surv2['survival_time']
    y_target_3 = X_train_surv3['survival_time']
    y_target_4 = X_train_amount['amount_spent']

    # drop
    X_train_surv1 = X_train_surv1.drop(['acc_id', 'survival_time', 'amount_spent'], axis=1, inplace=False)
    X_train_surv2 = X_train_surv2.drop(['acc_id', 'survival_time', 'amount_spent'], axis=1, inplace=False)
    X_train_surv3 = X_train_surv3.drop(['acc_id', 'survival_time', 'amount_spent'], axis=1, inplace=False)
    X_train_amount = X_train_amount.drop(['acc_id', 'survival_time', 'amount_spent'], axis=1, inplace=False)


    survival_category10 = LGBMClassifier(learning_rate=0.03, n_estimators=500, num_leaves=56, n_jobs=-1).fit(
        X_train_surv1, y_target_1)
    survival_category2 = LGBMClassifier(learning_rate=0.03, n_estimators=500, num_leaves=56, n_jobs=-1).fit(
        X_train_surv2, y_target_2)
    survival_reg = LGBMRegressor(n_estimators=500, num_leaves=64, n_jobs=-1).fit(X_train_surv3, y_target_3)


    amount_LGBM = LGBMRegressor(learning_rate=0.03, n_estimators=900, num_leaves=56, n_jobs=-1)
    amount_XGB = XGBRegressor(max_depth=7, learning_rate=0.03, n_estimators=500, num_leaves=56, random_state=0)
    amount_RF = RandomForestRegressor(max_depth=7, random_state=0)
    amount_gbm = GradientBoostingRegressor(n_estimators=500, random_state=0)

    amount_averaging = VotingRegressor(
        estimators=[('LGBM', amount_LGBM), ('XGB', amount_XGB), ('RF', amount_RF), ('gbm', amount_gbm)])
    amount_averaging.fit(X_train_amount, y_target_4)





    # save the model to disk
    pickle.dump(survival_category10, open('./Model/survival_category10.sav', 'wb'))
    pickle.dump(survival_category2, open('./Model/survival_category2.sav', 'wb'))
    pickle.dump(survival_reg, open('./Model/survival_reg.sav', 'wb'))
    # pickle.dump(amount_reg, open('./Model/amount_reg.sav', 'wb'))
    pickle.dump(amount_averaging, open('./Model/amount_reg.sav', 'wb'))

    print('create_model.py has been made')