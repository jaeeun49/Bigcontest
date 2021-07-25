# -*- coding: utf-8 -*-


import argparse
import os
import pickle

import pandas as pd
from pandas import Series, DataFrame
import numpy as np


# def for_category(df):
#     df['rich_monster2'] = df['rich_monster2'].astype('category')
#     df['npc_kill2'] = df['npc_kill2'].astype('category')
#     df['class'] = df['class'].astype('category')
#     df['level'] = df['level'].astype('category')
#     df['level_category'] = df['level_category'].astype('category')
#     df['first access day'] = df['first access day'].astype('category')
#
#     return df





if __name__ == "__main__":

    print('START survival time__predict')

    # From Preprocess folder
    y1 = pd.read_csv("./Preprocess/test1_preprocess.csv")
    y2 = pd.read_csv("./Preprocess/test2_preprocess.csv")

    # From Model folder
    survival_ca10 = pickle.load(open('./Model/survival_category10.sav', 'rb'))
    survival_ca2 = pickle.load(open('./Model/survival_category2.sav', 'rb'))
    survival_reg = pickle.load(open('./Model/survival_reg.sav', 'rb'))
    amount_reg = pickle.load(open('./Model/amount_reg.sav', 'rb'))


    category_list = ['rich_monster2', 'npc_kill2', 'class', 'level', 'level_category', 'first access day']


    # make categorical features
    y1 = pd.get_dummies(y1,
                        columns=['rich_monster2', 'npc_kill2', 'class', 'level', 'level_category', 'first access day'])
    y2 = pd.get_dummies(y2,
                        columns=['rich_monster2', 'npc_kill2', 'class', 'level', 'level_category', 'first access day'])


    # copy DataFrame
    y1_2 = y1.copy()
    y2_2 = y2.copy()


    #drop
    y1 = y1.drop('acc_id', axis=1, inplace=False)
    y2 = y2.drop('acc_id', axis=1, inplace=False)


    # predict
    pred_survival10_1 = survival_ca10.predict(y1)
    pred_survival10_2 = survival_ca10.predict(y2)
    pred_survival2_1 = survival_ca2.predict(y1)
    pred_survival2_2 = survival_ca2.predict(y2)

    # make DataFrame
    pred_survival1 = pd.DataFrame(
        {'acc_id': y1_2['acc_id'], 'survival_time': pred_survival10_1})
    pred_survival2 = pd.DataFrame(
        {'acc_id': y2_2['acc_id'], 'survival_time': pred_survival10_2})
    pred_survivalca1 = pd.DataFrame(
        {'acc_id': y1_2['acc_id'], 'survival_time_ca': pred_survival2_1})
    pred_survivalca2 = pd.DataFrame(
        {'acc_id': y2_2['acc_id'], 'survival_time_ca': pred_survival2_2})


    # fix the constant value
    predict1_sur = pred_survivalca1[pred_survivalca1['survival_time_ca'] == 1]
    predict1_sur['survival_time'] = 64
    predict1_sur = predict1_sur.drop('survival_time_ca', axis=1, inplace=False)

    predict2_sur = pred_survivalca2[pred_survivalca2['survival_time_ca'] == 1]
    predict2_sur['survival_time'] = 64
    predict2_sur = predict2_sur.drop('survival_time_ca', axis=1, inplace=False)

    # select churn users
    predict1_churn = pred_survivalca1[pred_survivalca1['survival_time_ca'] == 0]
    predict2_churn = pred_survivalca2[pred_survivalca2['survival_time_ca'] == 0]


    # Merge DataFrame
    predict1_churn = pd.merge(predict1_churn, y1_2, on='acc_id', how='left')
    predict_1 = predict1_churn.copy()
    predict_1 = predict_1.drop('acc_id', axis=1, inplace=False)
    predict_1 = predict_1.drop('survival_time_ca', axis=1, inplace=False)

    predict2_churn = pd.merge(predict2_churn, y2_2, on='acc_id', how='left')
    predict_2 = predict2_churn.copy()
    predict_2 = predict_2.drop('acc_id', axis=1, inplace=False)
    predict_2 = predict_2.drop('survival_time_ca', axis=1, inplace=False)

    #predict
    pred_surv1 = survival_reg.predict(predict_1)
    pred_surv2 = survival_reg.predict(predict_2)

    # make DataFrame
    predict1_churn = pd.DataFrame({'acc_id': predict1_churn['acc_id'], 'survival_time': pred_surv1})
    predict2_churn = pd.DataFrame({'acc_id': predict2_churn['acc_id'], 'survival_time': pred_surv2})

    # 64  or 0
    predict1_churn['survival_time'] = np.where(
        predict1_churn.survival_time >= 64, 64, predict1_churn.survival_time)
    predict1_churn['survival_time'] = np.where(
        predict1_churn.survival_time <= 0.5, 0.6, predict1_churn.survival_time)
    predict1_churn['survival_time'] = predict1_churn['survival_time'].apply(round)

    predict2_churn['survival_time'] = np.where(
        predict2_churn.survival_time >= 64, 64, predict2_churn.survival_time)
    predict2_churn['survival_time'] = np.where(
        predict2_churn.survival_time <= 0.5, 0.6, predict2_churn.survival_time)
    predict2_churn['survival_time'] = predict2_churn['survival_time'].apply(round)


    # concate DataFrame
    pred_survival1_2 = pd.concat([predict1_sur, predict1_churn])
    pred_survival2_2 = pd.concat([predict2_sur, predict2_churn])

    final_survival1 = pd.merge(pred_survival1, pred_survival1_2, on='acc_id')
    final_survival2 = pd.merge(pred_survival2, pred_survival2_2, on='acc_id')

    # Final survival time prediect
    a1 = final_survival1.loc[:, ['survival_time_x', 'survival_time_y']]
    a2 = final_survival2.loc[:, ['survival_time_x', 'survival_time_y']]

    # concate method with operator min
    final_survival1['survival_time'] = a1.apply('min', axis=1)
    final_survival2['survival_time'] = a2.apply('min', axis=1)

    final_survival1 = final_survival1.loc[:, ['acc_id', 'survival_time']]
    final_survival2 = final_survival2.loc[:, ['acc_id', 'survival_time']]
    print('END survival time__predict')


    # START Amount spent predict'
    print('START Amount spent predict')
    y1 = pd.read_csv("./preprocess/test1_preprocess.csv")
    y2 = pd.read_csv("./preprocess/test2_preprocess.csv")



    # make categorical features
    y1 = pd.get_dummies(y1,
                        columns=['rich_monster2', 'npc_kill2', 'class', 'level', 'level_category', 'first access day'])
    y2 = pd.get_dummies(y2,
                        columns=['rich_monster2', 'npc_kill2', 'class', 'level', 'level_category', 'first access day'])


    y1_2 = y1.copy()
    y2_2 = y2.copy()
    y1 = y1.drop('acc_id', axis=1, inplace=False)
    y2 = y2.drop('acc_id', axis=1, inplace=False)

    #predict amount
    pred_amou1 = amount_reg.predict(y1)
    pred_amou2 = amount_reg.predict(y2)

    final_amount1 = pd.DataFrame({'acc_id': y1_2['acc_id'], 'amount_spent': pred_amou1})
    final_amount2 = pd.DataFrame({'acc_id': y2_2['acc_id'], 'amount_spent': pred_amou2})
    final_amount1['amount_spent'] = np.where(final_amount1.amount_spent < 0, 0, final_amount1.amount_spent)
    final_amount2['amount_spent'] = np.where(final_amount2.amount_spent < 0, 0, final_amount2.amount_spent)


    # Merge survival & amount
    final_predict1 = pd.merge(final_survival1, final_amount1, on="acc_id")
    final_predict2 = pd.merge(final_survival2, final_amount2, on="acc_id")
    final_predict1['amount_spent'] = final_predict1['amount_spent'] / final_predict1['survival_time']
    final_predict2['amount_spent'] = final_predict2['amount_spent'] / final_predict2['survival_time']

    final_predict1.to_csv('./Predict/test1_predict.csv', index=False)
    final_predict2.to_csv('./Predict/test2_predict.csv', index=False)

    print('test1_predict.csv and test2_predict.csv have been made')
