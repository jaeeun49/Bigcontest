# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from tqdm import tqdm


def make_total_df(MODE):
    '''
    Merge combat_df + activity_df + pledge_df -> total_df
    input: combat_df + activity_df + pledge_df
    output: save total.csv
    '''
    activity_df = pd.read_csv('./raw/' + MODE + '_activity.csv')
    combat_df = pd.read_csv('./raw/' + MODE + '_combat.csv')
    pledge_df = pd.read_csv('./raw/' + MODE + '_pledge.csv')

    tmp = pd.merge(activity_df, combat_df,
                   on=["day", "acc_id", "char_id", "server"], how='left')

    total = pd.merge(tmp, pledge_df,
                     on=["day", "acc_id", "char_id", "server"], how='outer')

    # Give 8 class for acc_id who are not entered in class
    total = convert_sth(total, 'class', 8)
    # Give 18 level for charicter
    total = convert_sth(total, 'level', 18)
    total = total.fillna(0)
    # print('total.shape:', total.shape)
    total.to_csv("./Preprocess/" + MODE + "_total.csv", index=False)


def convert_sth(total,option,n):

    a = total.char_id[total[option].isnull()].unique()
    aaa= {1:n}

    for i in tqdm(range(len(a))):
        b=total[total["char_id"]==a[i]].loc[:,option].unique()
        if len(b)==2:
            if np.isnan(b[0]):
                aab= {a[i]:b[1]}
                aaa.update(aab)
            else:
                aab = {a[i]:b[0]}
                aaa.update(aab)
        else:
            aab = {a[i]:n}
            aaa.update(aab)

    for i in tqdm(range(total.shape[0])):
        if np.isnan(total.loc[i,option]) :
            total.loc[i,option] = aaa.get(total.loc[i,'char_id'])

    return total

def access_week(k):
    if 1 <= k <= 7:
        return 1
    elif 8<= k <=14:
        return 2
    elif 15<= k <=21:
        return 3
    else :
        return 4


def boss_monster(k):
    if 0 <= k <= 10 :
        return 'monster_0'
    elif 10 < k <= 30 :
        return 'monster_1'
    elif 30 < k <= 60 :
        return 'monster_2'
    else :
        return 'monster_3'

def kill_npc(k):
    if 0 <= k < 50:
        return 'npc_0'
    elif 50 <= k < 100:
        return 'npc_1'
    elif 100 <= k < 150:
        return 'npc_2'
    elif 150 <= k < 200:
        return 'npc_3'
    elif 200 <= k < 250:
        return 'npc_4'
    elif 250 <= k < 300:
        return 'npc_4'
    else:
        return 'npc_5'

def week(x) :
    if (x-1)//7 == 0 :
        return 1
    elif (x-1)//7 ==1 :
        return 2
    elif (x-1)//7 == 2 :
        return 3
    elif (x-1)//7 == 3:
        return 4

def spent_group(k):
    if 0 <= k < 2 :
        return 'spent_0'
    elif 2 <= k < 5 :
        return 'spent_1'
    elif 5 <= k < 8 :
        return 'spent_2'
    else :
        return 'spent_3'

def rank_pledge(k):
    if 0 <= k < 5:
        return 'pledge_rank_0'
    elif 5 <= k < 20:
        return 'pledge_rank_1'
    elif 20 <= k < 50 :
        return 'pledge_rank_2'
    else:
        return 'pledge_rank_3'


def recent(c, column) : ## 최근 4일치 데이터 반환해주는 함수
    c_0 = c
    c_0 = c_0.sort_values(by = ['acc_id', 'day'], ascending = False).reset_index(drop=True)
    c_0 = c_0[c_0[column]!=0]
    c_0 = c_0.reset_index() # index를 column으로 빼줘서 이 index를 가지고 이따가 색인하려고 이렇게 했다.

    c_1 = c_0.drop_duplicates(subset= ['acc_id'], inplace = False)  ## 28일기준 제일 최근접속
    c_1['recent']=4

    c_2 = c_0[~c_0['index'].isin(c_1['index'])]
    c_2 = c_2.drop_duplicates(subset = ['acc_id'], inplace = False)  ## 28일기준 두번째 최근접속
    c_2['recent']=3

    c_3 = c_0[~c_0['index'].isin(c_1['index'])]
    c_3 = c_3[~c_3['index'].isin(c_2['index'])]
    c_3 = c_3.drop_duplicates(subset = ['acc_id'], inplace = False) ## 28일 기준 세번째 최근접속
    c_3['recent']=2

    c_4 = c_0[~c_0['index'].isin(c_1['index'])]
    c_4 = c_4[~c_4['index'].isin(c_2['index'])]
    c_4 = c_4[~c_4['index'].isin(c_3['index'])]
    c_4 = c_4.drop_duplicates(subset = ['acc_id'], inplace = False) ## 28일 기준 네번째 최근접속
    c_4['recent']=1

    c_total = pd.concat([c_1,c_2,c_3,c_4], ignore_index = True) ##  acc_id, day, playtime, fishing, recent(1~4의값) 의 칼럼
    c_total = c_total.drop('index', axis =1, inplace= False).sort_values(by = ['acc_id', 'day'], ascending=False).reset_index(drop=True)
    return c_total


def fluc(df_activity):
    #
    c = df_activity.pivot_table(index=['acc_id', 'day']).reset_index(level=['acc_id', 'day'])
    c_playtime = c[['acc_id', 'day', 'playtime']]  ##  playtime,  대해서만 최근 4주반환 위한 데이터
    c_fishing = c[['acc_id', 'day', 'fishing']]  ##  fishing에 대해서만 최근 4주반환 위한 데이터

    # 여기까지는, fluctuation 구하기위한 최근 4일 접속때의 playtime, fishing 데이터 정리
    # 이제 이 데이터들 가지고 fluctuation 구하기
    c_play = recent(c_playtime, 'playtime')
    c_fish = recent(c_fishing, 'fishing')

    #     d = c_total.pivot_table(index = 'acc_id', columns = 'recent')

    d_playtime = c_play.pivot_table(index='acc_id', columns='recent')['playtime'].reset_index().fillna(0.00234)
    d_fishing = c_fish.pivot_table(index='acc_id', columns='recent')['fishing'].reset_index().fillna(0.000039)
    #     d_fishing = d['fishing'].fillna(0.000039).reset_index()   # fishing ==0 아닌 것중에 제일 최솟값
    #     d_fishing = d_fishing.replace(0,0.000039) ##   fishing ==0 인것들도 inf 방지 위해 제일 최솟값으로 채워
    #     d_playtime = d['playtime'].fillna(0.00234).reset_index() # playtime ==0 아닌 것중에 제일 최솟값
    #     d_playtime = d_playtime.replace(0, 0.00234) ## playtime==0 인애들이 inf 만들어서 이를 방지위해 최솟값 채워

    fluc_playtime = pd.DataFrame(d_playtime['acc_id'])
    fluc_playtime['play_fluc_1'] = (d_playtime[2] - d_playtime[1]) / d_playtime[1]
    fluc_playtime['play_fluc_2'] = (d_playtime[3] - d_playtime[2]) / d_playtime[2]
    fluc_playtime['play_fluc_3'] = (d_playtime[4] - d_playtime[3]) / d_playtime[3]

    fluc_fishing = pd.DataFrame(d_fishing['acc_id'])
    fluc_fishing['fish_fluc_1'] = (d_fishing[2] - d_fishing[1]) / d_fishing[1]
    fluc_fishing['fish_fluc_2'] = (d_fishing[3] - d_fishing[2]) / d_fishing[2]
    fluc_fishing['fish_fluc_3'] = (d_fishing[4] - d_fishing[3]) / d_fishing[3]

    fluc_total = pd.merge(fluc_playtime, fluc_fishing, on='acc_id', how='outer').fillna(0)
    return fluc_total




def make_df_1(total, values,col):
    '''
    :param total:
    :param values:
    :param col:
    :return: df
    '''
    df = pd.pivot_table(total, index='acc_id',
                        columns='day', values=values,
                        aggfunc=np.sum, fill_value=0).reset_index()

    df[col+'_1weak'] = df.iloc[:,1:8].mean(axis=1)
    df[col+'_2weak'] = df.iloc[:,8:15].mean(axis=1)
    df[col+'_3weak'] = df.iloc[:,15:22].mean(axis=1)
    df[col+'_4weak'] = df.iloc[:,22:29].mean(axis=1)

    return df

def make_df_2(total, values,col):
    df = pd.pivot_table(total, index='acc_id', columns='day',
                        values=values, aggfunc=np.sum,
                        fill_value=0)

    df[col+'_1weak'] = df.iloc[:,1:8].mean(axis=1)
    df[col+'_2weak']= df.iloc[:,8:15].mean(axis=1)
    df[col+'_3weak'] = df.iloc[:,15:22].mean(axis=1)
    df[col+'_4weak'] = df.iloc[:,22:29].mean(axis=1)

    return df


def trader(df):
    df['exp_wk1'] = df['solo_exp_1weak'] + df['party_exp_1weak'] + df['quest_exp_1weak']
    df['exp_wk2'] = df['solo_exp_2weak'] + df['party_exp_2weak'] + df['quest_exp_2weak']
    df['exp_wk3'] = df['solo_exp_3weak'] + df['party_exp_3weak'] + df['quest_exp_3weak']
    df['exp_wk4'] = df['solo_exp_4weak'] + df['party_exp_4weak'] + df['quest_exp_4weak']
    # 주차별  플레이시간 / 접속횟수
    df['playtime/attend_number_wk1'] = df['playtime_1weak'] / (df['attend_number_wk1'] + 1e-5)
    df['playtime/attend_number_wk2'] = df['playtime_2weak'] / (df['attend_number_wk2'] + 1e-5)
    df['playtime/attend_number_wk3'] = df['playtime_3weak'] / (df['attend_number_wk3'] + 1e-5)
    df['playtime/attend_number_wk4'] = df['playtime_4weak'] / (df['attend_number_wk4'] + 1e-5)

    # 주차별  경험치 획득량 / 플레이 시간
    df['exp_wk1/playtime_1weak'] = df['exp_wk1'] / (df['playtime_1weak'] + 1e-5)
    df['exp_wk2/playtime_2weak'] = df['exp_wk2'] / (df['playtime_2weak'] + 1e-5)
    df['exp_wk3/playtime_3weak'] = df['exp_wk3'] / (df['playtime_3weak'] + 1e-5)
    df['exp_wk4/playtime_4weak'] = df['exp_wk4'] / (df['playtime_4weak'] + 1e-5)

    # 주차별 money변화량 / 플레이시간
    df['game_money_change_1weak/playtime_1weak'] = df['game_money_change_1weak'] / (df['playtime_1weak'] + 1e-5)
    df['game_money_change_2weak/playtime_2weak'] = df['game_money_change_2weak'] / (df['playtime_2weak'] + 1e-5)
    df['game_money_change_3weak/playtime_3weak'] = df['game_money_change_3weak'] / (df['playtime_3weak'] + 1e-5)
    df['game_money_change_4weak/playtime_4weak'] = df['game_money_change_4weak'] / (df['playtime_4weak'] + 1e-5)

    # 주차별 money변화량 / 접속횟수
    df['game_money_change_1weak/attend_number_wk1'] = df['game_money_change_1weak'] / (
                df['attend_number_wk1'] + 1e-5)
    df['game_money_change_2weak/attend_number_wk2'] = df['game_money_change_2weak'] / (
                df['attend_number_wk2'] + 1e-5)
    df['game_money_change_3weak/attend_number_wk3'] = df['game_money_change_3weak'] / (
                df['attend_number_wk3'] + 1e-5)
    df['game_money_change_4weak/attend_number_wk4'] = df['game_money_change_4weak'] / (
                df['attend_number_wk4'] + 1e-5)

    # 주차별 private shop / playtime

    df['private_shop_wk1/playtime_wk1'] = df['private_shop_1weak'] / (df['playtime_1weak'] + 1e-5)
    df['private_shop_wk2/playtime_wk2'] = df['private_shop_2weak'] / (df['playtime_2weak'] + 1e-5)
    df['private_shop_wk3/playtime_wk3'] = df['private_shop_3weak'] / (df['playtime_3weak'] + 1e-5)
    df['private_shop_wk4/playtime_wk4'] = df['private_shop_4weak'] / (df['playtime_4weak'] + 1e-5)

    # 다 쓴 변수 드랍

    df.drop(['exp_wk1', 'exp_wk2', 'exp_wk3', 'exp_wk4'], axis=1, inplace=True)

    # 28일 쓴금액 / 28일 생존기간
    # df['sum_spent/attend_period'] = df['sum_spent'] / df['attend_period']

    return df


def fixed_moved_pledge(MODE):
    '''
    input: ex) MODE==test1
    output: DataFrame pledge_non_move_num
    ( columns: non_move_num, total_char_num)
    i.e. 각 혈맹에 총 캐릭터 수와 한번도 나가지 않은 고정 혈원의 수
    '''

    pledge_df = pd.read_csv('./raw/' + MODE + '_pledge.csv')

    # 1 캐릭터 별로 혈맹을 몇개 옮겨 다녔는지
    pledge_char = pd.DataFrame(pledge_df.groupby(['server', 'char_id', 'pledge_id'])[
                                   'play_char_cnt'].sum())
    pledge_char.drop('play_char_cnt', axis=1, inplace=True)

    # 2 pledge_char == 서버 별 char_id 와 그들의 pledge_id
    pledge_char = pledge_char.reset_index()
    pledge_char_move = pledge_char.pivot_table(
        index=['server', 'char_id'], aggfunc='count', values='pledge_id')
    pledge_char_move = pledge_char_move.reset_index()
    pledge_char_move.columns = ['server', 'char_id', 'pledge_id_num']

    # 3 한 pledge 에 pledge 옮기지 않았던 사람이 몇 명있는지
    pledge_pledge_char = pd.DataFrame(pledge_df.groupby(
        ['server', 'pledge_id'])['char_id'].value_counts())
    pledge_pledge_char.drop('char_id', axis=1, inplace=True)
    pledge_pledge_char = pledge_pledge_char.reset_index()

    # 4 pledge_id 별로 , 어떤 char_id가 있고, 걔들은 pledge 몇번 옮겼는지(pledge_id_num)를 구하기 위한 밑작업 + pledge 별 loyal 멤버 몇명있는지
    pledge_move_final = pd.merge(
        pledge_pledge_char, pledge_char_move, on=['server', 'char_id'])

    # 5 pledge 안옮겼던 loyal 멤버
    pledge_move_final_2 = pledge_move_final[pledge_move_final['pledge_id_num'] == 1]
    # pledge 옮겼던 멤버들
    pledge_move_final_2_1 = pledge_move_final[pledge_move_final['pledge_id_num'] != 1]

    # 6 final_3 == pledge 별로 loyal 멤버 몇명있는지
    pledge_move_final_3 = pledge_move_final_2.pivot_table(
        index=['server', 'pledge_id'], aggfunc='count', values='pledge_id_num')
    pledge_move_final_3 = pledge_move_final_3.reset_index()

    pledge_move_final_3_1 = pledge_move_final_2_1.pivot_table(
        index=['server', 'pledge_id'], aggfunc='count', values='pledge_id_num')
    pledge_move_final_3_1 = pledge_move_final_3_1.reset_index()
    copy_1 = pledge_move_final_3_1.copy()
    copy_1['pledge_id_num'] = 0

    # 7 len(pledge_move_final_3_2)  --> pledge_move_final_3_2 : server 별 pledge_id 별 loyal 멤버 수
    pledge_move_final_3_2 = pd.concat(
        [pledge_move_final_3, copy_1], ignore_index=True)
    pledge_move_final_3_2 = pledge_move_final_3_2.drop_duplicates(
        subset=['server', 'pledge_id'], keep='first')

    # pledge 별로 amout 구하기. pledge에 혈원이 몇명 있었는지
    pledge_move_final_4 = pledge_move_final.copy()
    pledge_move_final_4 = pledge_move_final_4.pivot_table(index=['server', 'pledge_id'], aggfunc='count',
                                                          values='char_id')
    pledge_move_final_4 = pledge_move_final_4.reset_index()  # -> final_4 -> pledge 에 혈원이 몇몇 있었는지
    pledge_non_move_num = pd.merge(pledge_move_final_3_2, pledge_move_final_4, on=['server', 'pledge_id'])
    pledge_non_move_num.columns = ['server', 'pledge_id', 'non_move_num', 'total_char_num']

    return pledge_non_move_num


def make_prepledge_df(total, mode):
    '''
    서버별 혈맹을 군집화 하기위한 작업
    '''
    # 1 Find member who have level 16,17
    pledge_highlevel = pd.DataFrame(
        columns=['server', 'pledge_id', 'highlevel'])


    for i in tqdm(total.server.unique()):
        server = total[total['server'] == i]
        df = server.pivot_table(values='char_id', index=server.pledge_id,
                            columns='level', aggfunc='count', fill_value=0).reset_index()
        df = df.drop([0.0])
        if 18.0 in df.columns:
            df = df.drop(18.0, axis=1, inplace=False)
        if len(df.pledge_id) != 0:
            if 17.0 in df.columns and 16.0 in df.columns:
                df3 = pd.DataFrame(
                {'server': i, 'pledge_id': df.pledge_id, 'highlevel': df[16.0] + df[17.0]})
            elif 17.0 in df.columns and 16.0 not in df.columns:
                df3 = pd.DataFrame(
                {'server': i, 'pledge_id': df.pledge_id, 'highlevel': df[17.0]})
            elif 17.0 not in df.columns and 16.0 in df.columns:
                df3 = pd.DataFrame(
                {'server': i, 'pledge_id': df.pledge_id, 'highlevel': df[16.0]})
            elif 17.0 not in df.columns and 16.0 not in df.columns:
                df3 = pd.DataFrame(
                {'server': i, 'pledge_id': df.pledge_id, 'highlevel': 0})
        pledge_highlevel = pledge_highlevel.append(df3)


    pledge_highlevel = pledge_highlevel.drop_duplicates(['server', 'pledge_id'])


    # 2 각 혈맹에 총 캐릭터 수와 한번도 나가지 않은 고정 혈원의 수
    pledge_non_move_num = fixed_moved_pledge(mode)


    # 3 각 혈맹이 접속한 횟수
    pledge_access_day = pd.DataFrame(
        columns=['server', 'pledge_id', 'pledge_access_day'])

    for i in tqdm(total.server.unique()):
        server = total[total['server'] == i]
        f = server.groupby('pledge_id')['day'].agg(
            [('pledge_access_day', lambda x: len(x.unique()))]).reset_index()
        f = f.drop([0.0])
        df = pd.DataFrame({'server': i, 'pledge_id': f.pledge_id,
                           'pledge_access_day': f.pledge_access_day})
        pledge_access_day = pledge_access_day.append(df)

    # 4 기존 혈맹 컬럼으로 변수 만들기
    # 4-1 prepare
    # columns_1 == need_columns which gonna be sum
    columns_1 = ['random_attacker_cnt_y', 'random_defender_cnt_y', 'same_pledge_cnt_y',
                 'temp_cnt_y', 'etc_cnt_y', 'combat_char_cnt', 'pledge_combat_cnt', 'combat_play_time',
                 'non_combat_play_time']

    # import total data
    total_df = total.pivot_table(
        index=['server', 'pledge_id'], aggfunc='sum', values=columns_1)

    # 4-2 delete the pledge zero row
    total_df = total_df.reset_index(['server', 'pledge_id'])

    # 4-3 find  index number of zero value
    cc = total_df.pledge_id == 0.0
    delete_index = []
    for _, i in enumerate(cc):
        if i == True:
            delete_index.append(_)

    # 4-4 drop delet_index
    total_df = total_df.drop(delete_index)

    # # 5 혈맹의 핵과금 혈맹이 몇명인지
    # pm = pd.read_csv('./rawdata/' + mode + '_payment.csv')
    #
    # f = pm.groupby('acc_id')['amount_spent'].agg(
    #     [('amount_spent_28', 'sum')]).reset_index()
    #
    # f['amount_spent_28_ca'] = f['amount_spent_28'].apply(spent_group)
    # total2 = pd.merge(total, f, on=['acc_id'], how='left')
    # total2 = total2.fillna(0)
    # pledge_highspent = pd.DataFrame(columns=['server', 'pledge_id', 'highspent'])
    #
    # for i in tqdm(total2.server.unique()):
    #     server = total2[total2['server'] == i]
    #     df = server.pivot_table(values='acc_id', index='pledge_id',
    #                             columns='amount_spent_28_ca', aggfunc='count')
    #
    #     df = df.fillna(0)
    #     df = df.reset_index()
    #     df = df.drop([0.0])
    #
    #     if 'spent_3' in df.columns:
    #         df3 = pd.DataFrame(
    #             {'server': i, 'pledge_id': df.pledge_id, 'highspent': df.spent_3})
    #     else:
    #         df3 = pd.DataFrame(
    #             {'server': i, 'pledge_id': df.pledge_id, 'highspent': 0})
    #     pledge_highspent = pledge_highspent.append(df3)
    #
    # pledge_highspent = pledge_highspent.drop_duplicates(['server', 'pledge_id'])


    # 6 merge DataFrame
    pledge1 = pd.merge(pledge_highlevel, pledge_non_move_num, on=['server', 'pledge_id'])
    pledge2 = pd.merge(pledge1, pledge_access_day, on=['server', 'pledge_id'])
    pledge_total = pd.merge(pledge2, total_df, on=['server', 'pledge_id'])
    # pledge_total = pd.merge(pledge3, pledge_highspent, on=['server', 'pledge_id'])

    print('--- pledge_total generated---')
    return pledge_total
