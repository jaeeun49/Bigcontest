# -*- coding: utf-8 -*-
# Copyright (c) 2019-
# Members: Hoe Sung Ryu( hoesungryu@yonsei.ac.kr )
#          Jeong jaeeun( today1255@gmail.com )

from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from bitwin_def.bitwin_test import *


def make_activity_features(total):
    '''

    '''
    features = []
    sequence_1 = [0, -4, -3, -2, -1]
    sequence_2 = [0, 1, 2, 3, 4]

    # 2-1 weekly playtime
    df = make_df_1(total, 'playtime', 'playtime')
    features.append(df.iloc[:, sequence_1])
    # print('[ 1 / 22 ] weekly playtime done')

    # 2-2 playtime MACD
    df = make_df_2(total, 'playtime', 'playtime_ma')
    idx = df.iloc[:, [-4, -3, -2, -1]].T
    idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    features.append(idx2.iloc[:, sequence_2])
    # print('[ 2 / 22 ] playtime MACD done')

    # 2-3 주당 솔로사냥획득 경험치
    df = make_df_1(total, 'solo_exp', 'solo_exp')
    features.append(df.iloc[:, sequence_1])
    # print('[ 3 / 22 ] weekly solo_exp done')

    # 2-4 주당 파티사냥획득 경험치
    df = make_df_1(total, 'party_exp', 'party_exp')
    features.append(df.iloc[:, sequence_1])
    # print('[ 4 / 22 ] weekly party_exp done')

    # 2-5 솔로사냥 이동평균
    # df = make_df_2(total, 'solo_exp', 'solo_exp_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx.iloc[:, sequence_2])
    # print('[ 5 / 22 ] weekly solo_exp MACD done')

    # 2-6 파티사냥 이동평균
    # df = make_df_2(total, 'party_exp', 'party_exp_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx.iloc[:, sequence_2])
    # print('[ 6 / 22 ] weekly party_exp MACD done')

    # 2-7 파티사냥획득 경험치/솔로사냥획득 경험치
    f = total.groupby('acc_id')['solo_exp'].agg([('solo_exp', 'sum')]).reset_index()
    v = total.groupby('acc_id')['party_exp'].agg([('party_exp', 'sum')]).reset_index()
    f.solo_exp = np.log(f.solo_exp + 2)
    v.party_exp = np.log(v.party_exp + 2)  # discard zero by add 2
    fv = pd.merge(f, v, on="acc_id")
    fv['ratio_sol_party'] = fv['solo_exp'] / fv['party_exp']
    fv = fv.fillna(0)
    features.append(fv.iloc[:, [0, 3]])
    # print('[ 7 / 22 ] weekly solo/party done')

    # 2-8 주당 평균 획득 경험치
    df = make_df_1(total, 'quest_exp', 'quest_exp')
    features.append(df.iloc[:, sequence_1])
    # print('[ 8 / 22 ] weekly quest_exp done')

    # # 2-9 주당 평균 획득 경험치 이동평균
    # df = make_df_2(total, 'quest_exp', 'quest_exp_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    # print('[ 9 / 22 ]  weekly quest_exp MACD done')

    # 2-10 주당 보스몬스터 타격횟수
    df = make_df_1(total, 'rich_monster', 'rich_monster')
    features.append(df.iloc[:, sequence_1])
    # print('[ 10 / 22 ] Hit the boss monster done')

    # 2-11 주당 보스몬스터 타격 이동평균
    # df = make_df_2(total, 'rich_monster', 'rich_monster_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    # print('[ 11 / 22] Hit boss monster MACD done ')

    # 2-12 보스몬스터 잔존활동과 무슨 연관이 있는지 확인해보기
    f = total.groupby('acc_id')['rich_monster'].agg([('rich_monster', 'sum')]).reset_index()
    f['rich_monster2'] = f['rich_monster'].apply(boss_monster)
    monster_title_replace = {'monster_0': 0, 'monster_1': 1, \
                     'monster_2': 2, 'monster_3': 3
                     }
    f['rich_monster2'] = f['rich_monster2'].apply(lambda x: monster_title_replace.get(x))
    features.append(f.iloc[:, [0, -1]])
    # print('[ 12 / 22 ] relation between boss and remain')


    # 2-13 주당 NPC 살해 횟수
    df = make_df_1(total, 'npc_kill', 'npc_kill')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'npc_kill', 'npc_kill_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    # print('[ 13 / 22] kill npc done ')

    # 2-14 NPC죽인 횟수와 잔존활동과 무슨 연관이 있는지 확인해보기
    f = total.groupby('acc_id')['npc_kill'].agg([('npc_kill', 'sum')]).reset_index()
    f['npc_kill2'] = f['npc_kill'].apply(kill_npc)
    npc_title_replace = {'npc_0': 0, 'npc_1': 1, 'npc_2': 1, \
                             'npc_3': 2, 'npc_4': 3, 'npc_5': 3}
    f['npc_kill2'] = f['npc_kill2'].apply(lambda x: npc_title_replace.get(x))
    features.append(f.iloc[:, [0, -1]])
    # print('[ 14 / 22] relation between kill the npc and remain')

    # 2-15 주별 캐릭터 사망&부할 관련&경험치 복구 횟수
    df = make_df_1(total, 'death', 'death')
    features.append(df.iloc[:, sequence_1])

    df = make_df_1(total, 'revive', 'revive')
    features.append(df.iloc[:, sequence_1])

    df = make_df_1(total, 'exp_recovery', 'exp_recovery')
    features.append(df.iloc[:, sequence_1])




    # df = make_df_2(total, 'death', 'death_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    #
    #
    # df = make_df_2(total, 'revive', 'revive_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    #
    # df = make_df_2(total, 'exp_recovery', 'exp_recovery_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    # print('[ 15 / 22 ] weekly revive,death recover exp')

    # 2-16 부활비율
    f = total.groupby('acc_id')['death'].agg([('death', 'sum')]).reset_index()
    v = total.groupby('acc_id')['revive'].agg([('revive', 'sum')]).reset_index()
    f.death = np.log(f.death + 2)
    v.revive = np.log(v.revive + 2)
    fv = pd.merge(f, v, on="acc_id")
    fv['ratio_revive'] = fv['revive'] / fv['death']
    fv = fv.fillna(0)
    features.append(fv.iloc[:, [0, 3]])
    # print('[ 16 / 22 ] ratio of ratio done ')

    # 2-17 주별 낚시 시간/개인 상점 운영시간-주당
    df = make_df_1(total, 'fishing', 'fishing')
    features.append(df.iloc[:, sequence_1])

    df = make_df_1(total, 'private_shop', 'private_shop')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'fishing', 'fishing_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    # df = make_df_2(total, 'private_shop', 'private_shop_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    # print('[ 17 / 22] fishing & shop ')

    # 2-18 주별 아데나 평균 변동량
    df = make_df_1(total, 'game_money_change', 'game_money_change')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'game_money_change', 'game_money_change_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])
    # print('[ 18 / 22 ] fluctuation of adena')

    # 2-19 아이템 강화 시도횟수
    f = total.groupby('acc_id')['enchant_count'].agg([('enchant_count', 'sum')]).reset_index()
    features.append(f)
    # print('[ 19 / 22 ] overall weapon ')

    # # 2-20 접속일수 (28일중 얼마나 접속했는지)
    # f = total.groupby('acc_id')['day'].agg([('number access', lambda x: len(x.unique()))]).reset_index()
    # features.append(f.iloc[:, [0, 1]])
    # print('[20/22]')

    # 2 - 20 각 주차별 접속 & 28일 전체 접속횟수 update !
    df = total.copy()

    a_days = df.groupby(['acc_id', 'day'])['rich_monster'].count()
    a_days_df = pd.DataFrame(a_days)
    a_days_df = a_days_df.reset_index(level=['acc_id', 'day'])
    a_days_df.drop('rich_monster', axis=1, inplace=True)

    b_days_df = a_days_df.copy()
    b_days_df['week'] = b_days_df['day'].apply(lambda x: week(x))

    b_days_df_1 = b_days_df.pivot_table(index='acc_id', columns='week', aggfunc='count', values='day')
    b_days_df_1 = b_days_df_1.fillna(0)
    b_days_df_1.columns = ['attend_number_wk1', 'attend_number_wk2', 'attend_number_wk3', 'attend_number_wk4']
    b_days_df_1 = b_days_df_1.reset_index()

    a_days_pivot = a_days_df.pivot_table(index='acc_id', aggfunc='min', values='day')
    a_days_pivot = a_days_pivot.reset_index()
    a_days_pivot.columns = ['acc_id', 'min_day']

    a_days_pivot['attend_period'] = 28 - a_days_pivot['min_day'] + 1

    a_2 = df.groupby('acc_id')['day'].nunique()
    a_2 = pd.DataFrame(a_2)
    a_2 = a_2.reset_index()
    a_2.columns = ['acc_id', 'attend_numbers']
    a_days_pivot = pd.merge(a_days_pivot, a_2, on='acc_id')
    a_days_pivot.drop(['min_day'], axis=1, inplace=True)
    period = pd.merge(a_days_pivot, b_days_df_1, on='acc_id')
    features.append(period)
    # print('[ 20 / 22 ] weekly and total access')


    # 2-21 장사꾼 판단 ; 총 활동시간중 개인상점 운영시간이 80%를 넘으면 장사꾼으로 판단
    big = total.pivot_table(index='acc_id', aggfunc='mean', values=['playtime', 'private_shop'])
    big = big.reset_index()
    big['shop/play'] = big['private_shop'] / big['playtime']
    big['trader'] = big['shop/play'].apply(lambda x: 1 if x > 0.8 else 0)  # 장사꾼 :1   일반유저 : 0
    features.append(big.iloc[:, [0, -1]])
    # print('[ 21 / 22 ] who are the merchant ')

    # 2-22 처음 접속한 주차
    f = total.groupby('acc_id')['day'].agg([('first access day', lambda x: x.min())]).reset_index()
    f['first access day'] = f['first access day'].apply(access_week)
    features.append(f)
    # print('[ 22 / 22 ] first access week ')



    final_df = pd.DataFrame({'acc_id': total.acc_id.unique()})

    for i in features:
        final_df = pd.merge(final_df, i, how='left', on='acc_id')

    activity_df = trader(final_df)

    tmp_total_df = fluc(total)

    final_activity_df = pd.merge(activity_df, tmp_total_df, how='left', on='acc_id')
    print('--- Activity Process Done ---')

    return final_activity_df


def make_combat_features(total):
    '''

    '''
    features = []
    sequence_1 = [0, -4, -3, -2, -1]
    sequence_2 = [0, 1, 2, 3, 4]

    # 3-1 한달간 활동 캐릭터 수
    # print('[ 1 / 12 ] the number of play character')
    f = total.groupby('acc_id')['char_id'].agg(
        [('num_of_char', lambda x: len(x.unique()))]).reset_index()
    features.append(f.iloc[:, [0, 1]])

    # 3-2 각 class에다 쏟는 활동시간/가장 많은 시간을 쏟는 캐릭터의 class
    # print('[ 2 / 12  ] find main character class')
    df = pd.pivot_table(total, index='acc_id', columns='class',
                        values='playtime', aggfunc=np.sum, fill_value=0).reset_index()
    df = df.drop(8.0, axis=1, inplace=False)

    for i in tqdm(range(df.shape[0])):
        a = max(df.iloc[i, [1, 2, 3, 4, 5, 6, 7, 8]])
        for j in range(1, 9):
            if df.iloc[i, j] == a:
                df.loc[i, 'class'] = df.columns[j]

    df.columns = ['acc_id', 'class0', 'class1', 'class2',
                  'class3', 'class4', 'class5', 'class6', 'class7', 'class']
    features.append(df)

    # 3-3 각 level에다 쏟는 활동시간/ 가장 많은 시간을 쏟는 캐릭터의 level/ 레벨을 초보/중수/고수로 나눠보기
    # print('[ 3 / 12 ] put an effort in each level')
    df = pd.pivot_table(total, index='acc_id', columns='level',
                        values='playtime', aggfunc=np.sum, fill_value=0).reset_index()
    df = df.drop(18.0, axis=1, inplace=False)

    for i in tqdm(range(df.shape[0])):
        a = max(df.iloc[i, [1, 2, 3, 4, 5, 6, 7, 8, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18]])
        for j in range(1, 19):
            if df.iloc[i, j] == a:
                df.loc[i, 'level'] = df.columns[j]

    # 0-10 : 초보 // 11 -14: 중수 //15-17: 고수
    for i in tqdm(range(df.shape[0])):
        if df.loc[i, 'level'] <= 10:
            df.loc[i, 'level_category'] = 1
        elif df.loc[i, 'level'] <= 14 and df.loc[i, 'level'] > 10:
            df.loc[i, 'level_category'] = 2
        else:
            df.loc[i, 'level_category'] = 3

    df.columns = ['acc_id', 'level_0', 'level_1', 'level_2',
                  'level_3', 'level_4', 'level_5', 'level_6',
                  'level_7', 'level_8', 'level_9', 'level_10',
                  'level_11', 'level_12', 'level_13', 'level_14',
                  'level_15', 'level_16', 'level_17', 'level', 'level_category'
                  ]
    features.append(df)

    # 3-4 얼마나 다양한 직업의 캐릭터를 가지고 있는지
    # print('[ 4 / 12 ] various character position ')
    f = total.groupby('acc_id')['class'].agg(
        [('occupational diversity', lambda x: len(x.unique()))]).reset_index()
    features.append(f.iloc[:, [0, 1]])

    # 3- 5 얼마나 다양한 레벨의 캐릭터를 가지고 있는지
    # print('[ 5 / 12 ] various character level')
    f = total.groupby('acc_id')['level'].agg(
        [('level diversity', lambda x: len(x.unique()))]).reset_index()
    features.append(f.iloc[:, [0, 1]])

    # 3- 6 주당 혈맹간 전투에 참여한 횟수
    # print('[ 6 / 12 ] the number of join the pledge battle ')
    df = make_df_1(total, 'pledge_cnt', 'pledge_cnt')
    features.append(df.iloc[:, sequence_1])
    #
    # df = make_df_2(total, 'pledge_cnt', 'pledge_cnt_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    # 3 - 7 주당 막피공격을 행한 횟수
    # print('[ 7 / 12 ] troll hit count')
    df = make_df_1(total, 'random_attacker_cnt_x', 'random_attacker_cnt_x')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'random_attacker_cnt_x', 'random_attacker_cnt_x_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    # 3-8 주 당 막피로부터 공격을 받은 횟수
    # print('[ 8 / 12 ] attacked by other player ')
    df = make_df_1(total, 'random_defender_cnt_x', 'random_defender_cnt_x')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'random_defender_cnt_x', 'random_defender_cnt_x_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    # 3-9 주 당 단발성 전투횟수
    # print('[ 9 / 12 ] instant battle')
    df = make_df_1(total, 'temp_cnt_x', 'temp_cnt_x')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'temp_cnt_x', 'temp_cnt_x_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    # 3-10 주당 동일 혈맹원간 전투 횟수
    # print('[ 10 / 12 ] combat with same pledge member')
    df = make_df_1(total, 'same_pledge_cnt_x', 'same_pledge_cnt_x')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'same_pledge_cnt_x', 'same_pledge_cnt_x_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    # 3-11 주당 기타전투 횟수
    # print('[ 11 / 12 ] weekly combat ')
    df = make_df_1(total, 'etc_cnt_x', 'etc_cnt_x')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'etc_cnt_x', 'etc_cnt_x_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    # 3-12 주당 전투 상대 캐릭터 수
    # print('[ 12 / 12 ] the number of battle')
    df = make_df_1(total, 'num_opponent', 'num_opponent')
    features.append(df.iloc[:, sequence_1])

    # df = make_df_2(total, 'num_opponent', 'num_opponent_ma')
    # idx = df.iloc[:, [-4, -3, -2, -1]].T
    # idx2 = idx.rolling(window=2, min_periods=1).mean().T.reset_index()
    # features.append(idx2.iloc[:, sequence_2])

    combat_df = pd.DataFrame({'acc_id': total.acc_id.unique()})

    for i in features:
        combat_df = pd.merge(combat_df, i, how='left', on='acc_id')

    print('--- Combat Process Done ---')

    return combat_df


def make_pledge_features(pledge_total,total):
    '''
    1. 서버별 혈맹 군집 나누기
    2. 혈맹 등급나눈걸로 유저별 변수 만들기
    '''
    features = []
    df2 = pd.DataFrame(columns=['server', 'pledge_id', 'group'])

    # 1 서버별 서버별 혈맹 군집 나누기
    for i in tqdm(pledge_total.server.unique()):
        server = pledge_total[pledge_total['server'] == i]
        server2 = server.drop(['server', 'pledge_id'], axis=1, inplace=False)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(server2)
        feature = scaler.fit_transform(server2)

        k = 3
        # create model and prediction
        model = KMeans(n_clusters=k, algorithm='auto')
        model.fit(feature)
        predict = pd.DataFrame(model.predict(feature))
        predict.columns = ['predict']

        # 결과 합치기
        final_df = pd.DataFrame(np.hstack((predict, feature)))
        # 컬럼명 지정
        cols = list(server2.columns.values)
        cols.insert(0, 'group')
        final_df.columns = cols

        server = server.reset_index()
        final_df = final_df.reset_index()

        raw_data = {'server': '{0}'.format(
            i), 'pledge_id': server['pledge_id'], 'group': final_df['group']}
        data = pd.DataFrame(raw_data)
        df2 = pd.concat([df2, data])

    grade_dic = df2.group.value_counts().sort_values().to_dict()

    def group2grade(group):
        if group == list(grade_dic.keys())[0]:
            return int(3)
        elif group == list(grade_dic.keys())[1]:
            return int(2)
        elif group == list(grade_dic.keys())[2]:
            return int(1)

    df2['grade'] = df2['group'].apply(group2grade)


    # 2 혈맹 등급나눈걸로 유저별 변수 만들기

    total2 = pd.merge(total, df2, on=['server', 'pledge_id'], how='left')
    a = total2.loc[:, ['server', 'acc_id', 'pledge_id', 'grade']]

    aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})

    a_group1 = a[a['grade'] == 1.0]
    f = a_group1.groupby('acc_id')['server'].agg(
        [('pledge_rank1', lambda x: len(x.unique()))]).reset_index()

    a_group2 = a[a['grade'] == 2.0]
    f2 = a_group2.groupby('acc_id')['server'].agg(
        [('pledge_rank2', lambda x: len(x.unique()))]).reset_index()

    a_group3 = a[a['grade'] == 3.0]
    f3 = a_group3.groupby('acc_id')['server'].agg(
        [('pledge_rank3', lambda x: len(x.unique()))]).reset_index()

    b = pd.merge(aa, f, how='left', on='acc_id')
    b2 = pd.merge(b, f2, how='left', on='acc_id')
    b3 = pd.merge(b2, f3, how='left', on='acc_id')
    b3 = b3.fillna(0)

    b3['pledge_rank_total'] = b3['pledge_rank1'] + \
                              b3['pledge_rank2'] * 5 + b3['pledge_rank3'] * 20


    features.append(b3)


    # 3 기존 혈맹 데이터에서 변수 만들기
    a = total.drop_duplicates(['acc_id', 'server', 'pledge_id'])
    a2 = pd.merge(a, pledge_total, on=['server', 'pledge_id'], how='left')
    a2 = a2.fillna(0)

    columns_1 = ['highlevel', 'non_move_num', 'total_char_num', 'pledge_access_day', 'random_attacker_cnt_y_y',
                 'random_defender_cnt_y_y', 'same_pledge_cnt_y_y', 'temp_cnt_y_y', 'etc_cnt_y_y', 'combat_char_cnt_y',
                 'pledge_combat_cnt_y', 'combat_play_time_y', 'non_combat_play_time_y']

    a2 = a2.pivot_table(index=['acc_id'], aggfunc='sum', values=columns_1)
    a2 = a2.reset_index(['acc_id'])
    features.append(a2)

    pledge_df = pd.DataFrame({'acc_id': total.acc_id.unique()})

    for i in features:
        pledge_df = pd.merge(pledge_df, i, how='left', on='acc_id')

    print('--- pledge Process Done ---')

    return pledge_df


def make_trade_df(total, MODE):
    '''
    output = trade_df
    '''
    features = []

    trade_df = pd.read_csv('./raw/' + MODE + '_trade.csv')


    # 1 아이템을 비슷한 특성을 지닌것끼리 묶어 카테고리화 시키기.
    # print('[ 1 / 12 ] Categorize the item ')
    replace_item_dic_1_sell = {'weapon': 'prop_1', 'armor': 'prop_1', 'accessory': 'prop_1', 'etc': 'prop_2',
                               'enchant_scroll': 'prop_2', 'spell': 'prop_2', 'adena': 'prop_3'}

    trade_df['item_type'] \
        = trade_df['item_type'].apply(lambda x: replace_item_dic_1_sell.get(x))

    # 2 해당 아이템을 거래(판매/구매) 한 횟수
    # print('[ 2 / 12 ] count the sell and buy')
    prop_list = ['prop_1', 'prop_2', 'prop_3']

    # for sell
    for i in prop_list:
        aa = pd.DataFrame(
            {'acc_id': total.drop_duplicates('acc_id')['acc_id']})
        trade_df2 = trade_df[trade_df['item_type'] == i]
        df = pd.pivot_table(trade_df2, index='source_acc_id', columns='day',
                            values='target_acc_id', aggfunc='count', fill_value=0).reset_index()
        df['{0}_sell_1weak'.format(i)] = df.iloc[:, 1:8].mean(axis=1)
        df['{0}_sell_2weak'.format(i)] = df.iloc[:, 8:15].mean(axis=1)
        df['{0}_sell_3weak'.format(i)] = df.iloc[:, 15:22].mean(axis=1)
        df['{0}_sell_4weak'.format(i)] = df.iloc[:, 22:29].mean(axis=1)
        df = df.rename(columns={'source_acc_id': 'acc_id'})
        tmp = pd.merge(aa, df, how='left', on='acc_id').fillna(0)
        features.append(tmp.iloc[:, [0, -4, -3, -2, -1]])

    # for buy
    for i in prop_list:
        aa = pd.DataFrame(
            {'acc_id': total.drop_duplicates('acc_id')['acc_id']})
        trade_df2 = trade_df[trade_df['item_type'] == i]
        df = pd.pivot_table(trade_df2, index='target_acc_id', columns='day',
                            values='source_acc_id', aggfunc='count', fill_value=0).reset_index()
        df['{0}_buy_1weak'.format(i)] = df.iloc[:, 1:8].mean(axis=1)
        df['{0}_buy_2weak'.format(i)] = df.iloc[:, 8:15].mean(axis=1)
        df['{0}_buy_3weak'.format(i)] = df.iloc[:, 15:22].mean(axis=1)
        df['{0}_buy_4weak'.format(i)] = df.iloc[:, 22:29].mean(axis=1)
        df = df.rename(columns={'target_acc_id': 'acc_id'})
        tmp = pd.merge(aa, df, how='left', on='acc_id').fillna(0)
        features.append(tmp.iloc[:, [0, -4, -3, -2, -1]])

    # # for sell MACD
    # for i in prop_list:
    #     aa = pd.DataFrame(
    #         {'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    #     trade_df2 = trade_df[trade_df['item_type'] == i]
    #     df = pd.pivot_table(trade_df2, index='source_acc_id', columns='day',
    #                         values='target_acc_id', aggfunc='count', fill_value=0).reset_index()
    #     df['{0}_sell_1weak_ma'.format(i)] = df.iloc[:, 1:8].mean(axis=1)
    #     df['{0}_sell_2weak_ma'.format(i)] = df.iloc[:, 8:15].mean(axis=1)
    #     df['{0}_sell_3weak_ma'.format(i)] = df.iloc[:, 15:22].mean(axis=1)
    #     df['{0}_sell_4weak_ma'.format(i)] = df.iloc[:, 22:29].mean(axis=1)
    #     df = df.rename(columns={'source_acc_id': 'acc_id'})
    #     tmp = pd.merge(aa, df, how='left', on='acc_id').fillna(
    #         0).set_index('acc_id')
    #     tmp2 = tmp.iloc[:, [-4, -3, -2, -1]].T
    #     tmp2 = tmp2.rolling(window=2, min_periods=1).mean().T.reset_index()
    #     features.append(tmp2.iloc[:, [0, 1, 2, 3, 4]])
    #
    # # for buy MACD
    # for i in prop_list:
    #     aa = pd.DataFrame(
    #         {'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    #     trade_df2 = trade_df[trade_df['item_type'] == i]
    #     df = pd.pivot_table(trade_df2, index='target_acc_id', columns='day',
    #                         values='source_acc_id', aggfunc='count', fill_value=0).reset_index()
    #     df['{0}_buy_1weak_ma'.format(i)] = df.iloc[:, 1:8].mean(axis=1)
    #     df['{0}_buy_2weak_ma'.format(i)] = df.iloc[:, 8:15].mean(axis=1)
    #     df['{0}_buy_3weak_ma'.format(i)] = df.iloc[:, 15:22].mean(axis=1)
    #     df['{0}_buy_4weak_ma'.format(i)] = df.iloc[:, 22:29].mean(axis=1)
    #     df = df.rename(columns={'target_acc_id': 'acc_id'})
    #     tmp = pd.merge(aa, df, how='left', on='acc_id').fillna(
    #         0).set_index('acc_id')
    #     tmp2 = tmp.iloc[:, [-4, -3, -2, -1]].T
    #     tmp2 = tmp2.rolling(window=2, min_periods=1).mean().T.reset_index()
    #     features.append(tmp2.iloc[:, [0, 1, 2, 3, 4]])

    # 3 - 3 해당 주차에 총 거래 횟수
    # print('[ 3/ 12 ] total trade amount')
    # 3-1 prepare
    id_sell_item_day = pd.pivot_table(trade_df, index='source_acc_id', columns='day',
                                      values='target_acc_id', aggfunc='count', fill_value=0).reset_index()

    id_buy_item_day = pd.pivot_table(trade_df, index='target_acc_id', columns='day',
                                     values='source_acc_id', aggfunc='count', fill_value=0).reset_index()

    # 3 - 4 유저 아이디 별 아이템 판 횟수
    # print('[ 4 / 12 ] number of times selling the items by acc_id  ')
    aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    id_sell_item_day['sell_item_1weak'] = id_sell_item_day.iloc[:, 1:9].mean(
        axis=1)
    id_sell_item_day['sell_item_2weak'] = id_sell_item_day.iloc[:, 9:16].mean(
        axis=1)
    id_sell_item_day['sell_item_3weak'] = id_sell_item_day.iloc[:, 16:23].mean(
        axis=1)
    id_sell_item_day['sell_item_4weak'] = id_sell_item_day.iloc[:, 23:30].mean(
        axis=1)
    id_sell_item_day = id_sell_item_day.rename(
        columns={'source_acc_id': 'acc_id'})
    b = pd.merge(aa, id_sell_item_day, how='left', on='acc_id')
    b = b.fillna(0)
    features.append(b.iloc[:, [0, -4, -3, -2, -1]])

    # 3 - 5 유저 아이디별 아이템 산 횟수
    # print('[ 5 / 12 ] number of times purchase the items by acc_id  ')
    aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    id_buy_item_day['buy_item_1weak'] = id_buy_item_day.iloc[:,
                                        1:9].mean(axis=1)
    id_buy_item_day['buy_item_2weak'] = id_buy_item_day.iloc[:,
                                        9:16].mean(axis=1)
    id_buy_item_day['buy_item_3weak'] = id_buy_item_day.iloc[:,
                                        16:23].mean(axis=1)
    id_buy_item_day['buy_item_4weak'] = id_buy_item_day.iloc[:,
                                        23:30].mean(axis=1)
    id_buy_item_day = id_buy_item_day.rename(
        columns={'target_acc_id': 'acc_id'})
    b = pd.merge(aa, id_buy_item_day, how='left', on='acc_id')
    b = b.fillna(0)
    features.append(b.iloc[:, [0, -4, -3, -2, -1]])

    # 3-6 해당 주차에 총 판매 횟수에 대한 MACD
    # print('[ 6 / 12 ] number of times purchase MACD ')
    # aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    # id_sell_item_day['sell_item_1weak_ma'] = id_sell_item_day.iloc[:, 1:9].mean(
    #     axis=1)
    # id_sell_item_day['sell_item_2weak_ma'] = id_sell_item_day.iloc[:, 9:16].mean(
    #     axis=1)
    # id_sell_item_day['sell_item_3weak_ma'] = id_sell_item_day.iloc[:, 16:23].mean(
    #     axis=1)
    # id_sell_item_day['sell_item_4weak_ma'] = id_sell_item_day.iloc[:, 23:30].mean(
    #     axis=1)
    # id_sell_item_day = id_sell_item_day.rename(
    #     columns={'source_acc_id': 'acc_id'})
    # b = pd.merge(aa, id_sell_item_day, how='left', on='acc_id')
    # b = b.fillna(0)
    # b = b.set_index('acc_id')
    # a = b.iloc[:, [-4, -3, -2, -1]].T
    # aa = a.rolling(window=2, min_periods=1).mean()
    # aa = aa.T
    # aa = aa.reset_index()
    # features.append(aa.iloc[:, [0, 1, 2, 3, 4]])

    # 3-7 해당 주차에 총 구매 횟수에 대한 MACD
    # aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    # id_buy_item_day['buy_item_1weak_ma'] = id_buy_item_day.iloc[:, 1:9].mean(
    #     axis=1)
    # id_buy_item_day['buy_item_2weak_ma'] = id_buy_item_day.iloc[:, 9:16].mean(
    #     axis=1)
    # id_buy_item_day['buy_item_3weak_ma'] = id_buy_item_day.iloc[:, 16:23].mean(
    #     axis=1)
    # id_buy_item_day['buy_item_4weak_ma'] = id_buy_item_day.iloc[:, 23:30].mean(
    #     axis=1)
    # id_buy_item_day = id_buy_item_day.rename(
    #     columns={'target_acc_id': 'acc_id'})
    # b = pd.merge(aa, id_buy_item_day, how='left', on='acc_id')
    # b = b.fillna(0)
    # b = b.set_index('acc_id')
    # a = b.iloc[:, [-4, -3, -2, -1]].T
    # aa = a.rolling(window=2, min_periods=1).mean()
    # aa = aa.T
    # aa = aa.reset_index()
    # features.append(aa.iloc[:, [0, 1, 2, 3, 4]])

    # 3- 8 주당 아이템 거래(판매,구매) 수량
    aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})

    # 3-9 주당 아이템 판매 수량
    df = pd.pivot_table(trade_df, index='source_acc_id', columns='day', values='item_amount', aggfunc='sum',
                        fill_value=0).reset_index()
    df['sell_amount_1weak'] = df.iloc[:, 1:9].mean(axis=1)
    df['sell_amount_2weak'] = df.iloc[:, 9:16].mean(axis=1)
    df['sell_amount_3weak'] = df.iloc[:, 16:23].mean(axis=1)
    df['sell_amount_4weak'] = df.iloc[:, 23:30].mean(axis=1)
    df = df.rename(columns={'source_acc_id': 'acc_id'})
    b = pd.merge(aa, df, how='left', on='acc_id')
    b = b.fillna(0)
    features.append(b.iloc[:, [0, -4, -3, -2, -1]])

    # 3 -10 주당 아이템 구매 수량
    # print('[ 10/ 12 ] the number of weekly buying')
    df = pd.pivot_table(trade_df, index='target_acc_id', columns='day', values='item_amount', aggfunc='sum',
                        fill_value=0).reset_index()
    df['buy_amount_1weak'] = df.iloc[:, 1:9].mean(axis=1)
    df['buy_amount_2weak'] = df.iloc[:, 9:16].mean(axis=1)
    df['buy_amount_3weak'] = df.iloc[:, 16:23].mean(axis=1)
    df['buy_amount_4weak'] = df.iloc[:, 23:30].mean(axis=1)
    df = df.rename(columns={'target_acc_id': 'acc_id'})
    b = pd.merge(aa, df, how='left', on='acc_id')
    b = b.fillna(0)
    features.append(b.iloc[:, [0, -4, -3, -2, -1]])

    # 3-11 주당 아이템 판매 이동평균
    # print('[ 11 / 12 ] the number of selling MACD')
    # df = pd.pivot_table(trade_df, index='source_acc_id', columns='day', values='item_amount', aggfunc='sum',
    #                     fill_value=0).reset_index()
    # df['sell_amount_1weak_ma'] = df.iloc[:, 1:9].mean(axis=1)
    # df['sell_amount_2weak_ma'] = df.iloc[:, 9:16].mean(axis=1)
    # df['sell_amount_3weak_ma'] = df.iloc[:, 16:23].mean(axis=1)
    # df['sell_amount_4weak_ma'] = df.iloc[:, 23:30].mean(axis=1)
    # df = df.rename(columns={'source_acc_id': 'acc_id'})
    # b = pd.merge(aa, df, how='left', on='acc_id')
    # b = b.fillna(0)
    # b = b.set_index('acc_id')
    # a = b.iloc[:, [-4, -3, -2, -1]].T
    # aa = a.rolling(window=2, min_periods=1).mean()
    # aa = aa.T
    # aa = aa.reset_index()
    # features.append(aa.iloc[:, [0, 1, 2, 3, 4]])

    # 3-12주당 아이템 구매 이동평균
    # print('[ 12 / 12 ] the number of buying MACD')
    # df = pd.pivot_table(trade_df, index='target_acc_id', columns='day', values='item_amount', aggfunc='sum',
    #                     fill_value=0).reset_index()
    # df['buy_amount_1weak_ma'] = df.iloc[:, 1:9].mean(axis=1)
    # df['buy_amount_2weak_ma'] = df.iloc[:, 9:16].mean(axis=1)
    # df['buy_amount_3weak_ma'] = df.iloc[:, 16:23].mean(axis=1)
    # df['buy_amount_4weak_ma'] = df.iloc[:, 23:30].mean(axis=1)
    # df = df.rename(columns={'target_acc_id': 'acc_id'})
    # b = pd.merge(aa, df, how='left', on='acc_id')
    # b = b.fillna(0)
    # b = b.set_index('acc_id')
    # a = b.iloc[:, [-4, -3, -2, -1]].T
    # aa = a.rolling(window=2, min_periods=1).mean()
    # aa = aa.T
    # aa = aa.reset_index()
    # features.append(aa.iloc[:, [0, 1, 2, 3, 4]])
    #
    trade_df = pd.DataFrame({'acc_id': total.acc_id.unique()})

    for i in features:
        trade_df = pd.merge(trade_df, i, how='left', on='acc_id')

    print('--- Trade_df Process Done ---')

    return trade_df


def make_payment_df(total, MODE):
    '''
    output = trade_df
    '''
    features = []

    pm = pd.read_csv('./raw/' + MODE + '_payment.csv')

    # 1 유저별 과금 횟수(28일동안)
    g_payment_day = pm.pivot_table(index='acc_id', aggfunc='count', values='day')
    g_payment_day = g_payment_day.reset_index()
    g_payment_day.columns = ['acc_id', 'pay_day_num']

    # 2 유저별 과금 최소금액
    g_payment_min = pm.pivot_table(
        index='acc_id', aggfunc='min', values='amount_spent')
    g_payment_min = g_payment_min.reset_index()
    g_payment_min.columns = ['acc_id', 'min_spent']

    # 3 유저별 과금 최대금액
    g_payment_max = pm.pivot_table(
        index='acc_id', aggfunc='max', values='amount_spent')
    g_payment_max = g_payment_max.reset_index()
    g_payment_max.columns = ['acc_id', 'max_spent']

    # 4 유저별 과금 합
    g_payment_sum = pm.pivot_table(
        index='acc_id', aggfunc='sum', values='amount_spent')
    g_payment_sum = g_payment_sum.reset_index()
    g_payment_sum.columns = ['acc_id', 'sum_spent']

    aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    b = pd.merge(aa, g_payment_day, how='left', on='acc_id')
    b2 = pd.merge(b, g_payment_min, how='left', on='acc_id')
    b3 = pd.merge(b2, g_payment_max, how='left', on='acc_id')
    b4 = pd.merge(b3, g_payment_sum, how='left', on='acc_id')
    b4 = b4.fillna(0)
    features.append(b4)

    # 5 주별 평균 결제 금액
    df = pd.pivot_table(pm, index='acc_id', columns='day', values='amount_spent', aggfunc=np.sum,
                        fill_value=0).reset_index()
    df['amount_spent_1weak'] = df.iloc[:, 1:8].mean(axis=1)
    df['amount_spent_2weak'] = df.iloc[:, 8:15].mean(axis=1)
    df['amount_spent_3weak'] = df.iloc[:, 15:22].mean(axis=1)
    df['amount_spent_4weak'] = df.iloc[:, 22:29].mean(axis=1)

    aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    b = pd.merge(aa, df, how='left', on='acc_id')
    b = b.fillna(0)
    features.append(b.iloc[:, [0, -4, -3, -2, -1]])

    # 6 주별 평균 결제 금액 이동평균
    # df = pd.pivot_table(pm, index='acc_id', columns='day', values='amount_spent', aggfunc=np.sum,
    #                     fill_value=0).reset_index()
    # df['amount_spent_1weak_ma'] = df.iloc[:, 1:8].mean(axis=1)
    # df['amount_spent_2weak_ma'] = df.iloc[:, 8:15].mean(axis=1)
    # df['amount_spent_3weak_ma'] = df.iloc[:, 15:22].mean(axis=1)
    # df['amount_spent_4weak_ma'] = df.iloc[:, 22:29].mean(axis=1)
    #
    # aa = pd.DataFrame({'acc_id': total.drop_duplicates('acc_id')['acc_id']})
    # b = pd.merge(aa, df, how='left', on='acc_id')
    # b = b.fillna(0)
    # b = b.set_index('acc_id')
    # a = b.iloc[:, [-4, -3, -2, -1]].T
    # aa = a.rolling(window=2, min_periods=1).mean()
    # aa = aa.T
    # aa = aa.reset_index()
    # features.append(aa.iloc[:, [0, 1, 2, 3, 4]])

    payment_df = pd.DataFrame({'acc_id': total.acc_id.unique()})

    for i in features:
        payment_df = pd.merge(payment_df, i, how='left', on='acc_id')

    print('--- Payment_df Process Done ---')

    return payment_df
