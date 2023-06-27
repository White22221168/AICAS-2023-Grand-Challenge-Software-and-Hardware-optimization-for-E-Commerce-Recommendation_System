import sys
import pandas as pd
import numpy as np
import datetime
import sys
# import time
import xgboost as xgb
# import pickle
# from sklearn.model_selection import KFold, train_test_split, GridSearchCV
# from sklearn.metrics import confusion_matrix, mean_squared_error
# from sklearn.datasets import load_iris, load_digits, load_boston
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import multiprocessing
from multiprocessing import Process, sharedctypes
import os
# import pickle
# import gc
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
import time
# import polars as pl
FEATURE_EXTRACTION_SLOT=10

def read_user_table(file):
    # 读取用户行为数据
    user_table = pd.read_csv(file, sep="\t",usecols=[0,1,2,4,5],header=None,names=["user_id","item_id","behavior_type","item_category","time"],dtype={'behavior_type':np.int8,"item_category":np.int16},nrows=100000)
    item_table = pd.read_csv('/aicas/project/tcdata/part2_item.txt',sep="\t",usecols=[0,1],header=None,names=["item_id","item_category"])
    user_table = user_table[user_table.item_id.isin(list(item_table.item_id))]
    user_table['days'] = user_table['time'].map(lambda x:x.split(' ')[0])
    user_table = user_table[(user_table['days'] != '2014-12-12') & (user_table['days'] != '2014-12-11')]
    user_table['hours'] = user_table['time'].map(lambda x:x.split(' ')[1])
    user_table = user_table.drop(columns=['time'],axis=1)
    user_table["daystime"] = pd.to_datetime(user_table["days"],format="%Y-%m-%d")
    user_table = user_table.drop(columns=["days"],axis=1)
    return user_table

# file_list = ['/aicas/project/tcdata/round2_user_0.txt', '/aicas/project/tcdata/round2_user_1.txt', '/aicas/project/tcdata/round2_user_2.txt', '/aicas/project/tcdata/round2_user_3.txt', '/aicas/project/tcdata/round2_user_4.txt']
# with Pool(processes=len(file_list)) as pool:
#     # 并行读取用户数据并进行处理
#     user_tables = pool.map(read_user_table, file_list)
# user_table = pd.concat(user_tables, ignore_index=True)
# print(len(user_table))

def test_1days(user_table,FEATURE_EXTRACTION_SLOT):
    print("开始读取1dayS")
    
        
    def get_label_testset(train_user,LabelDay,return_dict):
        time1 = time.time()
        # 测试集选为上一天所有的交互数据
        data_test = train_user[(train_user['daystime'] == LabelDay)]#&((train_user.behavior_type==3)|(train_user.behavior_type==2))
        data_test = data_test.drop_duplicates(['user_id', 'item_id'])
        # data_test[['user_id', 'item_id','item_category']].to_csv("./mid_data/get_label_testset1.csv")
        return_dict[0] = data_test[['user_id', 'item_id','item_category']]
        # return data_test[['user_id', 'item_id','item_category']]
        print('get_label_testset:'+str(time.time()-time1))

    def p_crosstab1(a,share,i):
        share[i] = pd.crosstab(a.item_category,a.behavior_type)
        
    def item_category_feture(data,end_time,beforeoneday,return_dict):
        time1 = time.time()
        manager = multiprocessing.Manager()
        share = manager.dict()
        # item_count = pd.crosstab(data.item_category,data.behavior_type)
        p1 = multiprocessing.Process(target=p_crosstab1,args=(data,share,0))
        p1.start()
        item_count_before5=None
        if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
            p2 = multiprocessing.Process(target=p_crosstab1,args=(beforefiveday,share,1))
            p2.start()
            # item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
        else:
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
            p2 = multiprocessing.Process(target=p_crosstab1,args=(beforefiveday,share,1))
            p2.start()
            # item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
        item_count_before_3=None
        if (((end_time-datetime.timedelta(days=3))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=3))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
            p3 = multiprocessing.Process(target=p_crosstab1,args=(beforethreeday,share,2))
            p3.start()
            # item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
            p3 = multiprocessing.Process(target=p_crosstab1,args=(beforethreeday,share,2))
            p3.start()
            # item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)

        item_count_before_2=None
        if (((end_time-datetime.timedelta(days=2))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=2))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2+2)]
            p4 = multiprocessing.Process(target=p_crosstab1,args=(beforethreeday,share,3))
            p4.start()
            # item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2)]
            p4 = multiprocessing.Process(target=p_crosstab1,args=(beforethreeday,share,3))
            p4.start()
            # item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
            
        # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
        beforeonedayitem_count = pd.crosstab(beforeoneday.item_category,beforeoneday.behavior_type)
        

        # buyRate_2 = pd.DataFrame()
        # buyRate_2['click'] = item_count_before5[1]/item_count_before5[4]
        # buyRate_2['skim'] = item_count_before5[2]/item_count_before5[4]
        # buyRate_2['collect'] = item_count_before5[3]/item_count_before5[4]
        # buyRate_2.index = item_count_before5.index

        # buyRate_3 = pd.DataFrame()
        # buyRate_3['click'] = item_count_before_3[1]/item_count_before_3[4]
        # buyRate_3['skim'] = item_count_before_3[2]/item_count_before_3[4]
        # buyRate_3['collect'] = item_count_before_3[3]/item_count_before_3[4]
        # buyRate_3.index = item_count_before_3.index

        # buyRate_5 = pd.DataFrame()
        # buyRate_5['click'] = item_count_before5[1]/item_count_before5[4]
        # buyRate_5['skim'] = item_count_before5[2]/item_count_before5[4]
        # buyRate_5['collect'] = item_count_before5[3]/item_count_before5[4]
        # buyRate_5.index = item_count_before5.index
        # buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
        # buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
        # buyRate_5 = buyRate_5.replace([np.inf, -np.inf], 0)
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        item_count = share[0]
        item_count_before5 = share[1]
        item_count_before_3 = share[2]
        item_count_before_2 = share[3]
        countAverage = item_count/FEATURE_EXTRACTION_SLOT
        buyRate = pd.DataFrame()
        buyRate['click'] = item_count[1]/item_count[4]
        buyRate['skim'] = item_count[2]/item_count[4]
        buyRate['collect'] = item_count[3]/item_count[4]
        buyRate.index = item_count.index
        buyRate = buyRate.replace([np.inf, -np.inf], 0)
        item_category_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
        item_category_feture = pd.merge(item_category_feture,countAverage,how='left',right_index=True,left_index=True)
        item_category_feture = pd.merge(item_category_feture,buyRate,how='left',right_index=True,left_index=True)
        item_category_feture = pd.merge(item_category_feture,item_count_before5,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
        item_category_feture = pd.merge(item_category_feture,item_count_before_3,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
        item_category_feture = pd.merge(item_category_feture,item_count_before_2,how='left',right_index=True,left_index=True,suffixes=('_left3', '_right3'))
        # item_category_feture = pd.merge(item_category_feture,buyRate_2,how='left',right_index=True,left_index=True)
        # item_category_feture = pd.merge(item_category_feture,buyRate_3,how='left',right_index=True,left_index=True)
        # item_category_feture = pd.merge(item_category_feture,buyRate_3,how='left',right_index=True,left_index=True)
        item_category_feture.fillna(0,inplace=True)
        item_category_feture = item_category_feture.reset_index()
        # item_category_feture.to_csv("./mid_data/item_category_feture1.csv")
        return_dict[7] = item_category_feture
        print('item_category_feture:'+str(time.time()-time1))


    def p_crosstab2(a,share,i):
        time1 = time.time()
        share[i] = a.groupby(['item_id', 'behavior_type'])['behavior_type'].count().unstack().fillna(0)
        print(f"{i} : {time.time()-time1}")
    def p_crosstab22(a,share,i):
        time1 = time.time()
        # a = a[['item_id', 'behavior_type', 'user_id']]
        share[i] = a.groupby(['item_id', 'behavior_type'])['user_id'].nunique().unstack()
        print(f"{i} : {time.time()-time1}")

    def item_id_feture(data,end_time,beforeoneday,return_dict):   
        time1 = time.time()
        manager = multiprocessing.Manager()
        share = manager.dict()
        p1 = multiprocessing.Process(target=p_crosstab2,args=(data,share,0))
        p1.start()
        item_count_before5=None
        if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
            p2 = multiprocessing.Process(target=p_crosstab2,args=(beforefiveday,share,1))
            p2.start()
        else:
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
            p2 = multiprocessing.Process(target=p_crosstab2,args=(beforefiveday,share,1))
            p2.start()

        item_count_before_3=None
        if (((end_time-datetime.timedelta(days=3))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=3))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
            p3 = multiprocessing.Process(target=p_crosstab2,args=(beforethreeday,share,2))
            p3.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
            p3 = multiprocessing.Process(target=p_crosstab2,args=(beforethreeday,share,2))
            p3.start()
        item_count_before_2=None
        if (((end_time-datetime.timedelta(days=2))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=2))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2+2)]
            p4 = multiprocessing.Process(target=p_crosstab2,args=(beforethreeday,share,3))
            p4.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2)]
            p4 = multiprocessing.Process(target=p_crosstab2,args=(beforethreeday,share,3))
            p4.start()
        
        # item_count_unq = data.groupby(by = ['item_id','behavior_type']).agg({"user_id":lambda x:x.nunique()})
        # item_count_unq = item_count_unq.unstack()
        # item_count_unq = item_count_unq.droplevel(level=0, axis=1)
        # item_count_unq = item_count_unq.rename_axis(columns=None)
        p5 = multiprocessing.Process(target=p_crosstab22,args=(data,share,4))
        p5.start()
        # item_count_unq.to_csv("./1.csv")
        # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
        beforeonedayitem_count = pd.crosstab(beforeoneday.item_id,beforeoneday.behavior_type)
        

        # buyRate_2 = pd.DataFrame()
        # buyRate_2['click'] = item_count_before5[1]/item_count_before5[4]
        # buyRate_2['skim'] = item_count_before5[2]/item_count_before5[4]
        # buyRate_2['collect'] = item_count_before5[3]/item_count_before5[4]
        # buyRate_2.index = item_count_before5.index

        # buyRate_3 = pd.DataFrame()
        # buyRate_3['click'] = item_count_before_3[1]/item_count_before_3[4]
        # buyRate_3['skim'] = item_count_before_3[2]/item_count_before_3[4]
        # buyRate_3['collect'] = item_count_before_3[3]/item_count_before_3[4]
        # buyRate_3.index = item_count_before_3.index

        # buyRate_5 = pd.DataFrame()
        # buyRate_5['click'] = item_count_before5[1]/item_count_before5[4]
        # buyRate_5['skim'] = item_count_before5[2]/item_count_before5[4]
        # buyRate_5['collect'] = item_count_before5[3]/item_count_before5[4]
        # buyRate_5.index = item_count_before5.index

        # buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
        # buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
        # buyRate_5 = buyRate_5.replace([np.inf, -np.inf], 0)
        print("主进程：{}".format(time.time()-time1))
        p1.join()
        
        item_count = share[0]
        
        countAverage = item_count/FEATURE_EXTRACTION_SLOT
        buyRate = pd.DataFrame()
        buyRate['click'] = item_count[1]/item_count[4]
        buyRate['skim'] = item_count[2]/item_count[4]
        buyRate['collect'] = item_count[3]/item_count[4]
        buyRate.index = item_count.index
        buyRate = buyRate.replace([np.inf, -np.inf], 0)
        item_id_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
        item_id_feture = pd.merge(item_id_feture,countAverage,how='left',right_index=True,left_index=True)
        item_id_feture = pd.merge(item_id_feture,buyRate,how='left',right_index=True,left_index=True)
        p5.join()
        item_count_unq = share[4]
        item_id_feture = pd.merge(item_id_feture,item_count_unq,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
        p2.join()
        p3.join()
        p4.join()
        item_count_before5 = share[1]
        item_count_before_3 = share[2]
        item_count_before_2 = share[3]
        item_id_feture = pd.merge(item_id_feture,item_count_before5,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
        item_id_feture = pd.merge(item_id_feture,item_count_before_3,how='left',right_index=True,left_index=True,suffixes=('_left3', '_right3'))
        item_id_feture = pd.merge(item_id_feture,item_count_before_2,how='left',right_index=True,left_index=True,suffixes=('_left4', '_right4'))
        # item_id_feture = pd.merge(item_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
        # item_id_feture = pd.merge(item_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
        # item_id_feture = pd.merge(item_id_feture,buyRate_5,how='left',right_index=True,left_index=True)
        item_id_feture.fillna(0,inplace=True)
        item_id_feture = item_id_feture.reset_index()
        # item_id_feture.to_csv("./mid_data/item_id_feture1.csv")
        return_dict[6] = item_id_feture
        print('item_id_feture:'+str(time.time()-time1))

    def p_crosstab3(a,share,i):
        share[i] = pd.crosstab(a.user_id,a.behavior_type)

    def user_id_feture(data,end_time,beforeoneday,return_dict):   
        time1 = time.time()
        manager = multiprocessing.Manager()
        share = manager.dict()
        # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
        p1 = multiprocessing.Process(target=p_crosstab3,args=(data,share,0))
        p1.start()
        item_count_before5=None
        if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
            p2 = multiprocessing.Process(target=p_crosstab3,args=(beforefiveday,share,1))
            p2.start()
        else:
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
            p2 = multiprocessing.Process(target=p_crosstab3,args=(beforefiveday,share,1))
            p2.start()

        item_count_before_3=None
        if (((end_time-datetime.timedelta(days=3))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=3))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
            p3 = multiprocessing.Process(target=p_crosstab3,args=(beforethreeday,share,2))
            p3.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
            p3 = multiprocessing.Process(target=p_crosstab3,args=(beforethreeday,share,2))
            p3.start()
        item_count_before_2=None
        if (((end_time-datetime.timedelta(days=2))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=2))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2+2)]
            p4 = multiprocessing.Process(target=p_crosstab3,args=(beforethreeday,share,3))
            p4.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2)]
            p4 = multiprocessing.Process(target=p_crosstab3,args=(beforethreeday,share,3))
            p4.start()
            
        # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
        beforeonedayuser_count = pd.crosstab(beforeoneday.user_id,beforeoneday.behavior_type)
        

        # buyRate_2 = pd.DataFrame()
        # buyRate_2['click'] = user_count_before5[1]/user_count_before5[4]
        # buyRate_2['skim'] = user_count_before5[2]/user_count_before5[4]
        # buyRate_2['collect'] = user_count_before5[3]/user_count_before5[4]
        # buyRate_2.index = user_count_before5.index

        # buyRate_3 = pd.DataFrame()
        # buyRate_3['click'] = user_count_before_3[1]/user_count_before_3[4]
        # buyRate_3['skim'] = user_count_before_3[2]/user_count_before_3[4]
        # buyRate_3['collect'] = user_count_before_3[3]/user_count_before_3[4]
        # buyRate_3.index = user_count_before_3.index

        # buyRate_5 = pd.DataFrame()
        # buyRate_5['click'] = user_count_before5[1]/user_count_before5[4]
        # buyRate_5['skim'] = user_count_before5[2]/user_count_before5[4]
        # buyRate_5['collect'] = user_count_before5[3]/user_count_before5[4]
        # buyRate_5.index = user_count_before5.index
        # buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
        # buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
        # buyRate_5 = buyRate_5.replace([np.inf, -np.inf], 0)
        long_online = pd.pivot_table(beforeoneday,index=['user_id'],values=['hours'],aggfunc=[np.min,np.max,np.ptp])
        long_online = long_online.droplevel(level=0, axis=1)
        long_online = long_online.rename_axis(columns=None)

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        user_count = share[0]
        user_count_before5 = share[1]
        user_count_before_3 = share[2]
        user_count_before_2 = share[3]
        countAverage = user_count/FEATURE_EXTRACTION_SLOT
        buyRate = pd.DataFrame()
        buyRate['click'] = user_count[1]/user_count[4]
        buyRate['skim'] = user_count[2]/user_count[4]
        buyRate['collect'] = user_count[3]/user_count[4]
        buyRate.index = user_count.index
        buyRate = buyRate.replace([np.inf, -np.inf], 0)
        user_id_feture = pd.merge(user_count,beforeonedayuser_count,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
        user_id_feture = pd.merge(user_id_feture,countAverage,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
        user_id_feture = pd.merge(user_id_feture,buyRate,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
        user_id_feture = pd.merge(user_id_feture,user_count_before5,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
        user_id_feture = pd.merge(user_id_feture,user_count_before_3,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
        user_id_feture = pd.merge(user_id_feture,user_count_before_2,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
        user_id_feture = pd.merge(user_id_feture, long_online, how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
        # user_id_feture = pd.merge(user_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
        # user_id_feture = pd.merge(user_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
        # user_id_feture = pd.merge(user_id_feture,buyRate_5,how='left',right_index=True,left_index=True)
        user_id_feture.fillna(0,inplace=True)
        user_id_feture = user_id_feture.reset_index()
        # user_id_feture.to_csv("./mid_data/user_id_feture1.csv")
        return_dict[5] = user_id_feture
        print('user_id_feture:'+str(time.time()-time1))


    def p_crosstab4(a,share,i):
        time1 = time.time()
        share[i] = a.groupby(['user_id', 'item_id', 'behavior_type'])['behavior_type'].count().unstack().fillna(0)
        print(f"user_item_feture_p{i+1}:"+str(time.time()-time1))

    def user_item_feture(data,end_time,beforeoneday,return_dict):   
        time1 = time.time()
        manager = multiprocessing.Manager()
        share = manager.dict()
        # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
        p1 = multiprocessing.Process(target=p_crosstab4,args=(data,share,0))
        p1.start()

        item_count_before5=None
        if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
            p2 = multiprocessing.Process(target=p_crosstab4,args=(beforefiveday,share,1))
            p2.start()
        else:
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
            p2 = multiprocessing.Process(target=p_crosstab4,args=(beforefiveday,share,1))
            p2.start()

        item_count_before_3=None
        if (((end_time-datetime.timedelta(days=3))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=3))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
            p3 = multiprocessing.Process(target=p_crosstab4,args=(beforethreeday,share,2))
            p3.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
            p3 = multiprocessing.Process(target=p_crosstab4,args=(beforethreeday,share,2))
            p3.start()
        item_count_before_2=None
        if (((end_time-datetime.timedelta(days=2))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=2))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2+2)]
            p4 = multiprocessing.Process(target=p_crosstab4,args=(beforethreeday,share,3))
            p4.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2)]
            p4 = multiprocessing.Process(target=p_crosstab4,args=(beforethreeday,share,3))
            p4.start()
            
        beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_id],beforeoneday.behavior_type)
        
    #    _live = user_item_long_touch(data) 
        max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['hours'],aggfunc=[np.min,np.max])
        max_touchtime = max_touchtime.droplevel(level=0, axis=1)
        max_touchtime = max_touchtime.rename_axis(columns=None)
        max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['behavior_type'],aggfunc=np.max)
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        user_item_count = share[0]
        user_item_count_5 = share[1]
        user_item_count_3 = share[2]
        user_item_count_2 = share[3]
        user_item_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)

        user_item_feture = pd.merge(user_item_feture,max_touchtime,how='left',right_index=True,left_index=True)
        user_item_feture = pd.merge(user_item_feture,max_touchtype,how='left',right_index=True,left_index=True)
    #    user_item_feture = pd.merge(user_item_feture,_live,how='left',right_index=True,left_index=True)

        user_item_feture = pd.merge(user_item_feture,user_item_count_5,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
        user_item_feture = pd.merge(user_item_feture,user_item_count_3,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
        user_item_feture = pd.merge(user_item_feture,user_item_count_2,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
        user_item_feture.fillna(0,inplace=True)
        user_item_feture = user_item_feture.reset_index()
        return_dict[9] = user_item_feture
        print('user_item_feture:'+str(time.time()-time1))

    def p_crosstab5(a,share,i):
        share[i] = a.groupby(['user_id', 'item_category', 'behavior_type'])['behavior_type'].count().unstack().fillna(0)
        
    def user_cate_feture(data,end_time,beforeoneday,return_dict):   
        time1 = time.time()
        manager = multiprocessing.Manager()
        share = manager.dict()
        # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
        p1 = multiprocessing.Process(target=p_crosstab5,args=(data,share,0))
        p1.start()
        item_count_before5=None
        if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
            p2 = multiprocessing.Process(target=p_crosstab5,args=(beforefiveday,share,1))
            p2.start()
        else:
            beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
            p2 = multiprocessing.Process(target=p_crosstab5,args=(beforefiveday,share,1))
            p2.start()

        item_count_before_3=None
        if (((end_time-datetime.timedelta(days=3))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=3))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
            p3 = multiprocessing.Process(target=p_crosstab5,args=(beforethreeday,share,2))
            p3.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
            p3 = multiprocessing.Process(target=p_crosstab5,args=(beforethreeday,share,2))
            p3.start()
        item_count_before_2=None
        if (((end_time-datetime.timedelta(days=2))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=2))>datetime.datetime(2014,12,10,0,0,0))):
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2+2)]
            p4 = multiprocessing.Process(target=p_crosstab5,args=(beforethreeday,share,3))
            p4.start()
        else:
            beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=2)]
            p4 = multiprocessing.Process(target=p_crosstab5,args=(beforethreeday,share,3))
            p4.start()
            
    #    _live = user_cate_long_touch(data)
        beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_category],beforeoneday.behavior_type)
        max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['hours'],aggfunc=[np.min,np.max])
        max_touchtime = max_touchtime.droplevel(level=0, axis=1)
        max_touchtime = max_touchtime.rename_axis(columns=None)
        max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['behavior_type'],aggfunc=np.max)
        # max_touchtype = max_touchtype.droplevel(level=0, axis=1)
        # max_touchtype = max_touchtype.rename_axis(columns=None)
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        user_item_count = share[0]
        user_cate_count_5 = share[1]
        user_cate_count_3 = share[2]
        user_cate_count_2 = share[3]
        user_cate_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)
        user_cate_feture = pd.merge(user_cate_feture,max_touchtime,how='left',right_index=True,left_index=True)
        user_cate_feture = pd.merge(user_cate_feture,max_touchtype,how='left',right_index=True,left_index=True)
    #    user_cate_feture = pd.merge(user_cate_feture,_live,how='left',right_index=True,left_index=True)
        user_cate_feture = pd.merge(user_cate_feture,user_cate_count_5,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
        user_cate_feture = pd.merge(user_cate_feture,user_cate_count_3,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
        user_cate_feture = pd.merge(user_cate_feture,user_cate_count_2,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
        user_cate_feture.fillna(0,inplace=True)
        user_cate_feture = user_cate_feture.reset_index()
        # user_cate_feture.to_csv("./mid_data/user_cate_feture1.csv")
        return_dict[8] = user_cate_feture
        print('user_cate_feture:'+str(time.time()-time1))

    def user_click(beforesomeday,return_dict):#用户在前几天各种操作在各个小时的计数
        time1 = time.time()
        user_act_count = pd.crosstab([beforesomeday.user_id,beforesomeday.behavior_type],beforesomeday.hours,dropna=False)
        user_act_count = user_act_count.unstack(fill_value = 0)
        user_act_count = user_act_count.droplevel(level=0, axis=1)
        user_act_count = user_act_count.rename_axis(columns=None)
        # user_act_count.to_csv("./mid_data/user_act_count1.csv")
        return_dict[1] = user_act_count
        print('user_act_count:'+str(time.time()-time1))


    def user_liveday(train_user_window1,return_dict):#用户各个行为活跃的天数
        time1 = time.time()
        user_live = train_user_window1.groupby(['user_id', 'behavior_type'])['daystime'].nunique().unstack(fill_value=0)
        user_live.columns = user_live.columns.astype(str)
        user_live = user_live.rename_axis(columns=None)
        # user_live.to_csv("./mid_data/user_live1.csv")
        return_dict[4] = user_live
        print('user_live:'+str(time.time()-time1))



    def user_item_click(beforesomeday,return_dict):
        time1 = time.time()
        user_item_act_count = pd.crosstab([beforesomeday.user_id,beforesomeday.item_id,beforesomeday.behavior_type],beforesomeday.hours)
        user_item_act_count = user_item_act_count.unstack(fill_value = 0)
        user_item_act_count = user_item_act_count.droplevel(level=0, axis=1)
        user_item_act_count = user_item_act_count.rename_axis(columns=None)
        # user_item_act_count.to_csv("./mid_data/user_item_act_count1.csv")
        return_dict[2] = user_item_act_count
        print('user_item_act_count:'+str(time.time()-time1))


    def user_cate_click(beforesomeday,return_dict):
        time1 = time.time()
        user_cate_act_count = pd.crosstab([beforesomeday.user_id,beforesomeday.item_category,beforesomeday.behavior_type],beforesomeday.hours)
        user_cate_act_count = user_cate_act_count.unstack(fill_value = 0)
        user_cate_act_count = user_cate_act_count.droplevel(level=0, axis=1)
        user_cate_act_count = user_cate_act_count.rename_axis(columns=None)
        # user_cate_act_count.to_csv("./mid_data/user_cate_act_count1.csv")
        return_dict[3] = user_cate_act_count
        print('user_cate_act_count:'+str(time.time()-time1))


    def user_item_long_touch(train_user_window1):
        _live = train_user_window1.groupby(by = ['user_id','item_id']).agg({"daystime":lambda x:(x.max()-x.min()).days})
        return _live

    def user_cate_long_touch(train_user_window1):
        _live = train_user_window1.groupby(by = ['user_id','item_category']).agg({"daystime":lambda x:(x.max()-x.min()).days})
        return _live
    
    

    def build_test(Data):
        LabelDay=datetime.datetime(2014,12,18,0,0,0)
        # Data = Data[user_table.item_id.isin(list(item_table.item_id))]
        Data["hours"] = Data["hours"].astype(np.int8)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p1 = multiprocessing.Process(target=get_label_testset,args=(Data,LabelDay,return_dict))
        p1.start()
        train_user_window1 =  Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT-1))) & (Data['daystime'] <= LabelDay )]
        beforeoneday = Data[Data['daystime'] == LabelDay]
        # beforetwoday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=2))) & (Data['daystime'] < LabelDay)]
        # beforefiveday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=5))) & (Data['daystime'] < LabelDay)]
        p2 = multiprocessing.Process(target=user_click,args=(beforeoneday,return_dict))
        p2.start()
        p3 = multiprocessing.Process(target=user_item_click,args=(beforeoneday,return_dict))
        p3.start()
        p4 = multiprocessing.Process(target=user_cate_click,args=(beforeoneday,return_dict))
        p4.start()
        # add_user_click_2 = user_click(beforetwoday)
        # add_user_click_5 = user_click(beforefiveday)
        p5 = multiprocessing.Process(target=user_liveday,args=(train_user_window1,return_dict))
        p5.start()
        p6 = multiprocessing.Process(target=user_id_feture,args=(train_user_window1, LabelDay,beforeoneday,return_dict))
        p6.start()
        p7 = multiprocessing.Process(target=item_id_feture,args=(train_user_window1, LabelDay,beforeoneday,return_dict))
        p7.start()
        p8 = multiprocessing.Process(target=item_category_feture,args=(train_user_window1, LabelDay,beforeoneday,return_dict))
        p8.start()
        p9 = multiprocessing.Process(target=user_cate_feture,args=(train_user_window1, LabelDay,beforeoneday,return_dict))
        p9.start()
        p10 = multiprocessing.Process(target=user_item_feture,args=(train_user_window1, LabelDay,beforeoneday,return_dict))
        p10.start()
        
        p1.join()
        test = return_dict[0]
        p2.join()
        add_user_click = return_dict[1]
        p3.join()
        add_user_item_click = return_dict[2]
        p4.join()
        add_user_cate_click = return_dict[3]
        p5.join()
        liveday = return_dict[4]
        p6.join()
        a = return_dict[5]
        p7.join()
        b = return_dict[6]
        p8.join()
        c = return_dict[7]
        p9.join()
        d = return_dict[8]
        p10.join()
        e = return_dict[9]
        
        test = pd.merge(test,a,on=['user_id'],how='left')
        test = pd.merge(test,b,on=['item_id'],how='left')
        test = pd.merge(test,c,on=['item_category'],how='left')
        test = pd.merge(test,d,on=['user_id','item_category'],how='left',suffixes=('_left11', '_right11'))
        test = pd.merge(test,e,on=['user_id','item_id'],how='left',suffixes=('_left22', '_right22'))
        test = pd.merge(test,add_user_click,left_on = ['user_id'],right_index=True,how = 'left' )
        # test = pd.merge(test,add_user_click_2,left_on = ['user_id'],right_index=True,how = 'left' )
        # test = pd.merge(test,add_user_click_5,left_on = ['user_id'],right_index=True,how = 'left' )
        test = pd.merge(test,add_user_item_click,left_on = ['user_id','item_id'],right_index=True,how = 'left' )
        test = pd.merge(test,add_user_cate_click,left_on = ['user_id','item_category'],right_index=True,how = 'left',suffixes=('_left333', '_right333') )
        test = pd.merge(test,liveday,left_on = ['user_id'],right_index=True,how = 'left' )
        test = test.fillna(0)
        # test = reduce_mem_usage(test)
        return test
    
    predicted_list = list()
    # cols  = ['user_id', 'item_id','item_category']+[x for x in range (426)]
    test=build_test(user_table)
    # test.to_csv("test2days.csv",index=False)
    # test.columns = cols
    # test = pd.read_csv("test1.csv")
    test_x = test.drop(['user_id', 'item_id','item_category'], axis=1).values
    print(len(test))
    dtest = xgb.DMatrix(test_x)
    # dtest.save_binary("dtest_2days.buffer")
    print("------start test------")
    bst = xgb.Booster()
    bst.load_model('/data/test/final/aicas_final_白白可爱666/weight/model_new_00_1000.bin')
    predicted_proba = bst.predict(dtest)
    #print(predicted_proba)
    # predicted_proba = model.predict_proba(test_x)
    print("------test finish------")
    predicted_proba = pd.DataFrame(predicted_proba)
    predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    predicted.columns = ['user_id','item_id','prob']
    #print(predicted)
    predicted = predicted.sort_values('prob',  axis=0,ascending=False)
    predicted_list.append(predicted.iloc[:32000,:2])
    # predicted.iloc[:50000,:].to_csv(("./submit11" + ".txt"), header=None, index=False,sep='\t')
    
    print("------start test------")
    bst = xgb.Booster()
    bst.load_model('/data/test/final/aicas_final_白白可爱666/weight/model_new_11_1000.bin')
    predicted_proba = bst.predict(dtest)
    #print(predicted_proba)
    # predicted_proba = model.predict_proba(test_x)
    print("------test finish------")
    predicted_proba = pd.DataFrame(predicted_proba)
    predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    predicted.columns = ['user_id','item_id','prob']
    #print(predicted)
    predicted = predicted.sort_values('prob',  axis=0,ascending=False)
    predicted_list.append(predicted.iloc[:32000,:2])
    # predicted.iloc[:50000,:].to_csv(("./submit11" + ".txt"), header=None, index=False,sep='\t')
    
    
    print("------start test------")
    bst = xgb.Booster()
    bst.load_model('/data/test/final/aicas_final_白白可爱666/weight/model_new_22_1000.bin')
    predicted_proba = bst.predict(dtest)
    #print(predicted_proba)
    # predicted_proba = model.predict_proba(test_x)
    print("------test finish------")
    predicted_proba = pd.DataFrame(predicted_proba)
    predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    predicted.columns = ['user_id','item_id','prob']
    #print(predicted)
    predicted = predicted.sort_values('prob',  axis=0,ascending=False)
    predicted_list.append(predicted.iloc[:32000,:2])
    # predicted.iloc[:50000,:].to_csv(("./submit11" + ".txt"), header=None, index=False,sep='\t')
    
    
    print("------start test------")
    bst = xgb.Booster()
    bst.load_model('/data/test/final/aicas_final_白白可爱666/weight/model_new_33_1000.bin')
    predicted_proba = bst.predict(dtest)
    #print(predicted_proba)
    # predicted_proba = model.predict_proba(test_x)
    print("------test finish------")
    predicted_proba = pd.DataFrame(predicted_proba)
    predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    predicted.columns = ['user_id','item_id','prob']
    #print(predicted)
    predicted = predicted.sort_values('prob',  axis=0,ascending=False)
    predicted_list.append(predicted.iloc[:32000,:2])
    # predicted.iloc[:50000,:].to_csv(("./submit11" + ".txt"), header=None, index=False,sep='\t')
    
    
    
    print("------start test------")
    bst = xgb.Booster()
    bst.load_model('/data/test/final/aicas_final_白白可爱666/weight/model_new_44_1000.bin')
    predicted_proba = bst.predict(dtest)
    #print(predicted_proba)
    # predicted_proba = model.predict_proba(test_x)
    print("------test finish------")
    predicted_proba = pd.DataFrame(predicted_proba)
    predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    predicted.columns = ['user_id','item_id','prob']
    #print(predicted)
    predicted = predicted.sort_values('prob',  axis=0,ascending=False)
    predicted_list.append(predicted.iloc[:32000,:2])
    result = pd.concat(predicted_list)
    result.to_csv(("/data/test/eval_script_v1/eval/submit" + ".txt"), header=None, index=False,sep='\t')