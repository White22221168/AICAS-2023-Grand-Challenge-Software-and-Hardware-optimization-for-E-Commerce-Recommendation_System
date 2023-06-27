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
import pickle
import gc

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in range(len(df.columns)):
        col_type = df.iloc[:,col].dtype
        if col_type != object:
            if str(col_type)[:3] == 'int':
                df.iloc[:,col] = pd.to_numeric(df.iloc[:,col], errors='coerce', downcast='integer') 
            elif str(col_type)[:5] == 'float':
                df.iloc[:,col] = pd.to_numeric(df.iloc[:,col], errors='coerce', downcast='float')
        else:
            df.iloc[:,col] = df.iloc[:,col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

traindata = pd.DataFrame()
FEATURE_EXTRACTION_SLOT = 10
def process():
    print("开始读取")
    # user_table0 = pd.read_csv('../tcdata/round2_user_0.txt',sep="\t",usecols=[1],header=None,names=["item_id"])
    # user_table1 = pd.read_csv('../tcdata/round2_user_1.txt',sep="\t",usecols=[1],header=None,names=["item_id"])
    # user_table2 = pd.read_csv('../tcdata/round2_user_2.txt',sep="\t",usecols=[1],header=None,names=["item_id"])
    # user_table3 = pd.read_csv('../tcdata/round2_user_3.txt',sep="\t",usecols=[1],header=None,names=["item_id"])	
    # user_table4 = pd.read_csv('../tcdata/round2_user_4.txt',sep="\t",usecols=[1],header=None,names=["item_id"])
    # user_table = pd.concat([user_table0,user_table1,user_table2,user_table3,user_table4],ignore_index=True)
    # user_table = user_table.drop_duplicates()
    # user_table.to_csv("./itemid.csv",index=False)
    # user_table1 = user_table.sample(frac=0.2)
    # user_table1.to_csv("./itemid_02.csv",index=False)
    # user_table2 = user_table.sample(frac=0.1)
    # user_table2.to_csv("./itemid_01.csv",index=False)
    # del user_table1
    # del user_table2
    # print(len(user_table))
    user_table0 = pd.read_csv('../tcdata/round2_user_0.txt',sep="\t",usecols=[0,1,2,4,5],header=None,names=["user_id","item_id","behavior_type","item_category","time"],dtype={'behavior_type':np.int8,"item_category":np.int16})
    user_table1 = pd.read_csv('../tcdata/round2_user_1.txt',sep="\t",usecols=[0,1,2,4,5],header=None,names=["user_id","item_id","behavior_type","item_category","time"],dtype={'behavior_type':np.int8,"item_category":np.int16})
    user_table2 = pd.read_csv('../tcdata/round2_user_2.txt',sep="\t",usecols=[0,1,2,4,5],header=None,names=["user_id","item_id","behavior_type","item_category","time"],dtype={'behavior_type':np.int8,"item_category":np.int16})
    user_table3 = pd.read_csv('../tcdata/round2_user_3.txt',sep="\t",usecols=[0,1,2,4,5],header=None,names=["user_id","item_id","behavior_type","item_category","time"],dtype={'behavior_type':np.int8,"item_category":np.int16})	
    user_table4 = pd.read_csv('../tcdata/round2_user_4.txt',sep="\t",usecols=[0,1,2,4,5],header=None,names=["user_id","item_id","behavior_type","item_category","time"],dtype={'behavior_type':np.int8,"item_category":np.int16})
    user_table = pd.concat([user_table0,user_table1,user_table2,user_table3,user_table4],ignore_index=True)
    item_table = pd.read_csv('./itemid_01.csv')
    user_table = user_table[user_table.item_id.isin(list(item_table.item_id))]
    # item_table = pd.read_csv('/tcdata/part2_item.txt',sep="\t",usecols=[0,1],header=None,names=["item_id","item_category"])
    print("读取完成")
    print(len(user_table))
    user_table['days'] = user_table['time'].map(lambda x:x.split(' ')[0])
    user_table['hours'] = user_table['time'].map(lambda x:x.split(' ')[1])
    # user_table['hours'] = user_table['hours'].astype(np.int8)
    print(len(user_table))
    user_table = user_table[user_table['days'] != '2014-12-12']
    print(len(user_table))
    user_table = user_table[user_table['days'] != '2014-12-11']
    print(len(user_table))
    user_table.to_csv("./drop1112_sub_item.csv",index=False)
    # return user_table

def get_train(train_user,end_time):
    # 取出label day 前一天的记录作为打标记录
    data_train = train_user[(train_user['daystime'] == (end_time-datetime.timedelta(days=1)))]#&((train_user.behavior_type==3)|(train_user.behavior_type==2))
    # 训练样本中，删除重复的样本
    data_train = data_train.drop_duplicates(['user_id', 'item_id'])
    data_train_ui = data_train['user_id'] / data_train['item_id']
#    print(len(data_train))

    # 使用label day 的实际购买情况进行打标
    data_label = train_user[train_user['daystime'] == end_time]
    data_label_buy = data_label[data_label['behavior_type'] == 4]
    data_label_buy_ui = data_label_buy['user_id'] / data_label_buy['item_id']

    # 对前一天的交互记录进行打标
    data_train_labeled = data_train_ui.isin(data_label_buy_ui)
    dict = {True: 1, False: 0}
    data_train_labeled = data_train_labeled.map(dict)

    data_train['label'] = data_train_labeled
    return data_train[['user_id', 'item_id','item_category', 'label']]

def get_label_testset(train_user,LabelDay):
    # 测试集选为上一天所有的交互数据
    data_test = train_user[(train_user['daystime'] == LabelDay)]#&((train_user.behavior_type==3)|(train_user.behavior_type==2))
    data_test = data_test.drop_duplicates(['user_id', 'item_id'])
    return data_test[['user_id', 'item_id','item_category']]



def item_category_feture(data,end_time,beforeoneday):
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    item_count = pd.crosstab(data.item_category,data.behavior_type)
    item_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
    item_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)

    item_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
        
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayitem_count = pd.crosstab(beforeoneday.item_category,beforeoneday.behavior_type)
    countAverage = item_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = item_count[1]/item_count[4]
    buyRate['skim'] = item_count[2]/item_count[4]
    buyRate['collect'] = item_count[3]/item_count[4]
    buyRate.index = item_count.index

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


    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    # buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    # buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    item_category_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,countAverage,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,buyRate,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,item_count_before5,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
    item_category_feture = pd.merge(item_category_feture,item_count_before_3,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
    item_category_feture = pd.merge(item_category_feture,item_count_before_2,how='left',right_index=True,left_index=True,suffixes=('_left3', '_right3'))
#    item_category_feture = pd.merge(item_category_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    item_category_feture = pd.merge(item_category_feture,buyRate_3,how='left',right_index=True,left_index=True)
    item_category_feture.fillna(0,inplace=True)
    return item_category_feture

def item_id_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    item_count = pd.crosstab(data.item_id,data.behavior_type)
    item_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        item_count_before5 = pd.crosstab(beforefiveday.item_id,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        item_count_before5 = pd.crosstab(beforefiveday.item_id,beforefiveday.behavior_type)

    item_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)

    item_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
    
    item_count_unq = data.groupby(by = ['item_id','behavior_type']).agg({"user_id":lambda x:x.nunique()})
    item_count_unq = item_count_unq.unstack()
    item_count_unq = item_count_unq.droplevel(level=0, axis=1)
    item_count_unq = item_count_unq.rename_axis(columns=None)
    # item_count_unq.to_csv("./1.csv")
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayitem_count = pd.crosstab(beforeoneday.item_id,beforeoneday.behavior_type)
    countAverage = item_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = item_count[1]/item_count[4]
    buyRate['skim'] = item_count[2]/item_count[4]
    buyRate['collect'] = item_count[3]/item_count[4]
    buyRate.index = item_count.index

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

    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    # buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    # buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    item_id_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,countAverage,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,buyRate,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_unq,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
    item_id_feture = pd.merge(item_id_feture,item_count_before5,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
    item_id_feture = pd.merge(item_id_feture,item_count_before_3,how='left',right_index=True,left_index=True,suffixes=('_left3', '_right3'))
    item_id_feture = pd.merge(item_id_feture,item_count_before_2,how='left',right_index=True,left_index=True,suffixes=('_left4', '_right4'))
#    item_id_feture = pd.merge(item_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    item_id_feture = pd.merge(item_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
    item_id_feture.fillna(0,inplace=True)
    return item_id_feture


def user_id_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_count = pd.crosstab(data.user_id,data.behavior_type)
    user_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        user_count_before5 = pd.crosstab(beforefiveday.user_id,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        user_count_before5 = pd.crosstab(beforefiveday.user_id,beforefiveday.behavior_type)

    user_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        user_count_before_3 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        user_count_before_3 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)

    user_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        user_count_before_2 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        user_count_before_2 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
        
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayuser_count = pd.crosstab(beforeoneday.user_id,beforeoneday.behavior_type)
    countAverage = user_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = user_count[1]/user_count[4]
    buyRate['skim'] = user_count[2]/user_count[4]
    buyRate['collect'] = user_count[3]/user_count[4]
    buyRate.index = user_count.index

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


    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    # buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    # buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)


    long_online = pd.pivot_table(beforeoneday,index=['user_id'],values=['hours'],aggfunc=[np.min,np.max,np.ptp])
    long_online = long_online.droplevel(level=0, axis=1)
    long_online = long_online.rename_axis(columns=None)

    user_id_feture = pd.merge(user_count,beforeonedayuser_count,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
    user_id_feture = pd.merge(user_id_feture,countAverage,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
    user_id_feture = pd.merge(user_id_feture,buyRate,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
    user_id_feture = pd.merge(user_id_feture,user_count_before5,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
    user_id_feture = pd.merge(user_id_feture,user_count_before_3,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
    user_id_feture = pd.merge(user_id_feture,user_count_before_2,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
    # print(user_id_feture.reset_index().index,long_online.index)
    # user_id_feture.reset_index().to_csv("./1.csv")
    # long_online.to_csv("./2.csv")
    user_id_feture = pd.merge(user_id_feture, long_online, how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
#    user_id_feture = pd.merge(user_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    user_id_feture = pd.merge(user_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
    user_id_feture.fillna(0,inplace=True)
    return user_id_feture



def user_item_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_item_count = pd.crosstab([data.user_id,data.item_id],data.behavior_type)
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    user_item_count_5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        user_item_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_id],beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        user_item_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_id],beforefiveday.behavior_type)
    user_item_count_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        user_item_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        user_item_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)

    user_item_count_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        user_item_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        user_item_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
        
    beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_id],beforeoneday.behavior_type)
    
#    _live = user_item_long_touch(data)
    
    
    max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['hours'],aggfunc=[np.min,np.max])
    max_touchtime = max_touchtime.droplevel(level=0, axis=1)
    max_touchtime = max_touchtime.rename_axis(columns=None)
    max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['behavior_type'],aggfunc=np.max)
    user_item_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,max_touchtime,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,max_touchtype,how='left',right_index=True,left_index=True)
#    user_item_feture = pd.merge(user_item_feture,_live,how='left',right_index=True,left_index=True)

    user_item_feture = pd.merge(user_item_feture,user_item_count_5,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
    user_item_feture = pd.merge(user_item_feture,user_item_count_3,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
    user_item_feture = pd.merge(user_item_feture,user_item_count_2,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
    user_item_feture.fillna(0,inplace=True)
    return user_item_feture

def user_cate_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_item_count = pd.crosstab([data.user_id,data.item_category],data.behavior_type)
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    
    user_cate_count_5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=(end_time-datetime.timedelta(days=5+2))]
        user_cate_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_category],beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=(end_time-datetime.timedelta(days=5))]
        user_cate_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_category],beforefiveday.behavior_type)
    user_cate_count_3 = None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=3+2))]
        user_cate_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=3))]
        user_cate_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)


    user_cate_count_2 = None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=7+2))]
        user_cate_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=7))]
        user_cate_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
        
#    _live = user_cate_long_touch(data)
    beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_category],beforeoneday.behavior_type)
    max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['hours'],aggfunc=[np.min,np.max])
    max_touchtime = max_touchtime.droplevel(level=0, axis=1)
    max_touchtime = max_touchtime.rename_axis(columns=None)
    max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['behavior_type'],aggfunc=np.max)
    # max_touchtype = max_touchtype.droplevel(level=0, axis=1)
    # max_touchtype = max_touchtype.rename_axis(columns=None)
    user_cate_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,max_touchtime,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,max_touchtype,how='left',right_index=True,left_index=True)
#    user_cate_feture = pd.merge(user_cate_feture,_live,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_5,how='left',right_index=True,left_index=True,suffixes=('_left', '_right'))
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_3,how='left',right_index=True,left_index=True,suffixes=('_left1', '_right1'))
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_2,how='left',right_index=True,left_index=True,suffixes=('_left2', '_right2'))
    user_cate_feture.fillna(0,inplace=True)
    return user_cate_feture

def user_click(beforesomeday):#用户在前几天各种操作在各个小时的计数
	user_act_count = pd.crosstab([beforesomeday.user_id,beforesomeday.behavior_type],beforesomeday.hours,dropna=False)
	user_act_count = user_act_count.unstack(fill_value = 0)
	return user_act_count

def user_liveday(train_user_window1):#用户各个行为活跃的天数
	user_live = train_user_window1.groupby(by = ['user_id','behavior_type']).agg({"daystime":lambda x:x.nunique()})
	user_live = user_live.unstack(fill_value = 0)
	return user_live


def user_item_click(beforesomeday):
	user_item_act_count = pd.crosstab([beforesomeday.user_id,beforesomeday.item_id,beforesomeday.behavior_type],beforesomeday.hours)
	user_item_act_count = user_item_act_count.unstack(fill_value = 0)
	return user_item_act_count

def user_cate_click(beforesomeday):
	user_cate_act_count = pd.crosstab([beforesomeday.user_id,beforesomeday.item_category,beforesomeday.behavior_type],beforesomeday.hours)
	user_cate_act_count = user_cate_act_count.unstack(fill_value = 0)
	return user_cate_act_count

def user_item_long_touch(train_user_window1):
	_live = train_user_window1.groupby(by = ['user_id','item_id']).agg({"daystime":lambda x:(x.max()-x.min()).days})
	return _live

def user_cate_long_touch(train_user_window1):
	_live = train_user_window1.groupby(by = ['user_id','item_category']).agg({"daystime":lambda x:(x.max()-x.min()).days})
	return _live


lock = multiprocessing.Lock()
def worker(df_shared,LabelDay,num):
    print("------第{}轮------".format(num))
    LabelDay = LabelDay-datetime.timedelta(days=num)
    if (LabelDay <= datetime.datetime(2014,12,13,0,0,0)):
            LabelDay = LabelDay-datetime.timedelta(days=3)
    train_user_window1 = None
    if (LabelDay >= datetime.datetime(2014,12,12,0,0,0)):
        train_user_window1 = df_shared[(df_shared['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT+2))) & (df_shared['daystime'] < LabelDay)]
    else:
        train_user_window1 = df_shared[(df_shared['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))) & (df_shared['daystime'] < LabelDay)]
#        train_user_window1 = Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))) & (Data['daystime'] < LabelDay)]
    beforeoneday = df_shared[df_shared['daystime'] == (LabelDay-datetime.timedelta(days=1))]
    # beforetwoday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=2))) & (Data['daystime'] < LabelDay)]
    # beforefiveday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=5))) & (Data['daystime'] < LabelDay)]
    x = get_train(df_shared, LabelDay)
    add_user_click_1 = user_click(beforeoneday)
    add_user_click_1 = add_user_click_1.droplevel(level=0, axis=1)
    add_user_click_1 = add_user_click_1.rename_axis(columns=None)
    add_user_item_click_1 = user_item_click(beforeoneday)
    add_user_item_click_1 = add_user_item_click_1.droplevel(level=0, axis=1)
    add_user_item_click_1 = add_user_item_click_1.rename_axis(columns=None)
    add_user_cate_click_1 = user_cate_click(beforeoneday)
    add_user_cate_click_1 = add_user_cate_click_1.droplevel(level=0, axis=1)
    add_user_cate_click_1 = add_user_cate_click_1.rename_axis(columns=None)
    # add_user_click_2 = user_click(beforetwoday)
    # add_user_click_5 = user_click(beforefiveday)
    liveday = user_liveday(train_user_window1)
    liveday = liveday.droplevel(level=0, axis=1)
    liveday = liveday.rename_axis(columns=None)
    # sys.exit()
    a = user_id_feture(train_user_window1, LabelDay,beforeoneday)
    a = a.reset_index()
    b = item_id_feture(train_user_window1, LabelDay,beforeoneday)
    b = b.reset_index()
    c = item_category_feture(train_user_window1, LabelDay,beforeoneday)
    c = c.reset_index()
    d = user_cate_feture(train_user_window1, LabelDay,beforeoneday)
    d = d.reset_index()
    e = user_item_feture(train_user_window1, LabelDay,beforeoneday)
    e = e.reset_index()
    x = pd.merge(x,a,on=['user_id'],how='left',suffixes=('_left111', '_right111'))
    x = pd.merge(x,b,on=['item_id'],how='left',suffixes=('_left222', '_right222'))
    x = pd.merge(x,c,on=['item_category'],how='left',suffixes=('_left333', '_right333'))
    x = pd.merge(x,d,on=['user_id','item_category'],how='left',suffixes=('_left11', '_right11'))
    x = pd.merge(x,e,on=['user_id','item_id'],how='left',suffixes=('_left22', '_right22'))
    x = pd.merge(x,add_user_click_1,left_on = ['user_id'],right_index=True,how = 'left' )
    # x = pd.merge(x,add_user_click_2,left_on = ['user_id'],right_index=True,how = 'left' )
    # x = pd.merge(x,add_user_click_5,left_on = ['user_id'],right_index=True,how = 'left' )
    x = pd.merge(x,add_user_item_click_1,left_on = ['user_id','item_id'],right_index=True,how = 'left' )
    x = pd.merge(x,add_user_cate_click_1,left_on = ['user_id','item_category'],right_index=True,how = 'left',suffixes=('_left333', '_right333') )
    x = pd.merge(x,liveday,left_on = ['user_id'],right_index=True,how = 'left' )
    x = x.fillna(0)
    print(num,LabelDay,len(x))
    # x = reduce_mem_usage(x)
    x_1 = x[x['label']==1]
    print(len(x_1))
    x_0 = x[x['label']==0]
    print(len(x_0))
    x_0 = x_0.sample(len(x_1)*90)
    x = pd.concat([x_1,x_0],axis=0)
    with lock:
        if not os.path.exists('./train.csv'):
            x.to_csv('./train.csv',mode='a',index=False)
        else:
            x.to_csv('./train.csv',mode='a',index=False,header=False)
    del a
    del b
    del c
    del d
    del e
    del x
    del df_shared
    gc.collect()

    
def build_test(Data):
    LabelDay=datetime.datetime(2014,12,18,0,0,0)
    Data = Data[user_table.item_id.isin(list(item_table.item_id))]
    test = get_label_testset(Data,LabelDay)
    train_user_window1 =  Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT-1))) & (Data['daystime'] <= LabelDay)]
    beforeoneday = Data[Data['daystime'] == LabelDay]
    # beforetwoday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=2))) & (Data['daystime'] < LabelDay)]
    # beforefiveday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=5))) & (Data['daystime'] < LabelDay)]
    add_user_click = user_click(beforeoneday)
    add_user_click = add_user_click.droplevel(level=0, axis=1)
    add_user_click = add_user_click.rename_axis(columns=None)
    add_user_item_click = user_item_click(beforeoneday)
    add_user_item_click = add_user_item_click.droplevel(level=0, axis=1)
    add_user_item_click = add_user_item_click.rename_axis(columns=None)
    add_user_cate_click = user_cate_click(beforeoneday)
    add_user_cate_click = add_user_cate_click.droplevel(level=0, axis=1)
    add_user_cate_click = add_user_cate_click.rename_axis(columns=None)
    # add_user_click_2 = user_click(beforetwoday)
    # add_user_click_5 = user_click(beforefiveday)
    liveday = user_liveday(train_user_window1)
    liveday = liveday.droplevel(level=0, axis=1)
    liveday = liveday.rename_axis(columns=None)
    a = user_id_feture(train_user_window1, LabelDay,beforeoneday)
    a = a.reset_index()
    b = item_id_feture(train_user_window1, LabelDay,beforeoneday)
    b = b.reset_index()
    c = item_category_feture(train_user_window1, LabelDay,beforeoneday)
    c = c.reset_index()
    d = user_cate_feture(train_user_window1, LabelDay,beforeoneday)
    d = d.reset_index()
    e = user_item_feture(train_user_window1, LabelDay,beforeoneday)
    e = e.reset_index()
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


if __name__=="__main__":
    # process()
    # Data = reduce_mem_usage(Data)
    print("process()结束")
    df_shared = pd.read_csv("./drop1112_sub_item.csv",dtype={'hours':np.int8})
    LabelDay = datetime.datetime(2014,12,18,0,0,0)
    p = list()
    # 将 DataFrame 转换为共享内存对象
    df_shared = df_shared.drop(columns=['time'],axis=1)
    # df_shared = sharedctypes.RawArray('b', Data.values.tobytes())
    # # 将共享内存对象转换回 DataFrame
    # df_shared = np.frombuffer(df_shared).reshape(Data.shape)
    # df_shared = pd.DataFrame(df_shared, columns=Data.columns)
    df_shared["daystime"] = pd.to_datetime(df_shared["days"],format="%Y-%m-%d")
    df_shared = df_shared.drop(columns=["days"],axis=1)
    print(df_shared.dtypes)
    print(len(df_shared))
    # del Data
    for i in range(15):
        p.append(multiprocessing.Process(target=worker,args=(df_shared,LabelDay,i)))
    # for k in range(3):
    #     for i in range(5*k,5*(k+1)):
    #         p[i].start()
    #     for i in range(5*k,5*(k+1)):
    #         p[i].join()
    for k in range(0,5):
        p[k].start()
    for k in range(0,5):
        p[k].join()
    for k in range(5,10):
        p[k].start()
    for k in range(5,10):
        p[k].join()
    for k in range(10,15):
        p[k].start()
    for k in range(10,15):
        p[k].join()
    # test=build_test(df_shared)
    print(gc.get_stats()[0]) 
    del df_shared
    gc.collect()
    print(gc.get_stats()[0]) 
    train_set = pd.read_csv("./train.csv")
    print(gc.get_stats()[0])
    print("--------数据集准备完成----------")
    print(len(train_set))
    # print(len(test))
    ###############采样
    # # train_set = train_set.iloc[:100000]
    # print(train_set.dtypes)
    # train_set.to_csv("./1.csv",index=False)
    # train_set_1 = train_set[train_set['label']==1]
    # train_set_0 = train_set[train_set['label']==0]
    # train_set_0 = train_set_0.sample(len(train_set_1)*90)
    # train_set = pd.concat([train_set_1,train_set[train_set['label']==0].sample(len(train_set_1)*90)],axis=0)
    # ###############
    train_y = train_set['label'].values
    train_x = train_set.drop(['user_id', 'item_id','item_category', 'label'], axis=1).values
    # test_x = test.drop(['user_id', 'item_id','item_category'], axis=1).values
    # train_set = train_set.sample(frac=0.00001)
    print("-----------数据处理完成----------")   
    num_round = 1000
    params = {'max_depth': 5, 'colsample_bytree': 0.8, 'subsample': 0.8, 'eta': 0.02, 'verbosity': 2,
                  'objective': 'binary:logistic','eval_metric':'error', 'min_child_weight': 2.5,
                   'seed': 10,'n_jobs':-1,'tree_method':'exact'} 
    plst = list(params.items())
    dtrain = xgb.DMatrix(train_x, label=train_y)
    # dtest = xgb.DMatrix(test_x)
    # del train_set
    # del test_x
    # # model = xgb.XGBClassifier(**params)
    # # print("------start cv------")
    # # cvresult = xgb.cv(plst, dtrain, num_boost_round=3000, nfold=3, metrics=['error'],
    # #      early_stopping_rounds=50, stratified=True, seed=10)
    print("------start train------")
    bst = xgb.train(plst, dtrain, num_round)
    bst.save_model('../weight/model.bin')
    # print("------start test------")
    # predicted_proba = bst.predict(dtest)
    # print("------test finish------")
    # predicted_proba = pd.DataFrame(predicted_proba)
    # predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    # predicted.columns = ['user_id','item_id','prob']
    # predicted = predicted.sort_values('prob',  axis=0,ascending=False)
    
    # predict2 = predicted.iloc[:30000, [0, 1]]
    # # 保存到文件
    # predict2.to_csv(("./result" + ".txt"), header=None, index=False,sep='\t')
    # print("-----save finish-----")
    # sys.exit()