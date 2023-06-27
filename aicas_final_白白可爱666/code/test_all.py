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
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
from test_1days import test_1days
from test_2days import test_2days
import time

pd.options.mode.chained_assignment = None
FEATURE_EXTRACTION_SLOT=10
print("开始读取")

def read_user_table(i,file):
    # 读取用户行为数据
    # l = [0,8,14,18,22,23,24]
    user_table = pd.read_csv(file, sep="\t",usecols=[0,1,2,4,5],skiprows=lambda x: x in range(0, i*40000000),nrows = 40000000,header=None,names=["user_id","item_id","behavior_type","item_category","time"],dtype={'behavior_type':np.int8,"item_category":np.int16})
    print(len(user_table))
    item_table = pd.read_csv('/data/final/part2_item.txt',sep="\t",usecols=[0],header=None,names=["item_id"])
    user_table = user_table[user_table.item_id.isin(list(item_table.item_id))]
    user_table['days'] = user_table['time'].map(lambda x:x.split(' ')[0])
    user_table = user_table[(user_table['days'] != '2014-12-12') & (user_table['days'] != '2014-12-11')]
    user_table['hours'] = user_table['time'].map(lambda x:x.split(' ')[1])
    user_table["daystime"] = pd.to_datetime(user_table["days"],format="%Y-%m-%d")
    user_table.drop(columns=['time',"days"],axis=1,inplace=True)
    return user_table

def p(param):
    # 读取用户行为数据
    return read_user_table(param[0],param[1])


file_list = [(0,'/data/final/round2_user_0.txt'), (0,'/data/final/round2_user_1.txt'), (0,'/data/final/round2_user_2.txt'), (0,'/data/final/round2_user_3.txt'), (0,'/data/final/round2_user_4.txt'),
             (1,'/data/final/round2_user_0.txt'), (1,'/data/final/round2_user_1.txt'), (1,'/data/final/round2_user_2.txt'), (1,'/data/final/round2_user_3.txt'), (1,'/data/final/round2_user_4.txt'),
             (2,'/data/final/round2_user_0.txt'), (2,'/data/final/round2_user_1.txt'), (2,'/data/final/round2_user_2.txt'), (2,'/data/final/round2_user_3.txt'), (2,'/data/final/round2_user_4.txt'),
             (3,'/data/final/round2_user_0.txt'), (3,'/data/final/round2_user_1.txt'), (3,'/data/final/round2_user_2.txt'), (3,'/data/final/round2_user_3.txt'), (3,'/data/final/round2_user_4.txt'),
             (4,'/data/final/round2_user_0.txt'), (4,'/data/final/round2_user_1.txt'), (4,'/data/final/round2_user_2.txt'), (4,'/data/final/round2_user_3.txt'), (4,'/data/final/round2_user_4.txt'),
             (5,'/data/final/round2_user_0.txt'), (5,'/data/final/round2_user_1.txt'), (5,'/data/final/round2_user_2.txt'), (5,'/data/final/round2_user_3.txt'), (5,'/data/final/round2_user_4.txt')]
with Pool(processes=len(file_list)) as pool:
    # 并行读取用户数据并进行处理
    user_tables = pool.map(p, file_list)

pool.close()  # 关闭池
pool.join()   # 等待所有进程完成

user_table = pd.concat(user_tables, ignore_index=True)

# p1 = multiprocessing.Process(target=test_1days,args=(user_table,FEATURE_EXTRACTION_SLOT))
# # p2 = multiprocessing.Process(target=test_2days,args=(user_table,FEATURE_EXTRACTION_SLOT))
# p1.start()
# # p2.start()
# p1.join()
# p2.join()
test_1days(user_table,FEATURE_EXTRACTION_SLOT)
# result1 = pd.read_csv("result_1QW.txt",names = ["1","2","3"],header=None,sep="\t")
# result2 = pd.read_csv("result_2days.txt",names = ["1","2","3"],header=None,sep="\t")
# result = pd.concat([result1,result2],ignore_index=True)
# result = result1[result1["3"]>=0.24]
# result = result.iloc[:40000].drop_duplicates(subset=['1','2'], keep='first')
# result[["1","2"]].to_csv(("./submit" + ".txt"), header=None, index=False,sep='\t')
print("全部结束")