# AICAS-2023-Grand-Challenge-Software-and-Hardware-optimization-for-E-Commerce-Recommendation_System
记录自己第一次在天池推荐系统比赛中的获奖经历，最终排名第4
# GNN4CAAI-BDSC2023-TASK1
### AICAS-2023-Grand-Challenge-Software-and-Hardware-optimization-for-E-Commerce-Recommendation_System
### 比赛链接:https://tianchi.aliyun.com/competition/entrance/532061/introduction?spm=a2c22.12281925.0.0.168371373MY56b
### 最终排名: 4/520   F1指标：0.0826

这个比赛最关键的两点是特征的构造和监督数据集的构建，下面我来具体的叙述一下在这个比赛中我的整体思路，第一次比赛获奖，可能会有很多不足之处，请多多体谅！

### 比赛说明：
使用机器学习或深度学习算法训练人工智能模型，以解决预测客户在某一天的购买行为的任务。数据集由以下两部分组成：数据集的第一部分被用作训练数据集，记录客户在30天内的行为。数据集的第二部分（产品子集）用作测试集，它由一组项目信息组成。要求参赛者建立和训练模型，根据用户在30天内的购买行为来预测用户在产品子集中的购买行为。
#### 用户的脱敏行为数据集：
字段  | 描述  | 注释
 ---- | ----- | ------  
 User_ID  | 用户标识 | 抽样&字段脱敏 
 Item_ID  | 商品标识 | 字段脱敏 
 Behavior_Type  | 用户对商品的行为类型 | 包括浏览、收藏、加购物车、购买，对应取值分别为1，2，3，4 
 User_Geohash  | 用户位置的空间标识 | 由经纬度通过保密的算法生成 
 Item_Category  | 商品分类标识 | 字段脱敏 
 Time  | 行为时间 | 精确到小时级别 
 #### 脱敏产品子集：
 字段  | 描述  | 注释
 ---- | ----- | ------  
 item_id  | 商品标识 | 抽样&字段脱敏 
 item_geohash  | 商品位置的空间信息，可以为空 | 由经纬度通过保密算法生成 
 item_category  | 商品分类标识 | 字段脱敏、
 
 #### 评估指标：
比赛采用经典的精确度(precision)、召回率(recall)和F1值作为评估指标。
    Accuracy score (F1 score): 30%
    Efficiency score (Inference Time): 70%
    CPU Core occupancy score: Alternative option
    F1 = 2 \times (precision \times recall) / (precision + recall) 
    Time score measures the total running time of the model on the test dataset. It is calculated as follows:
    If the team's running time is less than or equal to 5 minutes, the team scores 0.1 point.
    For every 1 minute beyond 5 minutes, the team loses 0.0005 points.
    The minimum time score a team can receive is 0


