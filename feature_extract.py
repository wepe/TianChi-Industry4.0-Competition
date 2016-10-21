#coding=utf-8
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Pool

os.mkdir('data/feature')

############     draft表 类别特征编码   ##############
######################################################
draft_train = pd.read_csv('data/draft_data_train.csv')
draft_test = pd.read_csv('data/draft_data_test_new.csv')
draft_test['key_index'] = -999
draft_train_test = pd.concat([draft_train,draft_test],axis=0)
category_var = ['draft_param1','draft_param2','draft_param3','draft_param10','draft_param11','draft_param4','draft_param6','draft_param7']

for var in category_var:
    var_dummies = pd.get_dummies(draft_train_test[var])
    var_dummies.columns = [var+'_'+str(i) for i in range(var_dummies.shape[1])]
    if var not in ['draft_param4','draft_param6','draft_param7']:
        draft_train_test.drop(var,axis=1,inplace=True)
    draft_train_test = pd.concat([draft_train_test,var_dummies],axis=1)

draft_train = draft_train_test[draft_train_test.key_index!=-999]
draft_test = draft_train_test[draft_train_test.key_index==-999]
draft_test.drop('key_index',axis=1,inplace=True)
draft_train.to_csv('data/feature/draft_train.csv',index=None)
draft_test.to_csv('data/feature/draft_test.csv',index=None)


############     param表 类别特征编码   ##############
######################################################
param_train = pd.read_csv('data/param_data_train.csv')
param_test = pd.read_csv('data/param_data_test_new.csv')
param_test['key_index'] = -999
param_train_test = pd.concat([param_train,param_test],axis=0)
category_var = ['param3','param4','param7','param8','param5','param9']

for var in category_var:
    var_dummies = pd.get_dummies(param_train_test[var])
    var_dummies.columns = [var+'_'+str(i) for i in range(var_dummies.shape[1])]
    if var not in ['param5','param9']:
        param_train_test.drop(var,axis=1,inplace=True)
    param_train_test = pd.concat([param_train_test,var_dummies],axis=1)


############     param表 数值型特征     ##############
######################################################
"""
param5 4,6
param9 3.6,  3.1,  3.
对'param5','param9'也进行了编码,但保留原始字段
"""
param_train = param_train_test[param_train_test.key_index!=-999]
param_test = param_train_test[param_train_test.key_index==-999]
param_test.drop('key_index',axis=1,inplace=True)
param_train.to_csv('data/feature/param_train.csv',index=None)
param_test.to_csv('data/feature/param_test.csv',index=None)



############       timevarying 表         ############
######################################################
"""
test数据都有时序参数记录
train数据14675个product只有6940个有时序参数记录
提取特征:对每个product,提取各个参数的最小,最大,中值,平均值,众数,方差,记录个数(分tparam1=50之前的记录和全部记录，更加细致地分10个时间段)
"""

params = ['tparam1', 'tparam18', 'tparam14', 'tparam2', 'tparam10', 'tparam7','tparam9', 'tparam3', 'tparam8', 'tparam11', 'tparam17', 'tparam4','tparam5', 'tparam6', 'tparam16', 'tparam15', 'tparam12', 'tparam13']
sts = ['_min','_max','_median','_mean','_std','_cnt','_max_min']

########  testset   ##########
##############################
tv_test = pd.read_csv('data/timevarying_param_test_new.csv',header=None)
tv_test.columns = ['product_no','param_name','param_value','add_time']
products = list(tv_test.product_no.unique())

#全部记录
tv_features = pd.DataFrame(columns=['product_no']+[p+s for s in sts for p in params])
def test_(product):
    d = {'product_no':product}
    tv_product = tv_test[tv_test.product_no==product]
    for param in params:
        tv_product_param = tv_product[tv_product.param_name==param]
        d[param+'_min'] = tv_product_param.param_value.min()
        d[param+'_max'] = tv_product_param.param_value.max()
        d[param+'_median'] = tv_product_param.param_value.median()
        d[param+'_mean'] = tv_product_param.param_value.mean()
        d[param+'_std'] = tv_product_param.param_value.std()
        d[param+'_cnt'] = tv_product_param.param_value.count()
        d[param+'_max_min'] = tv_product_param.param_value.max() - tv_product_param.param_value.min()
    this_tv_features = pd.DataFrame(d,index=[0])
    return this_tv_features

rst = []
pool = Pool(12)
for product in products:
    rst.append(pool.apply_async(test_,args=(product,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
    tv_features = pd.concat([tv_features,i],axis=0)
tv_features.to_csv('data/feature/tv_features_test.csv',index=None)

#进度50%之前的记录    
tv_features_half = pd.DataFrame(columns=['product_no']+[p+s for s in sts for p in params])
def test_(product):
    d = {'product_no':product}
    tv_product = tv_test[tv_test.product_no==product]
    tv_product.add_time = tv_product.add_time.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    half_time = int(tv_product[(tv_product.param_name=='tparam1')&(tv_product.param_value==50.0)].add_time)
    tv_product = tv_product[tv_product.add_time<half_time]
    if tv_product.shape[0]==0:
        print product
    for param in params:
        tv_product_param = tv_product[tv_product.param_name==param]
        d[param+'_min'] = tv_product_param.param_value.min()
        d[param+'_max'] = tv_product_param.param_value.max()
        d[param+'_median'] = tv_product_param.param_value.median()
        d[param+'_mean'] = tv_product_param.param_value.mean()
        d[param+'_std'] = tv_product_param.param_value.std()
        d[param+'_cnt'] = tv_product_param.param_value.count()
        d[param+'_max_min'] = tv_product_param.param_value.max() - tv_product_param.param_value.min()
    this_tv_features = pd.DataFrame(d,index=[0])
    return this_tv_features

rst = []
pool = Pool(12)
for product in products:
    print product
    rst.append(pool.apply_async(test_,args=(product,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
     tv_features_half = pd.concat([ tv_features_half,i],axis=0)
tv_features_half.to_csv('data/feature/tv_features_half_test.csv',index=None)

########  trainset   ##########
##############################
tv_train = pd.read_csv('data/timevarying_param_train.csv',header=None)
tv_train.columns = ['product_no','key_index','param_name','param_value','add_time']
products = list(tv_train.product_no.unique())

#全部记录
tv_features = pd.DataFrame(columns=['product_no']+[p+s for s in sts for p in params])
def train_(product):
    d = {'product_no':product}
    tv_product = tv_train[tv_train.product_no==product]
    for param in params:
        tv_product_param = tv_product[tv_product.param_name==param]
        d[param+'_min'] = tv_product_param.param_value.min()
        d[param+'_max'] = tv_product_param.param_value.max()
        d[param+'_median'] = tv_product_param.param_value.median()
        d[param+'_mean'] = tv_product_param.param_value.mean()
        d[param+'_std'] = tv_product_param.param_value.std()
        d[param+'_cnt'] = tv_product_param.param_value.count()
        d[param+'_max_min'] = tv_product_param.param_value.max() - tv_product_param.param_value.min()
    this_tv_features = pd.DataFrame(d,index=[0])
    return this_tv_features

rst = []
pool = Pool(12)
for product in products:
    print product
    rst.append(pool.apply_async(train_,args=(product,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
    tv_features = pd.concat([tv_features,i],axis=0)
tv_features.to_csv('data/feature/tv_features_train.csv',index=None)

#进度50%之前的记录    
tv_features_half = pd.DataFrame(columns=['product_no']+[p+s for s in sts for p in params])
def train_(product):
    d = {'product_no':product}
    tv_product = tv_train[tv_train.product_no==product]
    tv_product.add_time = tv_product.add_time.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    half_time = int(tv_product[(tv_product.param_name=='tparam1')&(tv_product.param_value==50.0)].add_time)
    tv_product = tv_product[tv_product.add_time<half_time]
    for param in params:
        tv_product_param = tv_product[tv_product.param_name==param]
        d[param+'_min'] = tv_product_param.param_value.min()
        d[param+'_max'] = tv_product_param.param_value.max()
        d[param+'_median'] = tv_product_param.param_value.median()
        d[param+'_mean'] = tv_product_param.param_value.mean()
        d[param+'_std'] = tv_product_param.param_value.std()
        d[param+'_cnt'] = tv_product_param.param_value.count()
        d[param+'_max_min'] = tv_product_param.param_value.max() - tv_product_param.param_value.min()
    this_tv_features = pd.DataFrame(d,index=[0])
    return this_tv_features

rst = []
pool = Pool(12)
for product in products:
    rst.append(pool.apply_async(train_,args=(product,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
     tv_features_half = pd.concat([ tv_features_half,i],axis=0)
tv_features_half.to_csv('data/feature/tv_features_half_train.csv',index=None)





################# 分10个时间段统计   #################
######################################################
stage = ['stg1_','stg2_','stg3_','stg4_','stg5_','stg6_','stg7_','stg8_','stg9_','stg10_']
params = ['tparam1', 'tparam18', 'tparam14', 'tparam2', 'tparam10', 'tparam7','tparam9', 'tparam3', 'tparam8', 'tparam11', 'tparam17', 'tparam4','tparam5', 'tparam6', 'tparam16', 'tparam15', 'tparam12', 'tparam13']
sts = ['_min','_max','_median','_mean','_std','_cnt','_max_min']

########  testset   ##########
##############################
tv_test = pd.read_csv('data/timevarying_param_test_new.csv',header=None)
tv_test.columns = ['product_no','param_name','param_value','add_time']
products = list(tv_test.product_no.unique())

tv_features = pd.DataFrame(columns=['product_no']+[stg+p+s for stg in stage for s in sts for p in params])
def test_(product):
    d = {'product_no':product}
    tv_product = tv_test[tv_test.product_no==product]
    tv_product.add_time = tv_product.add_time.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    t = tv_product.add_time.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    split_point = [0,int(t['10%']),int(t['20%']),int(t['30%']),int(t['40%']),int(t['50%']),int(t['60%']),int(t['70%']),int(t['80%']),int(t['90%']),1e11]
    for i in range(10):
        stg = stage[i]
        tv_product_stg = tv_product[(split_point[i]<tv_product.add_time)&(tv_product.add_time<split_point[i+1])]
        for param in params:
            tv_product_param = tv_product_stg[tv_product_stg.param_name==param]
            d[stg+param+'_min'] = tv_product_param.param_value.min()
            d[stg+param+'_max'] = tv_product_param.param_value.max()
            d[stg+param+'_median'] = tv_product_param.param_value.median()
            d[stg+param+'_mean'] = tv_product_param.param_value.mean()
            d[stg+param+'_std'] = tv_product_param.param_value.std()
            d[stg+param+'_cnt'] = tv_product_param.param_value.count()
            d[stg+param+'_max_min'] = tv_product_param.param_value.max() - tv_product_param.param_value.min()
    this_tv_features = pd.DataFrame(d,index=[0])
    return this_tv_features

rst = []
pool = Pool(12)
for product in products:
    print product
    rst.append(pool.apply_async(test_,args=(product,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
    tv_features = pd.concat([tv_features,i],axis=0)
tv_features.to_csv('data/feature/tv_stage10_test.csv',index=None)


########  trainset   ##########
##############################
tv_train = pd.read_csv('data/timevarying_param_train.csv',header=None)
tv_train.columns = ['product_no','key_index','param_name','param_value','add_time']
products = list(tv_train.product_no.unique())

tv_features = pd.DataFrame(columns=['product_no']+[stg+p+s for stg in stage for s in sts for p in params])
def train_(product):
    d = {'product_no':product}
    tv_product = tv_train[tv_train.product_no==product]
    tv_product.add_time = tv_product.add_time.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    t = tv_product.add_time.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    split_point = [0,int(t['10%']),int(t['20%']),int(t['30%']),int(t['40%']),int(t['50%']),int(t['60%']),int(t['70%']),int(t['80%']),int(t['90%']),1e11]
    for i in range(10):
        stg = stage[i]
        tv_product_stg = tv_product[(split_point[i]<tv_product.add_time)&(tv_product.add_time<split_point[i+1])]
        for param in params:
            tv_product_param = tv_product_stg[tv_product_stg.param_name==param]
            d[stg+param+'_min'] = tv_product_param.param_value.min()
            d[stg+param+'_max'] = tv_product_param.param_value.max()
            d[stg+param+'_median'] = tv_product_param.param_value.median()
            d[stg+param+'_mean'] = tv_product_param.param_value.mean()
            d[stg+param+'_std'] = tv_product_param.param_value.std()
            d[stg+param+'_cnt'] = tv_product_param.param_value.count()
            d[stg+param+'_max_min'] = tv_product_param.param_value.max() - tv_product_param.param_value.min()
    this_tv_features = pd.DataFrame(d,index=[0])
    return this_tv_features

rst = []
pool = Pool(12)
for product in products:
    print product
    rst.append(pool.apply_async(train_,args=(product,)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
    tv_features = pd.concat([tv_features,i],axis=0)
tv_features.to_csv('data/feature/tv_stage10_train.csv',index=None)


#####################  totaltime   #####################
######################################################
#for each product,get its max(add_time) - min(add_time)

tv_train = pd.read_csv('data/timevarying_param_train.csv',header=None)
tv_train.columns = ['product_no','key_index','param_name','param_value','add_time']
tv_train.drop('key_index',axis=1,inplace=True)
tv_test = pd.read_csv('data/timevarying_param_test_new.csv',header=None)
tv_test.columns = ['product_no','param_name','param_value','add_time']
tv = pd.concat([tv_train,tv_test],axis=0)
products = list(tv.product_no.unique())

def bar(product):
    this_product = tv[tv.product_no==product]
    this_product.add_time = this_product.add_time.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    total_time = this_product.add_time.max() - this_product.add_time.min()
    return product,total_time

product_totaltime = Pool(12).map(bar,products)
product = np.array([i[0] for i in product_totaltime])
totaltime = np.array([i[1] for i in product_totaltime])

product_totaltime = pd.DataFrame(product,columns=['product_no'])
product_totaltime['totaltime'] = totaltime
product_totaltime.to_csv('data/feature/continue_time.csv',index=None)

