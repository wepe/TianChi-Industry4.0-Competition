#coding=utf-8
import pandas as pd
import json


#生成draft_params_statistics，即trainset里所有的draft_param组合，按key_index均值排序
draft_params = ['draft_param1', 'draft_param2','draft_param3', 'draft_param4', 'draft_param5', 'draft_param6','draft_param7', 'draft_param9', 'draft_param10', 'draft_param11']
draft_data = pd.read_csv('../data/draft_data_train.csv')

draft_data.drop('product_no',axis=1,inplace=True)
keyindex_mean = draft_data.groupby(draft_params)['key_index'].agg('mean').reset_index()
keyindex_mean.rename(columns={'key_index':'keyindex_mean'},inplace=True)
keyindex_mean.sort_values('keyindex_mean',ascending=False,inplace=True)

keyindex_median = draft_data.groupby(draft_params)['key_index'].agg('median').reset_index()
keyindex_median.rename(columns={'key_index':'keyindex_median'},inplace=True)

keyindex_max = draft_data.groupby(draft_params)['key_index'].agg('max').reset_index()
keyindex_max.rename(columns={'key_index':'keyindex_max'},inplace=True)

keyindex_min = draft_data.groupby(draft_params)['key_index'].agg('min').reset_index()
keyindex_min.rename(columns={'key_index':'keyindex_min'},inplace=True)

draft_data.drop('key_index',axis=1,inplace=True)
draft_data['count'] = 1
keyindex_count = draft_data.groupby(draft_params)['count'].agg('sum').reset_index()

temp = pd.merge(keyindex_mean,keyindex_median,on=draft_params)
temp = pd.merge(temp,keyindex_max,on=draft_params)
temp = pd.merge(temp,keyindex_min,on=draft_params)
temp = pd.merge(temp,keyindex_count,on=draft_params)
temp.to_csv('draft_params_statistics.csv',index=None)


#统计Top K个组合的每个参数的count累计值
category_params = ['draft_param1', 'draft_param2','draft_param3', 'draft_param10', 'draft_param11']
numerical_params = ['draft_param4', 'draft_param5', 'draft_param6','draft_param7', 'draft_param9']
keyindex_mean = list(temp.keyindex_mean)

print "########################  top k = 20   #####################"
k = 20
topk = temp[temp.keyindex_mean>=keyindex_mean[k]]
for param in category_params:
    this_param = topk[[param,'count']].groupby(param)['count'].agg('sum').reset_index()
    this_param.index = this_param[param]
    this_param.drop(param,axis=1,inplace=True)
    print param,this_param.to_dict()['count']

for param in numerical_params:
    this_param = topk[[param,'count']].groupby(param)['count'].agg('sum').reset_index()
    this_param.index = this_param[param]
    this_param.drop(param,axis=1,inplace=True)
    this_dict = this_param.to_dict()['count']
    sum_ = 0
    cnt_ = 0
    for key,value in this_dict.iteritems():
        sum_ += key*value
        cnt_ += value
    print param,sum_/float(cnt_),this_dict

print "########################  top k = 30   #####################"
k = 30
topk = temp[temp.keyindex_mean>=keyindex_mean[k]]
for param in category_params:
    this_param = topk[[param,'count']].groupby(param)['count'].agg('sum').reset_index()
    this_param.index = this_param[param]
    this_param.drop(param,axis=1,inplace=True)
    print param,this_param.to_dict()['count']

for param in numerical_params:
    this_param = topk[[param,'count']].groupby(param)['count'].agg('sum').reset_index()
    this_param.index = this_param[param]
    this_param.drop(param,axis=1,inplace=True)
    this_dict = this_param.to_dict()['count']
    sum_ = 0
    cnt_ = 0
    for key,value in this_dict.iteritems():
        sum_ += key*value
        cnt_ += value
    print param,sum_/float(cnt_),this_dict
    
print "########################  top k = 40   #####################"
k = 40
topk = temp[temp.keyindex_mean>=keyindex_mean[k]]
for param in category_params:
    this_param = topk[[param,'count']].groupby(param)['count'].agg('sum').reset_index()
    this_param.index = this_param[param]
    this_param.drop(param,axis=1,inplace=True)
    print param,this_param.to_dict()['count']

for param in numerical_params:
    this_param = topk[[param,'count']].groupby(param)['count'].agg('sum').reset_index()
    this_param.index = this_param[param]
    this_param.drop(param,axis=1,inplace=True)
    this_dict = this_param.to_dict()['count']
    sum_ = 0
    cnt_ = 0
    for key,value in this_dict.iteritems():
        sum_ += key*value
        cnt_ += value
    print param,sum_/float(cnt_),this_dict


"""
########################  top k = 20   #####################
draft_param1 {0: 4, 1: 4, 3: 149, 4: 25, 5: 45}
draft_param2 {0: 214, 1: 13}
draft_param3 {0: 157, 1: 70}
draft_param10 {0: 111, 1: 104, 2: 12}
draft_param11 {0: 23, 1: 65, 3: 68, 4: 71}
draft_param4 342.982378855 {342: 4, 343: 223}
draft_param5 0.0851982378855 {0.065: 161, 0.075: 40, 0.06: 3, 0.285: 7, 0.085: 4, 0.28: 12}
draft_param6 0.846696035242 {1.0: 169, 0.4: 58}
draft_param7 1.04453744493 {1.04: 34, 1.0: 48, 1.05: 139, 1.3: 6}
draft_param9 0.313392070485 {0.0: 19, 0.34: 166, 0.35: 42}
########################  top k = 30   #####################
draft_param1 {0: 4, 1: 4, 3: 445, 4: 97, 5: 1229}
draft_param2 {0: 1694, 1: 85}
draft_param3 {0: 453, 1: 1326}
draft_param10 {0: 497, 1: 1266, 2: 16}
draft_param11 {0: 211, 1: 69, 3: 1324, 4: 175}
draft_param4 342.909499719 {342: 161, 343: 1618}
draft_param5 0.212998875773 {0.065: 422, 0.075: 132, 0.06: 3, 0.285: 7, 0.085: 4, 0.28: 1211}
draft_param6 0.557504215852 {1.0: 467, 0.4: 1312}
draft_param7 1.05450252951 {1.04: 34, 1.0: 48, 1.05: 1654, 1.3: 43}
draft_param9 0.107959527825 {0.0: 1218, 0.34: 429, 0.35: 132}
########################  top k = 40   #####################
draft_param1 {0: 4, 1: 62, 2: 57, 3: 464, 4: 142, 5: 3930}
draft_param2 {0: 4529, 1: 130}
draft_param3 {0: 587, 1: 4072}
draft_param10 {0: 497, 1: 3871, 2: 291}
draft_param11 {0: 276, 1: 81, 3: 4070, 4: 232}
draft_param4 342.949989268 {342: 233, 343: 4426}
draft_param5 0.128007083065 {0.065: 563, 0.075: 2857, 0.06: 3, 0.285: 7, 0.085: 4, 0.28: 1225}
draft_param6 0.829233741146 {1.0: 3333, 0.4: 1326}
draft_param7 1.19792444731 {1.04: 42, 1.0: 48, 1.05: 1801, 1.3: 2768}
draft_param9 0.256248121915 {0.0: 1232, 0.34: 559, 0.35: 2868}

"""


#生成提交文件
d1 = {'draft_param1':'3', 'draft_param2':'0','draft_param3':'0', 'draft_param4':'343', 'draft_param5':'0.065', 'draft_param6':'1.0','draft_param7':'1.045', 'draft_param9':'0.34', 'draft_param10':'1', 'draft_param11':'1'}
d2 = {'draft_param1':'5', 'draft_param2':'0','draft_param3':'1', 'draft_param4':'343', 'draft_param5':'0.07', 'draft_param6':'1.0','draft_param7':'1.3', 'draft_param9':'0.345', 'draft_param10':'1', 'draft_param11':'3'}
d3 = {'draft_param1':'5', 'draft_param2':'0','draft_param3':'1', 'draft_param4':'343', 'draft_param5':'0.075', 'draft_param6':'0.4','draft_param7':'1.05', 'draft_param9':'0.34', 'draft_param10':'1', 'draft_param11':'3'}
d = [d1,d2,d3]
with open('recommend_result.csv','w') as f:
    json.dump(d,f,sort_keys=True)
