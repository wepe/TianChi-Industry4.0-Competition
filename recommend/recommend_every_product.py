import pandas as pd

category_features = ['param3','param4','param7','param8']
numeric_features = ['param1','param2','param5','param6','param9']

param_train = pd.read_csv('../data/param_data_train.csv')
param_train = param_train[param_train.key_index>=0.97]
param_test = pd.read_csv('../data/param_data_test_new.csv')
param_test['key_index'] = -999
param_train_test = pd.concat([param_train,param_test],axis=0)

#scale to [0,1]
for f in numeric_features:
    param_train_test[f] = param_train_test[f]/param_train_test[f].max()

param_train = param_train_test[param_train_test.key_index!=-999]    
param_test = param_train_test[param_train_test.key_index==-999]

test_product = list(param_test.product_no)
train_product = []

#calculate distance and find the closest sample
for product_no in test_product:
    print "product_no",product_no
    this_product = param_test[param_test.product_no==product_no]
    temp = pd.merge(param_train,this_product,on=category_features)
    temp['distance'] = 0.24*(temp.param1_x - temp.param1_y)**2 + 0.2*(temp.param6_x - temp.param6_y)**2 + \
                       0.22*(temp.param2_x - temp.param2_y)**2 + 0.18*(temp.param5_x - temp.param5_y)**2 + 0.16*(temp.param9_x - temp.param9_y)**2
    temp.sort_values(by='distance',inplace=True)
    train_product.append(list(temp.product_no_x)[0])
    
#generate recommend result for every product_no in the testset
recommend_result = pd.DataFrame(test_product,columns=['test_product_no'])
recommend_result['train_product_no'] = train_product

draft_train = pd.read_csv('../data/draft_data_train.csv')
temp = pd.merge(recommend_result,draft_train,left_on='train_product_no',right_on='product_no')
temp = temp[['test_product_no','draft_param1','draft_param2','draft_param3','draft_param4','draft_param5','draft_param6','draft_param7','draft_param9','draft_param10','draft_param11']]
temp.rename(columns={'test_product_no':'product_no'})

temp.to_csv('recommend_every_product.csv',index=None)
