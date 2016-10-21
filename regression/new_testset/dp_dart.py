#coding=utf-8

import xgboost as xgb
import pandas as pd

#loading feature file and merge
draft_train = pd.read_csv('../../data/feature/draft_train.csv')
param_train = pd.read_csv('../../data/feature/param_train.csv')
train = pd.merge(draft_train,param_train,on=['product_no','key_index'])

draft_param_feature = list(train.columns)
draft_param_feature.remove('product_no')
draft_param_feature.remove('key_index')

print "all",train.shape
train = train[train.key_index>=0.85]
print '>0.85',train.shape

draft_test = pd.read_csv('../../data/feature/draft_test.csv')
param_test = pd.read_csv('../../data/feature/param_test.csv')
test = pd.merge(draft_test,param_test,on='product_no')

#drop nan columns
test['key_index'] = -999
train_test = pd.concat([train,test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
train = train_test[train_test.key_index!=-999]
test = train_test[train_test.key_index==-999]

train_y = train.key_index
train_x = train.drop(['key_index','product_no'],axis=1)
test_product_no = test.product_no
test_x = test.drop(['product_no','key_index'],axis=1)

#'p1_p2' is a new feature
train_x['p1_p2'] = train_x.param1 - train_x.param2
test_x['p1_p2'] = test_x.param1 - test_x.param2

#training xgboost
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

params={'booster':'gbtree',
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'max_depth':6,
	'lambda':100,
	'subsample':0.6,
	'colsample_bytree':0.6,
	'min_child_weight':5,#5~10
	'eta': 0.01,
	'sample_type':'uniform',
	'normalize':'tree',
	'rate_drop':0.1,
	'skip_drop':0.9,
	'seed':87,
	'nthread':12
	}

watchlist  = [(dtrain,'train')]

#通过cv找最佳的nround
cv_log = xgb.cv(params,dtrain,num_boost_round=25000,nfold=5,metrics='rmse',early_stopping_rounds=50,seed=1024)
bst_rmse= cv_log['test-rmse-mean'].min()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-rmse-mean']
bst_nb = cv_log.nb.to_dict()[bst_rmse]

watchlist  = [(dtrain,'train')]
model = xgb.train(params,dtrain,num_boost_round=bst_nb+50,evals=watchlist)

#predict test set
test_y = model.predict(dtest)
test_result = pd.DataFrame(test_product_no,columns=["product_no"])
test_result["score_all"] = test_y
test_result.to_csv("dp_dart_preds.csv",index=None,encoding='utf-8')

print bst_nb,bst_rmse
