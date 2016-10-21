#coding=utf-8

import xgboost as xgb
import pandas as pd

#loading feature file and merge
draft_train = pd.read_csv('../data/feature/draft_train.csv')
param_train = pd.read_csv('../data/feature/param_train.csv')
train = pd.merge(draft_train,param_train,on=['product_no','key_index'])

draft_param_feature = list(train.columns)
draft_param_feature.remove('product_no')
draft_param_feature.remove('key_index')

print "all",train.shape
train = train[train.key_index>=0.85]
print '>0.85',train.shape

draft_test = pd.read_csv('../data/feature/draft_test.csv')
param_test = pd.read_csv('../data/feature/param_test.csv')
test = pd.merge(draft_test,param_test,on='product_no')

test['key_index'] = -999
train_test = pd.concat([train,test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
train = train_test[train_test.key_index!=-999]
test = train_test[train_test.key_index==-999]

#generate label for each product_no
train.key_index = train.key_index.apply(lambda x: 1 if x<=0.92 else 0)
train_y = train.key_index
train_x = train.drop(['key_index','product_no'],axis=1)
test_product_no = test.product_no
test_x = test.drop(['product_no','key_index'],axis=1)

#"p1_p2" is a new feature
train_x['p1_p2'] = train_x.param1 - train_x.param2
test_x['p1_p2'] = test_x.param1 - test_x.param2

#training xgboost
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

params={'booster':'gbtree',
	'objective': 'binary:logistic',
	'scale_pos_weight':float(len(train_y)-sum(train_y))/sum(train_y),
	'eval_metric': 'auc',
	'max_depth':6,
	'lambda':100,
	'subsample':0.65,
	'colsample_bytree':0.65,
	'eta': 0.002,
	'seed':1024,
	'nthread':12
	}

watchlist  = [(dtrain,'train')]

#通过cv找最佳的nround
cv_log = xgb.cv(params,dtrain,num_boost_round=25000,nfold=5,metrics='auc',early_stopping_rounds=50,seed=1024)
bst_auc= cv_log['test-auc-mean'].max()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-auc-mean']
bst_nb = cv_log.nb.to_dict()[bst_auc]
#train
watchlist  = [(dtrain,'train')]
model = xgb.train(params,dtrain,num_boost_round=bst_nb+50,evals=watchlist)

#predict test set
test_y = model.predict(dtest)
test_result = pd.DataFrame(test_product_no,columns=["product_no"])
test_result["lt92_prob"] = test_y
test_result.to_csv("lt92_prob.csv",index=None,encoding='utf-8')

print bst_nb,bst_auc
