#coding=utf-8

import xgboost as xgb
import pandas as pd
import numpy as np

#loading feature file
train = pd.read_csv('../../../data/feature/draft_train.csv')
train = train[train.key_index>=0.85]
test = pd.read_csv('../../../data/feature/draft_test.csv')
test_old = pd.read_csv('../../../data/draft_data_test.csv')[['product_no','draft_param1']]
test = pd.merge(test_old,test,on='product_no',how='left')
test.drop('draft_param1',axis=1,inplace=True)
test['key_index'] = -999

train_test = pd.concat([train,test],axis=0)
draft_param5_dummies = pd.get_dummies(train_test.draft_param5)
draft_param5_dummies.columns = ['dp5_'+str(i) for i in range(draft_param5_dummies.shape[1])]

draft_param9_dummies = pd.get_dummies(train_test.draft_param9)
draft_param9_dummies.columns = ['dp9_'+str(i) for i in range(draft_param9_dummies.shape[1])]

train_test = pd.concat([train_test,draft_param5_dummies],axis=1)
train_test = pd.concat([train_test,draft_param9_dummies],axis=1)

train = train_test[train_test.key_index!=-999]
test = train_test[train_test.key_index==-999]

train_y = train.key_index
train_x = train.drop(['key_index','product_no'],axis=1)
test_product_no = test.product_no
test_x = test.drop(['key_index','product_no'],axis=1)

#training xgboost
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

params={'booster':'dart',
	'eval_metric': 'rmse',
	"objective":"reg:linear",
	'max_depth':6,
	'lambda':200,
	'subsample':0.65,
	'colsample_bytree':0.65,
	'eta': 0.01,
	'sample_type':'uniform',
	'normalize':'forest',
	'rate_drop':0.2,
	'skip_drop':0.8,
	'seed':1024,
	'nthread':12
	}

watchlist  = [(dtrain,'train')]

#通过cv找最佳的nround
cv_log = xgb.cv(params,dtrain,num_boost_round=25000,nfold=5,early_stopping_rounds=50,seed=1024)
bst_rmse= cv_log['test-rmse-mean'].min()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-rmse-mean']
bst_nb = cv_log.nb.to_dict()[bst_rmse]

watchlist  = [(dtrain,'train')]
model = xgb.train(params,dtrain,num_boost_round=bst_nb+50,evals=watchlist)

#predict test set
test_y = model.predict(dtest)
test_result = pd.DataFrame(test_product_no,columns=["product_no"])
test_result["score_draft"] = test_y
test_result.to_csv("draft_dart_preds.csv",index=None,encoding='utf-8')

print bst_nb,bst_rmse
