#coding=utf-8

import xgboost as xgb
import pandas as pd
#loading feature file
draft_train = pd.read_csv('../../../data/feature/draft_train.csv')
param_train = pd.read_csv('../../../data/feature/param_train.csv')
tv_train = pd.read_csv('../../../data/feature/tv_features_half_train.csv')
train = pd.merge(draft_train,param_train,on=['product_no','key_index'])
train = pd.merge(tv_train,train,on='product_no',how='left')
train = train[train.key_index>=0.85]

draft_test = pd.read_csv('../../../data/feature/draft_test.csv')
param_test = pd.read_csv('../../../data/feature/param_test.csv')
tv_test = pd.read_csv('../../../data/feature/tv_features_half_test.csv')
test_old = pd.read_csv('../../../data/draft_data_test.csv')[['product_no','draft_param1']]
test = pd.merge(draft_test,param_test,on='product_no')
test = pd.merge(tv_test,test,on='product_no',how='left')
test = pd.merge(test_old,test,on='product_no',how='left')
test.drop('draft_param1',axis=1,inplace=True)

train_y = train.key_index
train_x = train.drop(['key_index','product_no'],axis=1)
test_product_no = test.product_no
test_x = test.drop('product_no',axis=1)

#feature selection
feature_selected = list(pd.read_csv('half_xgb_feature_score.csv').feature)[0:150]
train_x = train_x[feature_selected]
test_x = test_x[feature_selected]

#training xgboost
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

params={'booster':'gbtree',
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'max_depth':8,
	'lambda':200,
	'subsample':0.75,
	'colsample_bytree':0.75,
	'eta': 0.002,#0.005
	'seed':1024,
	'nthread':8
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
test_result["score_half"] = test_y
test_result.to_csv("half_xgb_preds.csv",index=None,encoding='utf-8')

print bst_nb,bst_rmse
