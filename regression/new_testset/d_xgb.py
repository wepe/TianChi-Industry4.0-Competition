#coding=utf-8

import xgboost as xgb
import pandas as pd

#loading feature file
train = pd.read_csv('../../data/feature/draft_train.csv')
train = train[train.key_index>=0.85]
test = pd.read_csv('../../data/feature/draft_test.csv')

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

#training xgboost
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

params={'booster':'gbtree',
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'max_depth':4,
	'lambda':100,
	'subsample':0.65,
	'colsample_bytree':0.7,
	'min_child_weight':9,#8~10
	'eta': 0.05,
	'seed':77,
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
test_result["score_draft"] = test_y
test_result.to_csv("d_xgb_preds.csv",index=None,encoding='utf-8')

print bst_nb,bst_rmse
