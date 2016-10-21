from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import pandas as pd
import numpy as np

#loading feature file
train = pd.read_csv('../../data/feature/draft_train.csv')
train = train[train.key_index>=0.85]
test = pd.read_csv('../../data/feature/draft_test.csv')

#fill nan with train_test.median
test['key_index'] = -999
train_test = pd.concat([train,test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
train_test.fillna(train_test.median(),inplace=True)
train = train_test[train_test.key_index!=-999]
test = train_test[train_test.key_index==-999]

train_y = train.key_index
train_x = train.drop(['key_index','product_no'],axis=1)
test_product_no = test.product_no
test_x = test.drop(['product_no','key_index'],axis=1)

model = RandomForestRegressor(n_estimators=1000,criterion='mse',max_depth=5,max_features=0.75,min_samples_leaf=8,n_jobs=12,random_state=17)#min_samples_leaf: 5~10
scores = cross_val_score(model,train_x.values,train_y.values,cv=5,scoring='mean_squared_error')
print np.sqrt(-scores),np.mean(np.sqrt(-scores))

model.fit(train_x.values,train_y.values)
test_result = pd.DataFrame(test_product_no,columns=["product_no"])
test_result["score_draft"] = model.predict(test_x.values)
test_result.to_csv("d_rf_preds.csv",index=None,encoding='utf-8')
print test_result.describe()
