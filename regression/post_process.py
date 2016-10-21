import pandas as pd


#lt92 prob
lt92 = pd.read_csv('../classification/lt92_prob.csv')
lt92.sort_values(by='lt92_prob',inplace=True)
lt92_15 = lt92.tail(15)

#gt98 prob
gt98 = pd.read_csv('../classification/gt98_prob.csv')
gt98.sort_values(by='gt98_prob',inplace=True)
gt98_20 = gt98.tail(20)

# post process regression predict result
predict_result = pd.read_csv('new_testset/predict_result.csv')

predict_result = pd.merge(predict_result,lt92_15,on='product_no',how='left')
predict_result.fillna(-999,inplace=True)
predict_result = predict_result[predict_result.lt92_prob==-999]
predict_result = predict_result[['product_no','score_draft','score_half','score_all']]

predict_result = pd.merge(predict_result,gt98_20,on='product_no',how='left')
predict_result.fillna(-999,inplace=True)
predict_result = predict_result[predict_result.gt98_prob==-999]
predict_result = predict_result[['product_no','score_draft','score_half','score_all']]

lt92_15['score_draft'] =0.92
lt92_15['score_half'] = 0.92
lt92_15['score_all'] = 0.92
lt92_15.drop('lt92_prob',axis=1,inplace=True)

gt98_20['score_draft'] = 0.98
gt98_20['score_half'] = 0.98
gt98_20['score_all'] = 0.98
gt98_20.drop('gt98_prob',axis=1,inplace=True)

#generate submission file
submission = pd.concat([predict_result,lt92_15,gt98_20],axis=0)
submission.sort_values(by='product_no',inplace=True)
submission.to_csv('final_predict_result.csv',index=None)
print submission.describe()



