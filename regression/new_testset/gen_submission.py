import pandas as pd

#old testset predict result
old_testset = pd.read_csv('../old_testset/predict_result.csv')

#all/half predict result (use only draft_params and params feature)
dp_xgb_preds = pd.read_csv('dp_xgb_preds.csv')
dp_dart_preds = pd.read_csv('dp_dart_preds.csv')
dp_rf_preds = pd.read_csv('dp_rf_preds.csv')

dp_xgb_preds.score_all = 0.4*dp_xgb_preds.score_all + 0.3*dp_dart_preds.score_all + 0.3*dp_rf_preds.score_all
all_pred = dp_xgb_preds[['product_no','score_all']]

dp_xgb_preds['score_half'] = 0.4*dp_xgb_preds.score_all + 0.4*dp_dart_preds.score_all + 0.2*dp_rf_preds.score_all
half_pred = dp_xgb_preds[['product_no','score_half']]

#draft predict result
d_xgb_preds = pd.read_csv('d_xgb_preds.csv')
d_rf_preds = pd.read_csv('d_rf_preds.csv')
d_xgb_preds.score_draft = 0.6*d_xgb_preds.score_draft + 0.4*d_rf_preds.score_draft
draft_pred = d_xgb_preds

submission = pd.merge(draft_pred,half_pred,on='product_no')
submission = pd.merge(submission,all_pred,on='product_no')
submission_rest = pd.merge(submission,old_testset,on='product_no',how='left')
submission_rest.fillna(-999,inplace=True)
submission_rest = submission_rest[submission_rest.score_draft_y==-999]
submission_rest.rename(columns={'score_draft_x':'score_draft','score_half_x':'score_half','score_all_x':'score_all'},inplace=True)
submission_rest = submission_rest[['product_no','score_draft','score_half','score_all']]
submission = pd.concat([submission_rest,old_testset],axis=0)
submission.sort_values(by='score_all',inplace=True)
submission.to_csv('predict_result.csv',index=None)
print submission.describe()
