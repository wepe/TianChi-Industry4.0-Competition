import pandas as pd

all_xgb = pd.read_csv('all/all_xgb_preds.csv')
all_dart = pd.read_csv('all/all_dart_preds.csv')
all_xgb.score_all = 0.5*all_xgb.score_all + 0.5*all_dart.score_all
half_xgb = pd.read_csv('half/half_xgb_preds.csv')
draft_xgb = pd.read_csv('draft/draft_dart_preds.csv')

submission = pd.merge(draft_xgb,half_xgb,on='product_no')
submission = pd.merge(submission,all_xgb,on='product_no')

print submission.score_all.describe()
print submission.score_half.describe()
print submission.score_draft.describe()

submission.to_csv('predict_result.csv',index=None)
