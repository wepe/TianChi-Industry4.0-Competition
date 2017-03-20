# 工业4.0大数据竞赛 - 制造业质量控制

这是天池众智平台上的一道赛题：[链接](https://tianchi.shuju.aliyun.com/outsource/offer/projectdetails.htm?spm=0.0.0.0.4X6BsZ&id=12)，本方案取得了第一名的成绩，解决方案和代码现整理开源。

## 解决方案

[工业4.0大数据竞赛-技术方案总结](https://github.com/wepe/TianChi-Industry4.0-Competition/blob/master/%E5%B7%A5%E4%B8%9A4.0%E5%A4%A7%E6%95%B0%E6%8D%AE%E7%AB%9E%E8%B5%9B-%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88%E6%80%BB%E7%BB%93.pdf)

## 代码说明
### data

存放原始的数据文件，包括：

- 训练数据，`draft_data_train.csv`, `param_data_train.csv`, `timevarying_param_train.csv`
- 旧测试数据，`draft_data_test.csv`, `param_data_test.csv`, `timevarying_param_test.csv`
- 新测试数据，`draft_data_test_new.csv`, `param_data_test_new.csv`, `timevarying_param_test_new.csv`

### 数据分析与特征提取

- `plot_scatter.py`，画散点图

- `feature_extract.py`,提取特征，在`data`目录下生成`feature`目录，存放特征文件


### classification

- `xgb_gt98.py`，训练xgboost分类器，判断`product_no`的`key_index`是否大于0.98
- `xgb_lt92.py`，训练xgboost分类器，判断`product_no`的`key_index`是否小于0.92

### regression

- `old_testset`目录
    - `all`目录
        - `all_xgb.py`，使用了所有特征训练的xgboost，做了特征选择
        - `all_dart.py`，使用了所有特征训练的dart，做了特征选择
    - `half`目录
        - `xgb.py`，使用了加工进度50%之前的特征训练的xgboost
    - `draft`目录
        - `dart.py`，使用`draft_param`特征训练的dart
    - `gen_submission.py`，生成提交文件

- `new_testset`目录
    - `dp_xgb.py`，使用`draft_param`和`param`两种特征训练的xgboost
    - `dp_dart.py`，使用`draft_param`和`param`两种特征训练的dart
    - `dp_rf.py`，使用`draft_param`和`param`两种特征训练的rf
    - `d_xgb.py`,使用`draft_param`特征训练的xgboost
    - `d_rf.py`,使用`draft_param`特征训练的rf
    - `gen_submission.py`，生成提交文件，融合了旧测试数据的结果。根据题目要求，分`draft`、`half`、`all`三种预测。

- `post_process.py`，使用分类模型的预测结果，对回归预测的结果进行后处理

### recommend

- `recommend.py`,推荐三组工艺可调参数的预设值
- `recommend_every_product.py`,针对特定的工艺不可调整参数，对工艺可调参数进行推荐
