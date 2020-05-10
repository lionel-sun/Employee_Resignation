# Employee Resignation Rate Predict 离职率预测
## 收集数据

[Kaggle练习项目](https://www.kaggle.com/c/rs6-attrition-predict "kaggle链接")

- 训练集：train.csv

- 测试集：test.csv

## 特征选择
EmployeeNumber, Over18, Employee取值都一样直接删除掉。

## 数据预处理
- 使用onehot编码将离散型特征值转换为多列，使用lr预测的结果在leaderboard得分0.823
- 将特征值替换为数值0，1，2，3...，使用lr预测在leaderboard得分0.829
- 使用min-max规范化将所有特征值都映射到0-1之间

## 模型训练与预测
本部分说明都是采用离散特征值替换为数值的方法作为处理基础
- LR
使用LR模型默认参数和balanced采样的结果为0.829。
- LightGBM
使用GridSearchCV调整超参数，基于CPU运算。最好结果得分是0.838
