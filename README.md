# Employee Resignation Rate Predict 离职率预测
回归问题（不是分类问题）
## 收集数据

[Kaggle练习项目](https://www.kaggle.com/c/rs6-attrition-predict "kaggle链接")

- 训练集：train.csv

- 测试集：test.csv

## 特征选择


todo list:
去掉EmployeeNumber员工号码和预测无关

去掉Over 18（超过18岁）都是一样的值StandardHours employee count


需要数据可视化，onehot 编码后的查看L2P75重要特征。

尝试高阶组合特征

重写notebook

分类器模型输出的是predict_proba(是该分类的概率所有值非负，线性输出结果)

重写两种方法然后进行预测值