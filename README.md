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
Education:3.090254275690527 删除
HourlyRate:6.521479917934158

需要数据可视化，onehot 编码后的查看L2P75重要特征。

尝试高阶组合特征

重写notebook