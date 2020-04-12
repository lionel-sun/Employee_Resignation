
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

#加载数据
train_data = pd.read_csv('train.csv')
train_y = train_data['Attrition']
train_x = train_data.drop(columns=['user_id', 'Attrition'])
test_data = pd.read_csv('test.csv')
test_x = test_data.drop(columns=['user_id'])
#数据探索
print('查看数据信息：列名、非空个数、类型等')
print(train_data.info())
print('-'*30)
print('查看数据摘要')
print(train_data.describe())
print('-'*30)
print('查看离散数据分布')
print(train_data.describe(include=['O']))
print('-'*30)
print('查看前5条数据')
print(train_data.head())
print('-'*30)
print('查看后5条数据')
print(train_data.tail())

#特征转换，字符型转成不同列
dvec = DictVectorizer(sparse=False)
train_x = dvec.fit_transform(train_x.to_dict(orient='record'))
test_x = dvec.transform(test_x.to_dict(orient='record'))

#数值型进行规范化
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
test_x = min_max_scaler.transform(test_x)
print(train_x)