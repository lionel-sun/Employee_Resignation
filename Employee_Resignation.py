
from sklearn import preprocessing
import pandas as pd

#加载数据
train_data = pd.read_csv('train.csv')
train_y = train_data['Attrition']
train_x = train_data.drop(columns=['user_id', 'Attrition'])
#数值型进行规范化
min_max_scaler = preprocessing.MinMaxScaler()
train_x1 =min_max_scaler.fit_transform(train_x[['Age', 'DailyRate']])
print(train_x)