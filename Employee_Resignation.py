
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB # 高斯朴素贝叶斯 GaussianNB/MultinomialNB/BernoulliNB
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.ensemble import AdaBoostClassifier # AdaBoost
from xgboost import XGBClassifier # XGBoost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
train_data = pd.read_csv('train.csv')
train_y = train_data['Attrition']
train_x = train_data.drop(columns=['user_id', 'Attrition'])
test_data = pd.read_csv('test.csv')
test_x = test_data.drop(columns=['user_id'])
# 数据探索
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

# 显示特征之间的相关系数
features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
            'EmployeeCount', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobRole']
train_features = train_x[features]
plt.figure(figsize=(30, 30))
plt.title('Pearson Correlation between Features', y=1.05, size=15)
train_data_hot_encoded = train_features.drop('BusinessTravel', 1).join(train_features.BusinessTravel.str.get_dummies())
train_data_hot_encoded = train_features.drop('Department', 1).join(train_features.Department.str.get_dummies())
train_data_hot_encoded = train_features.drop('Gender', 1).join(train_features.Gender.str.get_dummies())
train_data_hot_encoded = train_features.drop('JobRole', 1).join(train_features.JobRole.str.get_dummies())
# 计算特征之间的Pearson系数，即相似度
sns.heatmap(train_data_hot_encoded.corr(),linewidths=0.1,vmax=1.0, fmt= '.2f', square=True,linecolor='white',annot=True)
plt.show()

# 特征转换，字符型转成不同列
dvec = DictVectorizer(sparse=False)
train_x = dvec.fit_transform(train_x.to_dict(orient='record'))
test_x = dvec.transform(test_x.to_dict(orient='record'))

# 采用min-max规范化
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
test_x = min_max_scaler.transform(test_x)
print(train_x)

'''
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(train_x, train_y)
print('IDS决策树训练集准确率 %.4lf' % clf.score(train_x, train_y))
print('ID3决策树训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_x, train_y, cv=10)))

#predict_y = clf.predict(test_x)
#print(predict_y)

# 创建LR分类器 0.8784
lr = LogisticRegression(solver='liblinear', multi_class='auto') #数据集比较小，使用liblinear，数据集大使用 sag或者saga
lr.fit(train_x, train_y)
print('LR准确率(基于训练集)： %.4lf' % lr.score(train_x, train_y))
print('LR训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(lr, train_x, train_y, cv=10)))
predict_lr = lr.predict(test_x)
#pd.DataFrame(predict_lr, columns=['predictions']).to_csv('predict_lr.csv')

# 创建线性 CART决策树分类器
model = DecisionTreeClassifier()
model.fit(train_x,train_y)
print('CART决策树准确率(基于训练集)： %.4lf' % model.score(train_x, train_y))
print('CART决策树训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_x, train_y, cv=10)))

# 创建LDA分类器 0.8750
model = LinearDiscriminantAnalysis(n_components=2)
model.fit(train_x,train_y)
print('LDA准确率(基于训练集)： %.4lf' % model.score(train_x, train_y))
print('LDA训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_x, train_y, cv=10)))
predict_lda = model.predict(test_x)
#pd.DataFrame(predict_lda, columns=['predictions']).to_csv('predict_lda.csv')

# 创建贝叶斯分类器
model = GaussianNB()
model.fit(train_x,train_y)
print('贝叶斯准确率(基于训练集)： %.4lf' % model.score(train_x, train_y))
print('贝叶斯训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_x, train_y, cv=10)))

# 创建SVM分类器
model = svm.SVC(kernel='rbf', C=1.0, gamma='auto')
model.fit(train_x,train_y)
print('SVM准确率(基于训练集)： %.4lf' % model.score(train_x, train_y))
print('SVM训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_x, train_y, cv=10)))

# 创建KNN分类器
model = KNeighborsClassifier()
model.fit(train_x,train_y)
print('KNN准确率(基于训练集)： %.4lf' % model.score(train_x, train_y))
print('KNN训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_x, train_y, cv=10)))

# 创建AdaBoost分类器
# 弱分类器
dt_stump = DecisionTreeClassifier(max_depth=5,min_samples_leaf=1)
dt_stump.fit(train_x, train_y)
#dt_stump_err = 1.0-dt_stump.score(test_x, test_y)
# 设置AdaBoost迭代次数
n_estimators=500
model = AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimators)
model.fit(train_x,train_y)
print('AdaBoost准确率(基于训练集)： %.4lf' % model.score(train_x, train_y))
print('AdaBoost训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_x, train_y, cv=10)))

# 创建XGBoost分类器
model = XGBClassifier()
model.fit(train_x,train_y)
print('XGBoost准确率(基于训练集)： %.4lf' % model.score(train_x, train_y))
print('XGBoost训练集cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_x, train_y, cv=10)))

# 使用TPOT自动机器学习工具对Titanic进行分类
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(train_x, train_y)
#print(tpot.score(train_x, train_y))
predict_tpot = tpot.predict(test_x)
pd.DataFrame(predict_tpot, columns=['predictions']).to_csv('predict_tpot.csv')
'''