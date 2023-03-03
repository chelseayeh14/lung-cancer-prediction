""" Libraries Used """
# %%
# 引入各種需要的 packages 和 函式
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras.optimizers import adam_v2

# %%

""" About Dataset"""

"""
GENDER :                M [Male] , F [Female]
AGE :                   Age of patients
SMOKING :               2 [Yes] , 1 [No]
YELLOW_FINGERS :        2 [Yes] , 1 [No]
ANXIETY :               2 [Yes] , 1 [No]
PEER_PRESSURE :         2 [Yes] , 1 [No]
CHRONIC DISEASE :       2 [Yes] , 1 [No]
FATIGUE :               2 [Yes] , 1 [No]
ALLERGY :               2 [Yes] , 1 [No]
WHEEZING :              2 [Yes] , 1 [No]
ALCOHOL CONSUMING :     2 [Yes] , 1 [No]
COUGHING :              2 [Yes] , 1 [No]
SHORTNESS OF BREATH :   2 [Yes] , 1 [No]
SWALLOWING DIFFICULTY : 2 [Yes] , 1 [No]
CHEST PAIN :            2 [Yes] , 1 [No]
LUNG_CANCER :           YES [Positive] , NO [Negative]
"""

""" Data Import """
# 匯入 survey lung cancer.csv 
data = pd.read_csv('survey lung cancer.csv', sep=',', encoding='UTF-8')

""" Basic Exploration """
# print(data.shape)           # Shape of The Dataset
# print(data.head())          # Glimpse of The Dataset
# print(data.info())          # Informations About The Dataset
# print(data.describe())      # Summary of This Dataset
# print(data.isna().sum())    # Checking null value

# Checking duplicate entry & remove
# print(data[data.duplicated()].shape[0])
data.drop_duplicates(keep='first', inplace=True) # keep='first 保留第一個出現, inplace=True 直接在原數據上修改
# print(data.shape[0])   

""" Custom Palette """
sns.set_style('whitegrid')
sns.set_context('poster', font_scale=0.7)
palette = ['#1d7874', '#679289', '#f4c095', '#ee2e31', '#ffb563', '#918450', '#f85e00', '#a41623', '#9a031e', '#d6d6d6', '#ffee32', '#ffd100', '#333533', '#202020']
sns.palplot(sns.color_palette(palette))
# plt.show()

""" Digging Deeper """
# Replace numeric values into categorical 
data_temp = data.copy(deep=True)
data_temp['GENDER'] = data_temp['GENDER'].replace({'M' : 'Male' , 'F' : 'Female'})
for column in data_temp.columns:
    data_temp[column] = data_temp[column].replace({2: 'Yes' , 1 : 'No'})
# print(data_temp.head())

# %%
# Positive Lung Cancer Cases
data_temp_pos = data_temp[data_temp['LUNG_CANCER'] == 'YES']
print((len(data_temp_pos) / len(data_temp)) * 100)
data_temp.LUNG_CANCER.value_counts().plot(kind='pie',figsize=(8, 8),autopct='%1.1f%%')
# print(data_temp_pos.head())

# %%
# Positives Cases Distribution about Age
fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True, sharey=True)
sns.histplot(data_temp_pos[data_temp_pos['GENDER'] == 'Male']['AGE'],color=palette[11], kde=True, ax=ax[0], bins=20, fill=True)
ax[0].lines[0].set_color(palette[12])
ax[0].set_title('\nPositive Male Cases Age Distribution\n', fontsize=20)
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Quantity')

sns.histplot(data_temp_pos[data_temp_pos['GENDER'] == 'Female']['AGE'], color=palette[12], kde=True, ax=ax[1], bins=20, fill=True)
ax[1].lines[0].set_color(palette[11])
ax[1].set_title('\nPositive Female Cases Age Distribution\n', fontsize=20)
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Quantity')

plt.tight_layout()  # 避免兩個子圖重疊
# plt.show()

# Stack together
plt.subplots(figsize=(20, 8))
p = sns.histplot(data=data_temp_pos, x='AGE', hue='GENDER', multiple='stack', palette=palette[11:13], kde=True, shrink=0.99, bins=20, fill=True)
p.axes.lines[0].set_color(palette[11])
p.axes.lines[1].set_color(palette[12])
p.axes.set_title('\nPositive Cases Age Distribution\n', fontsize=20)
plt.ylabel('Count')
plt.xlabel('Age')
# plt.show()

# Positives Cases Distribution about Gender
plt.subplots(figsize=(12, 12))

labels = 'Male', 'Female'
size = 0.5
wedges, texts, autotexts = plt.pie([len(data_temp_pos[data_temp_pos['GENDER'] == 'Male']['GENDER']),
                                    len(data_temp_pos[data_temp_pos['GENDER'] == 'Female']['GENDER'])],
                                    explode=(0, 0),
                                    textprops=dict(size=25, color='white'),
                                    autopct='%.2f%%', 
                                    pctdistance=0.7,
                                    radius=0.9, 
                                    colors=['#0f4c5c', '#FFC300'], 
                                    shadow=True,
                                    wedgeprops=dict(width=size, edgecolor='white', linewidth=5),
                                    startangle=0)
plt.legend(wedges, labels, title='Category', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
plt.title('\nPositive Cases Gender Distribution', fontsize=20)
# plt.show()

# Gender-wise Poistives Cases Reasons
fig, ax = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)

sns.countplot(data=data_temp_pos, x='GENDER', hue='SMOKING', hue_order=['Yes', 'No'], ax=ax[0], palette=['#0f4c5c', '#FFC300'], saturation=1)
ax[0].set_title('\nEffect of Smoking\n', fontsize=20)
ax[0].set_xlabel('Gender')
ax[0].set_ylabel('Quantity')
for container in ax[0].containers:
    ax[0].bar_label(container, label_type='center', padding=2, size=25, color='white', rotation=0)

sns.countplot(data=data_temp_pos, x='GENDER', hue='ALCOHOL CONSUMING', hue_order=['Yes', 'No'], ax=ax[1], palette=['#0f4c5c', '#FFC300'], saturation=1)
ax[1].set_title('\nEffect of Alcohol Consuming\n', fontsize=20)
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('Quantity')
for container in ax[1].containers:
    ax[1].bar_label(container, label_type='center', padding=2, size=25, color='white', rotation=0)

plt.tight_layout()
# plt.show()

# Gender-wise Positives Cases Symptoms
fig, ax = plt.subplots(2, 5, figsize=(12, 8), sharex=False, sharey=True)
loc = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]
col = ['YELLOW_FINGERS', 'ANXIETY', 'CHRONIC DISEASE', 'CHEST PAIN', 'FATIGUE ', 'WHEEZING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'ALLERGY ']
x_title = ['Yellow Fingers', 'Anxiety', 'Chronic Disease', 'Chest Pain', 'Fatigue', 'Wheezing', 'Coughing', 'Shortness of Breath', 'Swallowing Difficulty', 'Allergy']
for i in range(len(loc)):
    graph = ax[loc[i][0], loc[i][1]]
    sns.countplot(data=data_temp_pos, x='GENDER', hue=col[i].upper(), hue_order=['Yes', 'No'], ax=graph, palette=['#0f4c5c', '#FFC300'], saturation=1)
    graph.set_ylabel('Total')
    graph.legend(title=x_title[i], loc='upper right')
    for container in graph.containers:
        graph.bar_label(container, label_type='center', padding=2, size=17, color='white', rotation=0)
plt.tight_layout()
plt.show()

""" Correlation Heatmap """
# Converting 'LUNG_CANCER' column from Categorical to Numerical 
LabelEncoder = LabelEncoder()

data['GENDER'] = data['GENDER'].replace({'M' : 'Male', 'F' : 'Female'})
data['LUNG_CANCER'] = LabelEncoder.fit_transform(data['LUNG_CANCER'])
print(data['LUNG_CANCER']) # Yes => 1, No => 0

# Converting 'GENDER' column from Categorical to Numerical by using "One Hot Encoder" for avoiding unexpected gender bias.
data = pd.get_dummies(data, columns=['GENDER'])
data.rename(columns={'GENDER_Male' : 'MALE', 'GENDER_Female' : 'FEMALE', 'YELLOW_FINGERS' : 'YELLOW FINGERS', 'PEER_PRESSURE' : 'PEER PRESSURE', 'LUNG_CANCER' : 'LUNG CANCER', 'FATIGUE ' : 'FATIGUE', 'ALLERGY ' : 'ALLERGY'}, inplace=True)

# 將 Column 按照我們要的順序排列
data = data[["AGE","MALE","FEMALE","SMOKING","ALCOHOL CONSUMING","CHEST PAIN","SHORTNESS OF BREATH","COUGHING","PEER PRESSURE","CHRONIC DISEASE","SWALLOWING DIFFICULTY","YELLOW FINGERS","ANXIETY","FATIGUE","ALLERGY","WHEEZING","LUNG CANCER"]]
# print(data.head())

# Heatmap
plt.subplots(figsize=(14, 10))
p = sns.heatmap(data.corr(), cmap=palette, square=True, cbar_kws=dict(shrink=0.99),
                annot=True, vmin=-1, vmax=1, linewidths=0.1, linecolor='white',
                annot_kws=dict(fontsize=8))
p.axes.set_title('Pearson Correlation Of Features\n', fontsize=20)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# %%
""" Preprocessing for Classification"""
# Train Test Split
x = data.drop('LUNG CANCER', axis=1)
y = data['LUNG CANCER'] # Yes => 1, No => 0

scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x.std(), x.mean())
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=42)
print('Shape of training data : ', x_train.shape, y_train.shape)
print('Shape of testing data : ', x_test.shape, y_test.shape)

# %%

""" Machine Learning """
# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_conf = confusion_matrix(y_test, lr_pred)
lr_report = classification_report(y_test, lr_pred)
lr_recall = round(recall_score(y_test, lr_pred, average='macro') * 100, ndigits=2)
# print('Confusion Matrix : \n', lr_conf)
# print('Classification Report : \n', lr_report)
# print('The Accuracy of Logistic Regression is :', lr_acc, '%')

# %%
# Gaussian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_pred = gnb.predict(x_test)
gnb_conf = confusion_matrix(y_test, gnb_pred)
gnb_report = classification_report(y_test, gnb_pred)
gnb_recall = round(recall_score(y_test, gnb_pred, average='macro') * 100, ndigits=2)
# print('Confusion Matrix : \n', gnb_conf)
# print('Classification Report : \n', gnb_report)
# print('The Accuracy of Gaussian Naive Bayes is :', gnb_acc, '%')

# Bernoulli Naive Bayes Model
bnb = BernoulliNB()
bnb.fit(x_train, y_train)
bnb_pred = bnb.predict(x_test)
bnb_conf = confusion_matrix(y_test, bnb_pred)
bnb_report = classification_report(y_test, bnb_pred)
bnb_recall = round(recall_score(y_test, bnb_pred, average='macro') * 100, ndigits=2)
# print('Confusion Matrix : \n', bnb_conf)
# print('Classification Report : \n', bnb_report)
# print('The Accuracy of Bernoulli Naive Bayes is :', bnb_acc, '%')

# Support Vector Machine Model
svm = SVC(C=100, gamma=0.002, probability=True)
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
svm_conf = confusion_matrix(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred)
svm_recall = round(recall_score(y_test, svm_pred, average='macro') * 100, ndigits=2)
# print('Confusion Matrix : \n', svm_conf)
# print('Classification Report : \n', svm_report)
# print('The Accuracy of Support Vector Machine is :', svm_acc, '%')

# Random Forest Model
rfg = RandomForestClassifier(n_estimators=100, random_state=42) 
rfg.fit(x_train, y_train)
rfg_pred = rfg.predict(x_test)
rfg_conf = confusion_matrix(y_test, rfg_pred)
rfg_report = classification_report(y_test, rfg_pred)
rfg_recall = round(recall_score(y_test, rfg_pred, average='macro') * 100, ndigits=2)
# print('Confusion Matrix : \n', rfg_conf)
# print('Classification Report : \n', rfg_report)
# print('The Accuracy of Random Forest Classifier is :', rfg_acc, '%')

# K Nearest Neighbors Model
# 找到最佳 k值
k_value_range = range(3, 34)    # 設定欲找尋的k值範圍
# 裝測試結果的平均分數
# k_value_scores = []
# for k in k_value_range:
#     knn_model = KNeighborsClassifier(n_neighbors=k)
#     accuracy = cross_val_score(knn_model, x, y, cv=10, scoring='accuracy')
#     print('K值: ', k)
#     print('Accuracy: ', accuracy.mean())
#     k_value_scores.append(accuracy.mean())
# print(k_value_scores)
# print('最佳K值: ', k_value_scores.index(max(k_value_scores)) + 3) # 找一個最佳的k值，由於k的初始值我們設在3，所以要加三

knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_conf = confusion_matrix(y_test, knn_pred)
knn_report = classification_report(y_test, knn_pred)
knn_recall = round(recall_score(y_test, knn_pred, average='macro') * 100, ndigits=2)
# print('Confusion Matrix : \n', knn_conf)
# print('Classification Report : \n', knn_report)
# print('The Accuracy of K Nearest Neighbors Classifier is :', knn_acc, '%')

# %%
""" Deep Learning """
# Neural Network Architecture
regularization_parameter = 0.003
neural_model = Sequential() # 建立模型

# Add input layer : 確立 input 的格式、要經過幾層處理、每層要做什麼事
neural_model.add(Dense(units=32,    # 表示該層的 hidden layer 要有 32 個neuron
                       input_dim=(x_train.shape[-1]),   # 設定 input 格式
                       activation='relu',               # 激活函數使用 relu function
                       kernel_regularizer=regularizers.l1(regularization_parameter))) # 正則化：避免權重值過大，防止過擬合
neural_model.add(Dense(units=64, activation='relu', kernel_regularizer=regularizers.l1(regularization_parameter)))
neural_model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l1(regularization_parameter)))
neural_model.add(Dropout(0.3)) # 比例是 50%，意思是每輪訓練 10個 input 便隨機去掉3個变量
neural_model.add(Dense(units=16, activation='relu', kernel_regularizer=regularizers.l1(regularization_parameter)))

# Add output layer
neural_model.add(Dense(units=1, activation='sigmoid'))
print(neural_model.summary())

# 以compile函數定義損失函數(loss)、優化函數(optimizer)及成效衡量指標(mertrics)
adam = adam_v2.Adam(epsilon=1e-08) # 模糊因子
neural_model.compile(optimizer=adam,
                     loss='binary_crossentropy', 
                     metrics=[tf.keras.metrics.Recall()])

# 運用 Callbacks 函數監看訓練過程
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): # 在每個epoch结束時調用函數，將資料存在 log{} 裡面
        if(logs.get('recall') == 1.0):
            print('\nRecall Score is 100% so canceling training!')
            self.model.stop_training=True
callbacks = myCallback()

# 以compile函數進行訓練，指定訓練的樣本資料(x, y)，並撥一部分資料作驗證，還有要訓練幾個週期、訓練資料的抽樣方式
history = neural_model.fit(x_train, y_train,    
                           epochs=150,      # 訓練模型的迭代次數
                           verbose=1,       # 顯示模式：1 => 進度條
                           batch_size=64,   # 每次梯度更新的樣本數
                           validation_data=(x_test, y_test), # 用来評估损失，以及在每輪结 束時的任何模型度量指標
                           callbacks=[callbacks]) # 訓練時使用回條函數


neural_pred = neural_model.predict(x_test)
neural_pred = (neural_pred > 0.5)
print(neural_pred)
print(classification_report(y_test, neural_pred))

# Evaluation
scores = neural_model.evaluate(x_test, y_test)
neural_recall = np.round(scores[1] * 100, decimals=2)
print('The Recall Score of Neural Network Architecture is :', neural_recall, '%')

# All Recall Scores
recall_score = dict({'Logistic Regression' : lr_recall, 'Gaussian Naive Bayes Model' : gnb_recall,
                  'Bernoulli Naive Bayes Model' : bnb_recall, 'Support Vector Machine Model' : svm_recall,
                  'Random Forest Model' : rfg_recall, 'K Nearest Neighbors Model' : knn_recall,
                  'Neural Network Architecture' : neural_recall})
print(pd.Series(recall_score))
highest_score = max(recall_score.values())
highest_score_model = max(recall_score, key=recall_score.get)
print('The Highest Recall Score is ', highest_score, '%', 'by', highest_score_model)

# Visualization
x = recall_score.keys()
y = recall_score.values()
plt.bar(range(len(recall_score)), list(y))
plt.xticks(range(len(recall_score)), list(x), rotation=90)
plt.title('Model Recall Scores Comparision')
plt.show()

# %%
recall = history.history["recall"]
val_recall = history.history["val_recall"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(recall)) # number of epochs

plt.figure(figsize=(20, 12))
plt.subplot(2,1,1)
plt.plot(epochs, recall, "yellow", label= "Training Recall")
plt.plot(epochs, val_recall, "black", label= "Validation Recall")
plt.title("Training and validation Recall")
plt.legend()

plt.subplot(2,1,2)
plt.plot(epochs, loss, "yellow", label= "Training Loss")
plt.plot(epochs, val_loss, "black", label= "Validation Loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# %%
""" Application """
# 匯入我們自己的 my_dataset
my_data = pd.read_csv('my_dataset.csv', sep=',', encoding='UTF-8')
print(my_data)
# %%
# Data Preprocessing
my_data['GENDER'] = my_data['GENDER'].replace({'M' : 'Male', 'F' : 'Female'})
my_data2 = pd.get_dummies(my_data, columns=['GENDER'])
print(my_data2)
# %%
# 加入不同性別的欄位，以便後續順利預測
if 'GENDER_Male' in my_data2.columns:
    my_data2.insert(1, 'GENDER_Female', value=0)
else:
    my_data2.insert(1, 'GENDER_Male', value=0)
print(my_data2)
# %%
# 將 columns 重新命名
my_data2.rename(columns={'GENDER_Male' : 'MALE','GENDER_Female' : 'FEMALE','YELLOW_FINGERS' : 'YELLOW FINGERS', 'PEER_PRESSURE' : 'PEER PRESSURE', 'FATIGUE ' : 'FATIGUE', 'ALLERGY ' : 'ALLERGY'}, inplace=True)

# 將 Columns 按照我們要的順序排列好
my_data2 = my_data2[["AGE","MALE","FEMALE","SMOKING","ALCOHOL CONSUMING","CHEST PAIN","SHORTNESS OF BREATH","COUGHING","PEER PRESSURE","CHRONIC DISEASE","SWALLOWING DIFFICULTY","YELLOW FINGERS","ANXIETY","FATIGUE","ALLERGY","WHEEZING"]]
print(my_data2)

# %%
# 標準化
x_new = scaler.transform(my_data2)

# Prediction by Machine Learning
import warnings
warnings.filterwarnings("ignore")
print('lr :', lr.predict_proba(x_new))
print('gnb :', gnb.predict_proba(x_new))
print('bnb :', bnb.predict_proba(x_new))
print('SVM :', svm.predict_proba(x_new))
print('rfg :', rfg.predict_proba(x_new))
print('knn :', knn.predict_proba(x_new))

# Prediction by Deep learning
print('DL :', neural_model.predict(x_new))
# %%