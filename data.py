import numpy as np
import pandas as pd

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

# 輸入我們自己的資訊
data = pd.read_csv('survey lung cancer.csv', sep=',', encoding='UTF-8')

dict1 = {'Male': 'M', 'Female': 'F', 'Yes': 2, 'No': 1, 'Positive': 'Yes', 'Negative': 'No'}
dict2 = {}

while True:
    answer = input('請輸入性別 Male/Female : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['GENDER'] = dict1[answer]
        break

while True:
    answer = input('請輸入年齡 : ')
    if answer.isdigit() == False:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['AGE'] = int(answer)
        break

while True:
    answer = input('是否抽煙 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:    
        dict2['SMOKING'] = dict1[answer]
        break

while True:
    answer = input('是否有黃指甲現象 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['YELLOW_FINGERS'] = dict1[answer]
        break

while True:
    answer = input('是否有焦慮現象 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['ANXIETY'] = dict1[answer]
        break

while True:
    answer = input('是否有同儕壓力 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['PEER_PRESSURE'] = dict1[answer]
        break

while True:
    answer = input('是否有慢性病 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['CHRONIC DISEASE'] = dict1[answer]
        break

while True:
    answer = input('是否疲勞 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['FATIGUE'] = dict1[answer]
        break

while True:
    answer = input('是否過敏 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['ALLERGY'] = dict1[answer]
        break

while True:
    answer = input('是否呼吸困難 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['WHEEZING'] = dict1[answer]
        break

while True:
    answer = input('是否酗酒 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['ALCOHOL CONSUMING'] = dict1[answer]
        break

while True:
    answer = input('是否咳嗽 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['COUGHING'] = dict1[answer]
        break

while True:
    answer = input('是否呼吸急促 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['SHORTNESS OF BREATH'] = dict1[answer]
        break

while True:
    answer = input('是否吞嚥困難 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['SWALLOWING DIFFICULTY'] = dict1[answer]
        break

while True:
    answer = input('是否胸痛 Yes/No : ')
    if answer not in dict1:
        print('輸入錯誤，請重新輸入')
        continue
    else:
        dict2['CHEST PAIN'] = dict1[answer]
    break

print(dict2)

# 將 dict2 轉換為 DataFrame
my_data = pd.DataFrame(dict2, index=[0])
print(my_data)

# 將製作好的 DataFrame 輸出為 CSV 檔案
my_data.to_csv('my_dataset.csv', index=True)