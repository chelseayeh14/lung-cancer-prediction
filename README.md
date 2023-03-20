# Lung Cancer Prediction in Insurance based on Machine Learning

這份專案以肺癌作為疾病標的，選用七種不同的機器學習演算法預測一個人的肺癌的機率，並與肺癌相關險種做結合，以家族病史、居住地點、家庭責任等其他因素進行微調，產出不同方案的險種規劃。

## Purpose

希望可以解決保險銷售人員醫療知識不足的痛點😊

## Procedure

首先，使用 UCI Machine Learning Repository 上的 dataset 進行資料處理，包含刪除重複值、利用 One-Hot Encoding & Label Encoding 把分類變數變成數值，以及最後的資料標準化。  

接著，將資料分為 80% 的訓練集及 20% 的測試集，利用以下七種不同的機器學習演算法，並選擇 Recall 召回率最高的人工神經網路模型與後續的保險應用結合。

![image](model.png)

另外，我也在規劃中加入三種變數進行方案微調，分別為家族病史、居住地點、家庭責任，希望能更加貼近保險實務。  

最後，設計出三種不同的保險方案，可以作為保險銷售人員規劃保單的依據。
