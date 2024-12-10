# Week 1 submission report

- My research notebook is [on Google Colab](https://colab.research.google.com/drive/1VPrcj25Ie4X7XN7XucYMObkKzKjGpkht?usp=sharing). It includes most of the stuff I tried to implement the model.
- The final pipeline (directly generates Logistic regression and XGBoost model) can be found [here](https://colab.research.google.com/drive/11L9gn8RSPUJQL788nF5nCYfvsq4BC4A2?usp=sharing). Run all cells and download the generated `xgb_model.joblib` file for the API.
- Dataset used: [Google Drive link](https://drive.google.com/file/d/1_H0wTtfWyeo3lgA5q4yym4J6OkHjiWE8/view?usp=sharing),  [Kaggle link](https://www.kaggle.com/datasets/ealaxi/paysim1)

- How the pipeline works should be clear by the pipeline notebook comments

# The full approach
- I started by selecting an [appropriate dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). I uploaded the CSV file to Google Drive and synced it with Colab to load the dataset from it using pandas
- I analysed the fields in the data (there was one misspelled column), and checked for missing/NaN values (there were surprisingly none).
- I checked whether `type` had to do with the transaction being fraud. It showed that fraud transactions were only `CASH_OUT` and `TRANSFER`, so I encoded that field using one-hot encoding.
- I also looked into whether accounts (`nameOrig` and `nameDest`) committing fraud were unique. I found that the `nameOrig` of frauds were mostly not in common with those not committing fraud.
  - However, I couldn't figure out how to boil down so many unique account names into a simple feature. If I extracted the `nameOrig` of all frauds into a list and added a yes/no feature to the dataset, this would indirectly mean encoding the `isFraud` column in an indirect way.
  - I also figured out that the same `nameOrig` was highly unlikely to appear in multiple fraud transactions (most of them were from unique accounts). A totally new `nameOrig` could lead to false negatives. This is why I dropped the idea of using that as a feature.
- Then I drew some random plots to figure out some correlation in the data. There were 2 main things I found:
  1) Plotting amount vs oldbalanceOrig gives the fraud transaction in a strange pattern (see Figure 1)
  2) From the log scale plot it is clear that the data varies almost uniformly from 10^1 to 10^8, so linear scaling is not a good option, and taking log base 10 would linearize it pretty well. (see Figure 2)

  ![](https://raw.githubusercontent.com/0xDevansh/winter-of-code-2024/refs/heads/main/machine-learning/week-1/images/strange_relation.png)
  
  Figure 1

  ![](https://raw.githubusercontent.com/0xDevansh/winter-of-code-2024/refs/heads/main/machine-learning/week-1/images/log_scale_graphs.png)
  
  Figure 2

- As decided, I used log scaling, i.e. took the log (base10) for each value in the columns `oldbalanceOrig`, `oldbalanceDest`, `newbalanceOrig`, `newbalanceDest` and `amount`.

- Then, I used SMOTE to generate more samples for the fraud transactions, which were heavily under-represented in the dataset. This led to having 6354407 records for each, fraud and non-fraud.
- Next, I used a min-max scaler to prepare the data for logistic regression, in which case it is better to have uniformly scaled data.
- I finally used sklearn's `LogisticRegression` to train my model. This is surprisingly straightforward, and gave pretty good results.

- I later realised that I was supposed to apply SMOTE only on the training data, not the test data. This gives a very high value for false positives, i.e. a lot of actually genuine transactions are predicted as fraud. The same happens when using random forests or XGBoost.
- As I don't have much time left, I am leaving the model as it is due to lack of time... The final one being XGBoost as it gives better results overall (though there is not much improvement in the false positives)

## Challenges faced
- SMOTE created the problem of generating too many samples, which massively slowed down all training processes. By oversampling using SMOTE, the random forests model took me 2 hours to train.
  - A friend later suggested me to try undersampling, which solved both problems of class imbalance and too many samples.

- One blunder I had committed was applying SMOTE oversampling before the train and test split. This led to a literally perfect model where all scores were 0.99 (AUC, precision, f1 score, etc.). This seemed suspicious, and [I later found out](https://datascience.stackexchange.com/a/15633) that one should over/undersample only the train data, not the test one.
  - On fixing this, I have a lot of false positives in the final model, and I am not able to fix this, as I can't understand what action to take based on the SHAP analysis.

  ## The final (XGBoost) stats
  