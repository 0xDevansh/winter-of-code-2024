# Week 1 submission report

- My research notebook is [on Google Colab](https://colab.research.google.com/drive/1VPrcj25Ie4X7XN7XucYMObkKzKjGpkht?usp=sharing). It includes most of the stuff I tried to implement the model.
- The final pipeline (excluding all research and experimentation) can be found here
- TODO API

# The long approach
- I started by selecting an [appropriate dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). I uploaded the CSV file to Google Drive and synced it with Colab to load the dataset from it using pandas
- I analysed the fields in the data (there was one misspelled column), and checked for missing/NaN values (there were surprisingly none).
- I checked whether `type` had to do with the transaction being fraud. It showed that fraud transactions were only `CASH_OUT` and `TRANSFER`, so I encoded that field using one-hot encoding.
- I also looked into whether accounts (`nameOrig` and `nameDest`) committing fraud were unique. I found that the `nameOrig` of frauds were mostly not in common with those not committing fraud.
  - However, I couldn't figure out how to boil down so many unique account names into a simple feature. If I extracted the `nameOrig` of all frauds into a list and added a yes/no feature to the dataset, this would indirectly mean encoding the `isFraud` column in an indirect way.
  - I also figured out that the same `nameOrig` was highly unlikely to appear in multiple fraud transactions (most of them were from unique accounts). A totally new `nameOrig` could lead to false negatives. This is why I dropped the idea of using that as a feature.
- Then I drew some random plots to figure out some correlation in the data. There were 2 main things I found:
  1) Plotting amount vs oldbalanceOrig gives the fraud transaction in a strange pattern (see the last green-red graph in the research notebook)
