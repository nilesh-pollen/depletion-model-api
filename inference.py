import pandas as pd
import numpy as np
import warnings
import joblib
import plotly.express as px
from datetime import date, timedelta
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

model = joblib.load('depletion_model/model.pkl')
train_cols = joblib.load('depletion_model/train_cols.pkl')
X_val = joblib.load('depletion_model/X_val.pkl')


request = X_val.sample(1).values
print('request',request)

predictions = model.predict(request)
print('predictions', predictions)




def post_processing(predictions):
  for i in range(len(predictions)):
    if predictions[i] >= 1:
      predictions[i] = np.random.uniform(low=90, high=100, size=(1,))
    if predictions[i] <= 0:
      predictions[i] = np.random.uniform(low=0, high=10, size=(1,))
  return predictions

predictions = post_processing(predictions)


request_df = pd.DataFrame(request, columns = train_cols)
request_df['scores'] = predictions
print('request_df')


global_brand_ranking = request_df.groupby('brand').scores.mean().reset_index().sort_values(by = 'scores', ascending = False).brand.reset_index(drop = True)

print(global_brand_ranking)




seller_brand_wise_depletion = request_df.groupby(['seller_name', 'brand']).scores.mean().reset_index().sort_values(by = ['seller_name', 'scores'], ascending = False).reset_index(drop = True)

def get_seller_brand_rank(df):
  global rank
  if df['seller_name'] in seller_set:
    rank += 1
    return rank
  else:
    seller_set.add(df['seller_name'])
    rank = 0
    return rank

rank = -1
seller_set = set()
seller_brand_wise_depletion['rank'] = seller_brand_wise_depletion.apply(get_seller_brand_rank, axis = 1)
seller_brand_wise_depletion = seller_brand_wise_depletion.drop('scores', axis = 1)



seller_sku_wise_depletion = request_df.groupby(['seller_name', 'sku_number']).scores.mean().reset_index().sort_values(by = ['seller_name', 'scores'], ascending = False).reset_index(drop = True)

def get_seller_sku_rank(df):
  global rank
  if df['seller_name'] in seller_set:
    rank += 1
    return rank
  else:
    seller_set.add(df['seller_name'])
    rank = 0
    return rank

rank = -1  # resetting rank for seller_sku
seller_set = set()
seller_sku_wise_depletion['rank'] = seller_sku_wise_depletion.apply(get_seller_sku_rank, axis = 1)
seller_sku_wise_depletion = seller_sku_wise_depletion.drop('scores', axis = 1)
