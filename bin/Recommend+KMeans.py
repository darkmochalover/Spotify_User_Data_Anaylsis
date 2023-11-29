import numpy as np
import pandas as pd

from sklearn import preprocessing
from scipy.spatial import distance

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

import os
import math

############################################################################################
#       PREPROCESSING  : Load dataset and remove columns don't needed and N/A values.      #
############################################################################################

# Load data.
data_path = "data/dataset.csv"
df = pd.read_csv(data_path)

# Sorting and cleaning data.
cols = [ 'track_id', 'artists', 'album_name', 'track_name',
                'popularity',  'explicit', 'danceability', 'energy',
                'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']

df = df[cols]


df.drop_duplicates(subset=['track_id'], inplace=True)

df.dropna(axis=0) # 결측치 제거 (했는데 결측치가 없는 Clean 데이터였음)

# Normilizing data.
id_encoder = LabelEncoder()
df['track_id'] = id_encoder.fit_transform(df['track_id'])

genre_encoder = LabelEncoder()
df['track_genre'] = genre_encoder.fit_transform(df['track_genre'])

artists_encoder = LabelEncoder()
df['artists'] = artists_encoder.fit_transform(df['artists'])

album_encoder = LabelEncoder()
df['album_name'] = album_encoder.fit_transform(df['album_name'])

track_encoder = LabelEncoder()
df['track_name'] = track_encoder.fit_transform(df['album_name'])

# Scaling Numerical Features
numerical_features = ['explicit', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Person's coefficient 계산
corr_matrix = df.corr(method='pearson')

# 상관관계가 높은 feature 추출
# threshold 기준으로 자르려고 해봤으나, 1개의 feature만 나와서 top-N을 가져오는 방식 채택
person_corr = abs(corr_matrix['artists'])
top_n_list = person_corr.sort_values(ascending=False).index
top_n_list = top_n_list[:6]
print(top_n_list)


df = df[top_n_list]
new_cols = [x for x in top_n_list if x != 'artists']

## new_df는 인덱스를 artists로 함.
new_df = df
new_df['artists'] = artists_encoder.inverse_transform(df['artists'])
new_df.set_index('artists', inplace=True)

### Preprocessing 완료
print(new_df)
print(len(new_df))
print(new_df.columns)
quit()
################################################################################################3

####################################################################################
#       Collaborative Filtering  : Recommend based on collaborative filtering.     #
####################################################################################

from scipy.spatial import distance

# Choosing some artist
x = list(df.iloc[17])

# data_norm = pd.DataFrame(d, columns=names)
data_result = pd.DataFrame()
data_result['euclidean'] = [distance.euclidean(obj, x) for index, obj in df.iterrows()]
data_result['artists'] = df.index

# Taking 5 value
data_sug = data_result.sort_values(by=['euclidean']).iloc[:6]

# Getting info for the tracks.
data_big = df.set_index(df.loc[:, 'artist'])

artist_list = pd.DataFrame()
for i in list(data_sug.loc[:, 'artist']):
    if i in list(df.loc[:, 'artist']):
        track_info = data_big.loc[[i], ['track_name', 'artists']]
        #track_list = track_list.append(track_info)
        track_list = pd.concat([track_list, track_info], ignore_index=True)
        
recomended = track_list.values.tolist()
print(f"""You've just listened:   {recomended[0][0]} - {recomended[0][1]} 
Now you may listen : 
'{recomended[1][0]} - {recomended[1][1]}'
Or any of:
'{recomended[2][0]} - {recomended[2][1]}' 
'{recomended[3][0]} - {recomended[3][1]}'
'{recomended[4][0]} - {recomended[4][1]}'
'{recomended[5][0]} - {recomended[5][1]}'  """)