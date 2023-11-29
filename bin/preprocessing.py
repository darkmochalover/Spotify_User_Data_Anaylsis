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
top_n_list = top_n_list[:5]
print(top_n_list)

df = df[top_n_list]

### Preprocessing 완료
print(df)
print(len(df))
print(df.columns)
################################################################################################3

# Similarity Calculation
# Select the relevant columns for computing item similarities
cosine_sim = cosine_similarity(df, df)

# Output Function
def get_recommendations(title, N=5):
    indices = pd.Series(df.index, index=df['track_name']).drop_duplicates()

    try:
        idx = indices[title]
        try:
            len(idx)
            temp = 2
        except:
            temp = 1
    except KeyError:
        return "Song not found in the dataset."
    
    if temp == 2:
        idx = indices[title][0]
    else:
        idx = indices[title]
    
    # 유사도 점수
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    song_indices = [i[0] for i in sim_scores]
    
    recommended_songs = df[['track_name', 'artists', 'album_name']].iloc[song_indices]

    sim_scores_list = [i[1] for i in sim_scores]
    recommended_list = recommended_songs.to_dict(orient='records')
    for i, song in enumerate(recommended_list):
        song['similarity_score'] = sim_scores_list[i]
    
    return recommended_list

# Get the recommendations
recommended_songs = get_recommendations("Time", N=5)
if isinstance(recommended_songs, str):
    print(recommended_songs)
else:
    print("Recommended Songs:")
    for song in recommended_songs:
        print(f"Title: {song['track_name']}")
        print(f"Artist: {song['artists']}")
        print(f"Album: {song['album_name']}")
        print(f"Similarity Score: {song['similarity_score']:.2f}")
        print()