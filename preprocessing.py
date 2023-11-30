import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #graphing and visualizations
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
music_info = pd.read_csv('data/Music_Info.csv')
listening_info = pd.read_csv('data/User_Listening_History.csv')

# 병합
left = music_info.set_index('track_id')
right = listening_info.set_index('track_id')
music_listening_info = left.join(right)

music_listening_info.drop(columns=['genre', 'tags'])
music_listening_info.dropna(inplace=True) # 결측치 제거 

print(music_listening_info)


# genre_encoder = LabelEncoder()
# music_listening_info['genre'] = genre_encoder.fit_transform(music_listening_info['genre'])


# quit()
# Scaling Numerical Features
# 숫자형 열 가져오기
numerical_features = music_listening_info.select_dtypes(include=['float', 'int']).columns.tolist()
# print(numerical_features)
# quit()
numerical_features = ['year', 'duration_ms', 'danceability', 'energy', 'key', 'loudness',
'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
'valence', 'tempo', 'time_signature'] 
scaler = MinMaxScaler()
music_listening_info[numerical_features] = scaler.fit_transform(music_listening_info[numerical_features])

# print(music_listening_info)
# quit()

tag_encoder = LabelEncoder()
music_listening_info['genre'] = tag_encoder.fit_transform(music_listening_info['genre'])


# print(music_listening_info['genre'])
# quit()
###### PREPROCESSING 끝!

feature_cols = ['danceability', 'energy', 'key', 'loudness',
       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'time_signature', 'artist']


'''
Person's coefficient
'''
# # Person's coefficient 계산
df = music_listening_info[feature_cols]
artist_encoder = LabelEncoder()
df['artist'] = artist_encoder.fit_transform(df['artist'])


corr_matrix = df.corr(method='pearson')

# 상관관계가 높은 feature 추출
person_corr = abs(corr_matrix['artist'])
top_n_list = person_corr.sort_values(ascending=False).index
top_n_list = top_n_list[:5]
print(top_n_list)

# music_listening_info = music_listening_info[top_n_list]

# quit()
'''
K-Means
'''

from sklearn.cluster import KMeans

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df)

# X = music_listening_info[feature_cols]
# data = df.loc[:, ['artist']]

# 클러스터링 결과 시각화
df['artist'] = artist_encoder.inverse_transform(df['artist'])
plt.scatter(df[top_n_list[0]], df['artist'], c=kmeans.labels_)
plt.xlabel('numerical_features')
plt.ylabel('artist')
plt.show()


'''
Collaboarative Filltering
'''
# 아티스트별 피처 벡터 생성
# artist_vectors = []
artist_vectors = {}
artist_set = music_listening_info['artist'].unique()
for artist in artist_set:
    artist_df = music_listening_info[music_listening_info['artist'] == artist].reset_index(drop=False)
    
    feature_cols = ['danceability', 'energy', 'key', 'loudness',
       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'time_signature']
    artist_features = artist_df[feature_cols]
    # print(artist_features)
    # # quit()
    # print(artist_features.mean().values.astype(float))
    # print(type(artist_features.mean().values))
    artist_vectors[artist] = list(artist_features.mean().values.astype(float)) # numpy.ndarray

    # quit()

print(artist_vectors)
quit()
# 아티스트 간 유사도 계산
artist_similarity = cosine_similarity(list(artist_vectors.values()))

# 추천 함수
def recommend_artist(artist, n=5):
    index = music_listening_info[music_listening_info['artist'] == artist].index[0]
    similar_indices = artist_similarity[index].argsort()[-n:][::-1]
    similar_artists = [music_listening_info.iloc[i]['artist'] for i in similar_indices]
    return similar_artists

# 예시: 'Queen' 아티스트 추천
print(recommend_artist('RadioHead'))