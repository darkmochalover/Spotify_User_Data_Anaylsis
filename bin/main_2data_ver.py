import numpy as np
import pandas as pd
import os
import math



############################################################################################
#       PREPROCESSING  : Load dataset and remove columns don't needed and N/A values.      #
############################################################################################


#### 단일데이터 사용 


# data_path = "data/dataset.csv"
# df = pd.read_csv(data_path)
# cols = [ 'artists', 'album_name', 'track_name',
#                 'popularity',  'explicit', 'danceability', 'energy',
#                 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
#                 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']

# df = df[cols]
# df.dropna(axis=0) # 결측치 제거 (했는데 결측치가 없는 Clean 데이터였음)


#### 2개 데이터 사용 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

main_data_path = "data/dataset.csv"
sub_data_path = "data/data.csv"

#Main Raw Dataframe
df = pd.read_csv(main_data_path)
df.drop(columns='Unnamed: 0',inplace=True)

#Dataframe for getting year feature of songs
dfYear = pd.read_csv(sub_data_path)
dfYear = dfYear[['id','year']]
dfYear['track_id'] = dfYear['id']
dfYear.drop(columns='id',inplace=True)

#Merge 2 Dataframe
df = pd.merge(df,dfYear,on='track_id')

print(df)
print(df.columns)

cols = ['track_id', 'artists', 'album_name', 'track_name', 'popularity',
        'explicit', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'track_genre', 'year']



df = df[cols]

### Duplicate Check
df[df.duplicated('track_id')==True]
print(df[df.duplicated('track_id')==True])      # Duplicate data happened because of the song has 2 or more genres.


# Genre Encoding

# Crosstab Genre and Song

xtab_song = pd.crosstab(
    df['track_id'], 
    df['track_genre']
)

xtab_song = xtab_song*2

print(xtab_song.head(),len(xtab_song))

# Concatenate the encoded genre columns with the original dataframe

dfDistinct = df.drop_duplicates('track_id')
dfDistinct = dfDistinct.sort_values('track_id')
dfDistinct = dfDistinct.reset_index(drop=True)

xtab_song.reset_index(inplace=True)
data_encoded = pd.concat([dfDistinct, xtab_song], axis=1)
print(data_encoded.head(),len(data_encoded))


#######################################
#       Collaborative Filtering       #
#######################################


#######################################
#       Collaborative Filtering       #
#######################################
