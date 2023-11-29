import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #graphing and visualizations
import os

# Load dataset
music_info = pd.read_csv('data/Music Info.csv')
listening_info = pd.read_csv('data/User Listening History.csv')

# 병합
left = music_info.set_index('track_id')
right = listening_info.set_index('track_id')
music_listening_info = left.join(right)

artist_df = music_listening_info
artist_df = artist_df.dropna(axis=0) # 결측치 제거 

print(artist_df)
print(artist_df.columns)
quit()

###### PREPROCESSING 끝!


# Search by artist
def get_discography(artist_name):
    discography = music_info.loc[music_info.artist == artist_name]
    return discography

search = get_discography('Radiohead')
print(search.head())



# Search by tags
def get_tags(tags):
    search = music_info.loc[music_info.tags.isin(tags)]
    #search = music_info.tags.str.contains(tags)
    return search

search = get_tags(['electronic', 'alternative'])
print(search.head())



# Search by track_id
def get_song_info(id):
    artist = music_info.loc[music_info.track_id == id].artist
    name = music_info.loc[music_info.track_id == id].name
    info = pd.merge(artist, name, right_index=True, left_index=True)
    return info

track_id = 'TRLNZBD128F935E4D8'
get_song_info(track_id)