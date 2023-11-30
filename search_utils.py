import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Search by artist
def get_discography(df, artist_name):
    discography = df.loc[df.artist == artist_name]
    return discography


# Search by tags
def get_tags(df, tags):
    search = df.loc[df.tags.isin(tags)]
    #search = music_info.tags.str.contains(tags)
    return search


# Search by track_id
def get_song_info(df, id):
    artist = df.loc[df.track_id == id].artist
    name = df.loc[df.track_id == id].name
    info = pd.merge(artist, name, right_index=True, left_index=True)
    return info
