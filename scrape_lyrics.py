from dotenv import load_dotenv; load_dotenv()
from os import environ
import lyricsgenius
import requests
from time import sleep

def get_top_artists(page: int = 1, per_page: int = 50):
	res = requests.get(
		"https://genius.com/api/artists/chart",
		params=dict(
			time_period="all_time",
			chart_genre="all",
			page=page,
			per_page=per_page,
			text_format="html,markdown"
		)
	)
	res.raise_for_status()
	return res.json()['response']['chart_items']

token = environ['GENIUS_ACCESS_TOKEN']
genius = lyricsgenius.Genius(token)

top_100_artists = [
	artist
	for artist
	in
	(get_top_artists(page=1) + get_top_artists(page=2))
	# Ignore translation "artists"
	if artist['item']['id'] not in {196943, 1507735, 1376254}
]

def get_all_artist_songs(artist_id: int):
	page = 1
	songs = []
	while page:
		request = genius.artist_songs(artist_id, per_page=50, page=page)
		songs.extend(request['songs'])
		page = request['next_page']
		sleep(0.5)

	return songs

def get_all_song_lyrics(song_urls: list[str]):
	lyrics = []
	for url in song_urls:
		lyrics.append(
			genius.lyrics(song_url=url, remove_section_headers=True)
		)
		sleep(3)
	return lyrics

for index, artist in enumerate(top_100_artists):
	artist_name = artist['item']['name']
	artist_id = artist['item']['id']

	songs = get_all_artist_songs(artist_id)
	lyrics = get_all_song_lyrics(
		[
			"https://genius.com" + song['path'] for song in songs
		]
	)
	print(lyrics[0])
	break
