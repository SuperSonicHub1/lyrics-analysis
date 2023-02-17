from dotenv import load_dotenv; load_dotenv()
from os import environ
from requests import Session
from typing import Optional

key = environ['MUSIXMATCH_API_KEY']

BASE_URL = "https://api.musixmatch.com/ws/1.1/"

class Musixmatch:
	session: Session

	def __init__(self):
		self.session = Session()

	def call_api(self, path: str, **kwargs):
		res = self.session.get(BASE_URL + path, params=dict(apikey=key, **kwargs))
		res.raise_for_status()

		message = res.json()['message']
		assert message['header']['status_code'] == 200
		return message['body']
	
	def chart_artists_get(self, country: str = 'us', page: int = 1, page_size: int = 100):
		body = self.call_api("chart.artists.get", country=country, page=page, page_size=page_size)
		return body['artist_list']
	
	def artists_albums_get(self, artist_id: str, page: int = 1, page_size: int = 100):
		body = self.call_api("artist.albums.get", artist_id=artist_id, page=page, page_size=page_size)
		return body['album_list']
	
	def album_tracks_get(self, album_id: str, page: int = 1, page_size: int = 100):
		body = self.call_api("album.tracks.get", album_id=album_id, page=page, page_size=page_size)
		return body['track_list']

	def track_lyrics_get(self, track_id: str, page: int = 1, page_size: int = 100):
		body = self.call_api("track.lyrics.get", track_id=track_id, page=page, page_size=page_size)
		return body['lyrics']

if __name__ == '__main__':
	from pprint import pprint as print

	api = Musixmatch()
	# for i in range
	top_artists = api.chart_artists_get()
	for artist in top_artists:
		artist = artist['artist']
		print(artist['artist_name'])
		albums = api.artists_albums_get(artist['artist_id'])
		for album in albums:
			album = album['album']
			print(album['album_name'])
			tracks = api.album_tracks_get(album['album_id'])
			for track in tracks:
				track = track['track']
				print(track['track_name'])
				lyrics = api.track_lyrics_get(track['track_id'])
				print(lyrics['lyrics_body'])
				break


