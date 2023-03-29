from csv import DictReader
from pathlib import Path
import requests
from time import sleep
from pathlib import Path

for path in Path.cwd().glob("*.csv"):
	with open(path) as f:
		songs = {song['spotify'].split("/")[-1] for song in list(DictReader(f))[:5000]}

	songs_len = len(songs)

	base = Path.cwd() / "lyrics"
	base.mkdir(exist_ok=True)

	for index, track_id in enumerate(songs, 1):
		file = base / f"{track_id}.json"
		
		if file.exists():
			print(track_id, "already downloaded -- skipping")
			continue

		print(index, "/", songs_len, "songs:", track_id)
		res = requests.get("https://spotify-lyric-api.herokuapp.com/", params=dict(trackid=track_id))

		try:
			res.raise_for_status()
		except requests.exceptions.HTTPError as e:
			if res.status_code == 404:
				print(track_id, "doesn't have lyrics on Spotify -- skipping")
				continue
			else:
				raise e
		
		file.touch(exist_ok=True)
		file.write_text(res.text)

		sleep(0.5)

