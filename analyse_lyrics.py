import json
import gzip
from csv import DictReader
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import islice, groupby, chain
from operator import itemgetter
from multiprocessing import Pool
from collections import Counter, defaultdict
from pprint import pprint
import nltk
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
import matplotlib, mplcairo
import re

matplotlib.use("module://mplcairo.gtk")

lemmatizer = nltk.wordnet.WordNetLemmatizer()

parsings_dir = Path.cwd() / "parsings"

word_lists = {
	path.stem: [lemmatizer.lemmatize(word.strip()) for word in path.read_text("utf-8").splitlines()]
	for path
	in (Path.cwd() / "words").glob("*.txt")
}

# https://github.com/bfelbo/DeepMoji/blob/master/emoji_overview.png
# https://github.com/bfelbo/DeepMoji/blob/master/emoji_unicode.csv
# emoji_categories = dict(
# 	humor=set('😏😂😅🙈😋😉💀😜😈💁'),
# 	love={'😍', '❤', '😳', '💕', '😘', '♥', '💔', '♡', '💜', '💖', '💙', },
# 	music={'🎶', '🎧', },
# 	positive={'👌', '😊', '😁', '💯', '😌', '☺', '🙌', '🙏', '✌', '😎', '👍', '👏', '👀', '😄', '💪', '👊', '✨', },
# 	neutral={'😴', '😐', '✋', },
# 	negative={'😩', '😭', '😔', '😑', '😕', '😞', '😫', '😢', '😪', '😷', '🔫', '😣', '😓', '🙊', '😖', '🙅', '😬', },
# 	anger={'😒', '😡', '😤', '😠', },
# )

def detect_rhyme_scheme(scheme: str):
	if not scheme:
		return None
	
	# monorhyme: /A+/
	if len(set(scheme)) == 1:
		return 'monorhyme'

	# enclosed rhyme: ABBA
	if re.search(r'(?P<outer>\w)(?P<inner>\w)(?P=inner)(?P=outer)', scheme):
		return 'enclosed'
	
	# alternating rhyme: ABAB
	if re.search(r"(?P<first>\w)(?P<second>\w)(?P=first)(?P=second)", scheme):
		return 'alternating'

	# clumped rhyme: ABAB
	if re.search(r"(?P<first>\w)(?P=first)(?P<second>\w)(?P=second)", scheme):
		return 'clumped'

def extract_id(url: str):
	return url.split("/")[-1]

# Sources
def top_songs():
	f = open("top_songs.csv")

	try:
		yield from DictReader(f)
	finally:
		f.close()

def top_blues():
	f = open("top_blues.csv")
	try:
		yield from islice(DictReader(f), 5000)
	finally:
		f.close()

def top_pop():
	f = open("top_pop.csv")
	try:
		yield from islice(DictReader(f), 5000)
	finally:
		f.close()

def top_rock():
	f = open("top_rock.csv")
	try:
		yield from islice(DictReader(f), 5000)
	finally:
		f.close()

def top_rap():
	f = open("top_rap.csv")
	try:
		yield from islice(DictReader(f), 5000)
	finally:
		f.close()

source = sorted(top_songs(), key=itemgetter('release_year'))
# By year
# grouping = { k: list(v) for k, v in groupby(source, itemgetter('release_year')) }
# By decade
grouping = { k: list(v) for k, v in groupby(source, lambda x: x['release_year'][2] + '0s' if x['release_year'] else '') }

def load_word_frequencies(path: Path):
	with gzip.open(path, 'rt', encoding="utf-8") as f:
		parsing = json.load(f)
		return parsing['freqs']

def load_word_frequencies_filter_by_category(path: Path):
	with gzip.open(path, 'rt', encoding="utf-8") as f:
		parsing = json.load(f)
		stanzas_freqs = parsing['freqs']

	return (
		[ 
			{
				category: { k: v for k, v in stanza_freqs.items() if k in word_list }
				for category, word_list
				in word_lists.items()
			}
			for stanza_freqs
			in stanzas_freqs
		],
		sum(Counter(stanza_freqs).total() for stanza_freqs in stanzas_freqs),
	)

def load_rhyme_frequencies(path: Path):
	with gzip.open(path, 'rt', encoding="utf-8") as f:
		parsing = json.load(f)
		return Counter(frozenset(rhyme) for rhyme in chain.from_iterable(parsing['rhymes']))

def load_rhyme_schemes(path: Path):
	with gzip.open(path, 'rt', encoding="utf-8") as f:
		parsing = json.load(f)
		return Counter(detect_rhyme_scheme(rhyme_scheme) for rhyme_scheme in parsing['rhyme_structure'])

def frequency_by_group():
	for group, songs in grouping.items():
		if not group:
			continue

		print(group)
		track_ids = {extract_id(song['spotify']) for song in songs}

		word_freq = Counter()

		with Pool() as p:
			for index, stanza_freqs in enumerate(p.imap_unordered(
				load_word_frequencies,
				(file for track_id in track_ids if (file := parsings_dir / f"{track_id}.json.gz").exists())
			)):
				print(index)
				for stanza_freq in stanza_freqs:
					word_freq.update(stanza_freq)
		
		try:
			wc = WordCloud(width=1920*2, height=1080*2)
			wc.generate_from_frequencies(word_freq)
			wc.to_file(f"output/word_freq/{group} ({len(track_ids)} songs).png")
		except ValueError:
			print("No lyrics for this group. :(")

def category_frequency_by_group():
	storage = {}

	for group, songs in grouping.items():
		if not group or group in ['40s', '50s']:
			continue

		print(group)
		track_ids = {extract_id(song['spotify']) for song in songs}

		frequencies = {
			'total': 0,
			**{
				category: Counter()
				for category
				in word_lists.keys()
			}
		}

		with Pool() as p:
			for index, (stanza_freqs, total) in enumerate(p.imap_unordered(
				load_word_frequencies_filter_by_category,
				(file for track_id in track_ids if (file := parsings_dir / f"{track_id}.json.gz").exists())
			)):
				print(index)
				frequencies['total'] += total
				for stanza_freq in stanza_freqs:
					for category, counts in stanza_freq.items():
						frequencies[category].update(counts)
		
		storage[group] = frequencies

	percentages = {
		group: {
			category: freqs.total() / frequencies['total'] * 100
			for category, freqs
			in frequencies.items()
			if category != 'total'
		}
		for group, frequencies
		in storage.items()
	}

	fig, ax = plt.subplots()
	ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	for category in word_lists.keys():
		ax.plot(percentages.keys(), [percs[category] for percs in percentages.values()], label=category)

	ax.set_title("Word frequency by category over time")
	ax.set_xlabel("Decade")
	ax.set_ylabel("Percentage of corpus")
	ax.legend()

	fig.set_size_inches(11, 8.5)
	fig.savefig("output/frequency_by_category.png")

def most_common_rhymes():
	track_ids = {extract_id(song['spotify']) for song in source}

	rhyme_freqs = Counter()

	with Pool() as p:
		for index, rhyme_freq in enumerate(p.imap_unordered(
			load_rhyme_frequencies,
			(file for track_id in track_ids if (file := parsings_dir / f"{track_id}.json.gz").exists())
		)):
			# print(index)
			rhyme_freqs.update(rhyme_freq)

	print("Top 100 rhymes:")
	pprint(rhyme_freqs.most_common(100))

def load_sentiments(path: Path):
	with gzip.open(path, 'rt', encoding="utf-8") as f:
		parsing = json.load(f)
		sentiments = list(chain.from_iterable(parsing['sentiment']))

		counter = Counter()
		for line in sentiments:
			counter.update(Counter({ k: v * 100 for k, v in line.items() }))

		return (
			counter,
			len(sentiments) * 100,
		)

def sentiment_by_group():
	storage = {}

	for group, songs in grouping.items():
		if not group or group in ['40s', '50s']:
			continue

		print(group)
		track_ids = {extract_id(song['spotify']) for song in songs}

		frequencies = {
			'total': 0,
			'sentiments': Counter()
		}

		with Pool() as p:
			for index, (sentiments, total) in enumerate(p.imap_unordered(
				load_sentiments,
				(file for track_id in track_ids if (file := parsings_dir / f"{track_id}.json.gz").exists())
			)):
				frequencies['sentiments'].update(sentiments)
				frequencies['total'] += total

		storage[group] = frequencies

	percentages_ungrouped = {
		group: {
			category: freqs / sentiments['total'] * 100
			for category, freqs
			in sentiments['sentiments'].items()
		}
		for group, sentiments
		in storage.items()
	}

	percentages = {
		group: {
			emoji_category: sum(sentiments[emoji] for emoji in emojis)
			for emoji_category, emojis
			in emoji_categories.items()
		}
		for group, sentiments
		in percentages_ungrouped.items()
	}

	fig, ax = plt.subplots()
	ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	for emotion in percentages[list(storage.keys())[0]].keys():
		# I think this might be due to music note emojis
		# being included in lyrics to signify an instrumental
		# section...
		# if emotion in ['🎶', '🎧']:
		if emotion == 'music':
			continue
		ax.plot(percentages.keys(), [percs[emotion] for percs in percentages.values()], label=emotion)

	ax.set_title("Sentiment over time")
	ax.set_xlabel("Decade")
	ax.set_ylabel("Percentage of corpus")
	ax.legend()

	fig.set_size_inches(11, 8.5)
	fig.savefig("output/sentiment.png")

def most_common_rhyme_schemes():
	storage = {}

	for group, songs in grouping.items():
		if not group or group in ['40s', '50s']:
			continue

		print(group)
		track_ids = {extract_id(song['spotify']) for song in songs}

		rhyme_schemes_count = Counter()

		with Pool() as p:
			for index, rhyme_schemes in enumerate(p.imap_unordered(
				load_rhyme_schemes,
				(file for track_id in track_ids if (file := parsings_dir / f"{track_id}.json.gz").exists())
			)):
				rhyme_schemes_count.update(rhyme_schemes)

		storage[group] = rhyme_schemes_count

	percentages = {
		group: {
			scheme_type: amount / schemes.total() * 100
			for scheme_type, amount
			in schemes.items()
			if scheme_type
		}
		for group, schemes
		in storage.items()
	}
	
	fig, ax = plt.subplots()
	ax.yaxis.set_major_formatter(mtick.PercentFormatter())
	for scheme in percentages[list(storage.keys())[0]].keys():
		ax.plot(percentages.keys(), [percs[scheme] for percs in percentages.values()], label=scheme)
	
	ax.set_title("Rhyme schemes over time")
	ax.set_xlabel("Decade")
	ax.set_ylabel("Percentage of corpus")
	ax.legend()

	fig.set_size_inches(11, 8.5)
	fig.savefig("output/rhyme_schemes.png")

# frequency_by_group()
# category_frequency_by_group()
# most_common_rhymes()
# sentiment_by_group()
most_common_rhyme_schemes()

# def gen_sentiment_histogram():
# 	sentiments = list(chain.from_iterable(stanza_sentiment(stanza) for stanza in stanzas))
# 	plt.title("Sentiment of 'The Fever'")
# 	axes = plt.subplot()
# 	axes.hist(sentiments, bins=50)
# 	plt.show()
