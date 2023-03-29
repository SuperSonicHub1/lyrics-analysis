from collections import Counter
from datetime import timedelta
from difflib import SequenceMatcher
from itertools import combinations, pairwise, chain
import json
import re
from statistics import mean, median
from typing import List, NamedTuple, Optional, Tuple, Set, Dict
import nltk
from panphon import FeatureTable
from pathlib import Path
from multiprocessing import Pool
import requests
import logging
import gzip
logging.disable(logging.CRITICAL)

ft = FeatureTable()

lemmatizer = nltk.wordnet.WordNetLemmatizer()
# https://stackoverflow.com/questions/33587667/extracting-all-nouns-from-a-text-file-using-nltk
noun_tags = {"NN", "NNP", "NNS", "NNPS"}

class SpotifyLine(NamedTuple):
	start_time: timedelta
	text: str

class Line:
	line_no: int
	text: str
	# text sans adlib
	line: str
	adlib: Optional[str]

	def __init__(self, line_no: int, text: str):
		self.line_no = line_no
		self.text = text

		original_adlib = Line.find_adlib(text)
		self.adlib = original_adlib.strip() if original_adlib else None
		self.line = (text.replace("(" + original_adlib + ")", "") if self.adlib else text).strip()
	
	@staticmethod
	def find_adlib(text: str) -> Optional[str]:
		return match.group(1) if (match := re.search(r"\((.*)\)", text)) else None
	
	def __repr__(self) -> str:
		return f"Line({self.line_no=!r}, {self.text=!r}, {self.line=!r})"

Stanza = List[Line]
Stanzas = List[Stanza]

class Rhyme(NamedTuple):
	word_a: str
	line_a: Line
	word_b: str
	line_b: Line
	suffix: Tuple[str, ...]

def tag_stanza(stanza: List[Line]):
	return (nltk.pos_tag(nltk.word_tokenize(line.line)) for line in stanza)

def extract_nouns(stanza: List[Line]) -> set[str]:
	tagged_lines = tag_stanza(stanza)
	nouns_in_lines = ((lemmatizer.lemmatize(token[0].lower()) for token in tagged_line if token[1] in noun_tags) for tagged_line in tagged_lines)
	nouns = {noun for nouns in nouns_in_lines for noun in nouns}

	# https://stackoverflow.com/a/20879922
	return nouns

# Superceeded by deepmoji-server

# def penn_to_wn(tag: str):
# 	"""
# 	Convert between the PennTreebank tags and WordNet tags.
# 	https://stackoverflow.com/a/54588366
# 	"""
# 	if tag.startswith('J'):
# 		return wn.ADJ
# 	elif tag.startswith('N'):
# 		return wn.NOUN
# 	elif tag.startswith('R'):
# 		return wn.ADV
# 	elif tag.startswith('V'):
# 		return wn.VERB
# 	else:
# 		return None

# def get_token_sentiment(word: str, tag: str):
# 	"""
# 	https://stackoverflow.com/a/54588366
# 	"""
# 	wn_tag = penn_to_wn(tag)
# 	if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
# 		return None
	
# 	lemma = lemmatizer.lemmatize(word, pos=wn_tag)
# 	if not lemma:
# 		return None

# 	synsets = wn.synsets(word, pos=wn_tag)
# 	if not synsets:
# 		return None
	
# 	# Get the most common sense
# 	synset = synsets[0]

# 	return swn.senti_synset(synset.name())

# def stanza_sentiment(stanza: List[Line]) -> List[float]:
	# tagged_lines = tag_stanza(stanza)
	# sentimental_lines = (
	# 	(get_token_sentiment(word, tag) for word, tag in line)
	# 	for line
	# 	in tagged_lines
	# )
	# average_sentiment_per_line = (
	# 	mean(scores)
	# 	if len(scores := [token.pos_score() - token.neg_score() for token in line if token]) > 0
	# 	else None
	# 	for line
	# 	in sentimental_lines
	# )
	# lines_with_sentiment = [
	# 	score for score in average_sentiment_per_line if score != None 
	# ]

	# return lines_with_sentiment

def stanza_sentiment(stanza: Stanza) -> List[dict]:
	lines = [line.text + (f' ({line.adlib})' if line.adlib else "") for line in stanza if line.text or line.adlib]
	if not lines:
		return []
	res = requests.post("http://localhost:8080/", json=lines)
	res.raise_for_status()
	return res.json()

def syllable_is_stressed(syllable: str) -> bool:
	"""
	From https://www.poetryfoundation.org/learn/glossary-terms/foot:
	A foot usually contains one stressed syllable and at least one unstressed syllable.
	From https://en.wikipedia.org/wiki/Foot_(prosody)#Classical_meter:
	"stressed" = stressed/long syllable, "short" = unstressed/short syllable 
	"""

	# Does the vowel contain a stress or enlongation?
	return 'ˈ' in syllable or  'ˌ' in syllable or 'ː' in syllable
	# The almost-inverse of this would be
	# Does the vowel contain a shortening? (doesn't handled cases where vowel has no modifications)
	# return 'ˑ' in syllable or '̆' in syllable

def stanza_stress(stanza: List[Line]):
	"""
	TODO: Use the grapheme syllabifiers to accurately show where stress is in each line:
	- https://www.nltk.org/api/nltk.tokenize.LegalitySyllableTokenizer.html
	- https://www.nltk.org/api/nltk.tokenize.SyllableTokenizer.html
	"""
	phonemized_stanza = (nltk.word_tokenize(phonemizer(line, lang='en_us')) for line in stanza)
	syllabified_stanza = (
		(
			Pronunciation(word).syllables
			for word
			in line
		)
		for line
		in phonemized_stanza
	)
	stresses = [
		[
			[syllable_is_stressed(syllable) for syllable in word]
			for word
			in line
		]
		for line
		in syllabified_stanza
	]

	return stresses

def is_vowel(phone: str) -> bool:
	return phone[0] in ["A", "E", "I", "O", "U"]

def contains_vowels_cmu(rhyme: Tuple[str, ...]) -> bool:
	return any(is_vowel(phone) for phone in rhyme)

def phonemeize(*args: str):
	res = requests.post("http://localhost:9090/", json=args)
	res.raise_for_status()
	return res.json()

def rhymes(a: str, b: str) -> Optional[Tuple[str, ...]]:
	# When running this under multiprocessing, it recreates the tree every time. :/
	# if a in pronunciations_dict and b in pronunciations_dict:
	# 	rhyme = rhyme_trie.rhymes(a, b)
	# 	return tuple(rhyme) if rhyme and contains_vowels_cmu(rhyme) else None
	# else:
	# Use DeepPhonemeizer
	first_word, second_word = phonemeize(a, b)
	# Reverse words like in rhyming dictionaries to emphazise matches of endings 
	match = SequenceMatcher(None, first_word[::-1], second_word[::-1]).find_longest_match()

	# Match exists and ends of words are the same
	if match.size > 0 and match.a == 0 and match.b == 0:
		possible_rhyme = "".join(a[:match.size].split()[::-1])
		if any(phone.match({"syl": 1}) for phone in ft.word_fts(possible_rhyme)):
			return tuple(possible_rhyme)
		else:
			return None
	else:
		return None

def stanza_rhymes(stanza: List[Line]):

	ending_words = ((line, nltk.word_tokenize(line.line)[-1]) for line in stanza if line.line)
	rhyming_pairs = [
		Rhyme(word_a, line_a, word_b, line_b, suffix)
		for ((line_a, word_a), (line_b, word_b)) in combinations(ending_words, 2)
		if (suffix := rhymes(word_a, word_b))
	]
	return rhyming_pairs

def capital_letters():
	A = ord('A')
	return (chr(i) for i in range(A, A + 26))

def replace_at(base: str, replace: str, index: int):
	return ''.join([base[:index], replace, base[index + 1:]])

def rhyme_structure(rhymes: List[Rhyme], stanza_len: int):
	rhyme_dict: dict[Tuple[str, ...], Set[str]] = {}
	for rhyme in rhymes:
		if rhyme.suffix not in rhyme_dict:
			rhyme_dict[rhyme.suffix] = set()

		rhyme_dict[rhyme.suffix].update({rhyme.line_a.line_no, rhyme.line_b.line_no})
	
	structure_string = "*" * stanza_len
	for (indexes, letter) in zip(rhyme_dict.values(), capital_letters()):
		for index in indexes:
			structure_string = replace_at(structure_string, letter, index)

	return structure_string

def stanza_adlibs(stanza: List[Line]):
	return [(line_no, adlib) for line_no, adlib in parsed_adlibs if adlib]

def word_frequency(stanza: List[Line]):
	tokenized_stanza = (nltk.pos_tag(nltk.word_tokenize(line.line)) + (nltk.pos_tag(nltk.word_tokenize(line.adlib)) if line.adlib else []) for line in stanza)
	
	counter = Counter()

	for line in tokenized_stanza:
		words = (
			lemmatizer.lemmatize(token.lower())
			for (token, tag)
			in line
			if (
				tag.isalpha()
				# https://www.sketchengine.eu/penn-treebank-tagset/
				and (tag.startswith('V') or tag.startswith('R') or tag.startswith('N'))
			)
		)
		counter.update(words)

	return dict(counter)


def open_spotify_lyrics(path: str) -> Stanzas:
	with open(path) as f:
		lines = [
			{
				"start_time": timedelta(milliseconds=int(line['startTimeMs'])),
				"text": line['words']
			}
			for line in 
			json.load(f)['lines']
		]

	return parse_lyrics(lines)

def parse_lyrics(lines: List[SpotifyLine]) -> Stanzas:
	lines_len = len(lines)

	if lines_len == 0:
		return []
	if lines_len == 1:
		return [[Line(0, lines[0]['text'])]]
	

	mean_length = timedelta(
		seconds=mean(
			(right['start_time'] - left['start_time']).total_seconds()
			for left, right
			in pairwise(lines)
		)
	)

	stanzas = []
	new_stanza = []
	line_index = 0

	for i in range(lines_len):
		line = lines[i]

		if not (i == lines_len - 1):
			next_line = lines[i + 1]
			if next_line['start_time'] - line['start_time'] > mean_length * 1.5:
				stanzas.append(new_stanza)
				new_stanza = []
				line_index = 0

		converted_line = Line(line_index, line['text'])
		new_stanza.append(converted_line)
		line_index += 1
	
	stanzas.append(new_stanza)

	return stanzas

# stanzas = open_spotify_lyrics("lyrics/5lq6hpsabgw22xRYPHVV5c.json")


def get(path: Path):
	stanzas = open_spotify_lyrics(path)

	base_dir = Path.cwd() / "parsings"
	song_id = path.stem
	file = base_dir / f"{song_id}.json.gz"

	if file.exists():
		return

	ss_rhymes = [stanza_rhymes(stanza) for stanza in stanzas]

	parsing = dict(
		id=path.stem,
		freqs=[word_frequency(stanza) for stanza in stanzas],
		rhymes=[[(rhyme.word_a, rhyme.word_b) for rhyme in s_rhymes] for s_rhymes in ss_rhymes],
		rhyme_structure=[rhyme_structure(s_rhymes, len(stanza)) for s_rhymes, stanza in zip(ss_rhymes, stanzas)],
		sentiment=[stanza_sentiment(stanza) for stanza in stanzas]
	)

	with gzip.open(file, 'wt', encoding='utf-8') as f:
		json.dump(parsing, f)

def get_word_freq_of_dataset() -> Counter:
	word_freq = Counter()

	with Pool() as p:
		for index, stanza_freqs in enumerate(p.imap_unordered(
			get,
			(Path.cwd() / "lyrics").glob("*.json")
		)):
			print(index)
			for stanza_freq in stanza_freqs:
				word_freq.update(stanza_freq)
	
	return word_freq

def parse() -> dict:
	base_dir = Path.cwd() / "parsings"
	base_dir.mkdir(exist_ok=True)

	with Pool() as p:
		for index, _ in enumerate(p.imap_unordered(
			get,
			(Path.cwd() / "lyrics").glob("*.json")
		)):
			print(index)

	print("Done!")

parse()

# for index, stanza in enumerate(stanzas, 1):
# 	print("Stanza", index)
# 	# print("Nouns:", extract_nouns(stanza))
# 	# print("Sentiments:", stanza_sentiment(stanza))
# 	# print("Stresses:", stanza_stress(stanza))
# 	s_rhymes = stanza_rhymes(stanza)
# 	print("Rhymes:", s_rhymes)
# 	print("Rhyme structure:", rhyme_structure(s_rhymes, len(stanza)))
# 	print("Adlibs:", [line.adlib for line in stanza])
# 	print("Word frequency:", word_frequency(stanza))
