from difflib import SequenceMatcher
from itertools import combinations
import re
from statistics import mean
from typing import List, NamedTuple, Optional
from dp.phonemizer import Phonemizer
import nltk
from nltk.corpus import cmudict
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import rhyme_trie
from syllabify import Pronunciation

lemmatizer = nltk.wordnet.WordNetLemmatizer()
pronunciations_dict = cmudict.dict()
phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')
# https://stackoverflow.com/questions/33587667/extracting-all-nouns-from-a-text-file-using-nltk
noun_tags = {"NN", "NNP", "NNS", "NNPS"}

with open("the_fever.txt") as f:
	lyrics = f.read()

stanzas = [
	[
		line
		for line
		in stanza.split("\n")
	]
	for stanza
	in lyrics.split("\n\n")
]

def tag_stanza(stanza: list[str]):
	return (nltk.pos_tag(nltk.word_tokenize(line)) for line in stanza)

def extract_nouns(stanza: list[str]) -> set[str]:
	tagged_lines = tag_stanza(stanza)
	nouns_in_lines = ((lemmatizer.lemmatize(token[0].lower()) for token in tagged_line if token[1] in noun_tags) for tagged_line in tagged_lines)
	nouns = {noun for nouns in nouns_in_lines for noun in nouns}

	# https://stackoverflow.com/a/20879922
	return nouns

def penn_to_wn(tag):
	"""
	Convert between the PennTreebank tags and WordNet tags.
	https://stackoverflow.com/a/54588366
	"""
	if tag.startswith('J'):
		return wn.ADJ
	elif tag.startswith('N'):
		return wn.NOUN
	elif tag.startswith('R'):
		return wn.ADV
	elif tag.startswith('V'):
		return wn.VERB
	else:
		return None

def get_token_sentiment(word: str, tag: str):
	"""
	https://stackoverflow.com/a/54588366
	"""
	wn_tag = penn_to_wn(tag)
	if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
		return None
	
	lemma = lemmatizer.lemmatize(word, pos=wn_tag)
	if not lemma:
		return None

	synsets = wn.synsets(word, pos=wn_tag)
	if not synsets:
		return None
	
	# Get the most common sense
	synset = synsets[0]

	return swn.senti_synset(synset.name())

def stanza_sentiment(stanza: list[str]):
	tagged_lines = tag_stanza(stanza)
	sentimental_lines = (
		(get_token_sentiment(word, tag) for word, tag in line)
		for line
		in tagged_lines
	)
	average_sentiment_per_line = [
		mean(token.pos_score() - token.neg_score() if token else 0 for token in line)
		for line
		in sentimental_lines
	]
	return average_sentiment_per_line

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

def stanza_stress(stanza: list[str]):
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

class Rhyme(NamedTuple):
	word_a: str
	line_no_a: int
	word_b: str
	line_no_b: int
	suffix: List[str]

def rhymes(a: str, b: str) -> Optional[List[str]]:
	if a in pronunciations_dict and b in pronunciations_dict:
		return rhyme_trie.rhymes(a, b)
	else:
		# Use DeepPhonemeizer
		first_word = phonemizer(a, lang="en_us")
		second_word = phonemizer(b, lang="en_us")
		# Reverse words like in rhyming dictionaries to emphazise matches of endings 
		match = SequenceMatcher(None, first_word[::-1], second_word[::-1]).find_longest_match()

		# Match exists and ends of words are the same
		if match.size > 0 and match.a == 0 and match.b == 0:
			return a[:match.size].split()[::-1]
		else:
			return 0

def stanza_rhymes(stanza: list[str]):
	"""
	TODO: Prevent (')', ')') from showing up
	TODO: ('devastated', 'chord') is a rhyme?
	TODO: Support internal rhymes: https://en.wikipedia.org/wiki/Internal_rhyme
	"""

	ending_words = ((line_no, nltk.word_tokenize(line)[-1]) for line_no, line in enumerate(stanza))
	rhyming_pairs = [
		Rhyme(word_a, line_no_a, word_b, line_no_b, suffix)
		for ((line_no_a, word_a), (line_no_b, word_b)) in combinations(ending_words, 2)
		if (suffix := rhymes(word_a, word_b))
	]
	return rhyming_pairs

def stanza_adlibs(stanza: list[str]):
	parsed_adlibs = (
		(
			line_no,
			match.group(1) if (match := re.search(r"\((.*)\)", line)) else None
		)
		for line_no, line in
		enumerate(stanza)
	)

	return [(line_no, adlib) for line_no, adlib in parsed_adlibs if adlib]

for index, stanza in enumerate(stanzas, 1):
	print("Stanza", index)
	# print("Nouns:", extract_nouns(stanza))
	# print("Sentiments:", stanza_sentiment(stanza))
	# print("Stresses:", stanza_stress(stanza))
	print("Rhymes:", stanza_rhymes(stanza))
	# print("Adlibs:", stanza_adlibs(stanza))
