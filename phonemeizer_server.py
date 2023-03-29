from flask import Flask, request, jsonify
from dp.phonemizer import Phonemizer
from functools import cache

phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')

@cache
def phonemize(text: str) -> str:
	return phonemizer(text, lang="en_us")

app = Flask(__name__)

@app.get("/")
def index_get():
	return "Send a POST request to this URL with a JSON array of texts."

@app.post("/")
def index_post():
	texts = request.json
	return jsonify([phonemize(text) for text in texts])

app.run('0.0.0.0', 9090)
