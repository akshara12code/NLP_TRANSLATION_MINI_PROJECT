"""
NLP Language Translator using NLTK
====================================
NLP Mini Project

What NLP is happening here:
  1. Language Detection     - detect source language from character n-grams
  2. Tokenization           - split text into word/sentence tokens (NLTK)
  3. Stopword Removal       - filter noise words (NLTK stopwords corpus)
  4. POS Tagging            - tag each token: noun, verb, adjective, etc.
  5. Named Entity Recognition (NER) - find people, places, orgs in text
  6. Text Normalization     - lowercase, punctuation cleanup
  7. Translation            - send cleaned text to MyMemory API
  8. NLP Summary Report     - printed after every translation

Install dependencies:
    pip install nltk requests langdetect flask
"""

import re
import requests
import nltk
from langdetect import detect, LangDetectException

# ── Download required NLTK data (runs once) ─────────────────────────────────
print("Starting NLP Translator...", flush=True)
print("Downloading NLTK data (first run only)...", flush=True)

NLTK_PACKAGES = [
    "punkt", "punkt_tab", "stopwords",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab", "words"
]

for pkg in NLTK_PACKAGES:
    try:
        nltk.download(pkg, quiet=True)
        print(f"  ✓ {pkg}", flush=True)
    except Exception as e:
        print(f"  ✗ {pkg} failed: {e}", flush=True)

print("NLTK data ready.\n", flush=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

# ── Language menu ────────────────────────────────────────────────────────────
LANGUAGES = {
    "1":  ("English",              "en"),
    "2":  ("Hindi",                "hi"),
    "3":  ("French",               "fr"),
    "4":  ("German",               "de"),
    "5":  ("Spanish",              "es"),
    "6":  ("Italian",              "it"),
    "7":  ("Portuguese",           "pt"),
    "8":  ("Russian",              "ru"),
    "9":  ("Chinese (Simplified)", "zh"),
    "10": ("Japanese",             "ja"),
    "11": ("Korean",               "ko"),
    "12": ("Arabic",               "ar"),
    "13": ("Turkish",              "tr"),
    "14": ("Bengali",              "bn"),
    "15": ("Marathi",              "mr"),
    "16": ("Tamil",                "ta"),
    "17": ("Urdu",                 "ur"),
    "18": ("Dutch",                "nl"),
    "19": ("Polish",               "pl"),
    "20": ("Greek",                "el"),
}

POS_DESCRIPTIONS = {
    "NN": "Noun", "NNS": "Noun (plural)", "NNP": "Proper Noun",
    "NNPS": "Proper Noun (plural)", "VB": "Verb (base)",
    "VBD": "Verb (past)", "VBG": "Verb (gerund)",
    "VBN": "Verb (past participle)", "VBP": "Verb (present)",
    "VBZ": "Verb (3rd person)", "JJ": "Adjective",
    "JJR": "Adjective (comparative)", "JJS": "Adjective (superlative)",
    "RB": "Adverb", "RBR": "Adverb (comparative)",
    "RBS": "Adverb (superlative)", "PRP": "Pronoun",
    "PRP$": "Possessive Pronoun", "DT": "Determiner",
    "IN": "Preposition", "CC": "Conjunction", "CD": "Cardinal Number",
    "UH": "Interjection", "TO": "to", "MD": "Modal",
}


# ── NLP Pipeline ─────────────────────────────────────────────────────────────

def detect_language(text):
    """Step 1 — Language Detection using character n-gram statistics."""
    try:
        code = detect(text)
        name = next((v[0] for v in LANGUAGES.values() if v[1] == code), code.upper())
        return code, name
    except LangDetectException:
        return "unknown", "Unknown"


def normalize_text(text):
    """Step 2 — Text Normalization: strip extra whitespace and control chars."""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces
    text = re.sub(r"[^\x00-\x7F]+", lambda m: m.group(), text)  # keep unicode
    return text


def tokenize(text):
    """Step 3 — Tokenization: split into sentences, then words."""
    sentences = sent_tokenize(text)
    words     = word_tokenize(text)
    return sentences, words


def remove_stopwords(tokens):
    """Step 4 — Stopword Removal: filter common filler words."""
    try:
        stop_words = set(stopwords.words("english"))
    except OSError:
        stop_words = set()
    filtered = [t for t in tokens if t.lower() not in stop_words]
    return filtered, stop_words


def pos_tagging(tokens):
    """Step 5 — Part-of-Speech Tagging: label each token grammatically."""
    return pos_tag(tokens)


def named_entity_recognition(pos_tags):
    """Step 6 — Named Entity Recognition: find people, places, orgs."""
    tree      = ne_chunk(pos_tags, binary=False)
    entities  = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            entity_name  = " ".join(word for word, tag in subtree.leaves())
            entity_label = subtree.label()
            entities.append((entity_name, entity_label))
    return entities


def translate_via_api(text, src_code, tgt_code):
    """Step 7 — Translation via MyMemory API (after NLP preprocessing)."""
    url    = "https://api.mymemory.translated.net/get"
    params = {"q": text, "langpair": f"{src_code}|{tgt_code}"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("responseStatus") == 200:
            return data["responseData"]["translatedText"]
        return None
    except Exception as e:
        print(f"\n  Translation API error: {e}")
        return None


def run_nlp_pipeline(text, src_code, src_name, tgt_code, tgt_name):
    """Run the full NLP pipeline and print a detailed report."""

    print("\n" + "─" * 55)
    print("  NLP PIPELINE REPORT")
    print("─" * 55)

    # Step 1 — Language Detection
    detected_code, detected_name = detect_language(text)
    print(f"\n[1] Language Detection")
    print(f"    Detected  : {detected_name} ({detected_code})")
    print(f"    Selected  : {src_name} ({src_code})")
    if detected_code != src_code and detected_code != "unknown":
        print(f"    ⚠  Mismatch detected — proceeding with your selection.")

    # Step 2 — Normalization
    normalized = normalize_text(text)
    print(f"\n[2] Text Normalization")
    print(f"    Input     : {text}")
    print(f"    Normalized: {normalized}")

    # Step 3 — Tokenization (only meaningful for English/Latin scripts)
    sentences, word_tokens = tokenize(normalized)
    print(f"\n[3] Tokenization")
    print(f"    Sentences : {sentences}")
    print(f"    Words     : {word_tokens}")
    print(f"    Word count: {len(word_tokens)}")

    # Step 4 — Stopword Removal
    filtered_tokens, stop_words = remove_stopwords(word_tokens)
    removed = [t for t in word_tokens if t.lower() in stop_words]
    print(f"\n[4] Stopword Removal")
    print(f"    Removed   : {removed if removed else 'none'}")
    print(f"    Remaining : {filtered_tokens}")

    # Step 5 — POS Tagging
    alpha_tokens = [t for t in word_tokens if t.isalpha()]
    tags         = pos_tagging(alpha_tokens)
    print(f"\n[5] Part-of-Speech Tagging")
    for word, tag in tags:
        desc = POS_DESCRIPTIONS.get(tag, tag)
        print(f"    {word:<20} → {desc} ({tag})")

    # Step 6 — Named Entity Recognition
    entities = named_entity_recognition(tags)
    print(f"\n[6] Named Entity Recognition")
    if entities:
        for name, label in entities:
            print(f"    '{name}'  →  {label}")
    else:
        print("    No named entities found.")

    # Step 7 — Translation
    print(f"\n[7] Translation  ({src_name} → {tgt_name})")
    result = translate_via_api(normalized, src_code, tgt_code)
    if result:
        print(f"    Result: {result}")
    else:
        print("    Translation failed.")

    print("─" * 55 + "\n")
    return result


# ── UI Helpers ───────────────────────────────────────────────────────────────

def display_languages():
    print("\n" + "=" * 55)
    print("  Available Languages")
    print("=" * 55)
    items = list(LANGUAGES.items())
    for i in range(0, len(items), 2):
        left = f"  [{items[i][0]:>2}] {items[i][1][0]:<25}"
        right = f"[{items[i+1][0]:>2}] {items[i+1][1][0]}" if i+1 < len(items) else ""
        print(left + right)
    print("=" * 55)


def pick_language(prompt):
    while True:
        choice = input(prompt).strip()
        if choice in LANGUAGES:
            return LANGUAGES[choice]
        print(f"  Please enter a number between 1 and {len(LANGUAGES)}.")


# ── Main (CLI) ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("     NLP LANGUAGE TRANSLATOR")
    print("     NLTK Pipeline + MyMemory API")
    print("=" * 55)
    print("""
  NLP steps performed on every input:
    1. Language Detection   (langdetect)
    2. Text Normalization   (regex)
    3. Tokenization         (nltk.tokenize)
    4. Stopword Removal     (nltk.corpus.stopwords)
    5. POS Tagging          (nltk.pos_tag)
    6. Named Entity Recog.  (nltk.ne_chunk)
    7. Translation          (MyMemory API)
    """)

    display_languages()

    src_name, src_code = pick_language("\nSelect SOURCE language number: ")
    tgt_name, tgt_code = pick_language("Select TARGET language number: ")

    print(f"\n  Ready!  {src_name} → {tgt_name}")
    print("  Commands: 'switch' | 'langs' | 'quit'\n")

    while True:
        text = input("Enter text: ").strip()

        if not text:
            continue
        if text.lower() == "quit":
            print("\n  Goodbye!\n")
            break
        if text.lower() == "switch":
            src_name, src_code, tgt_name, tgt_code = tgt_name, tgt_code, src_name, src_code
            print(f"\n  Switched → {src_name} to {tgt_name}\n")
            continue
        if text.lower() == "langs":
            display_languages()
            src_name, src_code = pick_language("\nSelect SOURCE language number: ")
            tgt_name, tgt_code = pick_language("Select TARGET language number: ")
            print(f"\n  {src_name} → {tgt_name}\n")
            continue
        if len(text) > 500:
            print("  Max 500 characters. Please shorten your text.\n")
            continue

        run_nlp_pipeline(text, src_code, src_name, tgt_code, tgt_name)


# ── Flask Web API ─────────────────────────────────────────────────────────────
# Only imported/run when using web mode. CLI mode works exactly as before.

try:
    from flask import Flask, request, jsonify, render_template
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/translate", methods=["POST"])
    def translate_endpoint():
        body     = request.get_json(force=True)
        text     = body.get("text", "").strip()
        src_code = body.get("src", "en")
        tgt_code = body.get("tgt", "hi")

        if not text:
            return jsonify({"error": "No text provided"}), 400
        if len(text) > 500:
            return jsonify({"error": "Text too long (max 500 chars)"}), 400

        # ── Run the exact same NLP pipeline ──────────────────────────────────
        detected_code, detected_name = detect_language(text)
        normalized                   = normalize_text(text)
        sentences, word_tokens       = tokenize(normalized)
        filtered_tokens, stop_words  = remove_stopwords(word_tokens)
        removed_tokens               = [t for t in word_tokens if t.lower() in stop_words]
        alpha_tokens                 = [t for t in word_tokens if t.isalpha()]
        tags                         = pos_tagging(alpha_tokens)
        entities                     = named_entity_recognition(tags)
        translation                  = translate_via_api(normalized, src_code, tgt_code)

        src_name = next((v[0] for v in LANGUAGES.values() if v[1] == src_code), src_code)
        tgt_name = next((v[0] for v in LANGUAGES.values() if v[1] == tgt_code), tgt_code)

        # Also print to terminal (same as CLI)
        run_nlp_pipeline(text, src_code, src_name, tgt_code, tgt_name)

        return jsonify({
            "detected_code":   detected_code,
            "detected_name":   detected_name,
            "normalized":      normalized,
            "sentences":       sentences,
            "tokens":          word_tokens,
            "filtered_tokens": filtered_tokens,
            "removed_tokens":  removed_tokens,
            "pos_tags":        [[w, t] for w, t in tags],
            "entities":        [{"name": n, "label": l} for n, l in entities],
            "translation":     translation,
            "src_name":        src_name,
            "tgt_name":        tgt_name,
        })

except ImportError:
    app = None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--web" in sys.argv:
        if app is None:
            print("Flask not installed. Run: pip install flask flask-cors")
        else:
            print("\n  Starting web server at http://127.0.0.1:5000")
            print("  Press Ctrl+C to stop\n")
            app.run(debug=True, port=5000)
    else:
        main()