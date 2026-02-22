import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures


def setup_nltk():
    """Downloads necessary NLTK datasets silently."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def build_phrase_map(df, text_column):
    stop_words = set(stopwords.words("english"))
    all_words = []

    for text in df[text_column]:
        tokens = [
            w.lower()
            for w in word_tokenize(str(text))
            if w not in string.punctuation and w not in stop_words
        ]
        all_words.extend(tokens)

    bigram_finder = BigramCollocationFinder.from_words(all_words)
    top_bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 20)

    trigram_finder = TrigramCollocationFinder.from_words(all_words)
    top_trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, 10)

    return {" ".join(b): "_".join(b) for b in top_bigrams + top_trigrams}


def extract_keywords(sentence, phrase_map):
    stop_words = set(stopwords.words("english"))
    sentence_lower = str(sentence).lower()

    for phrase, replacement in phrase_map.items():
        if phrase in sentence_lower:
            sentence_lower = sentence_lower.replace(phrase, replacement)

    words = word_tokenize(sentence_lower)
    words = [w for w in words if w not in string.punctuation and w not in stop_words]

    tagged = pos_tag(words)
    keywords = [
        word for word, tag in tagged if tag.startswith("NN") or tag.startswith("VB")
    ]

    return list(set(keywords))


def clean_keyword(word):
    if not isinstance(word, str):
        return None
    word = word.lower().strip()
    word = re.sub(r"^[\W_]+|[\W_]+$", "", word)
    if word == "" or re.fullmatch(r"[\W_]+", word) or word.isdigit() or word == "nan":
        return None
    return word


def prepare_dataframe(df, text_column):
    # 1. Clean the raw text
    df["cleaned_interests"] = (
        df[text_column]
        .astype(str)
        .replace(r"[&/;\n|*+-]", ",", regex=True)
        .str.replace(r",+", ",", regex=True)
        .str.strip(", ")
    )

    # 2. Build map and extract keywords
    phrase_map = build_phrase_map(df, "cleaned_interests")
    df["keywords_list"] = df["cleaned_interests"].apply(
        lambda x: extract_keywords(x, phrase_map)
    )
    df["keywords"] = df["keywords_list"].apply(lambda x: ", ".join(x) if x else "")

    # 3. Sort for consistency
    dataset_sorted = df.sort_values(by="keywords").reset_index(drop=True)

    # 4. Generate the unique keywords dataframe
    keywords_df = (
        dataset_sorted["keywords_list"]
        .explode()
        .dropna()
        .to_frame(name="keywords_list")
    )
    keywords_df["keywords_list"] = keywords_df["keywords_list"].apply(clean_keyword)
    keywords_df = keywords_df.dropna().drop_duplicates().reset_index(drop=True)
    keywords_df = keywords_df[~keywords_df["keywords_list"].isin(["nan"])].reset_index(
        drop=True
    )

    return dataset_sorted, keywords_df
