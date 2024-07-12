from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

# Load the POS tagger
tagger = SequenceTagger.load("pos")

# Define the dictionary of POS tags and their meanings
POS_TAG_DICT = {
    "ADD": "Email",
    "AFX": "Affix",
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "HYPH": "Hyphen",
    "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NFP": "Superfluous punctuation",
    "NN": "Noun, singular or mass",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "NNS": "Noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
    "XX": "Unknown",
    ".": "Punctuation",
}

# Create a dictionary mapping tags to indices
tag_to_index = {tag: i for i, tag in enumerate(POS_TAG_DICT.keys())}


# Define a function to create a one-hot vector
def create_one_hot(tag: str, tag_dict: dict):
    vector = np.zeros(len(tag_dict))
    vector[tag_dict[tag]] = 1
    return vector


def create_feature_vector_from_sentence(sentence_text):
    sentence = Sentence(sentence_text)

    # Run POS tagging
    tagger.predict(sentence)

    # Initialize the feature vector
    feature_vector = []

    # Process each word in the sentence
    for token in sentence:
        pos = token.tag
        one_hot = create_one_hot(pos, tag_to_index)
        feature_vector.append(one_hot)

    return feature_vector


def create_feature_vectors_from_sentences(sentences: list[str]):
    return [create_feature_vector_from_sentence(sentence) for sentence in sentences]
