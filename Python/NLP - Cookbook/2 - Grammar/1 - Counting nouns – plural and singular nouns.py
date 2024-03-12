import os
import nltk
from nltk.stem import WordNetLemmatizer
import inflect
from nltk import pos_tag


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')


file = open(filename, "r", encoding="utf-8")
sherlock_holmes_text = file.read()
sherlock_holmes_text = sherlock_holmes_text.replace("\n", " ")

words = nltk.tokenize.word_tokenize(sherlock_holmes_text)
words_with_pos = pos_tag(words)


def get_nouns(words_with_pos):
    noun_set = ["NN", "NNS"]
    nouns = [word for word in words_with_pos if word[1] in noun_set]
    return nouns


nouns = get_nouns(words_with_pos)
print(nouns)


def is_plural_nltk(noun_info):
    pos = noun_info[1]
    if (pos == "NNS"):
        return True
    else:
        return False
    

def is_plural_wn(noun):
    wnl = WordNetLemmatizer()
    lemma = wnl.lemmatize(noun, 'n')
    plural = True if noun is not lemma else False
    return plural


def get_plural(singular_noun):
    p = inflect.engine()
    return p.plural(singular_noun)


def get_singular(plural_noun):
    p = inflect.engine()
    plural = p.singular_noun(plural_noun)
    if (plural):
        return plural
    else:
        return plural_noun
    

def plurals_wn(words_with_pos):
    other_nouns = []
    for noun_info in words_with_pos:
        word = noun_info[0]
        plural = is_plural_wn(word)
        if (plural):
            singular = get_singular(word)
            other_nouns.append(singular)
        else:
            plural = get_plural(word)
            other_nouns.append(plural)
    return other_nouns


other_nouns_wn = plurals_wn(nouns)
print(other_nouns_wn)