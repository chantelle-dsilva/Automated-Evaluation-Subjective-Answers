import re
from string import printable

import pyaspell
from config import stop_words

printable_characters = set([k for k in printable if k not in ("\n","\t","\r",";",'"',"'")])
spellchecker = pyaspell.Aspell(("lang", "en"))


def remove_all_non_printable(text):
    return "".join([k for k in text if k in printable_characters])


def remove_all_non_characters(text):
    return re.sub("[^a-zA-Z\s]"," ",text)


def spellcheck(text):
    correct_text = text
    for word in text.split():
        if not spellchecker.check(word):
            suggestion = spellchecker.suggest(word)
            if suggestion:
                correct_text = correct_text.replace(word,suggestion[0])
    return correct_text

def remove_multispaces(text):
    return re.sub("[\s]+"," ",text)

def remove_stopwords(text):
    return " ".join([k for k in text.split() if k not in stop_words])
