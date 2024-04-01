import os
import spacy
import matplotlib.pyplot as plt
# from Chapter01.dividing_into_sentences import read_text_file


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')

nlp = spacy.load("en_core_web_sm")
past_tags = ["VBD", "VBN"]
present_tags = ["VBG", "VBP", "VBZ"]


def read_text_file(text_file):
    file = open(text_file, "r", encoding="utf-8")
    text = file.read()
    text = text.replace("\n", " ")
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # sentences = [sentence.text for sentence in doc.sents]
    return text

def visualize_verbs(text_file):
    text = read_text_file(text_file)
    doc = nlp(text)
    verb_dict = {"Inf":0, "Past":0, "Present":0}
    for token in doc:
        if (token.tag_ == "VB"):
            verb_dict["Inf"] = verb_dict["Inf"] + 1
        if (token.tag_ in past_tags):
            verb_dict["Past"] = verb_dict["Past"] + 1
        if (token.tag_ in present_tags):
            verb_dict["Present"] = verb_dict["Present"] + 1
    plt.bar(range(len(verb_dict)), list(verb_dict.values()), align='center', color=["red","green","blue"])
    plt.xticks(range(len(verb_dict)), list(verb_dict.keys()))
    plt.show()


visualize_verbs(filename)
