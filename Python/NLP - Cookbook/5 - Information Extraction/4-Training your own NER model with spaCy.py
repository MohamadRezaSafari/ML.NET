import os
import spacy
from spacy.util import minibatch, compounding
from spacy.language import Language
import warnings
import random
from pathlib import Path


curerntPath = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(curerntPath, "model_output")
N_ITER=100

DATA = [
("A fakir from far-away India travels to Asterix's\
village and asks Cacofonix to save his land from\
drought since his singing can cause rain.",
{'entities':[(39, 46, "PERSON"),
(66, 75, "PERSON")]}),
("Cacofonix, accompanied by Asterix and Obelix,\
must travel to India aboard a magic carpet to\
save the life of the princess Orinjade, who is to\
be sacrificed to stop the drought.",
{'entities':[(0, 9, "PERSON"),
(26, 33, "PERSON"),
(38, 44, "PERSON"),
(61, 66, "LOC"),
(122, 130, "PERSON")]})
]



def save_model(nlp, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)


def load_model(input_dir):
    nlp = spacy.load(input_dir)
    return nlp


def create_model(model):
    if (model is not None):
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank("en")
    return nlp


def add_ner_to_model(nlp):
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    return (nlp, ner)

def add_labels(ner, data):
    for sentence, annotations in data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    return ner


def train_model(model=None):
    nlp = create_model(model)
    (nlp, ner) = add_ner_to_model(nlp)
    ner = add_labels(ner, DATA)
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if
    pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning,  module='spacy')
        if model is None:
            nlp.begin_training()
        for itn in range(N_ITER):
            random.shuffle(DATA)
            losses = {}
            batches = minibatch(DATA,
            size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.5, losses=losses)
            print("Losses", losses)
    return nlp




def test_model(nlp, data):
    for text, annotations in data:
        doc = nlp(text)
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)


def without_training(data=DATA):
    nlp = spacy.load("en_core_web_sm")
    test_model(nlp, data)
           

without_training()
