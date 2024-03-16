import spacy

nlp = spacy.load('en_core_web_sm')


sentences=["The big black cat stared at the small dog.",
"Jane watched her brother in the evenings."]


def get_subject_phrase(doc):
    for token in doc:
        if("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]


def get_object_phrase(doc):
    for token in doc:
        if("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]



# for sentence in sentences:
#     doc = nlp(sentence)
#     subject_phrase = get_subject_phrase(doc)
#     object_phrase = get_object_phrase(doc)
#     print(subject_phrase)
#     print(object_phrase)



def get_dative_phrase(doc):
    for token in doc:
        if ("dative" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]
        

def get_prepositional_phrase_objs(doc):
    prep_spans = []
    for token in doc:
        if ("pobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            prep_spans.append(doc[start:end])
    return prep_spans


for sentence in sentences:
    doc = nlp(sentence)
    dative_phrase = get_dative_phrase(doc)
    prepositional_phrase_objs = get_prepositional_phrase_objs(doc)
    print(dative_phrase)
    print(prepositional_phrase_objs)