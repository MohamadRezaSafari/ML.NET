import spacy


sentence = 'I have seldom heard him mention her under any other name.'

nlp = spacy.load('en_core_web_sm')
doc = nlp(sentence)

for token in doc:
    print(token.text, "\t", token.dep_, "\t")
    spacy.explain(token.dep_)

for token in doc:
    print(token.text)
    ancestors = [t.text for t in token.ancestors]
    print(ancestors)

for token in doc:
    print(token.text)
    childern = [t.text for t in token.children]
    print(childern)

for token in doc:
    print(token.text)
    subtree = [t.text for t in token.subtree]
    print(subtree)