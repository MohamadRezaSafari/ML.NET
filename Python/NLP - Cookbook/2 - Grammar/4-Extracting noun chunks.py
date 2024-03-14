import os
import spacy


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')


file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
# sentences = [sentence.text for sentence in doc.sents]

# for noun_chunk in doc.noun_chunks:
    # print(noun_chunk.text)


nlp = spacy.load('en_core_web_sm')
sentence = "All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind."

doc2 = nlp(sentence)

for noun_chunk in doc2.noun_chunks:
    # print(noun_chunk.text, "\t", noun_chunk.start, "\t", noun_chunk.end)
    # print(noun_chunk.text, "\t", noun_chunk.sent)
    print(noun_chunk.text, "\t", noun_chunk.root.text)


other_span = "emotions"
other_doc = nlp(other_span)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.similarity(other_doc))
