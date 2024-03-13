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
    print(noun_chunk)
