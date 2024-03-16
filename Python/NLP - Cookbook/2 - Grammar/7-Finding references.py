import spacy
import neuralcoref


nlp = spacy.load('en_core_web_sm')
# neuralcoref.add_to_pipe(nlp)


# text = "Earlier this year, Olga appeared on a new song. She was featured on one of the tracks. The singer is assuring that her next album will be worth the wait."


# doc = nlp(text)
# print(doc._.coref_resolved)


text = "Deepika has a dog. She loves him. The movie star has always been fond of animals."


neuralcoref.add_to_pipe(nlp, conv_dict={'Deepika': ['woman']})
