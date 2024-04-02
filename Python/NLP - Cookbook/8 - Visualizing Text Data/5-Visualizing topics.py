import os
import gensim
import pyLDAvis.gensim


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')


dictionary = gensim.corpora.Dictionary.load(os.path.join(curerntPath, 'gensim/id2word.dict'))
corpus = gensim.corpora.MmCorpus(os.path.join(curerntPath, 'gensim/corpus.mm'))
lda = gensim.models.ldamodel.LdaModel.load(os.path.join(curerntPath, 'gensim/lda_gensim.model'))


lda_prepared = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.show(lda_prepared)
pyLDAvis.save_html(lda_prepared, os.path.join(curerntPath, 'lda.html'))
                   
