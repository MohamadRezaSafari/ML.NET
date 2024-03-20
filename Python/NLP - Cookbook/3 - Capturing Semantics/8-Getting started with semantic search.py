from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, DATETIME
from whoosh.index import create_in
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
import csv
from Chapter03.word_embeddings import w2vec_model_path
from Chapter03.word_embeddings import load_model


imdb_dataset_path = "Chapter03/IMDB-Movie-Data.csv"
search_engine_index_path = "Chapter03/whoosh_index"


class IMDBSearchEngine:
    def query_engine(self, keywords):
        with self.index.searcher() as searcher:
            query = MultifieldParser(["title", "description"], self.index.schema).parse(keywords)
            results = searcher.search(query)
            print(results)
            print(results[0])
            return results


def get_similar_words(model, search_term):
    similarity_list = model.most_similar(search_term, topn=3)
    similar_words = [sim_tuple[0] for sim_tuple in similarity_list]
    return similar_words


search_engine = IMDBSearchEngine(search_engine_index_path, imdb_dataset_path, load_existing=False)
IMDBSearchEngine(search_engine_index_path, load_existing=True)

model = load_model(w2vec_model_path)
search_term = "gigantic"
other_words = get_similar_words(model, search_term)

results = search_engine.query_engine(" OR ".join([search_term] + other_words))
print(results[0])
