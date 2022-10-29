import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from gensim.models.doc2vec import TaggedDocument


# retrieve all the NIPS abstacts available
def load_nips_abstracts():
    avail_data = [2017, 2018, 2019, 2020, 2021]
    f_name = lambda year:  f'NIPS_data/{year}_data.pickle'
    nips_dct = {}
    for year in avail_data:
        with open(f_name(year), 'rb') as file:
            nips_dct[year] = pickle.load(file)
    return nips_dct


def process_abstract(abstract):
    clean_abstract = "".join([char for char in abstract if char not in string.punctuation])
    tokenized_abstract = word_tokenize(clean_abstract)

    stop_words = stopwords.words('english')
    filtered_abstract = [word for word in tokenized_abstract if word not in stop_words]

    porter = PorterStemmer()
    stemmed_abstract = [porter.stem(word) for word in filtered_abstract]
    return stemmed_abstract


def get_tagged_documents(raw_abstracts):
    tagged_documents = []
    for idx, (_, abstract) in enumerate(raw_abstracts):
        processed_abstract = process_abstract(abstract)
        tagged_abstract = TaggedDocument(processed_abstract, [idx])
        tagged_documents.append(tagged_abstract)
    return tagged_documents








