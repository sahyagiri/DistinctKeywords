from ast import keyword
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from nltk.corpus import wordnet
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import spacy
import joblib
from flashtext import KeywordProcessor
from sortedcontainers import SortedDict
from collections import Counter
import time
import string 
from datrie import Trie

class DistinctKeywords:
    def __init__(self,
    keyword_dictionary_file='hilbert_lookup_dictionary_simplewiki_17_2_22_v3.pickle.gz',
    keyword_processor_file='keyword_processor_simple_wiki2022.pickle',
    stop_words_set_file="stopwords.pickle") -> None:
        self.hilbert_lookup_dictionary=joblib.load(keyword_dictionary_file)
        self.hilbert_reverse_lookup_dictionary = {v: k for k, v in self.hilbert_lookup_dictionary.items()}
        self.hashlength=len(list(self.hilbert_lookup_dictionary.values())[0])
        self.nlp=spacy.load('en_core_web_sm')
        self.stop_words=joblib.load(stop_words_set_file)
        self.keyword_processor=joblib.load(keyword_processor_file)
        self.min_length=20
        self.sum=0
        self.count=0
    def __preprocess_no_lemmatization(self,x):
        x=str(x)
        x=x.lower()
        x=strip_numeric(x)
        x=strip_punctuation(x)
        x=strip_tags(x)
        x=strip_short(x,minsize=2)
        x=strip_non_alphanum(x)
        x=strip_multiple_whitespaces(x)
        x=' '.join([i for i in x.split() if i not in self.stop_words])
        x= self.keyword_processor.replace_keywords(x)
        return x
    

    def __get_wordnet_count(self,word):
        try:
            return wordnet.synsets(word)[0].lemmas()[0].count()
        except:
            return 0
   
    def get_keywords_from_text(self, 
                    input_document:str,
                    doc,
                    min_length=2,
                    include_proper_nouns=True,
                    max_proper_noun_count=5):

        input_text=self.__preprocess_no_lemmatization(input_document)
        trie=Trie(string.ascii_lowercase+string.digits)
        for word in input_text.split():
            if word in self.stop_words:
                continue 
            try:
                hashstring=self.hilbert_lookup_dictionary[word]
                prefix=hashstring[:min_length]
                if prefix in trie:
                    key=word.replace('_',' ')
                    value=self.__get_wordnet_count(key)
                    trie[prefix][key]=value
                else:
                    leaf_node=SortedDict()
                    key=word.replace('_',' ')
                    value=self.__get_wordnet_count(key)
                    leaf_node[key]=value
                    trie[prefix]=leaf_node
            except: # key not in the word vector
                continue 
        keywords=[]
        for i in trie.keys():
            keywords.append(trie[i].popitem(index=-1)[0])
        keywords=[i.replace('_',' ') for i in keywords if i in input_document]
        if include_proper_nouns:
            proper_nouns=[strip_multiple_whitespaces(strip_non_alphanum(tok.text)) for tok in doc.noun_chunks]
            proper_nouns=[i for i in proper_nouns if i.lower() not in self.stop_words]
            top_proper_nouns={i[0] for i in Counter(proper_nouns).most_common(max_proper_noun_count)}
            return list(set(keywords).union(top_proper_nouns))
        return keywords

    def get_keywords(self,
                    input_document:str,
                    min_length=2,
                    include_proper_nouns=True,
                    max_proper_noun_count=5):
        
        doc = self.nlp(input_document)
        keywords = self.get_keywords_from_text(input_document=input_document,
                                                doc=doc,
                                                min_length=min_length,
                                                include_proper_nouns=include_proper_nouns,
                                                max_proper_noun_count=max_proper_noun_count)

        return keywords

    def get_multiple_doc_keywords(self,
                                  docs:list,
                                  min_length=2,
                                  include_proper_nouns=True,
                                  max_proper_noun_count=5):

        keywords_for_all_input_samples = []

        for doc in self.nlp.pipe(docs):

            input_document = doc.text
            document_keywords = self.get_keywords_from_text(input_document=input_document,
                                                            doc=doc,
                                                            min_length=min_length,
                                                            include_proper_nouns=include_proper_nouns,
                                                            max_proper_noun_count=max_proper_noun_count)

            keywords_for_all_input_samples.append(document_keywords)

        return keywords_for_all_input_samples