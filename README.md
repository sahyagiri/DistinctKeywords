
#  DistinctKeywords

This is a utility function to extract semantically distinct keywords. This is an unsupervised method based on word2vec. Current implementation used a word2vec model trained in simplewiki.

Hilbert curve act as a Locality-sensitive hashing.

  

###  Supported Languages (jupyter notebooks available in examples)

1. English (default) using custom word2vec trained on simplewiki.

2. German (on test. Need support from native speakers). Using Word2vec from [this link](https://devmount.github.io/GermanWordEmbeddings/)

3. French (on test. Need support from native speakers). Using word2vec model from [this link](https://fauconnier.github.io/)

  

##  Installation Instructions

1. conda create -n keyphrases python=3.8 --no-default-packages

2. conda activate keyphrases

3. pip install distinct-keywords

4. python -m spacy download en_core_web_sm

5. conda install --channel=conda-forge nb_conda_kernels jupyter

6. jupyter notebook

  

##  Getting started

  

1. Clone the repository

2. Open the examples folder in jupyter notebook. The subfolders contain the respective language files.

3. Select the language you wanted to try out

  

##  Usage

```

from distinct_keywords.keywords import DistinctKeywords

  

doc = """

Supervised learning is the machine learning task of learning a function that

maps an input to an output based on example input-output pairs. It infers a

function from labeled training data consisting of a set of training examples.

In supervised learning, each example is a pair consisting of an input object

(typically a vector) and a desired output value (also called the supervisory signal).

A supervised learning algorithm analyzes the training data and produces an inferred function,

which can be used for mapping new examples. An optimal scenario will allow for the

algorithm to correctly determine the class labels for unseen instances. This requires

the learning algorithm to generalize from the training data to unseen situations in a

'reasonable' way (see inductive bias).

"""

  
  
  

distinct_keywords=DistinctKeywords()

  

distinct_keywords.get_keywords(doc)

  

```

##  Output

  

['machine learning',

'pairs',

'mapping',

'vector',

'typically',

'supervised',

'bias',

'supervisory',

'task',

'algorithm',

'unseen',

'training']

  

##  Methodology

  

After creating word2vec, the words are mapped to a hilbert space and the results are stored in a key-value pair (every word has a hilbert hash). Now for a new document, the words and phrases are cleaned, hashed using the dictionary. One word from each different prefix is then selected using wordnet ranking from NLTK (rare words are prioritized). The implementation of grouping and look up is made fast using Trie and SortedDict

![enter image description here](https://github.com/sahyagiri/DistinctKeywords/raw/main/steps_hilbert_hashing.png)


##  Benchmarks


Currently this is tested against KPTimes test dataset (20000 articles). A recall score of 31% is achieved when compared to the manual keywords given in the dataset.

Steps to arrive at the score:

1. Used both algorithms. Keybert was ran with additional parameter top_n=16 as the length of dstinct_keywords at 75% level was around 15.

2. Results of algorithms and original keywords were cleaned (lower case, space removal, character removal, but no lemmatization)

3. Take intersection of original keywords and generated keyword **word banks** (individual words)

4. For each prediction compare the length of intersecting words with length of total keyword words

Output is given below

![enter image description here](https://github.com/sahyagiri/DistinctKeywords/raw/main/benchmark_keybert_distinct_keywords_kptimes.png)
