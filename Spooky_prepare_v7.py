import numpy as np
import operator
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import string
import io
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import linalg
#from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('stopwords')

## Create tokens
#tokens_filtered = []
#for line in data:
def filter_line(line,L,wnl,my_string_punct, stopwords,perc):

    line = line.lower()
    for elemento in string.punctuation:
        line = line.replace(elemento," %s" % elemento)

    all_tokens = []
    all_tokens = line.split('\n')
    all_tokens = all_tokens[0]
    all_tokens = all_tokens.split(" ")

    ## Filtered tokens
    dict = {} # un dicionrio por linea
    s_token = []
    for index, token in enumerate(all_tokens):
        if index < len(all_tokens) and len(token) > L: # no consideres /n

            # Only Simple
            #token = wnl.stem(token)
            a = wnl.lemmatize(token)
            if a == token:
                token = wnl.lemmatize(token,'v')
            else:
                token = a

            if token not in dict.keys():
                dict[token]=1
            else:
                dict[token]=dict[token]+1
            s_token.append(token)

            # Only double
            if index+1 < len(all_tokens):
                token1 = all_tokens[index]
                token2 = all_tokens[index+1]

                a = wnl.lemmatize(token1)
                if a == token1:
                    token1 = wnl.lemmatize(token1,'v')
                else:
                    token1 = a

                a = wnl.lemmatize(token2)
                if a == token2:
                    token2 = wnl.lemmatize(token2,'v')
                else:
                    token2 = a

                token = token1 + '_' + token2

                if token not in dict.keys():
                    dict[token]=1
                else:
                    dict[token]=dict[token]+1

            # Only Triple
            if index+2 < len(all_tokens) and random.random() > perc:
                token1 = all_tokens[index]
                token2 = all_tokens[index+1]
                token3 = all_tokens[index+2]

                a = wnl.lemmatize(token1)
                if a == token1:
                    token1 = wnl.lemmatize(token1,'v')
                else:
                    token1 = a

                a = wnl.lemmatize(token2)
                if a == token2:
                    token2 = wnl.lemmatize(token2,'v')
                else:
                    token2 = a

                a = wnl.lemmatize(token3)
                if a == token3:
                    token3 = wnl.lemmatize(token3,'v')
                else:
                    token3 = a

                token = token1 + '_' + token2 + '_' + token3

                if token not in dict.keys():
                    dict[token]=1
                else:
                    dict[token]=dict[token]+1
                
    return dict, s_token

## Complete text
def filter_text(file,N,L,perc):
    lista = []
    lista_s = []
    ## Clean the data correctly
    my_string_punct = string.punctuation
    #my_string_punct = []

    ## Eliminate stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    #stopwords = []

    ## Lemmantization
    wnl = nltk.WordNetLemmatizer()

    ## Stemmer
    #wnl = PorterStemmer()
    #wnl = LancasterStemmer()
    #wnl = SnowballStemmer("english")

    for index,info in enumerate(file.readlines()):
        if index > 0: ## first line of file are labels
            info = info.split('","')
            line = info[1]
            if index < N:
                dict = []
                #dict = filter_line(line, stopwords, my_string_punct,L,wnl)
                dict,s_token = filter_line(line,L,wnl,my_string_punct,stopwords,perc)
                lista.append(dict)
                lista_s.append(s_token)
    return lista, lista_s

def dos_listas(N,L,perc): #top = 100, 90863; 150, 102123; 50, 760; 1000, 157239; 5000, 212290
    file_train = io.open('train.csv','r',encoding='utf-8')
    file_test = io.open('test.csv','r',encoding='utf-8')
    lista =[]
    lista1,lista_s1 = filter_text(file_train, N,L,perc)
    dim_train = len(lista1)
    lista2,lista_s2 = filter_text(file_test,N,L,perc)
    lista = lista1 + lista2
    lista_s = lista_s1 + lista_s2
    return lista, lista_s, dim_train
    """
    no string.punctuation:

    lemma, lower, 21469, 1 = 253334 -> 0.3609 with 10.

    lemma, lower, 100, 2 = 288116
    lemma, lower, 1000, 2 = 490715 > 64GB
    lemma, lower, 5000, 2 =
    lemma, lower, 10000, 2 = 700041 > 128GB

    lemma, lower, 100, 3 = 546967
    lemma, lower, 1000, 3 = 910361
    lemma, lower, 10000, 3 = 1246371
    """
def total_func(lista):
    dict_vectorizer = DictVectorizer(sparse=True)
    doc_term_mat = dict_vectorizer.fit_transform(lista)
    # Normalize doc term matrix
    Tfid_transformer = TfidfTransformer()
    matrix_transf = Tfid_transformer.fit_transform(doc_term_mat)
    return matrix_transf
