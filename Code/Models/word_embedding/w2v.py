from gensim.models import Word2Vec
import multiprocessing
from ...Utils.string_operations import clean_string,tokenize
from functools import partial
import numpy as np


class Word2vec:
    def __init__(self,vector_size=None,n_jobs=1,stopwords=set(),minsize=2,preprocess=clean_string):
        if n_jobs < 1:
            n_jobs = multiprocessing.cpu_count()
        self.n_jobs = n_jobs
        self.vector_size = vector_size
        self.preprocess = preprocess
        self.tokenizator = partial(tokenize,stopwords=stopwords,minsize=minsize)

    def fit(self,sentences):
        '''
        Trains the model using text
        '''
      #  with multiprocessing.Pool(processes=self.n_jobs) as pool:
           # aux = pool.map(self.preprocess,sentences)
          #  aux = pool.map(self.tokenizator,aux)
        self.model = Word2Vec(sentences,vector_size=self.vector_size,workers=self.n_jobs)

    def __encode(self,sentence):
        '''
        Returns the encoding of every word of a sentence
        '''
        clean_text = self.preprocess(sentence)
        tokens = self.tokenizator(clean_text)
        encoded = []
        for word in tokens:
            try:
                encoded.append(self.model.wv[word])
            except:
                continue
        try:
            encoded = np.stack(encoded)
        except:
            #raise Exception('No words in sentence')
            encoded = np.zeros((1,self.vector_size))
        return encoded

    def encode(self,sentences,mean=True):
        '''
        Returns the sentence encoding for each sentence in list by averaging word vectors.
        If sentences is not a list, return the encoding of string instead
        '''
        if type(sentences) == list:
            if self.n_jobs==1:
                return [self.encode(sentence) for sentence in sentences]
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                encodings = pool.map(self.encode,sentences)
            return encodings
        
        else:
            code = self.__encode(sentences)
            
            if mean:
                return np.mean(code,axis=0)
            return code
    
    def save(self,path):
        self.model.save(path)
    
    def load(self,path):
        self.model = Word2Vec.load(path)
        self.vector_size = self.model.vector_size