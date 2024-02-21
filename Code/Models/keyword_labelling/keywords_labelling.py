from nltk.stem.snowball import SnowballStemmer
import multiprocessing

from ...Utils.string_operations import clean_string,tokenize

class KeywordsLabelling:
    def __init__(self,keywords,stem=False):
        self.stemmer = SnowballStemmer('portuguese').stem if stem else None
        self.keywords = keywords

    #fit, predict and predict_proba implemented for consistency
    def fit(self,X,y):
        pass

    def predict(self,phrases):
        return [int(i>0) for i in self.predict_proba(phrases)]

    def predict_proba(self,phrases):
        if type(phrases) is not list:
            return [self.political_score(phrase=phrases)]
        return self.political_scores(phrases)
    

    def count_matches(self,phrase):
        '''
        find number of matches of keywords in phrase
        '''
        count=0
        for pattern in self.keywords:
            count+=KeywordsLabelling.__match(pattern,phrase)
        return count
    
    def political_score(self,phrase):
        if self.stemmer is not None:
            tokenized = [self.stemmer(t) for t in tokenize(clean_string(phrase))]
        else:
            tokenized = [t for t in tokenize(clean_string(phrase))]
        if len(tokenized) == 0:
            return 0.0
        return float(self.count_matches(tokenized))/len(tokenized)
    
    def political_scores(self,phrases,n_jobs=1):
        '''
        gets political scores from phrases in an iterable
        '''
        if n_jobs == 1:
            scores=[]
            for phrase in phrases:
                scores.append(self.political_score(phrase))
            return scores
        elif n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        elif n_jobs < 1:
            raise Exception("Invalid number of jobs")
        
        with multiprocessing.Pool(processes=n_jobs) as pool:
            scores=pool.map(self.political_score,phrases)
        return scores
    
    def __match(pattern,phrase):
        '''
        Find all matches of tokenized pattern in tokenized phrase
        '''
        ret = 0
        try:
            #find possible matches
            indices = [i for i, term in enumerate(phrase) if term == pattern[0]]
            for index in indices:
                #try to match pattern to tokenized phrase
                for i in range(len(pattern[1:])):
                    if phrase[index+i+1] != pattern[i+1]:
                        break
                #if it did not break(or pattern has len=1), pattern exists
                else:
                    ret+=1
            return ret
        except (ValueError,IndexError):
            #if the first element of pattern was not on list
            #or it was not found within list bounds, it does not exist
            return ret
        except Exception as e:
            print(e)
