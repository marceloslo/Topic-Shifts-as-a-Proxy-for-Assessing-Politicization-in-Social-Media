import multiprocessing
import json
import pandas as pd

from .keywords_labelling import KeywordsLabelling

class WeakLabelCreator:
    def __init__(self,path,platform='',keywords=[],stem = False,n_jobs=1):
        if platform == '':
            if 'Twitter' in path:
                platform='Twitter'
            elif 'Youtube' in path:
                platform='Youtube'
            elif 'TikTok' in path:
                platform='TikTok'
        self.platform=platform

        self.labeler = KeywordsLabelling(keywords=keywords,stem=stem)
        self.n_jobs = n_jobs if n_jobs >= 1 else multiprocessing.cpu_count()
        originals,comments = set(),set()
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                comment,original = self.__treat_line(line)
                originals.add(original)
                comments.add(comment)
        self.originals = list(originals)
        self.comments = list(comments)

    def __treat_line(self,line):
        line = json.loads(line)
        if self.platform == 'Twitter':
            return line['Comment'],line['OriginalComment']
        elif self.platform == 'Youtube':
            return line['Comment'],line['VideoTitle']+' '+line['VideoDescription']
        elif self.platform == 'TikTok':
            return line['Comment'],line['VideoDescription']
    
    def create_labelled_file(self,path_to_save):
        self.__label()
        with open(path_to_save,'a',encoding='utf-8') as save_file:
            for i in range(len(self.originals)):
                doc = {'text':self.originals[i],'label':int(self.original_scores[i]>0.0),'platform':self.platform,'kind':'original'}
                json.dump(doc,save_file,ensure_ascii=False)
                save_file.write('\n')
                save_file.flush()
            for j in range(len(self.comments)):
                doc = {'text':self.comments[j],'label':int(self.comment_scores[j]>0.0),'platform':self.platform,'kind':'comment'}
                json.dump(doc,save_file,ensure_ascii=False)
                save_file.write('\n')
                save_file.flush()

    def create_labelled_dataframe(self):
        self.__label()
        df = []
        for i in range(len(self.originals)):
            df.append({'text':self.originals[i],'label':int(self.original_scores[i]>0.0),'platform':self.platform,'kind':'original'})
        for j in range(len(self.comments)):
            df.append({'text':self.comments[j],'label':int(self.comment_scores[j]>0.0),'platform':self.platform,'kind':'comment'})
        return pd.DataFrame(df)
    
    def __label(self):
        self.original_scores = self.labeler.political_scores(self.originals,n_jobs=self.n_jobs)
        self.comment_scores = self.labeler.political_scores(self.comments,n_jobs=self.n_jobs)