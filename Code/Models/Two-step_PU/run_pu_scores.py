import sys,json,argparse,os
sys.path.insert(1,'../../../')
from Code.Utils.string_operations import clean_string
from Code.Models.word_embedding.w2v import Word2vec
from nltk.corpus import stopwords
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#create parser which can parse the following arguments: word2vec path, labelled data path, test path, parameter tuning method, model parameters and spy tolerance
parser = argparse.ArgumentParser()
parser.add_argument('--w2v', type=str, help='path to word2vec model',default='../../../Data/Keyword_Classification/300_dims_w2v_news.model')
parser.add_argument('--labelled_file', type=str, help='path to labelled data',default='../../../Data/labelled.jsonl')
parser.add_argument('--model_params', type=str, help='model parameters',default=None)
parser.add_argument('--parameter_path', type=str, help='path to parameter tuning results',default='./stats/model_stats.jsonl')
parser.add_argument('--spy_tolerance', type=float, help='spy tolerance',default=0.003)
parser.add_argument('--model_name',type=str,help='model name',default='all_kw')
args = parser.parse_args()

# read data and extract approximately 50000 comments and 50000 originals with probability of P equal to the dataset probability of P
print('Reading Data for model training')
df = pd.read_json(args.labelled_file,lines=True,encoding='utf-8')
df = pd.concat([df[df['kind']=='original'],df[df['kind']=='comment'].sample(len(df[df['kind']=='original']),random_state=1)])
df['text'] = df['text'].apply(lambda x: clean_string(x))
df = df[df['text'].apply(lambda x: len(str(x).split(' '))>5)]

# Start 2-step pu learning
# First step
print('First Step of PU Learning')

#Generate three separations -> unlabeled,positive and spies (counted as unlabeled by the classifier)
df['class'] = 'u'
df.loc[(df['label']==1),'class'] = 'p'
#spies are marked as negative
df.loc[df[df['label']==1].sample(frac=0.1,random_state=3).index,'class']='s'
df.loc[df['class']=='s','label']=0

#create TF-IDF features
vectorizer = TfidfVectorizer(stop_words=list(stopwords.words('portuguese')))
X = vectorizer.fit_transform(df['text'].to_list())
y = df['label'].to_list()

#Train naive bayes to discern u from Reliable Negatives(n)
clf = MultinomialNB()
clf.fit(X, y)
df['new_label'] = clf.predict_proba(X)[:,1]
spies = df[df['class']=='s']

#find threshold
t = 0.001
while len(spies[spies['new_label'] <= t])/len(spies) <= args.spy_tolerance:
    t += 0.001

#extract reliable negatives(those that have lower probability of P than all spies)
df.loc[(df['new_label']<=t) & (df['class']=='u'),'class']='n'

#notify if less than 10% of examples are of class n
percentage_of_negatives = len(df[df['class']=='n'])/len(df)
if percentage_of_negatives < 0.1:
    print(r'Less than 10% of examples are negative, it might be good to raise the spy tolerance')
    print('Current percentage of negatives:',percentage_of_negatives)

#turn spies back to positives
df.loc[df['class']=='s','class']='p'

# Second step
# sample equal amount of positive and negative examples
print('Second Step of PU Learning')
sample_size = min(len(df[df['class']=='p']),len(df[df['class']=='n']))
df = pd.concat([df[df['class']=='p'].sample(sample_size,random_state=1),df[df['class']=='n'].sample(sample_size,random_state=1)])

# load word2vec model
w2v = Word2vec()
w2v.load(args.w2v)

#prepare train data
y_train = df['class'].values
y_train[np.where(y_train=='n')]=0
y_train[np.where(y_train=='p')]=1
y_train = y_train.astype('int32')
X_train = w2v.encode(df['text'].tolist())

best_parameters = {}
if args.model_params:
    best_parameters = json.loads(args.model_params)
else:
    #load parameter tuning results
    with open(args.parameter_path,'r') as file:
        for line in file:
            model = json.loads(line)
            if model['model_name'] == args.model_name:
                best_parameters = model['model_parameters']
                break

print(f'Model parameters: {best_parameters}')
#train the final model
clf = XGBClassifier(objective='binary:logistic',tree_method='gpu_hist', gpu_id=0,random_state=7,**best_parameters)
clf.fit(X_train,y_train)

#apply model to all social media networks
for platform in ['Twitter','Youtube','TikTok']:
    print(f'Getting scores for {platform}')
    lines = []
    with open(f'../../../Data/{platform}/news.jsonl',encoding='utf-8') as news_file:
        for line in news_file:
            line=json.loads(line)
            if len(clean_string(line['Comment']).split(' '))>5:
                lines.append(line)
            if len(lines) >= 10**6:
                df = pd.DataFrame(lines)
                if platform=='Youtube':
                    df['Original'] = df['VideoTitle'] +' '+df['VideoDescription']
                elif platform=='Twitter':
                    df['Original'] = df['OriginalComment']
                else:
                    df['Original'] = df['VideoDescription']
                
                df = df[df['Original'].apply(lambda x: len(x.split(' '))>5)]
                comment_embeddings = w2v.encode(df['Comment'].to_list())
                comment_predictions = clf.predict_proba(comment_embeddings)
                del comment_embeddings

                original_embeddings = w2v.encode(df['Original'].to_list())
                original_predictions = clf.predict_proba(original_embeddings)
                del original_embeddings
                df = df.drop(columns=['Original'],errors='ignore')
                with open(f'../../../Data/two_step_pu/xgb_{platform}_{args.model_name}.jsonl','a',encoding='utf-8') as file:
                    for i,line in enumerate(df.to_dict(orient='records')):
                        line['CommentScore'] = float(comment_predictions[i][1])
                        line['OriginalScore'] = float(original_predictions[i][1])
                        json.dump(line,file,ensure_ascii=False)
                        file.write('\n')
                        file.flush()
                lines=[]
    with open(f'../../../Data/two_step_pu/xgb_{platform}_{args.model_name}.jsonl','a',encoding='utf-8') as file:
        df = pd.DataFrame(lines)
        if platform=='Youtube':
            df['Original'] = df['VideoTitle'] +' '+df['VideoDescription']
        elif platform=='Twitter':
            df['Original'] = df['OriginalComment']
        else:
            df['Original'] = df['VideoDescription']
        
        df = df[df['Original'].apply(lambda x: len(x.split(' '))>5)]
        comment_embeddings = w2v.encode(df['Comment'].to_list())
        comment_predictions = clf.predict_proba(comment_embeddings)
        del comment_embeddings

        original_embeddings = w2v.encode(df['Original'].to_list())
        original_predictions = clf.predict_proba(original_embeddings)
        del original_embeddings
        df = df.drop(columns=['Original'],errors='ignore')
        for i,line in enumerate(df.to_dict(orient='records')):
            line['CommentScore'] = float(comment_predictions[i][1])
            line['OriginalScore'] = float(original_predictions[i][1])
            json.dump(line,file,ensure_ascii=False)
            file.write('\n')
            file.flush()