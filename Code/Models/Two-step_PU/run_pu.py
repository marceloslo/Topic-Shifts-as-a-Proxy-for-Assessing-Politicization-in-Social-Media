import sys,json,argparse,os
sys.path.insert(1,'../../../')
from Code.Utils.string_operations import clean_string
from Code.Models.word_embedding.w2v import Word2vec
from nltk.corpus import stopwords
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ParameterSampler, cross_val_score
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,confusion_matrix

#create parser which can parse the following arguments: word2vec path, labelled data path, test path, parameter tuning method, model parameters and spy tolerance
parser = argparse.ArgumentParser()
parser.add_argument('--w2v', type=str, help='path to word2vec model',default='../../../Data/Keyword_Classification/300_dims_w2v_news.model')
parser.add_argument('--labelled_file', type=str, help='path to labelled data',default='../../../Data/labelled.jsonl')
parser.add_argument('--test_path', type=str, help='path to test data',default='../../../Data/labels/labeled_random_samples.xlsx')
parser.add_argument('--tuning_method', type=str, help='parameter tuning method',default='random')
parser.add_argument('--tuning_iterations', type=int, help='number of parameters to test for random search',default=500)
parser.add_argument('--model_params', type=str, help='model parameters',default=None)
parser.add_argument('--spy_tolerance', type=float, help='spy tolerance',default=0.003)
parser.add_argument('--model_name',type=str,help='model name',default='all_keywords')
args = parser.parse_args()

# read data and extract approximately 50000 comments and 50000 originals with probability of P equal to the dataset probability of P
print('Reading Data')
df = pd.read_json(args.labelled_file,lines=True,encoding='utf-8')
df = pd.concat([df[df['kind']=='original'],df[df['kind']=='comment'].sample(len(df[df['kind']=='original']),random_state=1)])
df['text'] = df['text'].apply(lambda x: clean_string(x))
df = df[df['text'].apply(lambda x: len(str(x).split(' '))>5)]


test = pd.read_excel(args.test_path)
remove_leaks = test['Original'].tolist() + test['Comment'].tolist()
remove_leaks = [clean_string(x) for x in remove_leaks]

df = df[~df['text'].isin(remove_leaks)]

# Start 2-step pu learning
# First step
print('First Step of PU Learning')
print(df['label'].value_counts())

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

print(df['class'].value_counts())

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

#prepare comment and original test data
X_test_comment = w2v.encode(test['Comment'].tolist())
X_test_original = w2v.encode(test['Original'].tolist())
y_test_comment = test['CommentTopic'].tolist()
y_test_original = test['OriginalTopic'].tolist()


def grid_search_xgb():
    best_score = 0
    best_parameters = {}
    for max_depth in tqdm(range(3,10,2)):
        for max_estimators in [50]+list(range(100,401,100)):
            for learning_rate in tqdm([0.01,0.05]+list(np.arange(0.1,1.01,0.1))):
                for reg_alpha in range(0,91,30):
                    for gamma in range(0,10,3):
                        for colsample_bytree in np.arange(0.5,1.01,0.25):
                            for min_child_weight in [0,1,4,8]:
                                clf = XGBClassifier(objective='binary:logistic',tree_method='gpu_hist', gpu_id=0,random_state=7,
                                                    max_depth = max_depth,n_estimators=max_estimators,learning_rate=learning_rate,
                                                reg_alpha=reg_alpha,gamma=gamma,colsample_bytree=colsample_bytree,min_child_weight=min_child_weight)
                                cv = cross_val_score(clf,X_train,y_train,cv=5,scoring='f1_weighted')
                                score = np.mean(cv)
                                if score >= best_score:
                                    best_parameters['max_depth']=max_depth
                                    best_parameters['n_estimators']=max_estimators
                                    best_parameters['learning_rate']=learning_rate
                                    best_parameters['reg_alpha']=reg_alpha
                                    best_parameters['gamma']=gamma
                                    best_parameters['colsample_bytree']=colsample_bytree
                                    best_parameters['min_child_weight']=min_child_weight
    return best_parameters

def random_search_xgb():
    param_grid = {'max_depth': range(3,10), 'n_estimators': list(range(50,501,50)), 'learning_rate': [0.01,0.05]+list(np.arange(0.1,1.01,0.1)),
                  'reg_alpha': range(0,91,15), 'gamma': range(0,10,3), 'colsample_bytree': np.arange(0.5,1.01,0.1), 'min_child_weight': [0,1,2,4,6,8]}
    param_list = list(ParameterSampler(param_grid, n_iter=args.tuning_iterations,random_state=7))
    best_score = 0
    best_parameters = {}
    for params in tqdm(param_list):
        xgb = XGBClassifier(objective='binary:logistic',tree_method='gpu_hist', gpu_id=0,random_state=7,**params)
        cv = cross_val_score(xgb,X_train,y_train,cv=5,scoring='f1_weighted')
        score = np.mean(cv)
        if score >= best_score:
            best_score=score
            best_parameters=params
    return best_parameters

if args.model_params is not None:   
    best_parameters = json.loads(args.model_params)
else:
    print(f'Tuning parameters, method = {args.tuning_method}')
    if args.tuning_method == 'grid':
        best_parameters = grid_search_xgb()
    elif args.tuning_method == 'random':
        best_parameters = random_search_xgb()
    print(f'Best parameters found: {best_parameters}')

#train the final model
clf = XGBClassifier(objective='binary:logistic',tree_method='gpu_hist', gpu_id=0,random_state=7,**best_parameters)
clf.fit(X_train,y_train)

#get statistics for final model and save data
comment_pred=clf.predict(X_test_comment)
original_pred=clf.predict(X_test_original)

def save_stats():
    stats = {}
    stats['f1_original'] = f1_score(test['OriginalTopic'].to_list(),original_pred, average='binary', pos_label=1)
    stats['f1_comment'] = f1_score(test['CommentTopic'].to_list(),comment_pred, average='binary', pos_label=1)
    stats['f1'] = (stats['f1_original']+stats['f1_comment'])/2
    stats['accuracy_original'] = accuracy_score(test['OriginalTopic'].to_list(), original_pred)
    stats['accuracy_comment'] = accuracy_score(test['CommentTopic'].to_list(), comment_pred)
    stats['accuracy'] = (stats['accuracy_original']+stats['accuracy_comment'])/2
    stats['precision_original'] = precision_score(test['OriginalTopic'].to_list(),original_pred, average='binary', pos_label=1)
    stats['precision_comment'] = precision_score(test['CommentTopic'].to_list(),comment_pred, average='binary', pos_label=1)
    stats['precision'] = (stats['precision_original']+stats['precision_comment'])/2
    stats['recall_original'] = recall_score(test['OriginalTopic'].to_list(),original_pred, average='binary', pos_label=1)
    stats['recall_comment'] = recall_score(test['CommentTopic'].to_list(),comment_pred, average='binary', pos_label=1)
    stats['recall'] = (stats['recall_original']+stats['recall_comment'])/2
    stats['true_positives_original'] = len(test[(test['OriginalTopic']==1) & (original_pred==1)])
    stats['true_positives_comment'] = len(test[(test['CommentTopic']==1) & (comment_pred==1)])
    stats['false_positives_original'] = len(test[(test['OriginalTopic']==0) & (original_pred==1)])
    stats['false_positives_comment'] = len(test[(test['CommentTopic']==0) & (comment_pred==1)])
    stats['true_negatives_original'] = len(test[(test['OriginalTopic']==0) & (original_pred==0)])
    stats['true_negatives_comment'] = len(test[(test['CommentTopic']==0) & (comment_pred==0)])
    stats['false_negatives_original'] = len(test[(test['OriginalTopic']==1) & (original_pred==0)])
    stats['false_negatives_comment'] = len(test[(test['CommentTopic']==1) & (comment_pred==0)])
    stats['model_name'] = args.model_name
    stats['model_parameters'] = best_parameters
    
    print('Model:'+args.model_name)
    print('Overall statistics: {}'.format({key:round(stats[key],3) for key in ['f1','accuracy','precision','recall']}))
    print('Original statistics: {}'.format({key:round(stats[key],3) for key in stats if key.endswith('original')}))
    print('Comment statistics: {}'.format({key:round(stats[key],3) for key in stats if key.endswith('comment')}))
    with open('./stats/model_stats.jsonl','a') as file:
        json.dump(stats,file)
        file.write('\n')

def save_predictions():
    predictions = {}
    predictions['original_predictions'] = original_pred.tolist()
    predictions['comment_predictions'] = comment_pred.tolist()
    predictions['model_name'] = args.model_name
    with open('./stats/model_predictions.jsonl','a') as file:
        json.dump(predictions,file)
        file.write('\n')

def save_confusion():
    def plot_confusion(y_true,y_pred,path):
        confusion = confusion_matrix(y_true,y_pred)
        sns.heatmap(confusion, annot=True,cmap='Blues',fmt='g')
        plt.xlabel('predicted',fontsize=14)
        plt.ylabel('true',fontsize=14)
        plt.savefig(path,format='eps')
        plt.clf()
    
    if not os.path.exists(f'./confusion/{args.model_name}'):
        os.makedirs(f'./confusion/{args.model_name}')
    
    plot_confusion(test['OriginalTopic'].to_list(),original_pred,f'./confusion/{args.model_name}/original.eps')
    plot_confusion(test['CommentTopic'].to_list(),comment_pred,f'./confusion/{args.model_name}/comment.eps')
    #plot confusion matrix for original and comment together
    plot_confusion(test['CommentTopic'].tolist()+test['OriginalTopic'].tolist(),comment_pred.tolist()+original_pred.tolist(),f'./confusion/{args.model_name}/overall.eps')

save_stats()
save_predictions()
save_confusion()

