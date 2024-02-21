import re
import unicodedata

def clean_string(string,keep_mentions=False,keep_links=False,keep_emojis=False,keep_punctuation=False,
                 keep_diacritics=False,lower=True,keep_hashtags=True):
    '''
    Converts string to lowercase and removes undesired characters, including emojis, 
    punctuation, links, mentions, diacritics, and hashtags, depending on the parameters.

    Default: - remove links, mentions, emojis, punctuation and diacritics
             - keep hashtags
             - convert string to lower  
    '''

    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    if lower:
        string = string.lower()
    #retirando newline/tabs
    string = string.replace('\\n',' ')
    string = string.replace('\n',' ')
    string = string.replace('\r',' ')
    string = string.replace('\t',' ')
    #retirando acentos
    if not keep_diacritics:
        string = unicodedata.normalize('NFKD', string).encode('ASCII','ignore').decode('ASCII')
    #retirando menções(@user)
    if not keep_mentions:
        string = re.sub(r'@[a-zA-Z0-9_]*','',string)
    #retirando links
    if not keep_links:
        string = re.sub(r'https?:[^\s]+','',string)
    #retirando pontuação(exceto hashtags)
    if not keep_punctuation:
        string = re.sub(r'[\\\,\.\!\?\/|()\'\=\*_\":{}\+;\[\]&\#]',' ',string)
        string = re.sub(r'[-]','',string)
    #retirando hashtags
    if not keep_hashtags:
        column = column.map(lambda text : re.sub(r'#[^\s]*','',text))
    #retirando emojis
    if not keep_emojis:
        string = emoj.sub(r'', string)
    #retirando vazios em excesso
    string = re.sub(' +',' ',string)
    return string

def remove_consecutive_letters(string):
    '''
    Example: lulaaaaa -> lula
    up to 2 'r' or 's' or 'e' are allowed due to portuguese sintax
    '''
    string = re.sub(r'([^eErRsS])\1+',r'\1',string)
    string = re.sub(r'([eErRsS])\1+',r'\1\1',string)
    return string

def tokenize(string,stopwords=set(),minsize=0):
    '''
    tokenizes a single string
    stopwords: set of words to remove from tokens, default = empty set
    minsize: minimum size of tokens, default = 0
    '''
    tokens = string.split(' ')
    return [token for token in tokens if token not in stopwords and len(token) >= minsize]