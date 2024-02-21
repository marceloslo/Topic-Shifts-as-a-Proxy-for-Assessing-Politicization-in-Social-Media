if __name__ == "__main__":
    import gc
    import pandas as pd
    from nltk.corpus import stopwords
    import sys,argparse
    sys.path.insert(1,'../../../')
    from Code.Models.word_embedding.w2v import Word2vec
    from Code.Utils.string_operations import clean_string,tokenize

    parser = argparse.ArgumentParser(description='Train w2v model')
    parser.add_argument('--vector_size', type=int, default=300, help='Size of the word vectors')
    parser.add_argument('--n_jobs', type=int, default=16, help='Number of jobs to use')
    args = parser.parse_args()
    stopwords = set(stopwords.words('portuguese'))

    trainset=[]
    #first, create auxiliary file
    for platform in ['TikTok','Twitter','Youtube']:
        print(f'Reading plaftorm: {platform}')
        for df in pd.read_json(f'../../../Data/{platform}/news.jsonl',encoding='utf-8',lines=True, chunksize=1000000):
            gc.collect()
            if platform == 'TikTok':
                df = df[['Url','Id','Comment','VideoDescription']]
                df.rename(columns={"Url":"OriginalId","VideoDescription":"Original"},inplace=True)
            elif platform == "Twitter":
                df = df[['ParentId','Id','Comment','OriginalComment']]
                df.rename(columns={"ParentId":"OriginalId",'OriginalComment':'Original'},inplace=True)
            else:
                df = df[['VideoId','Id','Comment','VideoTitle','VideoDescription']]
                df['Original'] = df['VideoTitle']+' '+df['VideoDescription'].fillna('')
                df.drop(columns=['VideoDescription','VideoTitle'],inplace=True)
                df.rename(columns={"VideoId":"OriginalId"},inplace=True)
            gc.collect()
            trainset.append(df)
        print(f'finished reading {platform}\n')
    trainset=pd.concat(trainset)
    gc.collect()

    comment_df = trainset[["Comment","Id"]].drop_duplicates(subset=['Comment'])
    comment_df = comment_df.rename(columns={"Comment":"Text"})
    original_df = trainset[["Original","OriginalId"]].drop_duplicates(subset=["Original"])
    original_df = original_df.rename(columns={"Original":"Text","OriginalId":"Id"})
    del trainset

    print("preprocessing docs")
    docs = []
    for line in comment_df.to_dict(orient="records"):
        line['Text'] = tokenize(clean_string(line['Text']),minsize=2,stopwords=stopwords)
        if len(line['Text']) < 2:
            continue
        docs.append(line['Text'])
    del comment_df

    for line in original_df.to_dict(orient="records"):
        line['Text'] = tokenize(clean_string(line['Text']),minsize=2,stopwords=stopwords)
        if len(line['Text']) < 2:
            continue
        docs.append(line['Text'])
    del original_df

    gc.collect()
    
    w2v = Word2vec(vector_size=args.vector_size,n_jobs=args.n_jobs)
    w2v.fit(docs)
    w2v.save(f'../../../Data/keyword_classification/{args.vector_size}_dims_w2v_news.model')