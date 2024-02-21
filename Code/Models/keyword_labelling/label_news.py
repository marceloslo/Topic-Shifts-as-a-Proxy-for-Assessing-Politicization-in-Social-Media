import sys
import argparse,os

if __name__ == "__main__":
    sys.path.insert(1,'../../../')
    from Code.Models.keyword_labelling.weak_label_creator import WeakLabelCreator

    parser = argparse.ArgumentParser(description='Label news')
    parser.add_argument('--save_path', type=str, default='../../../Data/labelled.jsonl', help='Path to save the labelled file')
    parser.add_argument('--keywords','-k',nargs='+', type=str, required=True, help='Keywords to use for labelling')
    parser.add_argument('--n_jobs', type=int, default=12, help='Number of jobs to use')
    args = parser.parse_args()

    #remove file with same name if it exists
    if os.path.exists(args.save_path):
        os.remove(args.save_path)

    keywords = [kw.split(' ') for kw in args.keywords]
    print(keywords)
    for platform in ['Youtube','TikTok','Twitter']:
        print('Creating labels for {}'.format(platform))
        creator = WeakLabelCreator(platform=platform,path=f'../../../Data/{platform}/news.jsonl',n_jobs=args.n_jobs,keywords=keywords)
        creator.create_labelled_file(path_to_save=args.save_path)
        print(f'End {platform}')