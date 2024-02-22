cd ../Code/Models/keyword_labelling
python label_news.py --save_path ../../../Data/labelled_football.jsonl -k flamengo corinthians palmeiras vasco gremio cruzeiro atletico santos futebol
cd ../Two-step_PU
python run_pu_scores.py --labelled_file ../../../Data/labelled_football.jsonl --model_name football_keywords --spy_tolerance 0.001
cd ../../../Scripts