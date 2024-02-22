cd ../Code/Models/keyword_labelling
python label_news.py --save_path ../../../Data/labelled_football.jsonl -k flamengo corinthians palmeiras vasco gremio cruzeiro atletico santos futebol
cd ../Two-step_PU
python run_pu.py --labelled_file ../../../Data/labelled_football.jsonl --model_name football_keywords --tuning_method random --tuning_iterations 5000
python run_pu_scores.py --labelled_file ../../../Data/labelled_football.jsonl --model_name football_keywords
cd ../../../Scripts