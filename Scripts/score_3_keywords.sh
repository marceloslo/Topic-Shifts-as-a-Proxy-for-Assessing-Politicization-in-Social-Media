cd ../Code/Models/keyword_labelling
python label_news.py --save_path ../../../Data/labelled_3kw.jsonl -k lula bolsonaro "#eleicoes2022"
cd ../Two-step_PU
python run_pu.py --labelled_file ../../../Data/labelled_3kw.jsonl --model_name 3_keywords --tuning_method random --tuning_iterations 5000
python run_pu_scores.py --labelled_file ../../../Data/labelled_3kw.jsonl --model_name 3_keywords 
cd ../../../Scripts