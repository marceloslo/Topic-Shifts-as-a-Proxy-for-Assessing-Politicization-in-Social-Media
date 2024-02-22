cd ../Code/Models/keyword_labelling
python label_news.py --save_path ../../../Data/labelled_all_kw.jsonl -k lula bolsonaro "#eleicoes2022" partido presidencia candidatura eleicoes eleitoral presidente debate eleicao
cd ../Two-step_PU
python run_pu.py --labelled_file ../../../Data/labelled_all_kw.jsonl --model_name all_keywords --tuning_method random --tuning_iterations 5000
python run_pu_scores.py --labelled_file ../../../Data/labelled_all_kw.jsonl --model_name all_keywords
cd ../../../Scripts