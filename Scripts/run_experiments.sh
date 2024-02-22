cd ../Code/Models/Two-step_PU
echo "Experiment with 3 keywords"
python run_pu.py --labelled_file ../../../Data/labelled_3kw.jsonl --model_name 3kw --model_params "{\"reg_alpha\": 45, \"n_estimators\": 100, \"min_child_weight\": 2, \"max_depth\": 5, \"learning_rate\": 0.1, \"gamma\": 3, \"colsample_bytree\": 0.9999999999999999}"
echo "Experiment with 6 keywords"
python run_pu.py --labelled_file ../../../Data/labelled_6kw.jsonl --model_name 6kw --model_params "{\"reg_alpha\": 0, \"n_estimators\": 400, \"min_child_weight\": 1, \"max_depth\": 5, \"learning_rate\": 0.1, \"gamma\": 0, \"colsample_bytree\": 0.9999999999999999}"
echo "Experiment with all keywords"
python run_pu.py --labelled_file ../../../Data/labelled_all_kw.jsonl --model_name all_kw --model_params "{\"reg_alpha\": 30, \"n_estimators\": 150, \"min_child_weight\": 6, \"max_depth\": 9, \"learning_rate\": 0.2, \"gamma\": 3, \"colsample_bytree\": 0.7999999999999999}"
cd ../../../Scripts