# Topic Shifts as a Proxy for Assessing Politicization in Social Media
Repository for the paper Topic Shifts as a Proxy for Assessing Politicization in Social Media accepted at ICWSM 2024.

The data used is available at https://doi.org/10.5281/zenodo.10689619. Rerunning the scripts for the labelling and word2vec training may lead to slightly different results due to randomness. 

## How to run the basic experiments

First, download the data from the link above and extract it in this repository.

Then, run the script run_experiments.sh and it should give you the results for all the Two-step PU models shown in the paper.

There are other additional scripts in the scripts folder, which run the code according to what was used in the paper.

## Code Usage

**WARNING:** running label_news.py and trainw2v.py with the sample parameters will lead to the dataset downloaded being overwritten, which may lead to different results.

### label_news.py
Script to create the file with keyword based labels used in the first step of PU learning.
In folder [keyword_labelling](Code/Models/keyword_labelling/).

Parameters:

- --save_path: Path to save the labelled file, default = '../../../Data/labelled.jsonl'.
- --n_jobs: Number of processes to use to create the labelled file, default = 12.
- --keywords or -k: list of keywords separated by space to use for labelling.

Sample usage:
```
python label_news.py --save_path ../../../Data/labelled_3kw.jsonl --n_jobs 4 -k lula bolsonaro #eleicoes2022
```

### trainw2v.py
Script to train the word2vec model used on the second step of PU learning. In folder [word_embedding](Code/Models/word_embedding/).

Parameters:

- --vector_size: Size of the resulting word embeddings, defailt = 300.
- --n_jobs: Number of jobs to use to train the word2vec model, default = 16.

Sample usage:
```
python trainw2v.py --vector_size 300 --n_jobs 4
```

The model always saves to the path '../../../Data/keyword_classification/{vector_size}_dims_w2v_news.model'

### run_pu.py
Script to tune the parameters and/or evaluate the resulting Two-step PU model in the test data.In folder [Two-step_PU](Code/Models/Two-step_PU/).

Parameters:

- --w2v: Path to word2vec model, default = '../../../Data/Keyword_Classification/300_dims_w2v_news.model'
- --labelled_file: Path to labelled data, default = '../../../Data/labelled.jsonl'.
- --test_path: Path to test data, default = '../../../Data/labels/labeled_random_samples.xlsx'
- --model_params: Json string containing the parameters to use for the model. If not given, the script will search for good parameters instead.
- --tuning_method: Parameter tuning method, possible values are "random", "grid" or "none". Note that if you choose "none", you must give a value to --model_params.
- --tuning_iterations: Number of iterations to run the random search for, default = 500.
- --spy_tolerance: The tolerance of spies to use in the first step of the PU learning, default = 0.003. A desirable tolerance should lead to 10-20% of your data being classified as negative.
- --model_name: The name of your model, used to save relevant stats and visualizations, default = all_keywords

Sample usage:
```
python run_pu.py --labelled_file ../../../Data/labelled_3kw.jsonl --model_name 3kw --model_params "{\"reg_alpha\": 45, \"n_estimators\": 100, \"min_child_weight\": 2, \"max_depth\": 5, \"learning_rate\": 0.1, \"gamma\": 3, \"colsample_bytree\": 0.9999999999999999}"
```

### run_pu_scores.py
Script to create the file with all PU scores for all platforms, which is used for the analysis.

Parameters:

- --w2v: Path to word2vec model, default = '../../../Data/Keyword_Classification/300_dims_w2v_news.model'
- --labelled_file: Path to labelled data, default = '../../../Data/labelled.jsonl'.
- --model_params: Json string containing the parameters to use for the model. If not given, the script will search for good parameters instead.
- --parameter_path: Path to the file produced by the run_pu.py script that contains information about the final model and its parameters. Can be used instead of --model_params, default = './stats/model_stats.jsonl'.
- --model_name: The name of your model, used to save the resulting files, as well as to recover the stats produced in run_pu.py, default = all_keywords.
- --spy_tolerance: The tolerance of spies to use in the first step of the PU learning, default = 0.003. A desirable tolerance should lead to 10-20% of your data being classified as negative.

Sample usage, after running the run_pu.py script:
```
python run_pu_scores.py --labelled_file ../../../Data/labelled_3kw.jsonl --model_name 3kw
```
