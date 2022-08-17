# GOALNET: Inferring Conjunctive Goal Predicates from Human Plan Demonstrations for Robot Instruction Following

This repository contains code implementation of the paper "GOALNET: Interleaving Neural Goal Predicate Inference with Classical Planning for Generalization in Robot Instruction Following".


Submitted to **37th AAAI Conference on Artifical Intelligence 2023**.

## Abstract

Our goal is to enable a robot to learn how to sequence its actions to perform tasks specified as natural language instructions, given successful demonstrations from a human partner. We build an iterative two-step approach that interleaves (i) inferring goal predicates implied by the language instruction for a given world state and (ii) synthesizing a feasible goal-reaching plan from that state. The agent executes the first action of the plan, andthe two steps are repeated. For goal prediction, we lever-age a neural network prediction model, while utilizinga classical planner for synthesizing plans. Our novel neuro-symbolic model, GOALNET, performs contextual and task dependent inference of goal predicatesfrom human demonstrations and a textual task description. GOALNET combines (i) learning, where dense representations are acquired for language instruction and the world state, enabling generalization to novel settings and (ii) planning, where the cause-effect modeling by the classical planner eschews irrelevant predicates, facilitating multi-stage decision making in large domains. GOALNET obtains 79% improvement in the goal reaching rate in comparison to a state-of-the-art rule- based approach on benchmark data with multi-stage instructions. Further, GOALNET can generalize to novel instructions for scenes with unseen objects.


## Installation
Install pip packages using
```bash
$ pip3 install -r requirements.txt
```

Run the following on python interpreter
```bash
$ import nltk
$ nltk.download('punkt')
$ nltk.download('averaged_perceptron_tagger')
```

## Table 2 results
To reproduce the results mentioned in Table 2, please run following commands

Tango Model (Table 2, Row 2)
```bash
$ python 
```

Aggregated model (Table 2, Row 3)
```bash
$ python 
```

GoalNet (Trained model also provided) (Table 2, Row 4)
```bash
$ python main.py
$ python eval.py
```

### Model Ablations
w/o Relational information (Table 2, Row 5)
```bash
$ python main.py
$ python eval.py
```

w/o Instance grounding (Table 2, Row 6)
```bash
$ python main.py
$ python eval.py
```

w/o δ− prediction(Table 2, Row 7)
```bash
$ python main.py
$ python eval.py
```

w/o δ+ prediction (Table 2, Row 8)
```bash
$ python main.py
$ python eval.py
```
w/o Temporal context encoding (Table 2, Row 9)
```bash
$ python main.py
$ python eval.py
```
w/o Grammar mask (Table 2, Row 10)
```bash
$ python main.py
$ python eval.py
```

### Model explorations
Instruction encoding : Conceptnet (Table 2, Row 11)
```bash
$ python main.py
$ python eval.py
```
Temporal Context (δ+t−1 ∪ δ−t−1) (Table 2, Row 12)
```bash
$ python main.py
$ python eval.py
```
Temporal Context (st+1) (Table 2, Row 13)
```bash
$ python main.py
$ python eval.py
```

Training using RINTANEN (Table 2, Row 14)
```bash
$ python main.py
$ python eval.py
```
GOALNET* (Table 2, Row 15)
```bash
$ python GoalNet_Star/main.py -m GoalNet_Star -e GoalNet_Star_exp -r train -v val -t test -o seen
$ python GoalNet_Star/eval.py -m GoalNet_Star -e GoalNet_Star_exp -t test -s True -o seen
```
## Table 3 results
To reproduce the results mentioned in Table 3, please run following commands

### Model training
GoalNet (Trained model provided <mention path>)
```bash
$ python 
```

GoalNet (modified to handle unseen object set) (Trained model provided <mention path>)
```bash
$ python GoalNet_Star/main.py -m GoalNet -e GoalNet_unseen_exp -r train -v val -t test -o seen 
```

GoalNet* (Trained model provided <mention path>)
```bash
$ python GoalNet_Star/main.py -m GoalNet_Star -e GoalNet_Star_exp -r train -v val -t test -o seen 
```


Tango - Verb Replacement, Paraphrasing and Unseen Objects (Table 3, Row 2)
```bash
$ python 
$ python 
$ python GoalNet_Star/eval.py -m Tango -e GoalNet_unseen_exp -t unseen_object_test -s True -o unseen
```
Aggregated - Verb Replacement, Paraphrasing and Unseen Objects (Table 3, Row 3)
```bash
$ python 
$ python 
$ python GoalNet_Star/eval.py -m Aggregated -e GoalNet_unseen_exp -t unseen_object_test -s True -o unseen
```
GoalNet - Verb Replacement, Paraphrasing and Unseen Objects (Table 3, Row 4)
```bash
$ python 
$ python 
$ python GoalNet_Star/eval.py -m GoalNet -e GoalNet_unseen_exp -t unseen_object_test -s True -o unseen 
```
GoalNet* - Verb Replacement, Paraphrasing and Unseen Objects (Table 3, Row 5)
```bash
$ python GoalNet_Star/eval.py -m GoalNet_Star -e GoalNet_Star_exp -t verb_replacement_test -s True -o seen  
$ python GoalNet_Star/eval.py -m GoalNet_Star -e GoalNet_Star_exp -t paraphrasing_test -s True -o seen  
$ python GoalNet_Star/eval.py -m GoalNet_Star -e GoalNet_Star_exp -t unseen_object_test -s True -o unseen  
```

## About the code
This command will train GOALNET* on the training dataset for NUM_EPOCHS epochs specified in main.py. It will save a checkpoint file results/GoalNet_Star_exp/GoalNet_Star_Model.pt after the EPOCH epoch. It will also save a training graph results/GoalNet_Star_exp/GoalNet_Star_graph.pdf where train and validation loss and accuracy can be visualized. In the end, it will output the epoch (say N) corresponding to the maximum validation accuracy using early stopping criteria. The dataset is loaded from data_clean folder. It has train, val and test folders.
This command will run inference on the trained model stored in results/GoalNet_Star_exp/ and output SJI, IED, GRR and F1 score.
-s parameter if True will save the planner output ("pred_delta" ,"pred_delta_inv" "planner_action" and "planner_state_dict") in a json file in results/GoalNet_Star_exp/eval_json folder. This can be used to quickly compute new evaluation metrics on final output without running the RINTANEN planner. To check evaluation metrics without running RINTANEN planner, set -s to False. It will load the pre-saved jsons from results/GoalNet_Star_exp/eval_json and compute metric values. -o parameter is seen when running evaluation for object set used in training and unseen for object set unseen in training



