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
$ python GoalNet/eval.py -m Tango -e GoalNet_exp -s True -t test 
```

Aggregated model (Table 2, Row 3)
```bash
$ python GoalNet/eval.py -m Aggregated -e GoalNet_exp -s True -t test
```

GoalNet (Trained model also provided) (Table 2, Row 4)
```bash
$ python GoalNet/main.py -m GoalNet -e GoalNet_exp -r train -v val -t test
$ python GoalNet/eval.py -m GoalNet -e GoalNet_exp -s True -t test
```

### Model Ablations
w/o Relational information (Table 2, Row 5)
```bash
$ python GoalNet/main.py -m GoalNet -e GoalNet_NoRelationInfo_exp -r train -v val -t test --no_relation
$ python GoalNet/eval.py -m GoalNet -e GoalNet_NoRelationInfo_exp -s True -t test
```

w/o Instance grounding (Table 2, Row 6)
```bash
$ python GoalNet/main.py -m GoalNet -e GoalNet_NoInstanceGrounding_exp -r train -v val -t test --no_instance_grounding
$ python GoalNet/eval.py -m GoalNet -e GoalNet_NoInstanceGrounding_exp -s True -t test
```

w/o δ− prediction(Table 2, Row 7)
```bash
$ python GoalNet_delta_g/main.py -m GoalNet -e GoalNet_delta_g_exp -r train -v val -t test
$ python GoalNet_delta_g/eval.py -m GoalNet -e GoalNet_delta_g_exp -s True -t test
```

w/o δ+ prediction (Table 2, Row 8)
```bash
$ python GoalNet_delta_g_inv/main.py -m GoalNet -e GoalNet_delta_g_inv_exp -r train -v val -t test
$ python GoalNet_delta_g_inv/eval.py -m GoalNet -e GoalNet_delta_g_inv_exp -s True -t test
```
w/o Temporal context encoding (Table 2, Row 9)
```bash
$ python GoalNet/main.py -m GoalNet -e GoalNet_NoTempContext_exp -r train -v val -t test --no_temporal_context
$ python GoalNet/eval.py -m GoalNet -e GoalNet_NoTempContext_exp -s True -t test
```
w/o Grammar mask (Table 2, Row 10)
```bash
$ python GoalNet/main.py -m GoalNet -e GoalNet_NoGrammarMask_exp -r train -v val -t test --no_grammar_mask
$ python GoalNet/eval.py -m GoalNet -e GoalNet_NoGrammarMask_exp -s True -t test
```

### Model explorations
Instruction encoding : Conceptnet (Table 2, Row 11)
```bash
$ python GoalNet/main.py -m GoalNet -e GoalNet_Conceptnet_exp -r train -v val -t test --conceptnet
$ python GoalNet/eval.py -m GoalNet -e GoalNet_Conceptnet_exp -s True -t test
```
Temporal Context (δ+t−1 ∪ δ−t−1) (Table 2, Row 12)
```bash
$ python GoalNet_tc_delta/main.py -m GoalNet -e GoalNet_tc_delta_exp -r train -v val -t test
$ python GoalNet_tc_delta/eval.py -m GoalNet -e GoalNet_tc_delta_exp -s True -t test
```
Temporal Context (st+1) (Table 2, Row 13)
```bash
$ python GoalNet_tc_state/main.py -m GoalNet -e GoalNet_tc_state_exp -r train -v val -t test
$ python GoalNet_tc_state/eval.py -m GoalNet -e GoalNet_tc_state_exp -s True -t test
```

Training using RINTANEN (Table 2, Row 14)
```bash
$ python GoalNet/main.py -m GoalNet -e GoalNet_Rintanen_exp -r train -v val -t test --rintanen
$ python GoalNet/eval.py -m GoalNet -e GoalNet_Rintanen_exp -s True -t test
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
$ python GoalNet/main.py -m GoalNet -e GoalNet_exp -r train -v val -t test 
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
$ python GoalNet/eval.py -m Tango -e GoalNet_exp -t verb_replacement_test -s True 
$ python GoalNet/eval.py -m Tango -e GoalNet_exp -t paraphrasing_test -s True 
$ python GoalNet_Star/eval.py -m Tango -e GoalNet_unseen_exp -t unseen_object_test -s True -o unseen
```
Aggregated - Verb Replacement, Paraphrasing and Unseen Objects (Table 3, Row 3)
```bash
$ python GoalNet/eval.py -m Aggregated -e GoalNet_exp -t verb_replacement_test -s True 
$ python GoalNet/eval.py -m Aggregated -e GoalNet_exp -t paraphrasing_test -s True 
$ python GoalNet_Star/eval.py -m Aggregated -e GoalNet_unseen_exp -t unseen_object_test -s True -o unseen
```
GoalNet - Verb Replacement, Paraphrasing and Unseen Objects (Table 3, Row 4)
```bash
$ python GoalNet/eval.py -m GoalNet -e GoalNet_exp -t verb_replacement_test -s True 
$ python GoalNet/eval.py -m GoalNet -e GoalNet_exp -t paraphrasing_test -s True
$ python GoalNet_Star/eval.py -m GoalNet -e GoalNet_unseen_exp -t unseen_object_test -s True -o unseen 
```
GoalNet* - Verb Replacement, Paraphrasing and Unseen Objects (Table 3, Row 5)
```bash
$ python GoalNet_Star/eval.py -m GoalNet_Star -e GoalNet_Star_exp -t verb_replacement_test -s True -o seen  
$ python GoalNet_Star/eval.py -m GoalNet_Star -e GoalNet_Star_exp -t paraphrasing_test -s True -o seen  
$ python GoalNet_Star/eval.py -m GoalNet_Star -e GoalNet_Star_exp -t unseen_object_test -s True -o unseen  
```

## About the code
```bash
python GoalNet/main.py -m GoalNet -e GoalNet_exp -r train -v val -t test -o seen
```
This command will train `GOALNET` on the training dataset for `NUM_EPOCHS` epochs specified in main.py. It will save a checkpoint file in `results/GoalNet_exp/GoalNet_Model.pt` after the EPOCH epoch. It will also save a training graph `results/GoalNet_exp/GoalNet_graph.pdf` where train and validation loss and accuracy can be visualized. In the end, it will output the epoch (say N) corresponding to the maximum validation accuracy using early stopping criteria. The dataset is loaded from `dataset` folder. It has train, val and test folders. It also contains the generalization dataset - verb_replacement_test, paraphrasing_test and unseen_object_test.

```bash
python GoalNet/eval.py -m GoalNet -e GoalNet_exp -t test -o seen -s True
```
This command will run inference on the trained model stored in `results/GoalNet_exp/` and output SJI, IED, GRR and F1 score.

### Input parameters
`-m` select the model to run - choices=['GoalNet','GoalNet_Star','Tango','Aggregated'] <br />
`-e` experiment name <br />
`-r` train dataset path (inside `dataset` folder) <br />
`-v` validation dataset path (inside `dataset` folder) <br />
`-t` test dataset path (inside `dataset` folder) <br />
`-s` parameter if `True` will save the planner output ("pred_delta" ,"pred_delta_inv" "planner_action" and "planner_state_dict") in a json file in results/GoalNet_exp/eval_json folder. <br />
`-o` parameter is `seen` when running evaluation for object set used in training and `unseen` for object set unseen in training




