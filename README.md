# GOALNET: Inferring Conjunctive Goal Predicates from Human Plan Demonstrations for Robot Instruction Following

This repository contains code implementation of the paper "GOALNET: Interleaving Neural Goal Predicate Inference with Classical Planning for Generalization in Robot Instruction Following".


Submitted to **37th AAAI Conference on Artifical Intelligence 2023**.

## Abstract

Our goal is to enable a robot to learn how to sequence its actions to perform tasks specified as natural language instructions, given successful demonstrations from a human partner. We build an iterative two-step approach that interleaves (i) inferring goal predicates implied by the language instruction for a given world state and (ii) synthesizing a feasible goal-reaching plan from that state. The agent executes the first action of the plan, andthe two steps are repeated. For goal prediction, we lever-age a neural network prediction model, while utilizinga classical planner for synthesizing plans. Our novel neuro-symbolic model, GOALNET, performs contextual and task dependent inference of goal predicatesfrom human demonstrations and a textual task description. GOALNET combines (i) learning, where dense representations are acquired for language instruction and the world state, enabling generalization to novel settings and (ii) planning, where the cause-effect modeling by the classical planner eschews irrelevant predicates, facilitating multi-stage decision making in large domains. GOALNET obtains 79% improvement in the goal reaching rate in comparison to a state-of-the-art rule- based approach on benchmark data with multi-stage instructions. Further, GOALNET can generalize to novel instructions for scenes with unseen objects.


## Getting Started

This implementation contains the GoalNet* mentioned in the paper for goal-constraint prediction along with action plan generation in case of unseen objects 

## Model Training 

The model mentioned in the paper can be trained through the command

```bash
$ python main.py -m GoalNet_Star -e GoalNet_Star_exp -r train -v val -t test -o seen 
```
This command will train GOALNET* on the training dataset for `NUM_EPOCHS` epochs specified in `main.py`. It will save a checkpoint file `results/GoalNet_Star_exp/GoalNet_Star_Model.pt` after the `EPOCH` epoch. It will also save a training graph `results/GoalNet_Star_exp/GoalNet_Star_graph.pdf` where train and validation loss and accuracy can be visualized. In the end, it will output the epoch (say `N`) corresponding to the maximum validation accuracy using early stopping criteria. The dataset is loaded from `data_clean` folder. It has `train`, `val` and `test` folders. <br />

In order to train the GOALNET model agnostic to object set size, we remove the adjacency matrix of the relation information from the encoder. To train this model, use the following command
```bash
$ python main.py -m GoalNet -e GoalNet_exp -r train -v val -t test -o seen
```

## Model Testing 

The model mentioned in the paper can be tested through the command

```bash
$ python eval.py -m GoalNet_Star -e GoalNet_Star_exp -t test -s True -o seen 
```

This command will run inference on the trained model stored in `results/GoalNet_Star_exp/` and output `SJI`, `IED`, `GRR` and `F1` score.<br />
`-s` parameter if `True` will save the planner output ("pred_delta" ,"pred_delta_inv" "planner_action" and "planner_state_dict") in a json file in `results/GoalNet_Star_exp/eval_json` folder. This can be used to quickly compute new evaluation metrics on final output without running the `RINTANEN` planner. To check evaluation metrics without running `RINTANEN` planner, set `-s` to `False`. It will load the pre-saved jsons from `results/GoalNet_Star_exp/eval_json` and compute metric values.
`-o` parameter is `seen` when running evaluation for object set used in training and `unseen` for object set unseen in training

Other commands to generate results mentione in paper <br />

Run GOALNET* with unseen object test set
```bash
$ python eval.py -m GoalNet_Star -e GoalNet_Star_exp -t unseen_object_test -s True -o unseen  
```
Run GOALNET* with verb replacement dataset
```bash
$ python eval.py -m GoalNet_Star -e GoalNet_Star_exp -t verb_replacement_test -s True -o seen 
```

Run GOALNET* with paraphrasing test set
```bash
$ python eval.py -m GoalNet_Star -e GoalNet_Star_exp -t paraphrasing_test -s True -o seen 
```

Run GOALNET with unseen object test set
```bash
$ python eval.py -m GoalNet -e GoalNet_exp -t unseen_object_test -s True -o unseen 
```

In order to run the two baselines mentioned in paper, `Tango` and `Aggregated` on unseen object test set, use the below two commands.
TANGO (Tuli et al. 2021), an imitation learning model that directly learns to predict actions from demonstrations. We use a refactored version of Tango assuming one to one correspondance between actions and goal predicates.
```bash
$ python eval.py -m Tango -e GoalNet_exp -t unseen_object_test -s True -o unseen
```

`Aggregated` is a variation of GOALNET, that excludes interleaving of goal prediction and planning. Here, SYMSIM is used to directly predict an aggregate predicate set, used by RINTANEN to output the final robot plan. Hence only evaluation part is changed.
```bash
$ python eval.py -m Aggregated -e GoalNet_exp -t unseen_object_test -s True -o unseen 
```

**Pre-trained models:** The pretrained model mentioned in the GoalNet paper can be found `results` folder.