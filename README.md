# GOALNET: Inferring Conjunctive Goal Predicates from Human Plan Demonstrations for Robot Instruction Following

This repository contains code implementation of the paper "GOALNET: Inferring Conjunctive Goal Predicates from Human Plan Demonstrations for Robot Instruction Following".

**Shreya Sharma, Jigyasa Gupta, Shreshth Tuli, Rohan Paul and Mausam**. Department of Computer Science and Engineering, Indian Institute of Techonology Delhi. Samsung R&D Institute India, Delhi. Department of Computing, Imperial College London, UK.

To appear in **Bridging the Gap Between AI Planning and Reinforcement Learning (PRL @ ICAPS) – Workshop at ICAPS 2022**.

## Abstract

Our goal is to enable a robot to learn how to sequence its actions to perform tasks specified as natural language instructions, given successful demonstrations from a human partner. The ability to plan high-level tasks can be factored as (i) inferring specific goal predicates that characterize the task implied by a language instruction for a given world state and (ii) synthesizing a feasible goal-reaching action-sequence with such predicates. For the former, we leverage a neural network prediction model, while utilizing a symbolic planner for the latter. We introduce a novel neuro-symbolic model, GOAL-NET, for contextual and task dependent inference of goal predicates from human demonstrations and linguistic task descriptions. GOALNET combines (i) learning, where dense representations are acquired for language instruction and the world state that enables generalization to novel settings and (ii) planning, where the cause-effect modeling by the symbolic planner eschews irrelevant predicates facilitating multi-stage decision making in large domains. GOALNET demonstrates a significant improvement (51%) in the task completion rate in comparison to a state-of-the-art rule-based approach on a benchmark data set displaying linguistic variations, particularly for multi-stage instructions.

<!-- ## Supplementary video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lUWU3rK1Gno/0.jpg)](https://www.youtube.com/watch?v=lUWU3rK1Gno) -->

## Getting Started

This implementation contains the GoalNet model mentioned in the paper for goal-constraint prediction along with action plan generation. 

## Model Training 

The model mentioned in the paper can be trained through the command

```bash
$ python3 main.py -m Simple -e $EXPERIMENT_NAME -r $TRAIN_DATA_PATH -v $VALIDATION_DATA_PATH -t $TEST_DATA_PATH
```
This command will train GOALNET on the training dataset for `NUM_EPOCHS` epochs specified in `main.py`. It will save a checkpoint file `results/EXPERIMENT_NAME/Simple_Model.pt` after the `EPOCH` epoch. It will also save a training graph `results/EXPERIMENT_NAME/Simple_graph.pdf` where train and validation loss and accuracy can be visualized. In the end, it will output the epoch (say `N`) corresponding to the maximum validation accuracy using early stopping criteria. The dataset is by default loaded from `data_clean` folder. It has `train`, `val` and `test` folders if not specified in the execution command.

## Model Testing 

The model mentioned in the paper can be tested through the command

```bash
$ python3 eval.py -m Simple -e $EXPERIMENT_NAME -t $TEST_DATA_PATH -s True/False
```

This command will run inference on the trained model stored in `results/EXPERIMENT_NAME/` and output `SJI`, `IED`, `GRR` and `F1` score.<br />
`-s` parameter if `True` will save the planner output ("pred_delta" ,"pred_delta_inv" "planner_action" and "planner_state_dict") in a json file in `results/EXPERIMENT_NAME/eval_json` folder. This can be later used to quickly compute new evaluation metrics on final output without running the `RINTANEN` planner again. To check evaluation metrics without running `RINTANEN` planner, set `-s` to `False`. It will load the pre-saved jsons from `results/EXPERIMENT_NAME/eval_json` and compute metric values.


## Sample Commands
```
python3 main.py -m Simple -e GoalNet_best -r train -v val -t test
```
```
python3 eval.py -m Simple -e GoalNet_best -t test -s True
```

**Pre-trained models:** The pretrained model mentioned in the GoalNet paper can be found `results/GoalNet_best` folder.


## Arxiv preprint
https://arxiv.org/abs/2205.07081.

## Cite this work
```bibtex
@article{sharma2022goalnet,
  title={GoalNet: Inferring Conjunctive Goal Predicates from Human Plan Demonstrations for Robot Instruction Following},
  author={Sharma, Shreya and Gupta, Jigyasa and Tuli, Shreshth and Paul, Rohan and Mausam},
  journal={International Conference on Automated Planning and Scheduling (ICAPS) - Planning and Reinforcement Learning Workshop},
  year={2022}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shreya Sharma, Jigyasa Gupta, Shreshth Tuli, Rohan Paul and Mausam
All rights reserved.

See License file for more details.
