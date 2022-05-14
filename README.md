# GOALNET: Inferring Conjunctive Goal Predicates from Human Plan Demonstrations for Robot Instruction Following

This repository contains code implementation of the paper "GOALNET: Inferring Conjunctive Goal Predicates from Human Plan Demonstrations for Robot Instruction Following".

**Shreya Sharma, Jigyasa Gupta, Shreshth Tuli, Rohan Paul and Mausam**. Department of Computer Science and Engineering, Indian Institute of Techonology Delhi. Department of Computing, Imperial College London, UK.

To appear in **Bridging the Gap Between AI Planning and Reinforcement Learning (PRL @ ICAPS) – Workshop at ICAPS 2022**.

## Abstract

Our goal is to enable a robot to learn how to sequence its actions to perform tasks specified as natural language instructions, given successful demonstrations from a human partner. The ability to plan high-level tasks can be factored as (i) inferring specific goal predicates that characterize the task implied by a language instruction for a given world state and (ii) synthesizing a feasible goal-reaching action-sequence with such predicates. For the former, we leverage a neural network prediction model, while utilizing a symbolic planner for the latter. We introduce a novel neuro-symbolic model, GOAL-NET, for contextual and task dependent inference of goal predicates from human demonstrations and linguistic task descriptions. GOALNET combines (i) learning, where dense representations are acquired for language instruction and the world state that enables generalization to novel settings and (ii) planning, where the cause-effect modeling by the symbolic planner eschews irrelevant predicates facilitating multi-stage decision making in large domains. GOALNET demonstrates a significant improvement (51%) in the task completion rate in comparison to a state-of-the-art rule-based approach on a benchmark data set displaying linguistic variations, particularly for multi-stage instructions.

<!-- ## Supplementary video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lUWU3rK1Gno/0.jpg)](https://www.youtube.com/watch?v=lUWU3rK1Gno) -->

## Getting Started

This implementation contains all the models mentioned in the paper for goal-constraint prediction along with action plan generation. This readme gives a broad idea of the work that has been accomplished. The code start point is `main.py`. For more details on replicating results, running the data collection platform and visualizing the collected dataset, refer to this [wiki](https://github.com/reail-iitd/goalnet/wiki).

<!-- For our GOALNET model, use $MODEL_NAME as **GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Tool_Action**. -->

## Arxiv preprint
<!-- https://arxiv.org/abs/2105.04556. -->

## Cite this work
```
@article{tuli2021tango,
  title={TANGO: Commonsense Generalization in Predicting Tool Interactions for Mobile Manipulators},
  author={Tuli, Shreshth and Bansal, Rajas and Paul, Rohan and Mausam},
  journal={arXiv preprint arXiv:2105.04556},
  year={2021}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shreshth Tuli, Rajas Basal, Rohan Paul, Mausam
All rights reserved.

See License file for more details.
