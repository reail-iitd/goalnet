# goal-prediction
Goal Constraint learning

## Todo
Critical:
1. Loss masking based on interaction predicted (teacher forcing initially) [done]
2. Instance agnostic state representation (max of states of all instances --> one-hot) [done]
3. Crowd-sourced data cleaning 

Code structure:
1. Dataset, conceptnet embeddings, vocab etc to be human readable [done]
2. Hard coded elements to be encoded in jsons.
3. Folder structure
4. Ensure same states for same objects with different instance [done]

New Issues - 
1. ConceptNet Embeddings -- Give same conceptnet for weird words like loveseat as seat only
2. Elif correction in  dence_vector
3. Structure the datapoint class
4. DGL dataset as list of datapoints
5. Vocab embeddings (other than avg)
6. DGL graph has all_object nodes irrespective of the domain --> prediction on all_objects --> masking
6. Masking based on object set (for likelihood scores)

Shreya TO DOs:
1. Try ovrefitting on training data
    a. Use Hetero GAT
    b. Hyperparameter Tuning
2. Resolve segfault issue in planner
3. Check BCE Loss correctness for number of predicates in relation

Issues fixed:
3. Line 274 -  277 in util.py
4. Fine tune model to remove training bias
5. Calculate f1, p, r for delta_g, g_p
6. Reduce teacher forcing inversely compared to accuracy
7. How is pruning working in ACL16
8. ACL16 is updating on clause phase
9. Pre-process the state output from planner to handle lower and upper case inconsistency
10. Non-fluent property predicated
