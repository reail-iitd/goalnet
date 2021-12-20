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
