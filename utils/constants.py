import pickle, os, json, torch
import numpy as np
from .util import opts

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DATASET PATH
data_file = "./data_clean/"

# LOADING DATASET SPECIFIC CONSTANTS
# couch - Loveseat, cd2 - Far Cry Game, Book_1 - Guiness Book
# objects + fluent attributes = all_objects
with open(f'{data_file}constants.json', 'r') as fh:
        constants_dict = json.load(fh)
        all_objects = constants_dict['all_objects']
        all_objects_kitchen = constants_dict['all_objects_kitchen']
        all_objects_living = constants_dict['all_objects_living']
        all_obj_prop = constants_dict['all_obj_prop']
        all_objects_spaced = constants_dict['all_objects_spaced']
        all_fluents = constants_dict['all_fluents']
        all_non_fluents = constants_dict['all_non_fluents']
        all_states = constants_dict['all_states']
        all_relations = constants_dict['all_relations']
        all_object_states = constants_dict['all_object_states']
        all_possible_actions = constants_dict['all_possible_actions']
        all_possible_actions = [i.lower() for i in all_possible_actions]
        instance_table = constants_dict['instance_table']
        graspableset = constants_dict['graspableset']
        all_objects_env = constants_dict['all_objects_env']
        IN_obj1 = constants_dict['IN_obj1']
        IN_obj2 = constants_dict['IN_obj2']
        ON_obj1 = constants_dict['ON_obj1']
        ON_obj2 = constants_dict['ON_obj2']
        Grasp_obj2 = constants_dict['Grasp_obj2']
        state_obj1 = constants_dict['state_obj1']

        universal_objects = constants_dict['universal_objects']
        universal_objects_kitchen = constants_dict['universal_objects_kitchen']
        universal_objects_living = constants_dict['universal_objects_living']
        universal_objects_prop = constants_dict['universal_objects_prop']
        universal_object_states = constants_dict['universal_object_states']

if(opts.run_option == "train"):
        universal_objects = all_objects
        universal_objects_kitchen = all_objects_kitchen
        universal_objects_living = all_objects_living
        universal_objects_prop = all_obj_prop
        universal_object_states = all_object_states

all_objects_lower = [i.lower() for i in all_objects]
universal_objects_lower = [i.lower() for i in universal_objects]
all_relations_lower = [i.lower() for i in all_relations]
all_fluents_lower = [i.lower() for i in all_fluents]
all_obj_prop_lower = {}
for obj in all_obj_prop: all_obj_prop_lower[obj.lower()] = all_obj_prop[obj]
all_object_states_lower = {}
for obj in all_object_states: all_object_states_lower[obj.lower()] = [i.lower() for i in all_object_states[obj]]
universal_object_states_lower = {}
for obj in universal_object_states: universal_object_states_lower[obj.lower()] = [i.lower() for i in universal_object_states[obj]]
MAX_REL = max([len(universal_object_states[obj]) for obj in universal_object_states.keys()])
N_STATES = len(all_states)
N_objects = len(universal_objects)
N_relations = len(all_relations)
N_fluents = len(all_fluents)
total_valid_constr0 = len(ON_obj1[0])*len(ON_obj2[0]) + len(IN_obj1[0])*len(IN_obj2[0])+ len(Grasp_obj2[0]) + len(all_objects_kitchen) + 31 #fluents of kitchen
total_valid_constr1 =  len(ON_obj1[1])*len(ON_obj2[1]) + len(IN_obj1[1])*len(IN_obj2[1]) + len(Grasp_obj2[1]) + N_objects + 7 # object fluents
total_valid_constr = [total_valid_constr0, total_valid_constr1]
mask_kitchen = torch.Tensor([1 if obj in universal_objects_kitchen else 0 for obj in universal_objects])
mask_living = torch.Tensor([1 if obj in universal_objects_living else 0 for obj in universal_objects])
mask_stateful = torch.Tensor([1 if obj in universal_object_states else 0 for obj in universal_objects])
state_masks = {}
for obj in universal_object_states:
        state_masks[obj] = torch.Tensor([(1 if state in universal_object_states[obj] else 0) for state in all_fluents])

# LOADING DATASET SPECIFIC VOCABULARY
with open(f'{data_file}vocab.json', 'r') as fh:
        VOCAB_VECT = json.load(fh)
        for obj in list(VOCAB_VECT):
                if '_' in obj:
                        VOCAB_VECT[obj.split('_')[0].lower()] = VOCAB_VECT[obj]
                VOCAB_VECT[obj.lower()] = VOCAB_VECT[obj]

# LOADING DATASET AGNOSTIC CONCEPTNET EMBEDDINGS
with open('./jsons/conceptnet.json', 'r') as fh:
        conceptnet_vectors = json.load(fh)

with open('./jsons/gen_obj.json', 'r') as fh:
        gen_obj_conceptnet = json.load(fh)

all_vectors = {}
for obj in conceptnet_vectors: all_vectors[obj.lower()] = np.array(conceptnet_vectors[obj])
for obj in VOCAB_VECT: all_vectors[obj.lower()] = np.array(VOCAB_VECT[obj])
def closest(word):
        word = word.lower()
        keys = list(all_vectors.keys())
        dists = [np.mean((all_vectors[word] - all_vectors[i]) ** 2) for i in all_vectors if i not in word]
        closest = np.argsort(dists)[:5]
        return keys[closest[0]], keys[closest[1]], keys[closest[2]], keys[closest[3]], keys[closest[4]]

# MODEL CONSTANTS
PRETRAINED_VECTOR_SIZE = 300
GRAPH_HIDDEN = 64
size, layers = (4, 2)
word_embed_size = PRETRAINED_VECTOR_SIZE
SBERT_VECTOR_SIZE = 384
batch_size = 32