import pickle, os, json

# LOADING DATASET SPECIFIC CONSTANTS
# couch - Loveseat, cd2 - Far Cry Game, Book_1 - Guiness Book
# objects + fluent attributes = all_objects
with open('./data/constants.json', 'r') as fh:
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
        instance_table = constants_dict['instance_table']
        graspableset = constants_dict['graspableset']
        all_objects_env = constants_dict['all_objects_env']
        IN_obj1 = constants_dict['IN_obj1']
        IN_obj2 = constants_dict['IN_obj2']
        ON_obj1 = constants_dict['ON_obj1']
        ON_obj2 = constants_dict['ON_obj2']
        Grasp_obj2 = constants_dict['Grasp_obj2']
        state_obj1 = constants_dict['state_obj1']
MAX_REL = max([len(all_object_states[obj]) for obj in all_object_states.keys()])
N_STATES = len(all_states)
N_objects = len(all_objects)
N_relations = len(all_relations)
N_fluents = len(all_fluents)
total_valid_constr0 = len(ON_obj1[0])*len(ON_obj2[0]) + len(IN_obj1[0])*len(IN_obj2[0])+ len(Grasp_obj2[0]) + len(all_objects_kitchen) + 31 #fluents of kitchen
total_valid_constr1 =  len(ON_obj1[1])*len(ON_obj2[1]) + len(IN_obj1[1])*len(IN_obj2[1]) + len(Grasp_obj2[1]) + N_objects + 7 # object fluents
total_valid_constr = [total_valid_constr0, total_valid_constr1]

# LOADING DATASET SPECIFIC VOCABULARY
with open('./data/vocab.json', 'r') as fh:
        VOCAB_VECT = json.load(fh)
        for obj in list(VOCAB_VECT):
                if '_' in obj:
                        VOCAB_VECT[obj.split('_')[0].lower()] = VOCAB_VECT[obj]

# LOADING DATASET AGNOSTIC CONCEPTNET EMBEDDINGS
with open('./jsons/conceptnet.json', 'r') as fh:
        conceptnet_vectors = json.load(fh)

# MODEL CONSTANTS
PRETRAINED_VECTOR_SIZE = 300
GRAPH_HIDDEN = 64
size, layers = (4, 2)
word_embed_size = PRETRAINED_VECTOR_SIZE