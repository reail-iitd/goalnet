from main import *

def loaddps(path):
    dps = []
    for f in tqdm(os.listdir(path), ncols=80):
        file_path = path + '/' + f
        with open(file_path, "r") as fh:
            dps.append(json.load(fh))
    return dps

def convertdp(dp, all_possible_rel, tochange = 1):
    rel_in_all_states = set(dp['initial_states'][0])
    for state in dp['initial_states']:
        rel_in_all_states.intersection_update(set(state))
    rel_in_delta = set(dp['delta_g'][0])
    for i, delta_g in enumerate(dp['delta_g']):
        rel_in_delta = rel_in_delta.union(set(dp['delta_g'][i]))
        rel_in_delta = rel_in_delta.union(set(dp['delta_g_inv'][i]))
    # remove a common relation from all states not in delta_g
    for _ in range(tochange):
        rel = random.choice(list(rel_in_all_states.difference(rel_in_delta)))
        for i, state in enumerate(dp['initial_states']):
            dp['initial_states'][i].remove(rel)
    # add another relation to all states
    for _ in range(tochange):
        rel = random.choice(list(all_possible_rel))
        for i, state in enumerate(dp['initial_states']):
            dp['initial_states'][i].append(rel) 
    return dp

def augment(all_possible_rel):
    test_path = data_file + 'test/'
    val_path = data_file + 'val/'
    train_path = data_file + 'train/'
    train, test, val = loaddps(train_path), loaddps(test_path), loaddps(val_path)
    for i in range(len(test)):
        dp = random.choice(train + val)
        dp = convertdp(dp, all_possible_rel)
        new_file_path = train_path + '/a_' + str(i) + '.json'
        with open(new_file_path, 'w') as fh:
            json.dump(dp, fh, indent=4)

if __name__ == '__main__':
    all_possible_rel = set()
    all_files = os.listdir(data_file + 'train/')
    for f in all_files:
        # create set of all relations
        with open(data_file + 'train/' + f, "r") as fh:
            dp = json.load(fh)
        for state in dp['initial_states']:
            for rel in state:
                all_possible_rel.add(rel)
        # remove previously augmented files
        if 'a_' in f:
            os.remove(data_file + 'train/' + f)
    # add augmented data
    augment(all_possible_rel)