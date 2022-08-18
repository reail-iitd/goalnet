from tqdm import tqdm
from .datapoint import *

# gives set of all datapoint graphs - complete set of datapoints
class DGLDataset():
    def __init__(self, program_dir, shuffle=False):
        self.dp_list = []
        all_files = os.listdir(program_dir)
        for f in tqdm(all_files, ncols=80):
            self.dp_list.append(Datapoint(program_dir + "/" + f))
        if shuffle: random.shuffle(self.dp_list)
        if opts.no_relational_info:
            self.features = len(all_non_fluents) + MAX_REL + word_embed_size
        else:
            self.features = len(all_non_fluents) + MAX_REL + word_embed_size + 1
