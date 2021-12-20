from tqdm import tqdm
from .datapoint import *

# gives set of all datapoint graphs - complete set of datapoints
class DGLDataset():
    def __init__(self, program_dir, shuffle=False):
        self.dp_list = []
        all_files = list(os.walk(program_dir))
        for path, dirs, files in tqdm(all_files):
            if shuffle: random.shuffle(files)
            for f in files:
                self.dp_list.append(Datapoint(path + "/" + f))
        self.features = len(all_non_fluents) + MAX_REL + word_embed_size