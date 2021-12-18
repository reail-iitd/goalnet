from random import shuffle
from tqdm import tqdm
from .datapoint import *

# gives set of all datapoint graphs - complete set of datapoints
class DGLDataset():
    def __init__(self, program_dir, embed, suffle):
        self.dp_list = []
        all_files = list(os.walk(program_dir))
        for path, dirs, files in tqdm(all_files):
            if shuffle:
                random.shuffle(files)
            for f in files:
                dp = Datapoint()
                dp.load_point(path + "/" + f)
                dp.encode_datapoint()
                self.dp_list.append(dp)

        if len(self.dp_list) > 0:
            self.features = self.dp_list[0].graph.ndata['feat'].shape[1]
        else:
            self.features = 0