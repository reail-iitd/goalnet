from .datapoint import *

# gives set of all datapoint graphs - complete set of datapoints
class DGLDataset():
    def __init__(self, program_dir, embed):
        self.graphs = []
        self.lang = []
        self.delta_g = []
        self.sents = []
        self.goalObjectsVec = []
        self.objects = []
        self.obj_states = []
        self.init = []
        self.nodes_name = []
        self.env_objects = []
        self.action_seq = []
        self.delta_g_inv = []
        self.file_name = []
        for path, dirs, files in tqdm(list(os.walk(program_dir))):
            if ("train" in program_dir): random.shuffle(files)
            for f in files:
                dp = encode_datapoint(path + "/" + f, embed)
                if dp is None: continue
                self.graphs.append(dp[0])
                self.lang.append(dp[1])
                self.delta_g.append(dp[2])
                self.sents.append(dp[3])
                self.goalObjectsVec.append(dp[4])
                self.objects.append(dp[5])
                self.obj_states.append(dp[6])
                self.nodes_name.append(dp[7])  # This is sent to model.py to analyse attention weights - these are node names
                self.init.append(dp[8])
                self.env_objects.append(dp[9])
                self.action_seq.append(dp[10])
                self.delta_g_inv.append(dp[11])
                self.file_name.append(dp[12])
        self.features = graphs[0].ndata['feat'].shape[1] if len(graphs) > 0 else 0