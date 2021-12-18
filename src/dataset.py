from .datapoint import *

# gives set of all datapoint graphs - complete set of datapoints
class DGLDataset():
    def __init__(self, program_dir, embed):
        graphs, lang_embed, delta_g, sents, goalObj_embed, objects, obj_states, nodes_name, initial_env, env_objects, action_seq = \
            [], [], [], [], [], [], [], [], [], [], []
        delta_g_inv = []
        file_name = []
        all_files = list(os.walk(program_dir))
        tmp_cnt = 0
        for path, dirs, files in tqdm(all_files):
            if ("train" in program_dir):
                random.shuffle(files)
            for f in files:
                tmp_cnt += 1
                dp = encode_datapoint(path + "/" + f, embed)
                if dp is None:
                    continue
                graphs.append(dp[0])
                lang_embed.append(dp[1])
                delta_g.append(dp[2])
                sents.append(dp[3])
                goalObj_embed.append(dp[4])
                objects.append(dp[5])
                obj_states.append(dp[6])
                nodes_name.append(dp[7])  # This is sent to model.py to analyse attention weights - these are node names
                initial_env.append(dp[8])
                env_objects.append(dp[9])
                action_seq.append(dp[10])
                delta_g_inv.append(dp[11])
                file_name.append(dp[12])
        self.graphs = graphs
        self.lang = lang_embed
        self.delta_g = delta_g
        self.sents = sents
        self.goalObjectsVec = goalObj_embed
        self.objects = objects
        self.obj_states = obj_states
        self.init = initial_env
        self.nodes_name = nodes_name
        self.env_objects = env_objects
        self.action_seq = action_seq
        self.delta_g_inv = delta_g_inv
        self.file_name = file_name
        if len(graphs) > 0:
            self.features = graphs[0].ndata['feat'].shape[1]
        else:
            self.features = 0