import sys
from utils.util import *
from src.model import *
from src.dataset import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 32

def backprop(data, optimizer, model, num_objects, epoch=1000, modelEnc=None, batch_size=1):
    total_loss = 0.0
    l = nn.BCELoss()
    acc = np.array([0, 0, 0, 0])
    batch_loss = 0.0

    for iter_num, graph in tqdm(list(enumerate(data.graphs)), ncols=80):
        lang_embed = torch.from_numpy(data.lang[iter_num])
        y_true, y_true_list = data.delta_g[iter_num][0], data.delta_g[iter_num]
        goalObjectsVec = torch.from_numpy(np.array(data.goalObjectsVec[iter_num]))
        if (y_true == ""):
            continue
        if (model_name == "baseline" or model_name == "baseline_withoutGraph"):
            y_pred = model(graph, lang_embed, True, torch.FloatTensor(string2vec(y_true)), epoch)
        if model_name == "GGCN_node_attn_sum" or  model_name == "GGCN_attn_pairwise" or model_name == "GAT_attn_pairwise":
            action, pred1_object, pred2_object, pred2_state = model(graph, lang_embed, goalObjectsVec, True,
                                                                    torch.FloatTensor(string2vec(y_true)), epoch, data.objects[iter_num],data.obj_states[iter_num])
            y_pred = torch.cat((action, pred1_object, pred2_object, pred2_state), 1).flatten().cpu()

        loss = loss_function(action, pred1_object, pred2_object, pred2_state, torch.FloatTensor(string2vec(y_true)), l)
        acc += accuracy_lenient(y_pred.detach().numpy().tolist(), y_true_list)[0]
        batch_loss += loss
        total_loss += loss
        if ((iter_num + 1) % batch_size == 0):
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            batch_loss = 0
    return (total_loss.item() / len(data.graphs)), acc / len(data.graphs)

result_folder = './results/'
os.makedirs(result_folder, exist_ok=True)
model_name = sys.argv[1]
if __name__ == '__main__':
    data_file = "data/"
    train_data = DGLDataset(data_file + 'train/')
    val_data = DGLDataset(data_file + 'val/')
    exit()

    if model_name == "baseline" or model_name == "baseline_obj1":
        model = Baseline_model(train_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                               ["Agent"] + all_relations[1:], torch.tanh, 0.5)
    if model_name == "GGCN_attn_pairwise":
        model = GGCN_Attn_pairwise(train_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                                   ["Agent"] + all_relations[1:], torch.tanh, 0.5)
    if model_name == "GAT_attn_pairwise":
        model = GAT_attn_pairwise(train_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                                   ["Agent"] + all_relations[1:], torch.tanh, 0.5)
    if model_name == "GGCN_node_attn_sum":
        model = GGCN_node_attn_sum(train_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                                   ["Agent"] + all_relations[1:], torch.tanh, 0.5)
    if model_name == "GGCN_node_attn_max":
        model = GGCN_node_attn_max(train_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                                   ["Agent"] + all_relations[1:], torch.tanh, 0.5)
    if model_name == "GGCN_node_attn_norm":
        model = GGCN_node_attn_norm(train_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                                    ["Agent"] + all_relations[1:], torch.tanh, 0.5)
    if model_name == "GGCN_node_attn_adv":
        model = GGCN_node_attn_adv(train_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                                   ["Agent"] + all_relations[1:], torch.tanh, 0.5)

    epoch = -1
    NUM_EPOCHS = 150

    best_val_acc = 0
    train_acc_arr = []
    val_acc_arr = []
    train_loss_arr = []
    val_loss_arr = []
    train_sji_arr, val_sji_arr = [], []
    last_train_acc = 0.0
    last_train_loss = 0.0
    loss_ref_pnt = -1.0

    terminal_file = open(result_folder + "terminal_out.txt", "w")
    for num_epochs in range(NUM_EPOCHS):
        # random.shuffle(train_data)
        print("EPOCH " + str(num_epochs))
        lrate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=1e-4, amsgrad=False)
        train_loss, train_acc = \
                backprop(train_data, optimizer, model, N_objects, num_epochs, batch_size)

        model.eval()
        with torch.no_grad():
            val_acc, val_exact_acc, val_sji, val_loss = eval_accuracy(model_name, val_data, model, result_folder, num_epochs)
        terminal_file.write("Epoch = " + str(num_epochs) + "\n")
        terminal_file.write("train loss = " + str(train_loss) + " val loss = " + str(val_loss) + "\n")
        terminal_file.write("train acc = " + str(train_acc) + " val acc = " + str(val_acc) + "\n")
        print("train loss = " + str(train_loss) + " val loss = " + str(val_loss))
        print("train acc = " + str(train_acc) + " val acc = " + str(val_acc))
        print("val exact acc = " + str(val_exact_acc))

        train_acc_arr.append(train_acc)
        train_loss_arr.append(train_loss)
        val_acc_arr.append(val_acc)
        val_loss_arr.append(val_loss)

        if (best_val_acc < val_acc[0]):
            best_val_acc = val_acc[0]
            torch.save(model.state_dict(),
                       result_folder + str(num_epochs) + "_" + str(round(train_acc[0], 2)) + "_" + str(
                           round(val_acc[0], 2)) + ".pt")
            torch.save(model.state_dict(), result_folder + "best_val.pt")
            model.eval()
        if (num_epochs % 10 == 0):
            plot_graphs(train_acc_arr, val_acc_arr)
    terminal_file.close()