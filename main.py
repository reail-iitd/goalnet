import sys
from src.util import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = int(sys.argv[3])
batch_size = 32

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def plot_grad_flow(named_parameters, filename):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if type(p.grad) == type(None):
                print("None gradient ", n)
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="g")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="b" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=4)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([line.Line2D([0], [0], color="g", lw=4),
                line.Line2D([0], [0], color="r", lw=4),
                line.Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(result_folder + filename)
    plt.close('all')

def loss_1oML(action, pred1_obj, pred2_obj, pred2_state, y_true, l):
    l_sum = l(action.cpu()[0], y_true[: N_relations])
    l_sum += l(pred1_obj.cpu()[0], y_true[N_relations: N_relations + N_objects])
    l_sum += l(pred2_obj.cpu()[0], y_true[N_relations + N_objects: N_relations + N_objects + N_objects + 1])
    l_sum += l(pred2_state.cpu()[0], y_true[N_relations + N_objects + N_objects + 1:])
    return l_sum

def sample(logits):
        log_probs = F.log_softmax(logits, dim=-1)

        # probs = F.softmax(logits, dim=-1)
        # val = torch.multinomial(probs, 1).item()
        val = logits.detach().numpy().tolist().index(max(logits.detach().numpy().tolist()))
        # print("VAL -------------> ", log_probs[0][val])
        return val, log_probs[0][val]

def loss_rl(action, pred1_obj, pred2_obj, pred2_state, reward, max_index):
    # scale reward from 0 - 1 --> -1 - 1
    reward = (reward - 0.5)*2
    _, p_act = sample(action)
    _, p_obj1 = sample(pred1_obj)
    _, p_obj2 = sample(pred2_obj)
    _, p_state = sample(pred2_state)

    loss = -1*(p_act + p_obj1 + p_obj2 + p_state)*reward
    return loss


def backprop(data, optimizer, model, num_objects, epoch=1000, modelEnc=None, batch_size=1):
    total_loss = 0.0
    l = nn.BCELoss()
    # l = nn.CrossEntropyLoss()
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
            # if epoch==1:
            #     plot_grad_flow(model.named_parameters(),  "gradient_graph_pre.jpg")
            batch_loss.backward()
            # if epoch==1:
            #     plot_grad_flow(model.named_parameters(),  "gradient_graph_post.jpg")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            batch_loss = 0

    return (total_loss.item() / len(data.graphs)), acc / len(data.graphs)
    # return (total_loss / len(data.graphs)), acc / len(data.graphs)


def plot_graphs(train_acc_arr, val_acc_arr):
    fig, axs = plt.subplots(2)
    fig.suptitle('Loss and Acc')
    train_acc_arr_np = np.array(train_acc_arr)
    val_acc_arr_np = np.array(val_acc_arr)
    axs[0].plot(train_loss_arr, label='train')
    axs[0].plot(val_loss_arr, label='val')
    axs[1].plot(train_acc_arr_np[:, 0], label='train')
    axs[1].plot(val_acc_arr_np[:, 0], label='val')
    axs[0].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[1].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[0].title.set_text("Loss")
    axs[1].title.set_text("Overall acc")
    plt.savefig(result_folder + "graphs_overall.jpg")
    plt.close('all')

    fig, axs = plt.subplots(3)
    fig.suptitle('Acc. of individual pred')
    axs[0].plot(train_acc_arr_np[:, 1], label='train')
    axs[0].plot(val_acc_arr_np[:, 1], label='val')
    axs[1].plot(train_acc_arr_np[:, 2], label='train')
    axs[1].plot(val_acc_arr_np[:, 2], label='val')
    axs[2].plot(train_acc_arr_np[:, 3], label='train')
    axs[2].plot(val_acc_arr_np[:, 3], label='val')
    axs[0].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[1].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[2].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[0].title.set_text("State acc")
    axs[1].title.set_text("Obj1 acc")
    axs[2].title.set_text("Obj2 acc")
    plt.savefig(result_folder + "graphs_indiv.jpg")
    plt.close('all')


result_folder = "./" + sys.argv[2] + "/"
# result_folder = "./" +  argv[1] + "/"
try:
    os.makedirs(result_folder)
except:
    pass
model_name = sys.argv[1]
if __name__ == '__main__':
    """
        load data for train and val file
    """
    data_file = "data/"
    # data_file = "Type6_Final_train_test_0203/"
    train_data = DGLDataset(data_file + 'train_acl/', conceptnet_vectors)
    val_data = DGLDataset(data_file + 'val_acl/', conceptnet_vectors)
    print("DGL graph created")
    """
        load model
    """
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

    # for name, param in model.named_parameters():
    #   print (name)
    #   print(param.requires_grad)
    # if torch.cuda.is_available():
    #     model.cuda()
    #     print("Moved tensors to cuda")

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
    #  for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name + ": " + str(param.data))
    terminal_file = open(result_folder + "terminal_out.txt", "w")
    for num_epochs in range(NUM_EPOCHS):
        # random.shuffle(train_data)
        print("EPOCH " + str(num_epochs))
        # lr = 0.01
        # lrate = max(lr * (1.0 / (1 + int(num_epochs/20))), 0.001)
        lrate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=1e-4, amsgrad=False)
        train_loss, train_acc = \
                backprop(train_data, optimizer, model, N_objects, num_epochs, batch_size)

        # t
        # train_acc = eval_accuracy(train_data, True, model)
        model.eval()
        with torch.no_grad():
            val_acc, val_exact_acc, val_sji, val_loss = eval_accuracy(model_name, val_data, model, result_folder, num_epochs)
        # Write terminal putput in outfile !
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
            # print final predicted constraints
            # with torch.no_grad():
            #     print_final_constraints(model_name, val_data, False, model, result_folder)

        if (num_epochs % 10 == 0):
            plot_graphs(train_acc_arr, val_acc_arr)
    terminal_file.close()