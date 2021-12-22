import sys
from utils.util import *
from src.model import *
from src.dataset import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 32

def backprop(data, optimizer, scheduler, model, num_objects, epoch=1000, modelEnc=None, batch_size=1, train=True):
    total_loss = 0.0
    l = nn.BCELoss()
    acc = 0

    for iter_num, dp in tqdm(list(enumerate(data.dp_list)), leave=False, ncols=80):
        dp_loss, dp_acc = 0, 0
        for i in range(len(dp.states)):
            delta_g_true = dp.delta_g_embed[i]
            action, pred1_object, pred2_object, pred2_state, l_h = model(dp.states[i], dp.sent_embed, dp.goal_obj_embed, l_h if i else None)
            loss = loss_function(action, pred1_object, pred2_object, pred2_state, dp.delta_g_embed[i], dp.delta_g[i], l)
            pred_delta = vect2string(action, pred1_object, pred2_object, pred2_state, dp.env_domain)
            dp_acc_i = int((pred_delta == '' and dp.delta_g[i] == []) or pred_delta in dp.delta_g[i]) 
            # if epoch > 70 and dp_acc_i == 0: print(pred_delta, dp.delta_g[i])
            dp_loss += loss; dp_acc += dp_acc_i
        if train and ((iter_num + 1) % batch_size == 0):
            optimizer.zero_grad(); dp_loss.backward(); 
            if epoch==2: plot_grad_flow(model.named_parameters(), f'gradients_{iter_num}.pdf')
            optimizer.step()
        acc += (dp_acc / len(dp.states)); dp_loss /= len(dp.states); total_loss += dp_loss
    scheduler.step()
    return (total_loss.item() / len(data.dp_list)), acc / len(data.dp_list)

result_folder = './results/'
print("Continue...")
os.makedirs(result_folder, exist_ok=True)
if __name__ == '__main__':
    # train = sys.argv[0] + "/"
    # val = sys.argv[1] + "/"
    # model_type = sys.argv[2]
    model_type = "GGCN"
    data_file = "data/"
    train_data = DGLDataset(data_file + "train/")
    val_data = DGLDataset(data_file + "val/")
    test_data = DGLDataset(data_file + 'val/')

    if model_type == "simple":
        model = Simple_Model(train_data.features, 2 * GRAPH_HIDDEN, N_objects, len(all_fluents), ["Empty"] + all_relations[1:])
    elif model_type == "GGCN":
        model = GGCN_Model(train_data.features, 2 * GRAPH_HIDDEN, N_objects, len(all_fluents), ["Empty"] + all_relations[1:])
    elif model_type == "HAN":
        model = HAN_model(train_data.features, 2 * GRAPH_HIDDEN, N_objects, len(all_fluents), ["Empty"] + all_relations[1:])

    device = 'cpu'
    checkpoint = torch.load(result_folder + "/" + sys.argv[1], map_location=device)
    model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.eval()
    test_sji, test_god, test_ied = eval_accuracy(val_data, model, verbose = True) 
    tqdm.write(f'Test SJI : {"{:.8f}".format(test_sji)}\tTest GOD : {"{:.8f}".format(test_god)}\tTest IED : {"{:.8f}".format(test_ied)}')
            