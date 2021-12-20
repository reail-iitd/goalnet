import sys
from utils.util import *
from src.model import *
from src.dataset import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 32

def backprop(data, optimizer, model, num_objects, epoch=1000, modelEnc=None, batch_size=1):
    total_loss = 0.0
    l = nn.MSELoss()
    acc = 0

    for iter_num, dp in tqdm(list(enumerate(data.dp_list)), leave=False, ncols=80):
        dp_loss, dp_acc = 0, 0
        for i in range(len(dp.states)):
            delta_g_true = dp.delta_g_embed[i]
            action, pred1_object, pred2_object, pred2_state = model(dp.states[i], dp.sent_embed, dp.goal_obj_embed)
            loss = loss_function(action, pred1_object, pred2_object, pred2_state, dp.delta_g_embed[i], dp.delta_g[i], l)
            pred_delta = vect2string(action, pred1_object, pred2_object, pred2_state)
            # print(pred_delta, dp.delta_g[i], dp.delta_g_embed[i])
            dp_acc += int(pred_delta in dp.delta_g[i]) 
            dp_loss += loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        acc += (dp_acc / len(dp.states)); dp_loss /= len(dp.states); total_loss += dp_loss
    return (total_loss.item() / len(data.dp_list)), acc / len(data.dp_list)

result_folder = './results/'
os.makedirs(result_folder, exist_ok=True)
if __name__ == '__main__':
    data_file = "data/"
    train_data = DGLDataset(data_file + 'test/')
    # val_data = DGLDataset(data_file + 'val/')
    # test_data = DGLDataset(data_file + 'test/')

    model = Simple_Model(train_data.features, GRAPH_HIDDEN, N_objects, len(all_fluents), ["Empty"] + all_relations[1:])

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

    for num_epochs in trange(NUM_EPOCHS, ncols=80):
        # random.shuffle(train_data)
        lrate = 0.0005
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=1e-4, amsgrad=False)
        train_loss, train_acc = \
                backprop(train_data, optimizer, model, N_objects, num_epochs, batch_size)

        tqdm.write(f'Training Loss: {train_loss}\tTraining Acc : {train_acc}')
        # model.eval()
        # with torch.no_grad():
        #     val_acc, val_exact_acc, val_sji, val_loss = eval_accuracy(model_name, val_data, model, result_folder, num_epochs)
        # print("train loss = " + str(train_loss) + " val loss = " + str(val_loss))
        # print("train acc = " + str(train_acc) + " val acc = " + str(val_acc))
        # print("val exact acc = " + str(val_exact_acc))

        # train_acc_arr.append(train_acc)
        # train_loss_arr.append(train_loss)
        # val_acc_arr.append(val_acc)
        # val_loss_arr.append(val_loss)

        # if (best_val_acc < val_acc[0]):
        #     best_val_acc = val_acc[0]
        #     torch.save(model.state_dict(),
        #                result_folder + str(num_epochs) + "_" + str(round(train_acc[0], 2)) + "_" + str(
        #                    round(val_acc[0], 2)) + ".pt")
        #     torch.save(model.state_dict(), result_folder + "best_val.pt")
        #     model.eval()
        # if (num_epochs % 10 == 0):
        #     plot_graphs(train_acc_arr, val_acc_arr)
