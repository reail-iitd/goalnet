import sys, random
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
        teacher_forcing = random.random()
        state = dp.states[0]
        for i in range(len(dp.states)):
            if teacher_forcing < 0.8 or epoch < 100 or i == 0:
                state, state_dict = dp.states[i], dp.state_dict[i]
            else:
                _, state, state_dict = run_planner_simple(state_dict, dp, pred_delta, True)
            delta_g_true = dp.delta_g_embed[i]
            action, pred1_object, pred2_object, pred2_state, l_h = model(state, dp.sent_embed, dp.goal_obj_embed, l_h if i else None)
            loss = loss_function(action, pred1_object, pred2_object, pred2_state, dp.delta_g_embed[i], dp.delta_g[i], l)
            pred_delta = vect2string(state_dict, action, pred1_object, pred2_object, pred2_state, dp.env_domain)
            dp_acc_i = int((pred_delta == '' and dp.delta_g[i] == []) or pred_delta in dp.delta_g[i]) 
            # if epoch > 70 and dp_acc_i == 0: print(pred_delta, dp.delta_g[i])
            dp_loss += loss; dp_acc += dp_acc_i
        if train:
            optimizer.zero_grad(); dp_loss.backward(); #plot_grad_flow(model.named_parameters(), f'gradients_{iter_num}.pdf')
            optimizer.step()
        acc += (dp_acc / len(dp.states)); dp_loss /= len(dp.states); total_loss += dp_loss
    scheduler.step()
    return (total_loss.item() / len(data.dp_list)), acc / len(data.dp_list)

result_folder = './results/'
os.makedirs(result_folder, exist_ok=True)
if __name__ == '__main__':
    model_type = sys.argv[1]
    data_file = "data/"
    train_data = DGLDataset(data_file + "val/")
    val_data = DGLDataset(data_file + "val/")
    test_data = DGLDataset(data_file + 'val/')

    model = eval(model_type + '_Model(train_data.features, 2 * GRAPH_HIDDEN, N_objects, len(all_fluents), ["Empty"] + all_relations[1:])')

    epoch = -1
    NUM_EPOCHS = 500

    best_val_acc = 0
    best_model = None
    train_acc_arr = []
    val_acc_arr = []
    train_loss_arr = []
    val_loss_arr = []
    train_sji_arr, val_sji_arr = [], []
    last_train_acc = 0.0
    last_train_loss = 0.0
    loss_ref_pnt = -1.0

    for num_epochs in trange(NUM_EPOCHS, ncols=80):
        random.shuffle(train_data.dp_list)
        lrate = 0.0005 # val keep 0.0005 and train 0.00005
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        train_loss, train_acc = backprop(train_data, optimizer, scheduler, model, N_objects, num_epochs)
        with torch.no_grad():
            val_loss, val_acc = backprop(val_data, optimizer, scheduler, model, N_objects, num_epochs, train=False)
        tqdm.write(f'Train Loss: {"{:.8f}".format(train_loss)}\tTrain Acc : {"{:.8f}".format(train_acc)}\tVal Loss: {"{:.8f}".format(val_loss)}\tVal Acc : {"{:.8f}".format(val_acc)}')
        train_loss_arr.append(train_loss); train_acc_arr.append(train_acc)
        val_loss_arr.append(val_loss); val_acc_arr.append(val_acc)
        if num_epochs % 50 == 49:
            plot_graphs(result_folder, model_type + "_graph", train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr)
            with torch.no_grad():
                test_loss, test_acc = backprop(test_data, optimizer, scheduler, best_model, N_objects, num_epochs, train=False)        
                test_sji, test_f1, test_ied = eval_accuracy(test_data, best_model, verbose = False)
            tqdm.write(f'Test Loss: {"{:.8f}".format(test_loss)}\tTest Acc : {"{:.8f}".format(test_acc)}\tTest SJI : {"{:.8f}".format(test_sji)}\tTest F1 : {"{:.8f}".format(test_f1)}\tTest IED : {"{:.8f}".format(test_ied)}')
            
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)
            
    torch.save(best_model.state_dict(), result_folder + model.name + ".pt")
    print(f'Best validation accuracy: {max(val_acc_arr)} at epoch {np.argmax(val_acc_arr)}')
