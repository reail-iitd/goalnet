import sys, random
from utils.util import *
from src.model import *
from src.dataset import *

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
            pred_delta = vect2string(state_dict, action, pred1_object, pred2_object, pred2_state, dp.env_domain, dp.arg_map)
            dp_acc_i = int((pred_delta == '' and dp.delta_g[i] == []) or pred_delta in dp.delta_g[i]) 
            # if epoch > 200 and dp_acc_i == 0: print(pred_delta, dp.delta_g[i])
            dp_loss += loss; dp_acc += dp_acc_i
        if train:
            optimizer.zero_grad(); dp_loss.backward(); 
            # if epoch==1: plot_grad_flow(model.named_parameters(), f'gradients_{iter_num}.pdf')
            optimizer.step()
        acc += (dp_acc / len(dp.states)); dp_loss /= len(dp.states); total_loss += dp_loss
        fname = dp.file_path.split('/')[-1]
        # tqdm.write(f'{fname},{dp_acc / len(dp.states)}')
    if train: scheduler.step()
    return (total_loss.item() / len(data.dp_list)), acc / len(data.dp_list)

result_folder = './results/'
os.makedirs(result_folder, exist_ok=True)
if __name__ == '__main__':
    model_type = opts.model
    result_folder_exp = result_folder + opts.expname + "/"
    print("Result folder: ", result_folder_exp)
    os.makedirs(result_folder_exp, exist_ok=True)
    train_data = DGLDataset(data_file + opts.train + "/")
    val_data = DGLDataset(data_file + opts.val + "/")
    test_data = DGLDataset(data_file + opts.test + '/')

    model = eval(model_type + '_Model(train_data.features, 2 * GRAPH_HIDDEN, N_objects, len(all_fluents), ["Empty"] + all_relations[1:])')

    epoch = -1
    NUM_EPOCHS = 80

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

    lrate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for num_epochs in trange(NUM_EPOCHS, ncols=80):
        crossval(train_data, val_data)
        train_loss, train_acc = backprop(train_data, optimizer, scheduler, model, N_objects, num_epochs)
        with torch.no_grad():
            val_loss, val_acc = backprop(val_data, optimizer, scheduler, model, N_objects, num_epochs, train=False)
        tqdm.write(f'Epoch {num_epochs} Train Loss: {"{:.8f}".format(train_loss)}\tTrain Acc : {"{:.8f}".format(train_acc)}\tVal Loss: {"{:.8f}".format(val_loss)}\tVal Acc : {"{:.8f}".format(val_acc)}')
        train_loss_arr.append(train_loss); train_acc_arr.append(train_acc)
        val_loss_arr.append(val_loss); val_acc_arr.append(val_acc)
        if num_epochs % 20 == 19:
            plot_graphs(result_folder_exp, model_type + "_graph", train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr)
            # with torch.no_grad():
            #     test_loss, test_acc = backprop(test_data, optimizer, scheduler, best_model, N_objects, num_epochs, train=False)        
            #     test_sji, test_f1, test_ied, test_fb, test_fbs = eval_accuracy(test_data, best_model, verbose = False)
            # tqdm.write(f'Test Loss: {"{:.8f}".format(test_loss)}\tTest Acc : {"{:.8f}".format(test_acc)}\tTest SJI : {"{:.8f}".format(test_sji)}\tTest F1 : {"{:.8f}".format(test_f1)}\tTest IED : {"{:.8f}".format(test_ied)}\tTest FB : {"{:.8f}".format(test_fb)}\tTest FBS : {"{:.8f}".format(test_fbs)}')
            torch.save(best_model.state_dict(), result_folder_exp + model.name + ".pt")
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)

    print(f'Best validation accuracy: {max(val_acc_arr)} at epoch {np.argmax(val_acc_arr)}')
