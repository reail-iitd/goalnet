from main import *

if __name__ == '__main__':
    result_folder = './results/' + opts.expname + '/'
    model_type = opts.model
    if(model_type == "Tango" or model_type == "Aggregated"):
        model_type = "GoalNet"
    test_data = DGLDataset(data_file + opts.test + '/')

    best_model = eval(model_type + '_Model(test_data.features, 4 * GRAPH_HIDDEN, N_objects, len(all_fluents), ["Empty"] + all_relations[1:])')
    checkpoint = torch.load(result_folder + best_model.name + ".pt", map_location='cpu')
    best_model.load_state_dict(checkpoint); best_model.eval()

    with torch.no_grad():
        test_loss, test_acc = backprop(test_data, None, None, best_model, N_objects, 0, train = False)        
        test_sji, test_f1, test_ied, test_fb, test_fbs, test_grr = eval_accuracy(test_data, best_model, verbose = False)
    tqdm.write(f'SJI : {"{:.3f}".format(test_sji)}\tIED : {"{:.3f}".format(test_ied)}\tF1 : {"{:.3f}".format(test_f1)}\t GRR : {"{:.3f}".format(test_grr)}')
