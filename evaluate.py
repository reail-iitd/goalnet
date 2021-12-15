from util import *
import sys
  # return cf

result_folder = sys.argv[3]
#y_true = test_set.delta_g[iter_num][0]
# "0102_GGCN_attn_2/" - 66_0.9_0.49.pt
# "0202_GGCN_attn_2/" - 16_0.7_0.39.pt
#  "0202_GGCN_attn/" - 49_0.9_0.49.pt
# result_folder = "./" +  argv[1] + "/"
try:
  os.makedirs(result_folder)
except:
  pass


if __name__ == '__main__':
  # Type6_Final_train_test_0902_SOPPOS_2
  val_data = DGLDataset('data_acl16/'+sys.argv[2], conceptnet_vectors)
  print("MN = ", sys.argv[1])
  model_name = sys.argv[1]
  print("DGL graph created")
  epoch=1000
  # load model
  if(model_name == "baseline"):
    model = Baseline_model(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+all_relations[1:], torch.tanh, 0.5)
  if(model_name == "baseline_objectConstrained"):
    model = Baseline_model_objectConstrained(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+all_relations[1:], torch.tanh, 0.5)
  if(model_name == "baseline_model_recurrent"):
    model = Baseline_model_recurrent(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+all_relations[1:], torch.tanh, 0.5)
  if model_name  == "GGCN_attn":
    model = GGCN_Attn(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+all_relations[1:], torch.tanh, 0.5)
  if model_name == "GGCN_attn_pairwise":
    model = GGCN_Attn_pairwise(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 2, ["Agent"]+all_relations[1:], torch.tanh, 0.5)
  if model_name == "GAT_attn_pairwise":
    model = GGCN_Attn_pairwise(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 2, ["Agent"]+all_relations[1:], torch.tanh, 0.5)
  if model_name == "GGCN_node_attn_adv":
    model = GGCN_node_attn_adv(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3,
                               ["Agent"] + all_relations[1:], torch.tanh, 0.5)
  if model_name == "GGCN_node_attn_sum":
    model = GGCN_node_attn_sum(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 2,
                               ["Agent"] + all_relations, torch.tanh, 0.5)
  # if model_name == "GGCN_node_attn":
  #   model = GGCN_node_attn(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"] + [all_relations][1:], torch.tanh, 0.5)
  if model_name == "GGCN_attn_pairwise_2attn":
    model = GGCN_Attn_pairwise_2attn(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+[all_relations][1:], torch.tanh, 0.5)
  if model_name == "GGCN_attn_pairwise_dropout":
    model = GGCN_Attn_pairwise_dropout(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+[all_relations][1:], torch.tanh, 0.5)
  if model_name == "GGCN_attn_pairwise_auxloss":
    model = GGCN_Attn_pairwise_auxloss(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+[all_relations][1:], torch.tanh, 0.5)
  if model_name == "GGCN_attn_dropout":
    # this class is yet to be completely written
    model = GGCN_Attn_dropout(val_data.features, N_objects, 2 * GRAPH_HIDDEN, len(all_fluents), 3, ["Agent"]+[all_relations][1:], torch.tanh, 0.5)

  device = 'cpu'
  # if torch.cuda.is_available():
  #   device = 'cuda'
  checkpoint = torch.load(result_folder + "/" + sys.argv[4], map_location=device)
  model.load_state_dict(checkpoint)

  # if torch.cuda.is_available():
  #   model.cuda()
  #   print("Moved tensors to cuda")

  optimizer = torch.optim.Adam(model.parameters() , lr=0.0005, weight_decay=1e-5)
  model.eval()
  test_acc,test_exact_acc, test_sji,_ = eval_accuracy(model_name,val_data, model) 
  print("SOP acc " + str(test_acc))
  print("Exact acc = "+str(test_exact_acc))
  print("SJI score = " + str(test_sji))
  # outfile = open(result_folder+"outfile_"+sys.argv[2]+".txt","w")
  with torch.no_grad():
    print_final_constraints(model_name, val_data, False, model, result_folder, result_folder + "/pddl_" + sys.argv[2]+"/", True)
  # outfile.close()
