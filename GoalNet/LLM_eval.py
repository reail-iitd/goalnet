from src.dataset import *
import re


def parse_llm_output(input_str):
    pred_delta, pred_delta_inv = "", ""
    # Use regular expressions to extract the sets
    set1_match = re.search(r'Set1 = \((.*)\);?\n', input_str, re.DOTALL)
    set2_match = re.search(r'Set2 = \((.*)\);?', input_str, re.DOTALL)

    # print(set1_match.group(1))
    # print(set2_match.group(1))

    # Extract the contents of the sets
    if set1_match:
        set1_content = set1_match.group(1)
        pred_delta = [f'({element})' for element in re.findall(r'\((.*?)\)', set1_content) if len(element.split())==3]
    if set2_match:
        set2_content = set2_match.group(1)
        pred_delta_inv = [f'({element})' for element in re.findall(r'\((.*?)\)', set2_content) if len(element.split())==3]
    
    return pred_delta, pred_delta_inv




data = DGLDataset(data_file + "test" + '/')
# print(test_data[0])
verbose = False
sji, f1, ied, fb, fbs, grr = 0, 0, 0, 0, 0, 0
max_len = max([len(dp.states) - 1 for dp in data.dp_list])
pred_delta, pred_delta_inv = '',''

output = open("/home/shreya/goalnet/results/llm_output2.txt", "a")

for iter_num, dp in tqdm(list(enumerate(data.dp_list)), leave=False, ncols=80):

    state = dp.states[0]; state_dict = dp.state_dict[0]
    adj_matrix = dp.adj_matrix[0]
    init_state_dict = dp.state_dict[0]
    action_seq = []
    json_file_name = dp.file_path.split("/")[-1].split(".")[0] + "_eval.json"
    print("filename = ", dp.file_path )

    json_dict = {}
    # break 

    # Read deltas from the LLM output
    # pred_delta, pred_delta_inv = LLM_output_reader(dp.file_path)
    llm_out = "/home/shreya/goalnet/GoalNet/LLM_outputs2/" +  dp.file_path.split("/")[-1]

    print(dp.file_path.split("/")[-1])

    with open(llm_out, "r") as json_file:
        llm_data = json.load(json_file)
    
    output.write(f'file: {dp.file_path.split("/")[-1]} \n')
    output.write(f'str: {llm_data["sent"]} \n')
    pred_delta, pred_delta_inv = parse_llm_output(llm_data["response"])

    print(pred_delta, pred_delta_inv)
    output.write(f'{pred_delta} \n{pred_delta_inv}')


    planner_action, state, state_dict, adj_matrix = run_planner_llm(state, state_dict, adj_matrix, dp, pred_delta, pred_delta_inv, verbose=verbose)
    action_seq.extend(planner_action)

    action_seq_gt = []
    action_seq_gt = dp.action_seq[:-1]

    sji_val =  get_sji(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0], verbose=verbose)
    ied__val = get_ied(action_seq, action_seq_gt)
    grr_val = get_goal_reaching(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0])
    f1_val = get_f1_index(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0])

    sji += sji_val
    f1 += f1_val
    fb += get_fbeta(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0])
    fbs += get_fbeta_state(state_dict, dp.state_dict[-1])
    ied += ied__val
    grr += grr_val

    output.write(f'sji: {sji_val},  ied: {ied__val},   GRR: {grr_val},  F1: {f1_val}, \n')

    print("SJI ------------ ", sji_val)
    print("IED ------------ ", ied__val)
    print("GRR ------------ ", grr_val)
    print("F1 ------------ ", f1_val)

    # break
        

sji, f1, ied, fb, fbs, grr = sji / len(data.dp_list), f1 / len(data.dp_list), ied / len(data.dp_list), fb / len(data.dp_list), fbs / len(data.dp_list), grr / len(data.dp_list)
output.write(f'\n \n Avg sji: {sji}, \n Avg ied: {ied}, \n  Avg GRR: {grr}, \n Avg F1: {f1}, \n \n')