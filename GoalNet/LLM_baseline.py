import os
import json
# import replicate

import os
import openai

openai.api_key = os.getenv("GPT_API_KEY")


env_info = "all_objects : [IceCream_1, Fork_1, Microwave, Armchair_4, Xbox_1, Counter_1, GarbageBin_1, StoveFire_4, Book_1, Pillow_2, Cd_2, InstantRamen_1, Glass_1, Book_2, Kettle, Salt_1, Shelf_1, Beer_1, LongCup_1, Pillow_4, Plate_1, Tv_1Remote_1, StoveFire_1, Loveseat_1, Book_3, Armchair_2, Plate_2, Tv_1, SnackTable_1, BoiledEgg_1, Sink, Cd_1, Bowl_1, Loveseat_2, StoveFire_3, Pillow_1, Fridge, Studytable_1, CanadaDry_1, Spoon_1, Syrup_1, GarbageBag_1, Mug_1, Pillow_3, XboxController_1, Coke_1, SinkKnob, BagOfChips_1, Syrup_2, Armchair_3, EnergyDrink_1, Stove, LongCup_2, Armchair_1, CoffeeTable_1, StoveFire_2, Ramen_1, Shelf_2, Robot],all_obj_prop : {Robot:  [], Counter_1:  [IsPlaceableOn], Sink:  [IsPlaceableOn], Stove:  [Turnable], Mug_1:  [IsGraspable, IsPourable], MicrowaveDoor:  [], Microwave:  [Openable, Pressable, IsPlaceableIn], Fridge:  [Pressable, Openable, IsPlaceableIn], FridgeLeftDoor:  [], FridgeRightDoor:  [], Spoon_1:  [IsGraspable, IsPlacable, IsScoopable], IceCream_1:  [IsGraspable], Kettle:  [IsGraspable, IsPourable], Ramen_1:  [IsGraspable, IsAddable], Syrup_1:  [IsGraspable, IsSqueezeable], Glass_1:  [IsGraspable, IsPourable], LongCup_1:  [IsGraspable, IsPourable], LongCup_2:  [IsGraspable, IsPourable], Fork_1:  [IsGraspable, IsPlacable], EnergyDrink_1:  [IsGraspable, IsPourable], Coke_1:  [IsGraspable, IsPourable], CanadaDry_1:  [IsGraspable, IsPourable], Plate_1:  [IsGraspable], Plate_2:  [IsGraspable], Syrup_2:  [IsGraspable, IsSqueezeable], InstantRamen_1:  [IsGraspable, IsPourable], BoiledEgg_1:  [IsGraspable, IsAddable], Salt_1:  [IsGraspable, IsAddable], Light_1:  [], StoveKnob_1:  [], StoveKnob_2:  [], StoveKnob_3:  [], StoveKnob_4:  [], StoveFire_1:  [IsPlaceableOn], StoveFire_2:  [IsPlaceableOn], StoveFire_3:  [IsPlaceableOn], StoveFire_4:  [IsPlaceableOn], FridgeButton:  [], MicrowaveButton:  [], SinkKnob:  [IsTurnable], IceCreamScoop:  [], Bowl_1:  [IsGraspable], Loveseat_1:  [IsPlaceableOn], Armchair_1:  [IsPlaceableOn], Armchair_2:  [IsPlaceableOn], CoffeeTable_1:  [IsPlaceableOn], TvTable_1:  [], Tv_1:  [Pressable], Tv_1Remote_1:  [IsGraspable], Pillow_1:  [IsGraspable], Pillow_2:  [IsGraspable], Pillow_3:  [IsGraspable], SnackTable_1:  [IsPlaceableOn], BagOfChips_1:  [IsGraspable], GarbageBag_1:  [IsGraspable, IsPlaceableIn], GarbageBin_1:  [IsGraspable, IsPlaceableOn, IsPlaceableIn], Shelf_1:  [IsPlaceableOn], Shelf_2:  [IsPlaceableOn], Book_1:  [IsGraspable], Book_2:  [IsGraspable], Beer_1:  [IsGraspable], XboxController_1:  [IsGraspable], Xbox_1:  [IsGraspable], Cd_2:  [IsGraspable], Cd_1:  [IsGraspable], Armchair_3:  [IsPlaceableOn], Armchair_4:  [IsPlaceableOn], Pillow_4:  [IsGraspable], Studytable_1:  [IsPlaceableOn], Lamp_1:  [], Tv_1PowerButton:  [], Tv_1ChannelUpButton:  [], Tv_1ChannelDownButton:  [], Tv_1VolumeUpButton:  [], Tv_1VolumeDownButton:  [], Tv_1Remote_1PowerButton:  [], Tv_1Remote_1ChannelUpButton:  [], Tv_1Remote_1ChannelDownButton:  [], Tv_1Remote_1VolumeUpButton:  [], Tv_1Remote_1VolumeDownButton:  [], Tv_1Remote_1MuteButton:  [], Loveseat_2:  [IsPlaceableOn], Counter1_1:  [IsPlaceableOn], Book_3:  [IsGraspable], Flower_1:  []}, \n all_fluents : [IsOpen, Channel2, IceCream, Egg, spoon, MicrowaveIsOn, Ramen, LeftDoorIsOpen, StoveFire3, IsOn, Coffee, ScoopsLeft, CD, Water, Channel3, TapIsOn, HasChips, RightDoorIsOpen, StoveFire2, Vanilla, Channel4, Volume, StoveFire1, DoorIsOpen, StoveFire4, WaterDispenserIsOpen, Chocolate, Channel1], \n all_non_fluents : [IsTurnable, IsPlaceableIn, IsPlaceableOn, Pressable, IsSqueezeable, Turnable, IsPlacable, IsAddable, IsPourable, IsScoopable, IsGraspable, Openable], \n all_states : [IsPlacable, IsOpen, Channel2, IsPlaceableIn, IsAddable, IsSqueezeable, IceCream, Egg, IsGraspable, MicrowaveIsOn, Ramen, LeftDoorIsOpen, StoveFire3, Pressable, IsOn, Coffee, ScoopsLeft, CD, Water, Channel3, Openable, TapIsOn, IsPourable, HasChips, RightDoorIsOpen, IsTurnable, StoveFire2, Vanilla, Channel4, Volume, IsPlaceableOn, Turnable, StoveFire1, IsScoopable, DoorIsOpen, StoveFire4, WaterDispenserIsOpen, Chocolate, Channel1], \n all_relations : [state, Near, On, In, Grasping] , \n all_object_states : {IceCream_1: [ScoopsLeft], Fork_1: [Water], Microwave: [DoorIsOpen, MicrowaveIsOn], Xbox_1: [CD], Cd_2: [CD], InstantRamen_1: [Water, Coffee], Glass_1: [Chocolate, Water, Coffee], Kettle: [Water, Ramen, Coffee], LongCup_1: [IceCream, Chocolate, Water, Coffee], Plate_1: [Ramen, IceCream, Chocolate, Egg, Water], Tv_1Remote_1: [Channel2, Volume, Channel3, IsOn, Channel1, Channel4], Plate_2: [Ramen, IceCream, Chocolate, Egg, Water], Tv_1: [Channel2, Volume, Channel3, IsOn, Channel1, Channel4], Cd_1: [CD], Bowl_1: [HasChips, Ramen, Water, IceCream], Fridge: [RightDoorIsOpen, LeftDoorIsOpen, WaterDispenserIsOpen], Spoon_1: [ScoopsLeft, Water], Syrup_1: [Vanilla, Chocolate], Mug_1: [Chocolate, Water, IceCream, Coffee, spoon], SinkKnob: [TapIsOn], BagOfChips_1: [IsOpen, HasChips], Syrup_2: [Vanilla, Chocolate], Stove: [StoveFire4, StoveFire3, StoveFire1, StoveFire2], LongCup_2: [IceCream, Chocolate, Water, Coffee]},"

# Specify the folder path where your JSON files are located
folder_path = '../dataset/test/'

# List all files in the folder
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]


outfile_path = './LLM_outputs2/'  # Change to the path and name of your file

existing_json_files = set([f for f in os.listdir(outfile_path) if f.endswith('.json')])

def dict2str(data_dict):
    for key, values in data_dict.items():
        data_string = ""
        # Join the list of values into a comma-separated string
        values_str = ', '.join(values)
        
        # Format the key-value pair
        key_value_str = f"{key} : [{values_str}]"
        
        # Append to the final string
        data_string += key_value_str
        return data_string

# Iterate over each JSON file and read its content
for file_name in json_files:
    if file_name in existing_json_files:
        continue
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as json_file:
        try:
            data = json.load(json_file)
            # Process the data from the JSON file as needed
            sent = data['sent']
            arg_mapping = data['arg_mapping']
            arg_map_str = dict2str(arg_mapping)
            initial_states = data['initial_states']
            initial_states_str = ', '.join(initial_states[0])

            print("For sent ", sent, " file ", file_name )
            # 
            # These goals will be given to an robotic agent to execute in a home/factory like environemnt where it could perform the following actions - .....
            pre_prompt = f'Your goal is to determine two distinct sets of predicates: one set for the predicates that should be true in the desired goal state, and another set for the predicates that should be false in the goal state. Here goal state is the final state reached on following a language instruction that describes a task given the current state of the environment. \
                These goals will be given to a robotic agent to execute in a home/factory like environemnt where it could perform the following actions - are grasp, release, moveto, press_MicrowaveButton, press_FridgeButton, press_Tv_1PowerButton, press_Tv_1Remote_1PowerButton, press_Tv_1ChannelUpButton, press_Tv_1Remote_1ChannelUpButton, press_Tv_1ChannelDownButton, press_Tv_1Remote_1ChannelDownButton, press_Tv_1VolumeUpButton, press_Tv_1Remote_1VolumeUpButton, press_Tv_1VolumeDownButton, press_Tv_1Remote_1VolumeDownButton, press_Tv_1Remote_1MuteButton, turn_SinkKnob, turn_StoveKnob_1, turn_StoveKnob_2, turn_StoveKnob_3, turn_StoveKnob_4, open_Microwave, open_FridgeLeftDoor, open_FridgeRightDoor, open_BagOfChips_1, close_Microwave, close_FridgeLeftDoor, close_FridgeRightDoor, keep_On_StoveFire1, keep_On_StoveFire2, keep_On_StoveFire3, keep_On_StoveFire4, keep_On_Sink, keep_In_Microwave, keep_In_FridgeLeft, keep_In_FridgeRight, keep_In_GarbageBag_1, keep_In_GarbageBin_1, on_keep, in_keep, near_keep, keep, pour, pour, add_Salt_1, add_Ramen_1, add_BoiledEgg_1, add_IceCreamScoop, squeeze, scoop, place_Fork_1, place_Spoon_1, insert \
                You have the following information about this environment - {env_info} \n. The output has to be a set of predicates of the format - (Relation, Object1, Object2) where Relation is Near / On / In / Grasping, or the format can be (state, Object, state of the object)\
                Here are examples to illustrate this: \
                    1. For language instruction: put the pillows on the couch, output is Set1 =  ((On Pillow_1 Loveseat_1), (On Pillow_2 Loveseat_1), (On Pillow_3 Loveseat_1), (On Pillow_4 Loveseat_1) )); Set2 = ((On Pillow_1 Shelf_1), (On Pillow_2 Shelf_1), (On Pillow_3 Shelf_1), (On Pillow_4 Shelf_1)) \
                    2. For language instruction: take the icecream out of the freezer, output is Set1 =((Near Robot IceCream_1), (Grasping Robot IceCream_1)); Set2 = ((In IceCream_1 Fridge), (Near Robot Stove)) \
                    3. For language instruction: turn on the left stove, output is Set1 = ((state Stove StoveFire4)); Set2 = () \
                The system needs to use this information about the environment to generate these sets of predicates in response to the provided language instructions.'
            # pre_prompt = ""
            # language_instr = "put the pillows on the couch"

            prompt = f'What would the output predicate set be for language instruction - \n{sent} \n given the initial state of the environment - \
                \n {initial_states_str} \n and argument mapping - \n {arg_map_str} \n'

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": pre_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            print("Output = ",response["choices"][0]["message"]["content"] )

            json_out = {"sent": sent, "pre_prompt": pre_prompt, "prompt": prompt, "response": response["choices"][0]["message"]["content"]}
            o_file_path = outfile_path + file_name
            with open(o_file_path, "w") as json_file:
                json.dump(json_out, json_file, indent=4)  # indent for pretty formatting (optional)

            # outfile.write(f'filename: {file_name}\n')
            # outfile.write(f'sent: {sent}\n')
            # outfile.write(f'output: {response["choices"][0]["message"]["content"]} \n \n')
            # break


        except json.JSONDecodeError:
            print(f"Failed to parse {file_name} as JSON.")
# outfile.close()