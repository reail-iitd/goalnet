(define 
(problem tmp) 
(:objects icecream_1 fork_1 microwave armchair_4 xbox_1 garbagebin_1 stovefire_4 book_1 pillow_2 cd_2 instantramen_1 glass_1 book_2 kettle salt_1 shelf_1 beer_1 longcup_1 pillow_4 plate_1 tv_1remote_1 stovefire_1 loveseat_1 book_3 armchair_2 plate_2 tv_1 snacktable_1 boiledegg_1 sink cd_1 bowl_1 loveseat_2 stovefire_3 pillow_1 fridge studytable_1 canadadry_1 spoon_1 syrup_1 garbagebag_1 mug_1 pillow_3 xboxcontroller_1 coke_1 sinkknob bagofchips_1 syrup_2 armchair_3 energydrink_1 stove longcup_2 armchair_1 coffeetable_1 stovefire_2 ramen_1 shelf_2 robot ) 
(:init (state stove stovefire1) (state mug_1 coffee) (state fridge leftdoorisopen) (state icecream_1 scoopsleft) (state kettle water) (state syrup_1 vanilla) (state syrup_2 chocolate) (on mug_1 stovefire_2) (in spoon_1 fridge) (on longcup_1 stovefire_3) (in longcup_2 fridge) (in energydrink_1 fridge) (in coke_1 fridge) (in canadadry_1 fridge) (on plate_2 stovefire_4) (on salt_1 stovefire_3) (near robot sink) (near robot stove) (near robot fridge) ) 
(:goal (AND (grasping robot plate_1) )) 
)