; PDDL rules which define the domain knowledge of the environment.
; PDDL rules so far use predicates, for and if conditions. Plan to add
; continuous variables and time is being considered. 
; The PDDL rules are based on the notion of environment as list of objects
; and relationship between them. Each objects has a set of states which have
; different values. Different objects can have different states however in this
; version we are dealing only with binary stateValues. At the moment
; there could be a finite set of pre-defined relationship between objects ex:
; object on top of other object or In etc. Also each object has statically
; defined affordances like cup is graspable and pourable but stove is not.
; ==========================================================================
; Predicate List -
;    state object-name state-name                       = True if object-name has state-name with value 1 and F if 0                  
;                                                         this is allowed since we only handle binary state values                     
;    affordance-type object-name                        = True if object-name has the given affordance-type
;    relationship-type object-name object-name          = True if the given relationship between the two given object names
;        
;
; Action List
;    Predefined set of primitive actions
; ==========================================================================
; Authors: Dipendra Misra (dkm@cs.cornell.edu)
; Notes  : Rules for naming actions. Base action names are in camelsCap and 
;          in order to add parameter specific functionality. I have added the 
;          notation that parameter name be attached to the base action name as 
;          baseaction_specificparameter. At this point its only for one specific
;          parameter but in future more could be added. Thus first index of _ 
;          decides the base action name. Alternatively use startsWith.
; ==========================================================================
;

(define (domain simulator)
(:predicates  (state ?x ?y)
              (IsGraspable ?x)
	      (IsSqueezeable ?x)
	      (Grasping ?x ?y)
	      (Near ?x ?y)
	      (In ?x ?y)
	      (On ?x ?y)
	      (Above ?x ?y)
)


; Grasping Functionality
; Description: Grasp something if not already grasping, its near and not In another object.
;              In which case you grasp it and nullify the relationship with other objects.

(:action grasp :parameters (?o)
:precondition (and (IsGraspable ?o) (Near Robot ?o) (not (Grasping Robot ?o)) )
:effect (and (Grasping Robot ?o)
			 (forall (?otherobj)
                (when (not (= ?o ?otherobj))
                      (and (not (In ?o ?otherobj)) (not (On ?o ?otherobj)) (not (Near ?o ?otherobj))
			   (not (In ?otherobj ?o)) (not (On ?otherobj ?o)) (not (Near ?otherobj ?o)))
			    )
			 )
	    )
)




 ; Release functionality
 ; Description: Simply ungrasp it. Not handling the new spatial relations that might develop
 ;              because of the release ex: it might fall on a stove and get heated

(:action release :parameters (?o)
:precondition (Grasping Robot ?o)
:effect (not (Grasping Robot ?o) ))





; Moving functionality
; Description: Move if not already close and the object exists. In which case you become near to that
; object and become far from other objects. Its not entirely correct, you can still be near to other objects
; but no harm in making additional moves.

(:action moveto :parameters (?o)
:effect (and (Near Robot ?o)
             (forall (?otherobj)
                (when (and (not (= ?o ?otherobj)) (not (= Robot ?otherobj)) (not (Grasping Robot ?otherobj)) )
                      (not (Near ?o ?otherobj))
			    )
             )
        )
)




; Pressing functionality
; Description: Object specific functionality  

; Pressing Microwave. If close then turn on and heat all objects In it
;                     If already on then close it

(:action press_MicrowaveButton :parameters ()
:precondition (Near Robot Microwave)
:effect (and (when (not (state Microwave MicrowaveIsOn))
			       (and (state Microwave MicrowaveIsOn) (forall (?otherobj)
													   (when (In Microwave ?otherobj)
															 (state ?otherobj temperatureHigh)
													   )
											   ))
              )
			  (when (state Microwave MicrowaveIsOn)
			       (not (state Microwave MicrowaveIsOn))
              )
	    )
)

; Press fridge water dispenser. If close then turn and fill grasped objects else close

(:action press_FridgeButton :parameters ()
:precondition (Near Robot Fridge)
:effect (and (when (not (state Fridge WaterDispenserIsOpen))
			       (and (state Fridge WaterDispenserIsOpen) (forall (?otherobj)
														            (when (Grasping Robot ?otherobj)
															              (state ?otherobj Water)
													                )
											                ))
             )
			 (when (state Fridge WaterDispenserIsOpen)
			       (not (state Fridge WaterDispenserIsOpen))
             )
	    )
)

; Press Tv power button

(:action press_Tv_1PowerButton :parameters ()
:precondition (Near Robot Tv_1)
:effect (and (when (not (state Tv_1 IsOn))
			       (state Tv_1 IsOn) 
             )
			 (when (state Tv_1 IsOn)
			       (not (state Tv_1 IsOn))
             )
	    )
)

; Press Tv remote power button

(:action press_Tv_1Remote_1PowerButton :parameters ()
:precondition (Grasping Robot Tv_1Remote_1)
:effect (and (when (not (state Tv_1 IsOn))
			       (state Tv_1 IsOn) 
             )
			 (when (state Tv_1 IsOn)
			       (not (state Tv_1 IsOn))
             )
	    )
)

; Press Tv channel up

(:action press_Tv_1ChannelUpButton :parameters ()
:precondition (Near Robot Tv_1)
:effect (when (state Tv_1 IsOn) (and (when (state Tv_1 Channel1) (and (state Tv_1 Channel2) (not (state Tv_1 Channel1))))
			             (when (state Tv_1 Channel2) (and (state Tv_1 Channel3) (not (state Tv_1 Channel2))))
			             (when (state Tv_1 Channel3) (and (state Tv_1 Channel4) (not (state Tv_1 Channel3))))
			             (when (state Tv_1 Channel4) (and (state Tv_1 Channel5) (not (state Tv_1 Channel4))))
			             (when (state Tv_1 Channel5) (and (state Tv_1 Channel6) (not (state Tv_1 Channel5))))
			             (when (state Tv_1 Channel6) (and (state Tv_1 Channel1) (not (state Tv_1 Channel6))))
	                        )
        )
)

; Press Tv Remote channel up

(:action press_Tv_1Remote_1ChannelUpButton :parameters ()
:precondition (Grasping Robot Tv_1Remote_1)
:effect (when  (state Tv_1 IsOn) (and (when (state Tv_1 Channel1) (and (state Tv_1 Channel2) (not (state Tv_1 Channel1))))
			 	      (when (state Tv_1 Channel2) (and (state Tv_1 Channel3) (not (state Tv_1 Channel2))))
                    		      (when (state Tv_1 Channel3) (and (state Tv_1 Channel4) (not (state Tv_1 Channel3))))
			              (when (state Tv_1 Channel4) (and (state Tv_1 Channel5) (not (state Tv_1 Channel4))))
				      (when (state Tv_1 Channel5) (and (state Tv_1 Channel6) (not (state Tv_1 Channel5))))
				      (when (state Tv_1 Channel6) (and (state Tv_1 Channel1) (not (state Tv_1 Channel6))))
	                         )
         )
)

; Press Tv channel down

(:action press_Tv_1ChannelDownButton :parameters ()
:precondition (Near Robot Tv_1)
:effect (when (state Tv_1 IsOn) (and (when (state Tv_1 Channel1) (and (state Tv_1 Channel6) (not (state Tv_1 Channel1))))
			 (when (state Tv_1 Channel2) (and (state Tv_1 Channel1) (not (state Tv_1 Channel2))))
			 (when (state Tv_1 Channel3) (and (state Tv_1 Channel2) (not (state Tv_1 Channel3))))
			 (when (state Tv_1 Channel4) (and (state Tv_1 Channel3) (not (state Tv_1 Channel4))))
			 (when (state Tv_1 Channel5) (and (state Tv_1 Channel4) (not (state Tv_1 Channel5))))
			 (when (state Tv_1 Channel6) (and (state Tv_1 Channel5) (not (state Tv_1 Channel6))))
	         )
        )
)

; Press Tv Remote channel down

(:action press_Tv_1Remote_1ChannelDownButton :parameters ()
:precondition (Grasping Robot Tv_1Remote_1)
:effect (when (state Tv_1 IsOn) (and (when (state Tv_1 Channel1) (and (state Tv_1 Channel6) (not (state Tv_1 Channel1))))
			 (when (state Tv_1 Channel2) (and (state Tv_1 Channel1) (not (state Tv_1 Channel2))))
			 (when (state Tv_1 Channel3) (and (state Tv_1 Channel2) (not (state Tv_1 Channel3))))
			 (when (state Tv_1 Channel4) (and (state Tv_1 Channel3) (not (state Tv_1 Channel4))))
			 (when (state Tv_1 Channel5) (and (state Tv_1 Channel4) (not (state Tv_1 Channel5))))
			 (when (state Tv_1 Channel6) (and (state Tv_1 Channel5) (not (state Tv_1 Channel6))))
	      )
        )
)

; Press Tv Volume Up
(:action press_Tv_1VolumeUpButton :parameters ()
:precondition (Near Robot Tv_1)
:effect (when (state Tv_1 IsOn) (state Tv_1 Volume))
)

; Press Tv Remote Volume Up
(:action press_Tv_1Remote_1VolumeUpButton :parameters ()
:precondition (Grasping Robot Tv_1Remote_1)
:effect (when (state Tv_1 IsOn) (state Tv_1 Volume))
)

; Press Tv Volume Down
(:action press_Tv_1VolumeDownButton :parameters ()
:precondition (Near Robot Tv_1)
:effect (when (state Tv_1 IsOn) (not (state Tv_1 Volume)))
)

; Press Tv Remote Volume Down
(:action press_Tv_1Remote_1VolumeDownButton :parameters ()
:precondition (Grasping Robot Tv_1Remote_1)
:effect (when (state Tv_1 IsOn) (not (state Tv_1 Volume)))
)

; Press Tv Mute
(:action press_Tv_1Remote_1MuteButton :parameters ()
:precondition (Near Robot Tv_1)
:effect (when (state Tv_1 IsOn) (not (state Tv_1 Volume)))
)




; Turning functionality pddls

(:action turn_SinkKnob :parameters ()
:precondition (Near Robot Sink)
:effect (and (when (not (state SinkKnob TapIsOn))    
                        (and (state SinkKnob TapIsOn)  (forall (?otherobj)
												            (when (On ?otherobj Sink)
														          (state ?otherobj Water)
												            )
				                                    )
					    ) 
		     )
			 (when (state SinkKnob TapIsOn)    
						 (not (state SinkKnob TapIsOn))
		     )
		)
)

(:action turn_StoveKnob_1 :parameters ()
:precondition (Near Robot Stove)
:effect (and (when (not (state Stove StoveFire1))    
                        (and (state Stove StoveFire1)  (forall (?otherobj)
												            (when (On ?otherobj StoveFire1)
														          (state ?otherobj temperatureHigh)
												            )
				                                       )
					    ) 
		     )
			 (when (state Stove StoveFire1)    
						 (not (state Stove StoveFire1))
		     )
		)
)

(:action turn_StoveKnob_2 :parameters ()
:precondition (Near Robot Stove)
:effect (and (when (not (state Stove StoveFire2))    
                        (and (state Stove StoveFire2)  (forall (?otherobj)
												            (when (On ?otherobj StoveFire2)
														          (state ?otherobj temperatureHigh)
												            )
				                                       )
					    ) 
		     )
			 (when (state Stove StoveFire2)    
						 (not (state Stove StoveFire2))
		     )
		)
)

(:action turn_StoveKnob_3 :parameters ()
:precondition (Near Robot Stove)
:effect (and (when (not (state Stove StoveFire3))    
                        (and (state Stove StoveFire3)  (forall (?otherobj)
												            (when (On ?otherobj StoveFire3)
														          (state ?otherobj temperatureHigh)
												            )
				                                       )
					    ) 
		     )
			 (when (state Stove StoveFire3)    
						 (not (state Stove StoveFire3))
		     )
		)
)

(:action turn_StoveKnob_4 :parameters ()
:precondition (Near Robot Stove)
:effect (and (when (not (state Stove StoveFire4))    
                        (and (state Stove StoveFire4)  (forall (?otherobj)
												            (when (On ?otherobj StoveFire4)
														          (state ?otherobj temperatureHigh)
												            )
				                                       )
					    ) 
		     )
			 (when (state Stove StoveFire4)    
						 (not (state Stove StoveFire4))
		     )
		)
)



; Opening functionality
; you open either a microwave door, fridge door or open a bag of chips

(:action open_Microwave :parameters ()
:precondition (and (Near Robot Microwave) (not (state Microwave DoorIsOpen)))
:effect (and (state Microwave DoorIsOpen)
		     (not (state Microwave MicrowaveIsOn))
	    )
)

(:action open_FridgeLeftDoor :parameters ()
:precondition (and (Near Robot Fridge) (not (state Fridge LeftDoorIsOpen)))
:effect (and (state Fridge LeftDoorIsOpen)
             (not (state Fridge WaterDispenserIsOpen))
	    )
)

(:action open_FridgeRightDoor :parameters ()
:precondition (and (Near Robot Fridge) (not (state Fridge RightDoorIsOpen)))
:effect (state Fridge RightDoorIsOpen)
)

(:action open_BagOfChips_1 :parameters ()
:precondition (and (Grasping Robot BagOfChips_1))
:effect (state BagOfChips_1 IsOpen))


; Closing functionality

(:action close_Microwave :parameters ()
:precondition (and (Near Robot Microwave) (state Microwave DoorIsOpen))
:effect (not (state Microwave DoorIsOpen)))

(:action close_FridgeLeftDoor :parameters ()
:precondition (and (Near Robot Fridge) (state Fridge LeftDoorIsOpen))
:effect (not (state Fridge LeftDoorIsOpen) ))

(:action close_FridgeRightDoor :parameters ()
:precondition (and (Near Robot Fridge) (state Fridge RightDoorIsOpen))
:effect (not (state Fridge RightDoorIsOpen)))


; Keeping functionality below

; Keeping something on stove 

(:action keep_On_StoveFire1 :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Stove))
:effect (and (not (Grasping Robot ?x)) (On ?x StoveFire1) (when (state Stove StoveFire1)
																  (state ?x temperatureHigh)
                                                            )
        )
)

(:action keep_On_StoveFire2 :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Stove))
:effect (and (not (Grasping Robot ?x)) (On ?x StoveFire2) (when (state Stove StoveFire2)
																  (state ?x temperatureHigh)
                                                            )
        )
)

(:action keep_On_StoveFire3 :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Stove))
:effect (and (not (Grasping Robot ?x)) (On ?x StoveFire3) (when (state Stove StoveFire3)
																  (state ?x temperatureHigh)
                                                            )
        )
)

(:action keep_On_StoveFire4 :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Stove))
:effect (and (not (Grasping Robot ?x)) (On ?x StoveFire4) (when (state Stove StoveFire4)
																  (state ?x temperatureHigh)
                                                            )
        )
)

; Keeping something on the sink

(:action keep_On_Sink :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Sink))
:effect (and (not (Grasping Robot ?x)) (On ?x Sink) (when (state SinkKnob TapIsOn)
																  (and (state ?x Water) (not (state ?x temperatureHigh)))
                                                            )
		)
)

; Keeping something In the microwave

(:action keep_In_Microwave :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Microwave) (state Microwave DoorIsOpen))
:effect (and (not (Grasping Robot ?x)) (In ?x Microwave)) )

; keep for fridge

(:action keep_In_FridgeLeft :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Fridge) (state Fridge LeftDoorIsOpen))
:effect (and (not (Grasping Robot ?x)) (In ?x FridgeLeft) ) )

(:action keep_In_FridgeRight :parameters (?x)
:precondition (and (Grasping Robot ?x) (Near Robot Fridge) (state Fridge RightDoorIsOpen))
:effect (and (not (Grasping Robot ?x)) (In ?x FridgeRight) ) )

; keeping for garbage_bag

(:action keep_In_GarbageBag_1 :parameters (?x)
:precondition (and (Grasping Robot ?x)  (Grasping Robot GarbageBag_1))
:effect (and (not (Grasping Robot ?x)) (In ?x GarbageBag_1) ))

; keeping for garbage_bin

(:action keep_In_GarbageBin_1 :parameters (?x)
:precondition (and (Grasping Robot ?x)  (Near Robot GarbageBin_1))
:effect (and (not (Grasping Robot ?x)) (In ?x GarbageBin_1) ))

(:action on_keep :parameters (?x ?z)
:precondition (and (Grasping Robot ?x)  (Near Robot ?z))
:effect (and (not (Grasping Robot ?x)) (On ?x ?z) ))

(:action in_keep :parameters (?x ?z)
:precondition (and (Grasping Robot ?x)  (Near Robot ?z))
:effect (and (not (Grasping Robot ?x)) (In ?x ?z) ))

(:action near_keep :parameters (?x ?z)
:precondition (and (Grasping Robot ?x)  (Near Robot ?z))
:effect (and (not (Grasping Robot ?x)) (Near ?x ?z) ))

; Generic keep

;(:action keep :parameters (?x ?rel ?z)
;:precondition (and (Grasping Robot ?x) (Near Robot ?z))
;:effect (and  (not (Grasping Robot ?x)) 
;              (when (= ?rel In) (In ?x ?z) )
;			  (when (= ?rel On) (On ?x ?z) )
;			  (when (= ?rel Near) (Near ?x ?z) )
;	    )
;)


; Pouring and adding functioality below

; pour something completely 

(:action pour :parameters(?x)
:precondition (Grasping Robot ?x)
:effect  (and (not (state ?x Water)) (not (state ?x Coffee)) (not (state ?x Coke))
              (not (state ?x EnergyDrink)) (not (state ?x IceCream)) (not (state ?x Chocolate))
			  (not (state ?x CanadaDry)) (not (state ?x Vanilla)) (not (state ?x Salt))
			  (not (state ?x Ramen)) (not (state ?x Egg)) 
		 )
)

; pour something from one cup to another

(:action pour :parameters(?x ?y)
:precondition (and (Grasping Robot ?x) (Grasping Robot ?y))
:effect  (and (when (state ?x Water)
					(state ?y Water)
              )
			  (when (state ?x Coffee)
					(state ?y Coffee)
              )
			  (when (state ?x Coke)
					(state ?y Coke)
              )
			  (when (state ?x EnergyDrink)
					(state ?y EnergyDrink)
              )
			  (when (state ?x IceCream)
					(state ?y IceCream)
              )
			  (when (state ?x Chocolate)
					(state ?y Chocolate)
              )
			  (when (state ?x CanadaDry)
					(state ?y CanadaDry)
              )
			  (when (state ?x Vanilla)
					(state ?y Vanilla)
              )
			  (when (state ?x Salt)
					(state ?y Salt)
              )
			  (when (state ?x Ramen)
					(state Ramen ?y)
              )
			  (when (state ?x Egg)
					(state Egg ?y)
              )
		  )
)


; adding functionality

(:action add_Salt_1 :parameters(?x)
:precondition (and (Grasping Robot Salt_1) (Grasping Robot ?x))
:effect (state ?x Salt)
)

(:action add_Ramen_1 :parameters(?x)
:precondition (and (Grasping Robot Ramen_1) (Grasping Robot ?x))
:effect (and (state ?x Ramen) (not (Grasping Robot Ramen_1)) (In Ramen_1 ?x))
)

(:action add_BoiledEgg_1 :parameters(?x)
:precondition (and (Grasping Robot BoiledEgg_1) (Grasping Robot ?x))
:effect (and (state ?x Egg) (not (Grasping Robot BoiledEgg_1)) (In BoiledEgg_1 ?x))
)

(:action add_IceCreamScoop :parameters(?x)
:precondition (and (Grasping Robot Spoon_1) (state Spoon_1 ScoopsLeft))
:effect (and (state ?x IceCream) (not (state Spoon_1 ScoopsLeft)) )
)


; Squeezing functionality 

(:action squeeze :parameters(?x ?y)
:precondition (and (Grasping Robot ?x) (Grasping Robot ?y) (IsSqueezeable ?x))
:effect (and (when (state ?y Vanilla)
				   (state ?y Vanilla)
		     )
			 (when (state ?x Chocolate)
				   (state ?y Chocolate)
		     )
		)
)


; Scooping Functionality

(:action scoop :parameters(?x ?y)
:precondition (and (Grasping Robot ?x) (Grasping Robot ?y) (state ?y ScoopsLeft))
:effect (state ?x ScoopsLeft)
)


; Place Functionality

(:action place_Fork_1 :parameters(?y)
:precondition (Grasping Robot ?y)
:effect (and (not (Grasping Robot ?y)) (state ?y Fork) (In Fork_1 ?y))
)

(:action place_Spoon_1 :parameters(?y)
:precondition (Grasping Robot ?y)
:effect (and (not (Grasping Robot ?y)) (state ?y Spoon) (In Spoon_1 ?y))
)


; Insert Functionality

(:action insert :parameters(?x ?y)
:precondition (and (state ?x CD) (Grasping Robot ?x) (Grasping Robot ?y))
:effect (and (state ?y CD) (not (Grasping Robot ?x)) (In ?x ?y))
)

)
