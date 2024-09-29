# TEXT_PROMPT = "bicycle . cycle . pedal cycle . pushbike . push bike . bike . sedan car . car . hatchback . hatchback car . convertible car . convertible . jeep\
#  . jeep car . sedan . suv . suv car . pickup . pick-up truck . pickup truck . human . man . woman . child . kid . boy . girl . pedestrian\
#  . person . truck . semi . semi-trailer . semitrailer . eighteen-wheeler . lorry . lorry truck . bus . autobus . motorbus ."


# TEXT_PROMPT = "bicycle . cycle .\
#  sedan car . car . sedan . suv . pick-up truck .\
#  human . man . woman . pedestrian . person .\
#  truck . semi . lorry .\
#  bus .\
#  traffic cone .\
#  barrier . road barrier . traffic barrier .\
#  construction vehicle . dumptruck . dump truck . cement mixer . bulldozer . crane . forklift .\
#  motorcycle . motorbike .\
#  trailer . rv . camper . truck trailer ."


# List of synonyms for each class
TEXT_PROMPT_MAPS = {
    "bicycle": [
        "bicycle",
        "cycle",
        # "rider on bicycle",
    ],
    "car": [ # car was changed
        "sedan car",
        "car",
        "sedan",
        "suv",
    ],
    "pedestrian": [
        "human",
        "man",
        "woman",
        "pedestrian",
        "person",
        # "child"
    ],
    "truck": [ # truck was changed
        "truck",
        "semi",
        "lorry",
        # "pick-up truck",
        "pickup truck",
        # "semi truck",
        # "front of semi-trailer truck",
        # "dumptruck",
        # "dump truck",
        # "cement mixer"
    ],
    "bus": [
        "bus",
        # "shuttle bus"
    ],
    "traffic_cone": [
        "traffic cone"
    ],
    "barrier": [
        "road barrier",
        "traffic barrier",
        # "jersey barrier",
        # "water-filled barrier",
        # "crowd control barrier",
        # "pedestrian barrier",
        # "temporary traffic barrier",
        # "temporary road barrier",
        # "traffic barrier on road",
        # "road barrier on road"
    ],
    "construction_vehicle": [
        "construction vehicle",
        "bulldozer",
        "excavator",
        # "forklift"
    ],
    "motorcycle": [
        "motorcycle",
        "motorbike",
        # "vespa",
        # "scooter",
        # "moped",
        # "rider on motorcycle",
        # "rider on scooter",
    ],
    "trailer": [
        "truck trailer",
        # "travel trailer",
        # "trailer behind truck",
        # "trailer towed by truck",
    ]
}

# TEXT_PROMPT_MAPS = {
#     "bicycle": [
#         "bicycle",
#         "cycle",
#         "rider on bicycle",
#     ],
#     "car": [
#         "sedan car",
#         "car",
#         "sedan",
#         "suv",
#     ],
#     "pedestrian": [
#         "human",
#         "man",
#         "woman",
#         "pedestrian",
#         "person",
#         "child"
#     ],
#     "truck": [
#         "truck",
#         "semi",
#         "lorry",
#         "pick-up truck",
#         "pickup truck",
#         "semi truck",
#         "front of semi-trailer truck",
#         "dumptruck",
#         "dump truck",
#         "cement mixer"
#     ],
#     "bus": [
#         "bus",
#         "shuttle bus"
#     ],
#     "traffic_cone": [
#         "traffic cone"
#     ],
#     "barrier": [
#         "road barrier",
#         "traffic barrier",
#         "jersey barrier",
#         "water-filled barrier",
#         "crowd control barrier",
#         "pedestrian barrier",
#         "temporary traffic barrier",
#         "temporary road barrier",
#         "traffic barrier on road",
#         "road barrier on road"
#     ],
#     "construction_vehicle": [
#         "construction vehicle",
#         "bulldozer",
#         "excavator",
#         "forklift"
#     ],
#     "motorcycle": [
#         "motorcycle",
#         "motorbike",
#         "vespa",
#         "scooter",
#         "moped",
#         "rider on motorcycle",
#         "rider on scooter",
#     ],
#     "trailer": [
#         "truck trailer",
#         "travel trailer",
#         "trailer behind truck",
#         "trailer towed by truck",
#     ]
# }


def create_text_prompt(prompt_map):
    prompt = ""
    for cls in prompt_map:
        for synonym in prompt_map[cls]:
            prompt += synonym + " . "
    return prompt

def create_reverse_maps(prompt_map):
    maps = {}
    for cls in prompt_map:
        for synonym in prompt_map[cls]:
            maps[synonym] = cls
    return maps


TEXT_PROMPT = create_text_prompt(TEXT_PROMPT_MAPS)

MAPS = create_reverse_maps(TEXT_PROMPT_MAPS)


# utility vehicle?, coach?
# bike for bicycle?
# children included in pedestrian?

OLD_MAPS = {
    "bicycle": "bicycle",
    # "bike": "bicycle",
    "cycle": "bicycle",
    "pedal cycle": "bicycle",
    "push bike": "bicycle",
    "pushbike": "bicycle",
    "car": "car",
    "hatchback": "car",
    "convertible": "car",
    "jeep": "car",
    "sedan": "car",
    "sedan car": "car",
    "suv": "car",
    "suv car": "car",
    "hatchback car": "car",
    "convertible car": "car",
    "jeep car": "car",
    "pickup truck": "truck",
    "pickup": "truck",
    "pick-up truck": "truck",
    "pickup_truck": "truck",
    "human": "pedestrian",
    "man": "pedestrian",
    "woman": "pedestrian",
    "child": "pedestrian",
    "kid": "pedestrian",
    "boy": "pedestrian",
    "girl": "pedestrian",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "truck": "truck",
    "semi": "truck",
    "semitrailer": "trailer",
    "semi_trailer": "trailer",
    "tank_trailer": "trailer",
    "semi-trailer": "trailer",
    "eighteen-wheeler": "trailer",
    "lorry": "truck",
    "lorry truck": "truck",
    "bus": "bus",
    "autobus": "bus",
    # "coach": "bus",
    "motorbus": "bus",
    "traffic cone": "traffic_cone",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "road barrier": "barrier",
    "road_barrier": "barrier",
    "traffic barrier": "barrier",
    "traffic_barrier": "barrier",
    "construction vehicle": "construction_vehicle",
    "construction_vehicle": "construction_vehicle",
    "dumptruck": "truck",
    "dump truck": "truck",
    "forklift": "construction_vehicle",
    "cement mixer": "construction_vehicle",
    "bulldozer": "construction_vehicle",
    "crane": "construction_vehicle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "trailer": "trailer",
    "rv": "trailer",
    "camper": "trailer",
    "truck trailer": "trailer",
    "truck_trailer": "trailer",
}


BOX_THRESHOLDS = {
    "car": 0.10,
    "truck": 0.10,
    "bus": 0.10,
    "bicycle": 0.10,
    "pedestrian": 0.10,
    "trailer": 0.10,
    "barrier": 0.10,
    "construction_vehicle": 0.10,
    "traffic_cone": 0.10,
    "motorcycle": 0.10,
}

TEXT_THRESHOLDS = {
    "car": 0.10,
    "truck": 0.10,
    "bus": 0.10,
    "bicycle": 0.10,
    "pedestrian": 0.10,
    "trailer": 0.10,
    "barrier": 0.10,
    "construction_vehicle": 0.10,
    "traffic_cone": 0.10,
    "motorcycle": 0.10,
}