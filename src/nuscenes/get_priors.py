from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Quaternion
from nuscenes.utils.data_classes import Box

import os
import sys
import numpy as np
import json


ver_name = "v1.0-trainval"
input_path = "../../data/nuScenes"
nusc = NuScenes(ver_name, input_path, True)

path = os.getcwd()

scene_names = []
for scene in nusc.scene:
    scene_names.append(scene["name"])


size_dict = {}


for scene_name in scene_names:
    try:
        scene_token = nusc.field2token("scene", "name", scene_name)[0]
    except Exception:
        print("\n Not a valid scene name for this dataset!")
        exit(2)

    scene = nusc.get("scene", scene_token)
    sample = nusc.get("sample", scene["first_sample_token"])
    frame_num = 0

    while sample["next"] != "":
        # Extract all the relevant data from the nuScenes dataset for our scene. The variable 'sample' is the frame
        # Note: This is NOT multithreaded for nuScenes data because each scene is small enough that this runs relatively quickly.
        # Get translation, rotation, dimensions, and origins for bounding boxes for each annotation

        sizes = []
        annotation_names = []
        for i in range(0, len(sample["anns"])):
            token = sample["anns"][i]
            annotation_metadata = nusc.get("sample_annotation", token)

            # Store data obtained from annotation
            sizes.append(annotation_metadata["size"])
            annotation_names.append(annotation_metadata["category_name"])

            if annotation_metadata["category_name"] not in size_dict:
                size_dict[annotation_metadata["category_name"]] = []
            else:
                size_dict[annotation_metadata["category_name"]].append(
                    annotation_metadata["size"]
                )

            # Confidence for ground truth data is always 101 # changed this to 101 since predictions may have confidence of 100

        frame_num += 1
        sample = nusc.get("sample", sample["next"])


avg_sizes = {}

# print the average size of each category
for key in size_dict:
    print("{:<10} {:<10}".format(key, np.mean(size_dict[key], axis=0)))
    avg_sizes[key] = list(np.mean(size_dict[key], axis=0))

with open("shape_priors.json", "w") as outfile:
    json.dump(avg_sizes, outfile)