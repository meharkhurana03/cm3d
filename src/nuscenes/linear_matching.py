from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.splits import mini_val, mini_train, train_detect, train, val
from eval_custom import NuScenesEval
from nuscenes.eval.common.config import config_factory
from tqdm import tqdm

import os
# import tqdm
import tensorflow as tf
import numpy as np
import json
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


"""Tools for computing box matching using Open Dataset criteria."""
import dataclasses
from typing import Optional

# import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2

TensorLike = tf.types.experimental.TensorLike
Type = label_pb2.Label.Box.Type
LongitudinalErrorTolerantConfig = (
    metrics_pb2.Config.LongitudinalErrorTolerantConfig
)


@dataclasses.dataclass
class MatchResult:
    """Object holding matching result.

    A single i-th element in each tensor describes a match between predicted and
    groundtruth box, the associated IOU (2D IOU, 3D IOU, or the LET variant), and
    the index in the batch where the batch was made.
    """

    # Indices into the prediction boxes, as if boxes were reshaped as [-1, dim].
    prediction_ids: TensorLike  # [N]
    # Indices into the groundtruth boxes, as if boxes were reshaped as [-1, dim].
    groundtruth_ids: TensorLike  # [N]
    # Matching quality (IOU variant depends on the matching config).
    ious: TensorLike  # [N]
    # Longitudinal affinity for a given match.
    longitudinal_affinities: TensorLike  # [N]


def match(
    prediction_boxes: TensorLike,
    groundtruth_boxes: TensorLike,
    iou: float,
    box_type: 'label_pb2.Label.Box.Type',
    matcher_type: metrics_pb2.MatcherProto.Type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN,
    let_metric_config: Optional[LongitudinalErrorTolerantConfig] = None,
) -> MatchResult:
    """Returns a matching between predicted and groundtruth boxes.

    Matching criteria and thresholds are specified through the config. Boxes are
    represented as [center_x, center_y, bottom_z, length, width, height, heading].
    The function treats "zeros(D)" as an invalid box and will not try to match it.

    Args:
      prediction_boxes: [B, N, D] or [N, D] tensor.
      groundtruth_boxes: [B, M, D] or [N, D] tensor.
      iou: IOU threshold to use for matching.
      box_type: whether to perform matching in 2D or 3D.
      matcher_type: the matching algorithm for the matcher. Default to Hungarian.
      let_metric_config: Optional config describing how LET matching should be
        done.

    Returns:
      A match result struct with flattened tensors where i-th element describes a
      match between predicted and groundtruth box, the associated IOU
      (AA_2D, 2D IOU, 3D IOU, or the LET variant), and the index in the batch
      where the match was made:
        - indices of predicted boxes [Q]
        - corresponding indices of groundtruth boxes [Q]
        - IOUs for each match. [Q]
        - index in the input batch [Q] with values in {0, ..., B-1}.
    """
    with tf.name_scope('open_dataset_matcher/match'):
        config = _create_config(iou, box_type, matcher_type, let_metric_config)
        if tf.rank(prediction_boxes) == 2:
            prediction_boxes = tf.expand_dims(prediction_boxes, 0)
        if tf.rank(groundtruth_boxes) == 2:
            groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
        tf.debugging.assert_shapes([
            (prediction_boxes, ('b', 'n', 'd')),
            (groundtruth_boxes, ('b', 'm', 'd')),
        ])
        pred_ids, gt_ids, ious, la = py_metrics_ops.match(
            prediction_boxes, groundtruth_boxes, config=config.SerializeToString()
        )
        return MatchResult(
            prediction_ids=pred_ids,
            groundtruth_ids=gt_ids,
            ious=ious,
            longitudinal_affinities=la,
        )


def _create_config(
    iou: float,
    box_type: 'label_pb2.Label.Box.Type',
    matcher_type: metrics_pb2.MatcherProto.Type,
    let_metric_config: Optional[LongitudinalErrorTolerantConfig] = None,
) -> metrics_pb2.Config:
    return metrics_pb2.Config(
        score_cutoffs=[0.0],
        box_type=box_type,
        difficulties=[metrics_pb2.Difficulty(levels=[])],
        breakdown_generator_ids=[breakdown_pb2.Breakdown.ONE_SHARD],
        matcher_type=matcher_type,
        iou_thresholds=[0, iou, iou, iou, iou],
        let_metric_config=let_metric_config,
    )



INPUT_PATH = "../../data/nuScenes"
selected_waymo_locations = None

pred_load_dir = '../../outputs/nuscenes/ablation_1new_val_0_150_detic.json'
sam3d_load_dir = '../../outputs/sam3d_results_nusc_val.json'

scene_list = val

sam3d_objects = json.load(open(sam3d_load_dir))
pred_objects = json.load(open(pred_load_dir))


ALPHAS = []
BETAS = []



if __name__ == "__main__":

    # Validating the sam3d preds
    nusc = NuScenes('v1.0-trainval', INPUT_PATH, True)

    # convert sam3d boxes to tensors
    sam3d_box_dict = {}
    sam3d_supp_dict = {}
    
    sam3d_max_conf = -1e7
    sam3d_min_conf = 1e7
    pred_max_conf = -1e7
    pred_min_conf = 1e7

    print("Parsing through SAM3D predictions...")
    for sample in sam3d_objects["results"]:
        if sample not in sam3d_box_dict.keys():
            sam3d_box_dict[sample] = []
        if sample not in sam3d_supp_dict.keys():
            sam3d_supp_dict[sample] = []

        for obj in sam3d_objects["results"][sample]:
            sam3d_box_dict[sample].append(np.array([
                obj["translation"][0],
                obj["translation"][1],
                obj["translation"][2] - obj["size"][2] / 2,
                obj["size"][0],
                obj["size"][1],
                obj["size"][2],
                R.from_quat(obj["rotation"]).as_euler('xyz', degrees=False)[0],
            ], dtype=float))

            sam3d_supp_dict[sample].append(
                [
                    obj["attribute_name"],
                    obj["detection_score"],
                    obj["velocity"],
                    obj["detection_name"]
                ]
            )

            # Update conf range
            if obj["detection_score"] > sam3d_max_conf:
                sam3d_max_conf = obj["detection_score"]
            if obj["detection_score"] < sam3d_min_conf:
                if obj["detection_score"] != 0:
                    sam3d_min_conf = obj["detection_score"]
                else:
                    print("Box score is zero, omitting")

    pred_box_dict = {}
    pred_supp_dict = {}

    print("Parsing through ZS3D predictions...")
    for sample in pred_objects["results"]:
        if sample not in pred_box_dict.keys():
            pred_box_dict[sample] = []
        if sample not in pred_supp_dict.keys():
            pred_supp_dict[sample] = []

        for obj in pred_objects["results"][sample]:
            pred_box_dict[sample].append(np.array([
                obj["translation"][0],
                obj["translation"][1],
                obj["translation"][2] - obj["size"][2] / 2,
                obj["size"][0],
                obj["size"][1],
                obj["size"][2],
                R.from_quat(obj["rotation"]).as_euler('xyz', degrees=False)[0],
            ], dtype=float))


            pred_supp_dict[sample].append(
                [
                    obj["attribute_name"],
                    obj["detection_score"],
                    obj["velocity"],
                    obj["detection_name"]
                ]
            )

            # Update conf range
            if obj["detection_score"] > pred_max_conf:
                pred_max_conf = obj["detection_score"]
            if obj["detection_score"] < pred_min_conf:
                pred_min_conf = obj["detection_score"]



    pred_matches_dict = {}
    sam3d_matches_dict = {}

    print("Computing matches...")
    for timestamp in pred_box_dict.keys():
        if timestamp not in pred_matches_dict:
            pred_matches_dict[timestamp] = []
        if timestamp not in sam3d_matches_dict:
            sam3d_matches_dict[timestamp] = []

        try:
            sam3d_boxes = np.array(sam3d_box_dict[timestamp], dtype=float)
        except KeyError:
            continue

        pred_boxes = np.array(pred_box_dict[timestamp], dtype=float)

        sam3d_boxes = tf.convert_to_tensor(sam3d_boxes, dtype=float)
        pred_boxes = tf.convert_to_tensor(pred_boxes, dtype=float)


        matches = match(pred_boxes, sam3d_boxes, 0.2, Type.TYPE_2D)


        if matches.ious.shape[0] > 0:
        
            for i in range(matches.ious.shape[0]):
                pred_matches_dict[timestamp].append(int(matches.prediction_ids[i]))
                sam3d_matches_dict[timestamp].append(int(matches.groundtruth_ids[i]))



    # Create ALPHAS and BETAS for grid search
    # Learning the scaling (alpha) and offset (beta) values, such that sam3d_conf = sam3d_conf*alpha + beta, for linear matching
    best_alpha = 0
    best_beta = 1e7
    best_score = -1


    ALPHAS = np.arange(
        pred_min_conf / sam3d_max_conf,
        pred_max_conf / (sam3d_min_conf),
        0.04,
        dtype=float
    )
    ALPHAS = list(ALPHAS)


    for i_a, alpha in tqdm(enumerate(ALPHAS)):
            matched_objects = {
                "meta": {
                    "use_camera": True,
                    "use_lidar": True,
                    "use_radar": False,
                    "use_map": True,
                    "use_external": False,
                },
                "results": {
                }
            }

            num_samples = 0
            num_pred_boxes = 0
            num_sam3d_boxes = 0
            num_sam3d_samples = 0
            num_matched_boxes = 0

            for timestamp in pred_box_dict.keys():
                for i, pred_box in enumerate(pred_box_dict[timestamp]):
                    if i in pred_matches_dict[timestamp]:
                        continue

                    box_dict = {
                        "sample_token": timestamp,
                        "translation": [],
                        "size": [],
                        "rotation": [],
                        "velocity": [0, 0],
                        "detection_name": pred_supp_dict[timestamp][i][3],
                        "detection_score": pred_supp_dict[timestamp][i][1],
                        "attribute_name": pred_supp_dict[timestamp][i][0]
                    }

                    heading = pred_box[6]
                    rot_matrix = np.eye(3)
                    rot_matrix[0:2, 0:2] = [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]

                    box_dict["translation"].append(float(pred_box[0]))
                    box_dict["translation"].append(float(pred_box[1]))
                    box_dict["translation"].append(float(pred_box[2]) + float(pred_box[5]) / 2)
                    box_dict["size"].append(float(pred_box[3]))
                    box_dict["size"].append(float(pred_box[4]))
                    box_dict["size"].append(float(pred_box[5]))
                    box_dict["rotation"] = list(Quaternion(matrix=rot_matrix))


                    if timestamp not in matched_objects["results"]:
                        matched_objects["results"][timestamp] = []
                    matched_objects["results"][timestamp].append(box_dict)

                    num_pred_boxes += 1
                num_samples += 1
            

            for timestamp in sam3d_box_dict.keys():
                for i, sam3d_box in enumerate(sam3d_box_dict[timestamp]):
                    if i in sam3d_matches_dict[timestamp]:
                        continue

                    box_dict = {
                        "sample_token": timestamp,
                        "translation": [],
                        "size": [],
                        "rotation": [],
                        "velocity": [0, 0],
                        "detection_name": sam3d_supp_dict[timestamp][i][3],
                        "detection_score": np.clip(sam3d_supp_dict[timestamp][i][1] * alpha, 0, 1),
                        "attribute_name": sam3d_supp_dict[timestamp][i][0]
                    }

                    heading = sam3d_box[6]
                    rot_matrix = np.eye(3)
                    rot_matrix[0:2, 0:2] = [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]

                    box_dict["translation"].append(float(sam3d_box[0]))
                    box_dict["translation"].append(float(sam3d_box[1]))
                    box_dict["translation"].append(float(sam3d_box[2]) + float(sam3d_box[5]) / 2)
                    box_dict["size"].append(float(sam3d_box[3]))
                    box_dict["size"].append(float(sam3d_box[4]))
                    box_dict["size"].append(float(sam3d_box[5]))
                    box_dict["rotation"] = list(Quaternion(matrix=rot_matrix))


                    if timestamp not in matched_objects["results"]:
                        matched_objects["results"][timestamp] = []
                    matched_objects["results"][timestamp].append(box_dict)

                    num_sam3d_boxes += 1
                num_sam3d_samples += 1
                



            for timestamp in pred_matches_dict:
                
                for i, pred_id in enumerate(pred_matches_dict[timestamp]):

                    sam3d_id = sam3d_matches_dict[timestamp][i]
                    
                    pred_box = pred_box_dict[timestamp][pred_id]
                    sam3d_box = sam3d_box_dict[timestamp][sam3d_id]

                    sam3d_box_score = sam3d_supp_dict[timestamp][sam3d_id][1] * alpha
                    pred_box_score = pred_supp_dict[timestamp][pred_id][1]

                    if sam3d_box_score > pred_box_score:
                        box_dict = {
                            "sample_token": timestamp,
                            "translation": [],
                            "size": [],
                            "rotation": [],
                            "velocity": [0, 0],
                            "detection_name": pred_supp_dict[timestamp][pred_id][3],
                            "detection_score": np.clip(sam3d_box_score, 0, 1),
                            "attribute_name": pred_supp_dict[timestamp][pred_id][0]
                        }


                        heading = sam3d_box[6]
                        rot_matrix = np.eye(3)
                        rot_matrix[0:2, 0:2] = [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]

                        box_dict["translation"].append(float(sam3d_box[0]))
                        box_dict["translation"].append(float(sam3d_box[1]))
                        box_dict["translation"].append(float(sam3d_box[2]) + float(sam3d_box[5]) / 2)
                        box_dict["size"].append(float(sam3d_box[3]))
                        box_dict["size"].append(float(sam3d_box[4]))
                        box_dict["size"].append(float(sam3d_box[5]))
                        box_dict["rotation"] = list(Quaternion(matrix=rot_matrix))
                    
                    else:
                        box_dict = {
                            "sample_token": timestamp,
                            "translation": [],
                            "size": [],
                            "rotation": [],
                            "velocity": [0, 0],
                            "detection_name": pred_supp_dict[timestamp][pred_id][3],
                            "detection_score": pred_supp_dict[timestamp][pred_id][1],
                            "attribute_name": pred_supp_dict[timestamp][pred_id][0]
                        }


                        heading = pred_box[6]
                        rot_matrix = np.eye(3)
                        rot_matrix[0:2, 0:2] = [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]

                        box_dict["translation"].append(float(pred_box[0]))
                        box_dict["translation"].append(float(pred_box[1]))
                        box_dict["translation"].append(float(pred_box[2]) + float(pred_box[5]) / 2)
                        box_dict["size"].append(float(pred_box[3]))
                        box_dict["size"].append(float(pred_box[4]))
                        box_dict["size"].append(float(pred_box[5]))
                        box_dict["rotation"] = list(Quaternion(matrix=rot_matrix))

                    

                    if timestamp not in matched_objects["results"]:
                        matched_objects["results"][timestamp] = []
                    matched_objects["results"][timestamp].append(box_dict)

                    num_matched_boxes += 1

            
            
            print("num_samples", num_samples)
            print("num_pred_boxes", num_pred_boxes)
            print("num_sam3d_boxes", num_sam3d_boxes)
            print("num_sam3d_samples", num_sam3d_samples)
            print("num_matched_boxes", num_matched_boxes)

            with open("../../outputs/matched_pseudolabels_nusc_train_0322.json", "w") as f:
                json.dump(matched_objects, f)

            cfg_ = config_factory('detection_cvpr_2019')
            result_path_ = "../../outputs/matched_pseudolabels_nusc_train_0322.json"
            eval_set_ = "val"
            # TODO: change here for val/train
            output_dir_ = "../../outputs/nuscenes/"
            verbose_ = False
            drivable_filtering_ = False
            object_only_ = False
            # run evaluation
            nusc_eval = NuScenesEval(
                nusc,
                config=cfg_,
                result_path=result_path_,
                eval_set=eval_set_,
                output_dir=output_dir_,
                verbose=verbose_,
                drivable_filtering=drivable_filtering_,
                object_only=object_only_
            )

            metrics_summary = nusc_eval.main(
                plot_examples=False,
                render_curves=False
            )

            map_score = float(metrics_summary["mean_ap"])

            if map_score > best_score:
                best_score = map_score
                best_alpha = alpha
                # best_beta = beta
                with open("../../outputs/best_matched_pseudolabels_nusc_train_0322.json", "w") as f:
                    json.dump(matched_objects, f)

            
            print(f"Curr Score: {map_score},  Curr Alpha: {alpha}")
            print(f"Best Score: {best_score}, Best Alpha: {best_alpha}")
            print("-"*80)
    
    print()
    
