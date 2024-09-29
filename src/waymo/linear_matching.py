from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2
import os
from tqdm import tqdm
import tensorflow as tf
import numpy as np

def _create_pd_file(frame):
    o_list = []

    for laser_label in frame.laser_labels:
        # import ipdb
        # ipdb.set_trace()
        o = metrics_pb2.Object()
        o.context_name = frame.context.name
        o.frame_timestamp_micros = frame.timestamp_micros

        o.object.box.CopyFrom(laser_label.box)
        o.object.metadata.CopyFrom(laser_label.metadata)
        o.object.id = laser_label.id
        o.object.num_lidar_points_in_box = laser_label.num_lidar_points_in_box
        o.object.type = laser_label.type
        o.score = 0.5
        o_list.append(o)
    return o_list

import subprocess

"""Tools for computing box matching using Open Dataset criteria."""
import dataclasses
from typing import Optional

import tensorflow as tf

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


selected_waymo_locations = None
pred_load_dir = '../outputs/waymo/'
sam3d_load_dir = '../../../SAM3D/pred_outputs/sam3d_outputs/'
OUTPUT_DIR = '../../outputs/waymo/'


sam3d_load_path = os.path.dirname(sam3d_load_dir)
print(f'loading {sam3d_load_path}/waymo-test-1.bin ...')
sam3d_objects = metrics_pb2.Objects()

# sam3d_file = open(f'{sam3d_load_path}/waymo-test-1.bin', 'rb')
# sam3d_file = open(f'{sam3d_load_path}/waymo-test-2.bin', 'rb')
sam3d_file = open(f'{sam3d_load_path}/waymo-train.bin', 'rb')
sam3d_objects_data = sam3d_file.read()
sam3d_objects.ParseFromString(sam3d_objects_data)
sam3d_file.close()

pred_load_path = os.path.dirname(pred_load_dir)
print(f'loading {pred_load_path}/pred_1_euler2_final.bin ...')
pred_objects = metrics_pb2.Objects()

# pred_file = open(f'{pred_load_path}/pred_1_euler2_final.bin', 'rb')
# pred_file = open(f'{pred_load_path}/pred_only_2.bin', 'rb')
pred_file = open(f'../../outputs/waymo/pseudolabels_waymo_0307_train_0_798.bin', 'rb')
pred_objects_data = pred_file.read()
pred_objects.ParseFromString(pred_objects_data)
pred_file.close()



if __name__ == "__main__":
    # print(sam3d_objects)
    # print(pred_objects)

    # convert waymo objects object to a tensor of boxes
    sam3d_box_dict = {}
    sam3d_supp_dict = {}

    sam3d_max_conf = -1e7
    sam3d_min_conf = 1e7
    pred_max_conf = -1e7
    pred_min_conf = 1e7

    print("Parsing through SAM3D predictions...")
    for o in tqdm(sam3d_objects.objects):
        if (o.context_name, o.frame_timestamp_micros) not in sam3d_box_dict.keys():
            sam3d_box_dict[(o.context_name, o.frame_timestamp_micros)] = []
        if (o.context_name, o.frame_timestamp_micros) not in sam3d_supp_dict.keys():
            sam3d_supp_dict[(o.context_name, o.frame_timestamp_micros)] = []

        sam3d_box_dict[(o.context_name, o.frame_timestamp_micros)].append(np.array([
            o.object.box.center_x,
            o.object.box.center_y,
            o.object.box.center_z - o.object.box.height / 2,
            o.object.box.length,
            o.object.box.width,
            o.object.box.height,
            o.object.box.heading
        ], dtype=float))

        sam3d_supp_dict[(o.context_name, o.frame_timestamp_micros)].append(
            [
                o.context_name,
                o.score,
                o.object.id,
                o.object.type
            ]
        )

        # Update conf range
        if o.score > sam3d_max_conf:
            sam3d_max_conf = o.score
        if o.score < sam3d_min_conf:
            if o.score != 0:
                sam3d_min_conf = o.score
            else:
                print("Box score is zero, omitting")

    # print(sam3d_box_dict)
    print(len(sam3d_box_dict.keys()))
    print(len(sam3d_supp_dict.keys()))

    pred_box_dict = {}
    pred_supp_dict = {}
    print("Parsing through ZS3D predictions...")
    for o in tqdm(pred_objects.objects):
        if (o.context_name, o.frame_timestamp_micros) not in pred_box_dict.keys():
            pred_box_dict[(o.context_name, o.frame_timestamp_micros)] = []
        if (o.context_name, o.frame_timestamp_micros) not in pred_supp_dict.keys():
            pred_supp_dict[(o.context_name, o.frame_timestamp_micros)] = []
        
        pred_box_dict[(o.context_name, o.frame_timestamp_micros)].append(
            np.array([
                o.object.box.center_x,
                o.object.box.center_y,
                o.object.box.center_z - o.object.box.height / 2,
                o.object.box.length,
                o.object.box.width,
                o.object.box.height,
                o.object.box.heading
            ], dtype=float)
        )

        pred_supp_dict[(o.context_name, o.frame_timestamp_micros)].append(
            [
                o.context_name,
                o.score,
                o.object.id,
                o.object.type
            ]
        )

        # Update conf range
        if o.score > pred_max_conf:
            pred_max_conf = o.score
        if o.score < pred_min_conf:
            pred_min_conf = o.score


    # print(pred_box_dict)
    print(len(pred_box_dict.keys()))
    print(len(pred_supp_dict.keys()))

    


    pred_matches_dict = {}
    sam3d_matches_dict = {}

    print("Computing matches...")
    for timestamp in tqdm(pred_box_dict.keys()):
        
        sam3d_boxes = np.array(sam3d_box_dict[timestamp], dtype=float)
        pred_boxes = np.array(pred_box_dict[timestamp], dtype=float)

        sam3d_boxes = tf.convert_to_tensor(sam3d_boxes, dtype=float)
        pred_boxes = tf.convert_to_tensor(pred_boxes, dtype=float)
        # print(pred_boxes)

        # print("SAM3D boxes:", sam3d_boxes.shape)
        # print("Pred boxes: ", pred_boxes.shape)

        matches = match(pred_boxes, sam3d_boxes, 0.2, Type.TYPE_2D)

        if timestamp not in pred_matches_dict:
            pred_matches_dict[timestamp] = []
        if timestamp not in sam3d_matches_dict:
            sam3d_matches_dict[timestamp] = []

        if matches.ious.shape[0] > 0:
          
            for i in range(matches.ious.shape[0]):
                # print(float(matches.ious[i]))
                # print('\t', int(matches.prediction_ids[i]), ":", list(pred_boxes[matches.prediction_ids[i]].numpy()))
                # print('\t', int(matches.groundtruth_ids[i]), ":", list(sam3d_boxes[matches.groundtruth_ids[i]].numpy()))
                # print('\t', float(matches.longitudinal_affinities[i]))


                pred_matches_dict[timestamp].append(int(matches.prediction_ids[i]))
                sam3d_matches_dict[timestamp].append(int(matches.groundtruth_ids[i]))

        #         print('\tconf_pred: ', pred_supp_dict[timestamp][int(matches.prediction_ids[i])][1])
        #         print('\tconf_sam3d:', sam3d_supp_dict[timestamp][int(matches.groundtruth_ids[i])][1])
        # print('-----------------')

    # Create ALPHAS and BETAS for grid search
    # Learning the scaling (alpha) and offset (beta) values, such that sam3d_conf = sam3d_conf*alpha + beta, for linear matching
    best_alpha = 0
    best_beta = 1e7
    best_score = -1

    print("pred_min_conf = ", pred_min_conf)
    print("pred_max_conf = ", pred_max_conf)
    print("sam3d_min_conf = ", sam3d_min_conf)
    print("sam3d_max_conf = ", sam3d_max_conf)

    print(pred_min_conf / sam3d_max_conf)
    print(pred_max_conf / (sam3d_min_conf))

    ALPHAS = np.arange(
        pred_min_conf / sam3d_max_conf,
        pred_max_conf / (sam3d_min_conf) + 0.04,
        0.04,
        dtype=float
    )
    ALPHAS = list(ALPHAS)[::-1][3:]
    print("alphas:", ALPHAS)

    for i_a, alpha in tqdm(enumerate(ALPHAS)):
        # for beta in BETAS:
            print(f"For alpha = {alpha} ({i_a})")

            print("pred_min_conf = ", pred_min_conf)
            print("pred_max_conf = ", pred_max_conf)
            print("sam3d_min_conf = ", sam3d_min_conf * alpha)
            print("sam3d_max_conf = ", sam3d_max_conf * alpha)

            matched_objects = metrics_pb2.Objects()

            num_samples = 0
            num_pred_boxes = 0
            num_sam3d_boxes = 0
            num_sam3d_samples = 0
            num_matched_boxes = 0


            for timestamp in pred_box_dict.keys():
                for i, pred_box in enumerate(pred_box_dict[timestamp]):
                    if i in pred_matches_dict[timestamp]:
                        continue

                    o = metrics_pb2.Object()
                    o.context_name = pred_supp_dict[timestamp][i][0]
                    o.frame_timestamp_micros = timestamp[1]
                    
                    waymo_box = label_pb2.Label.Box()
                    waymo_box.center_x = float(pred_box[0])
                    waymo_box.center_y = float(pred_box[1])
                    waymo_box.center_z = float(pred_box[2]) + float(pred_box[5]) / 2
                    waymo_box.length = float(pred_box[3])
                    waymo_box.width = float(pred_box[4])
                    waymo_box.height = float(pred_box[5])
                    waymo_box.heading = float(pred_box[6])

                    o.object.box.CopyFrom(waymo_box)

                    o.score = pred_supp_dict[timestamp][i][1]
                    o.object.id = pred_supp_dict[timestamp][i][2]
                    o.object.type = pred_supp_dict[timestamp][i][3]

                    matched_objects.objects.append(o)

                    num_pred_boxes += 1
                num_samples += 1

            print("Number of objects after adding preds:", len(matched_objects.objects))
            
            for timestamp in sam3d_box_dict.keys():
                for i, sam3d_box in enumerate(sam3d_box_dict[timestamp]):
                    if timestamp in sam3d_matches_dict:
                        if i in sam3d_matches_dict[timestamp]:
                            continue

                    o = metrics_pb2.Object()
                    o.context_name = sam3d_supp_dict[timestamp][i][0]
                    o.frame_timestamp_micros = timestamp[1]
                    
                    waymo_box = label_pb2.Label.Box()
                    waymo_box.center_x = float(sam3d_box[0])
                    waymo_box.center_y = float(sam3d_box[1])
                    waymo_box.center_z = float(sam3d_box[2]) + float(sam3d_box[5]) / 2
                    waymo_box.length = float(sam3d_box[3])
                    waymo_box.width = float(sam3d_box[4])
                    waymo_box.height = float(sam3d_box[5])
                    waymo_box.heading = float(sam3d_box[6])

                    o.object.box.CopyFrom(waymo_box)

                    o.score = np.clip(sam3d_supp_dict[timestamp][i][1] * alpha, 0, 1)
                    
                    o.object.id = sam3d_supp_dict[timestamp][i][2]
                    o.object.type = sam3d_supp_dict[timestamp][i][3]

                    matched_objects.objects.append(o)

                    num_sam3d_boxes += 1
                num_sam3d_samples += 1

            print("Number of objects after adding sam3d boxes:", len(matched_objects.objects))

            for timestamp in pred_matches_dict:
                # print(len(pred_matches_dict[timestamp]), len(sam3d_matches_dict[timestamp]))
                # print(zip(pred_matches_dict[timestamp], sam3d_matches_dict[timestamp]))
                for i, pred_id in enumerate(pred_matches_dict[timestamp]):
                    # print(pred_id)
                    sam3d_id = sam3d_matches_dict[timestamp][i]
                    
                    pred_box = pred_box_dict[timestamp][pred_id]
                    sam3d_box = sam3d_box_dict[timestamp][sam3d_id]

                    sam3d_box_score = sam3d_supp_dict[timestamp][sam3d_id][1] * alpha
                    pred_box_score = pred_supp_dict[timestamp][pred_id][1]

                    if sam3d_box_score > pred_box_score:
                        o = metrics_pb2.Object()
                        o.context_name = sam3d_supp_dict[timestamp][sam3d_id][0]
                        o.frame_timestamp_micros = timestamp[1]
                        
                        waymo_box = label_pb2.Label.Box()
                        waymo_box.center_x = float(sam3d_box[0])
                        waymo_box.center_y = float(sam3d_box[1])
                        waymo_box.center_z = float(sam3d_box[2]) + float(sam3d_box[5]) / 2
                        waymo_box.length = float(sam3d_box[3])
                        waymo_box.width = float(sam3d_box[4])
                        waymo_box.height = float(sam3d_box[5])
                        waymo_box.heading = float(sam3d_box[6])

                        o.object.box.CopyFrom(waymo_box)

                        o.score = np.clip(sam3d_box_score, 0, 1)
                        o.object.id = sam3d_supp_dict[timestamp][sam3d_id][2]
                        o.object.type = pred_supp_dict[timestamp][pred_id][3]

                    else:
                        o = metrics_pb2.Object()
                        o.context_name = sam3d_supp_dict[timestamp][sam3d_id][0]
                        o.frame_timestamp_micros = timestamp[1]
                        
                        waymo_box = label_pb2.Label.Box()
                        waymo_box.center_x = float(pred_box[0])
                        waymo_box.center_y = float(pred_box[1])
                        waymo_box.center_z = float(pred_box[2]) + float(pred_box[5]) / 2
                        waymo_box.length = float(pred_box[3])
                        waymo_box.width = float(pred_box[4])
                        waymo_box.height = float(pred_box[5])
                        waymo_box.heading = float(pred_box[6])

                        o.object.box.CopyFrom(waymo_box)

                        o.score = pred_box_score
                        o.object.id = pred_supp_dict[timestamp][sam3d_id][2]
                        o.object.type = pred_supp_dict[timestamp][pred_id][3]

                    matched_objects.objects.append(o)

                    num_matched_boxes += 1

            print("num_samples", num_samples)
            print("num_pred_boxes", num_pred_boxes)
            print("num_sam3d_boxes", num_sam3d_boxes)
            print("num_sam3d_samples", num_sam3d_samples)
            print("num_matched_boxes", num_matched_boxes)

            f = open(f'{OUTPUT_DIR}/matched_pseudolabels_waymo_train_0310.bin', 'wb')
            f.write(matched_objects.SerializeToString())
            f.close()

            ## TODO: Evaluate.
            

            eval_str = '~/mmdetection3d/mmdet3d/evaluation/functional/waymo_utils/' + \
                f'compute_detection_metrics_main {OUTPUT_DIR}/matched_pseudolabels_waymo_train_0310.bin ' + \
                f'../../data/waymo-v1.4.2/waymo_format/gt-training.bin'
            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print(ret_texts)

            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Overall/L1 mAP': 0,
                'Overall/L1 mAPH': 0,
                'Overall/L2 mAP': 0,
                'Overall/L2 mAPH': 0
            }
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall/L1 mAP'] = \
                (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
                    ap_dict['Cyclist/L1 mAP']) / 3
            ap_dict['Overall/L1 mAPH'] = \
                (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                    ap_dict['Cyclist/L1 mAPH']) / 3
            ap_dict['Overall/L2 mAP'] = \
                (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                    ap_dict['Cyclist/L2 mAP']) / 3
            ap_dict['Overall/L2 mAPH'] = \
                (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                    ap_dict['Cyclist/L2 mAPH']) / 3

            map_score = ap_dict['Overall/L2 mAP']

            if map_score > best_score:
                best_score = map_score
                best_alpha = alpha
                # best_beta = beta

                f = open(f'{OUTPUT_DIR}/best_matched_pseudolabels_waymo_train_0310.bin', 'wb')
                f.write(matched_objects.SerializeToString())
                f.close()

            print(f"Curr Score: {map_score},  Curr Alpha: {alpha}")
            print(f"Best Score: {best_score}, Best Alpha: {best_alpha}")
            print("-"*80)

    # print(pred_matches_dict)
    # print(sam3d_matches_dict)


    
          
          

        
