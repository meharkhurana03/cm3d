# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import List, Tuple, Dict, Any, Callable, Optional
import tqdm

import numpy as np
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox, DetectionMetrics, \
    DetectionMetricDataList, DetectionMetricData
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import train, val, test, mini_train, mini_val, train_detect, train_track

from shapely.geometry import Point



class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int):

        # assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)


def add_center_dist(nusc: NuScenes,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record['translation'][0],
                               box.translation[1] - pose_record['translation'][1],
                               box.translation[2] - pose_record['translation'][2])
            # if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
            box.ego_translation = ego_translation
            # else:
            #     raise NotImplementedError

    return eval_boxes

class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        # assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        # assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            # 'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])



def category_to_detection_name_rare(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        # 'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.child': 'child',
        'human.pedestrian.stroller': 'stroller',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None

# def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
#         -> Tuple[EvalBoxes, Dict]:
#     """
#     Loads object predictions from file.
#     :param result_path: Path to the .json result file provided by the user.
#     :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
#     :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
#     :param verbose: Whether to print messages to stdout.
#     :return: The deserialized results and meta data.
#     """

#     # Load from file and check that the format is correct.
#     with open(result_path) as f:
#         data = json.load(f)
#     assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
#                               'See https://www.nuscenes.org/object-detection for more information.'

#     # Deserialize results and get meta data.
#     all_results = EvalBoxes.deserialize(data['results'], box_cls)
#     meta = data['meta']
#     if verbose:
#         print("Loaded results from {}. Found detections for {} samples."
#               .format(result_path, len(all_results.sample_tokens)))

#     # Check that each sample has no more than x predicted boxes.
#     for sample_token in all_results.sample_tokens:
#         assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
#             "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

#     return all_results, meta

def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val,
                    'train_detect': train_detect, 'train_track': train_track,
                    'train_detect50': train_detect[:50],
                    'train_detect25': train_detect[175:200],
                    'val25': val[:25],
                    'train25': train[:10]}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits

def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False, rare: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track', 'train_detect50', 'train_detect25', 'val25', 'train25'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                if rare:
                    detection_name = category_to_detection_name_rare(sample_annotation['category_name'])
                else:
                    detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue
                # if detection_name in ["child", "stroller", "construction_vehicle", "barrier", "trailer"]:
                # if detection_name == "trailer":
                #     scene_token = sample['scene_token']
                #     scene_record = nusc.get('scene', scene_token)
                #     print(f"{detection_name} found in ", scene_record["name"])
                

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations

def get_nusc_map(nusc, scene):
    # Get scene location
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    # Get nusc map
    nusc_map = NuScenesMap(dataroot='../../data/nuScenes', map_name=location)

    return nusc_map

# modified from nuscenes.eval.common.loaders
def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field

# modified from nuscenes.eval.common.loaders
def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      drivable_filtering: bool = True,
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter, drivable_filter = 0, 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get('sample', sample_token)['anns']
        bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                         nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("> Original number of boxes: %d" % total)
        print("> After distance based filtering: %d" % dist_filter)
        print("> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("> After bike rack filtering: %d" % bike_rack_filter)

    if drivable_filtering:

        sample = nusc.get('sample', eval_boxes.sample_tokens[0])
        scene = nusc.get('scene', sample['scene_token'])
        nusc_map = get_nusc_map(nusc, scene)

        drivable_records = nusc_map.drivable_area

        drivable_polygons = []
        for record in drivable_records:
            # print(record)
            polygons = [nusc_map.extract_polygon(token) for token in record["polygon_tokens"]]
            drivable_polygons.extend(polygons)
        print(eval_boxes)

        for ind, sample_token in enumerate(eval_boxes.sample_tokens):
            
            filtered_boxes = []
            for box in eval_boxes[sample_token]:
                m_x, m_y, m_z = box.translation
                point = Point(m_x, m_y)
                is_drivable = False
                for polygon in drivable_polygons:
                    if point.within(polygon):
                        is_drivable = True
                        break
                if is_drivable:
                    filtered_boxes.append(box)

            eval_boxes.boxes[sample_token] = filtered_boxes
            drivable_filter += len(eval_boxes.boxes[sample_token])

        if verbose:
            print("> After drivable area filtering: %d" % drivable_filter)



        

    return eval_boxes


def calc_recall(md):
    rec = np.copy(md.recall)
    return rec

def accumulate_object_class(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
            #    class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------
    class_name = "object"
    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))

            if not gt_box_match.detection_name in ['traffic_cone', 'barrier']:
                match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            else:
                match_data['vel_err'].append(np.nan)
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            # period = np.pi if class_name == 'barrier' else 2 * np.pi
            period = np.pi
            if not gt_box_match.detection_name in ['traffic_cone']:
                match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))
            else:
                match_data['orient_err'].append(np.nan)

            if not gt_box_match.detection_name in ['barrier', 'traffic_cone']:
                match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            else:
                match_data['attr_err'].append(np.nan)
            
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_actual = np.max(rec)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err']), rec_actual

def accumulate_with_recall(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return 0, DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return 0, DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    # print(rec, tp)
    rec_actual = np.max(rec)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return rec_actual, DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 drivable_filtering: bool = True,
                 object_only: bool = True,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.drivable_filtering = drivable_filtering
        self.object_only = object_only
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        if len(self.cfg.class_range.values()) > 10:
            self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose, rare=True)
        else:
            self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
        print(len(self.pred_boxes.sample_tokens), len(self.gt_boxes.sample_tokens))
        # self.pred_boxes.sample_tokens = 
        # assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
        #     "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose,
                                            drivable_filtering=drivable_filtering)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose,
                                          drivable_filtering=drivable_filtering)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        recall_list = []
        metric_data_list = DetectionMetricDataList()

        if self.object_only:
            recall_th_list = []
            for dist_th in self.cfg.dist_ths:
                md, rec = accumulate_object_class(self.gt_boxes, self.pred_boxes, self.cfg.dist_fcn_callable, dist_th, verbose=True)
                metric_data_list.set("object", dist_th, md)
                recall_th_list.append(rec)
            print(recall_th_list)
            recall_list.append(sum(recall_th_list)/len(recall_th_list))
                
        else:
            for class_name in self.cfg.class_names:
                recall_th_list = []
                for dist_th in self.cfg.dist_ths:
                    rec, md = accumulate_with_recall(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th, verbose=True)
                    metric_data_list.set(class_name, dist_th, md)
                    recall_th_list.append(rec)
                recall_list.append(sum(recall_th_list)/len(recall_th_list))



        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        if self.object_only:
            # Compute AP
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[("object", dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap("object", dist_th, ap)
            
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[("object", self.cfg.dist_th_tp)]
                # if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                #     tp = 1
                # elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                #     tp = 1
                # else:
                tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp("object", metric_name, tp)
        
        else:
            for class_name in self.cfg.class_names:
                # Compute APs.
                for dist_th in self.cfg.dist_ths:
                    metric_data = metric_data_list[(class_name, dist_th)]
                    ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                    metrics.add_label_ap(class_name, dist_th, ap)

                # Compute TP metrics.
                for metric_name in TP_METRICS:
                    metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                    if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                        tp = np.nan
                    elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                        tp = np.nan
                    else:
                        tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                    metrics.add_label_tp(class_name, metric_name, tp)
        
        

        # Compute recall.
        # recall_lists = []
        # for dth in self.cfg.dist_ths:
        #     md = metric_data_list[("object", dth)]
        #     rec = [calc_recall(md)]
        #     recall_lists.append(rec)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list, recall_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list, recall_list = self.evaluate()


        print(metrics.tp_scores)
        print(metrics._label_tp_errors)

        # # Render PR and TP curves.
        # if render_curves:
        #     self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('mRec: %.4f' % (sum(recall_list)/len(recall_list)))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE', 'avgRec'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for i, class_name in enumerate(class_aps.keys()):
            print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err'],
                    recall_list[i],
                    ))
        
        # print(recall_list)

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--drivable_filtering', type=int, default=1,
                        help='Whether to filter out boxes outside the drivable regions.')
    parser.add_argument('--object_only', type=int, default=0,
                        help='Whether to calculate the generic-object AP and recall.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    drivable_filtering_ = bool(args.drivable_filtering)
    object_only_ = bool(args.object_only)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_, drivable_filtering=drivable_filtering_,
                              object_only=object_only_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)