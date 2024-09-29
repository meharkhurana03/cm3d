import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torchvision
import copy
import multiprocessing as mp
# import pointops
import random
import argparse
import json
import pickle
import pycocotools
from pycocotools import mask
# print(getattr(pycocotools))
# import hdbscan
from inspect import getmembers, isfunction
# from nms.nms import rboxes, malisiewicz
from pyquaternion import Quaternion as pQuaternion

# import groundingdino.datasets.transforms as T
# from groundingdino.models import build_model
# from groundingdino.util.slconfig import SLConfig
# from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image, ImageDraw, ImageFont
from os.path import join

# import trimesh
import sys
from shapely.geometry import Point, box
# TODO: handle this
# from zoedepth.utils.geometry import depth_to_points, create_triangles
# from zoedepth.utils.misc import get_image_from_url, colorize

from functools import partial
# import tempfile

# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import Box
# from pyquaternion import Quaternion
# from nuscenes.utils.data_classes import Quaternion as nQuaternion
#from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from utils.pcd import LidarPointCloud, view_points
# from nuscenes.nuscenes import NuScenes, NuScenesExplorer
# # from nuscenes.utils.geometry_utils import view_points
# from nuscenes.map_expansion.map_api import NuScenesMap
# from nuscenes.map_expansion import arcline_path_utils
# from nuscenes.map_expansion.bitmap import BitMap
# from nuscenes.utils.splits import mini_val, mini_train, train_detect, train, val

from kitti_object import kitti_object


# from zoedepth.models.builder import build_model as build_model_zoe
# from zoedepth.utils.config import get_config

from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import scipy
import time
import tqdm

from cfg.prompt_cfg import MAPS

# from mmdetection3d.mmdet3d.models.layers import box3d_multiclass_nms



# VER_NAME = "v1.0-mini"
# INPUT_PATH = "../nusc-mini/"
VER_NAME = "v1.0-trainval"
# INPUT_PATH = "../../../nuScenes/"
# INPUT_PATH = "/ssd0/nperi/nuScenes/"
# INPUT_PATH = "/data2/mehark/nuScenes/nuScenes/"
INPUT_PATH = "/data2/mehark/kitti/"


CAM_LIST = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
ATTRIBUTE_NAMES = {
    "barrier": "",
    "traffic_cone": "",
    "bicycle": "cycle.without_rider",
    "motorcycle": "cycle.without_rider",
    "pedestrian": "pedestrian.standing",
    "car": "vehicle.stopped",
    "bus": "vehicle.stopped",
    "construction_vehicle": "vehicle.stopped",
    "trailer": "vehicle.stopped",
    "truck": "vehicle.stopped",
}

KITTI_CLASS_MAPS = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "truck": "Truck",
    "bus": "Tram",
    "traffic_cone": "Misc",
    "construction_vehicle": "Misc",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
    "trailer": "Misc",
    "barrier": "Misc",
}


OUTPUT_DIR = "../../outputs/kitti/"
PRED_DIR = "/data2/mehark/kitti/training/pred/"
PSEUDO_DIR = "/data2/mehark/kitti/training/pseudo/"
# INPUT_DIR = "/ssd0/mehark/zs3d_outputs/nuscenes/"
INPUT_DIR = "/data2/mehark/zs3d_outputs/kitti_detic_wo_2d_nms/"  # ran nms for this actually 
# INPUT_DIR = "../wider_model_outputs"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
# zoe_model = (torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(DEVICE).eval())



# utils
def count_frames(nusc, sample):
    frame_count = 1

    if sample["next"] != "":
        frame_count += 1

        # Don't want to change where sample['next'] points to since it's used later, so we'll create our own pointer
        sample_counter = nusc.get('sample', sample['next'])

        while sample_counter['next'] != '':
            frame_count += 1
            sample_counter = nusc.get('sample', sample_counter['next'])
    
    return frame_count

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def clusters_hdbscan(points_set):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=10, cluster_selection_epsilon=1
                                )
    clusterer.fit(points_set)
    labels = clusterer.labels_.copy()
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    try:
        cluster_info = cluster_info[cluster_info[:, 1].argsort()]
    except:
        return None
    clusters_labels = cluster_info[::-1][:, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1
    return labels


def get_medoid(points):
    dist_matrix = torch.cdist(points.T, points.T, p=2)
    # print(dist_matrix)
    return torch.argmin(dist_matrix.sum(axis=0))


def get_detection_name(name):
    if name not in ["trafficcone", "constructionvehicle", "human"]:
        detection_name = name
    elif name == "trafficcone":
        print("TRAFFIC_CONE DETECTED xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        detection_name = "traffic_cone"
    elif name == "constructionvehicle":
        print("CONSTRUCTION_VEHICLE DETECTED xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        detection_name = "construction_vehicle"
    elif name == "human":
        detection_name = "pedestrian"

    detection_name = KITTI_CLASS_MAPS[detection_name]
    
    return detection_name

def get_shape_prior(shape_priors, name, chatgpt=True):
    # add priors for more categories
    if not chatgpt:
        if name == "car":
            return shape_priors["vehicle.car"]
        elif name == "bicycle":
            return shape_priors["vehicle.bicycle"]
        elif name == "bus":
            return shape_priors["vehicle.bus.rigid"]
        elif name == "truck":
            return shape_priors["vehicle.truck"]
        elif name == "pedestrian":
            return shape_priors["human.pedestrian.adult"]
        elif name == "traffic_cone":
            return shape_priors["movable_object.trafficcone"]
        elif name == "construction_vehicle":
            return shape_priors["vehicle.construction"]
        elif name == "motorcycle":
            return shape_priors["vehicle.motorcycle"]
        elif name == "trailer":
            return shape_priors["vehicle.trailer"]
        elif name == "child":
            return shape_priors["human.pedestrian.child"]
        elif name == "stroller":
            return shape_priors["human.pedestrian.adult"]
    
    else:
        return shape_priors[name]


def push_centroid(centroid, extents, rot_quaternion, poserecord=None, ego_frame=False):
    centroid = np.squeeze(centroid)
    # print("centroid", centroid)

    if not ego_frame:
        av_centroid = poserecord["translation"]
        ego_centroid = centroid - av_centroid
    else:
        ego_centroid = centroid

    # print("av_centroid", av_centroid)

    # print(ego_centroid)
    l = extents[0]
    w = extents[1]

    # w = extents[0]
    # l = extents[1]
    
    angle = R.from_quat(list(rot_quaternion)).as_euler('xyz', degrees=False)
    # print(angle)
    theta = -angle[0]

    #TODO: check this
    if np.isnan(theta):
        theta = 0.5 * np.pi
    
    # print("theta", theta)
    # print(ego_centroid)
    
    alpha = np.arctan(np.abs(ego_centroid[1]) / np.abs(ego_centroid[0]))
    # alpha = np.arctan(ego_centroid[1] / ego_centroid[0])

    if ego_centroid[0] < 0:
        if ego_centroid[1] < 0:
            alpha = -np.pi + alpha # do something else
        else:
            alpha = np.pi - alpha
    else:
        if ego_centroid[1] < 0:
            alpha = -alpha

    # if ego_centroid[0] < 0:
    #     alpha = np.pi + alpha
    

    # if alpha < 0:
    #     alpha = np.pi + alpha
    
    # print("alpha", alpha)

    # print(w / (2*np.sin(theta - alpha)), l / (2*np.cos(theta - alpha)))
    offset = np.min( [np.abs(w / (2*np.sin(theta - alpha))), np.abs(l / (2*np.cos(theta - alpha)))] )

    # if offset == np.abs(w / (2*np.sin(theta - alpha))):
    #     offset = w / (2*np.sin(theta - alpha))
    # elif offset == np.abs(l / (2*np.cos(theta - alpha))):
    #     offset = l / (2*np.cos(theta - alpha))
    # else:
    #     assert False == True, "incorrect offset calculation"
    
    # print("centroid before:", centroid)
    # print(offset)
    x_dash = centroid[0] + offset * np.cos(alpha)
    y_dash = centroid[1] + offset * np.sin(alpha)

    # print(x_dash, y_dash, centroid[2])
    pushed_centroid = np.array([x_dash, y_dash, centroid[2]])
    # print("centroid after:", pushed_centroid)

    return pushed_centroid

# def get_vertex_pairs(centroid, theta, extents):
#     x1 = centroid[0] + np.cos(theta)*extents[1]/2 - np.sin(theta)*extents[0]/2
#     y1 = centroid[1] + np.sin(theta)*extents[1]/2 + np.cos(theta)*extents[0]/2
#     x2 = centroid[0] - np.cos(theta)*extents[1]/2 + np.sin(theta)*extents[0]/2
#     y2 = centroid[1] - np.sin(theta)*extents[1]/2 - np.cos(theta)*extents[0]/2
#     x3 = centroid[0] - np.cos(theta)*extents[1]/2 - np.sin(theta)*extents[0]/2
#     y3 = centroid[1] - np.sin(theta)*extents[1]/2 + np.cos(theta)*extents[0]/2
#     x4 = centroid[0] + np.cos(theta)*extents[1]/2 + np.sin(theta)*extents[0]/2
#     y4 = centroid[1] + np.sin(theta)*extents[1]/2 - np.cos(theta)*extents[0]/2

#     pairs = ((x1, y1, x2, y2), (x3, y3, x4, y4))
#     print(pairs)
#     return pairs

def get_top_left_vertex(centroid, extents, theta):
    # x1 = centroid[0] - np.cos(theta)*extents[1]/2 - np.sin(theta)*extents[0]/2
    # y1 = centroid[1] - np.sin(theta)*extents[1]/2 + np.cos(theta)*extents[0]/2

    # x2 = centroid[0] + np.cos(theta)*extents[1]/2 - np.sin(theta)*extents[0]/2
    # y2 = centroid[1] + np.sin(theta)*extents[1]/2 + np.cos(theta)*extents[0]/2

    # x3 = centroid[0] + np.cos(theta)*extents[1]/2 + np.sin(theta)*extents[0]/2
    # y3 = centroid[1] + np.sin(theta)*extents[1]/2 - np.cos(theta)*extents[0]/2

    # x4 = centroid[0] - np.cos(theta)*extents[1]/2 + np.sin(theta)*extents[0]/2
    # y4 = centroid[1] - np.sin(theta)*extents[1]/2 - np.cos(theta)*extents[0]/2

    # x_list = sorted([x1, x2, x3, x4])
    # y_list = sorted([x1, x2, x3, x4])

    # if (x1 == x_list[0] and y1 == y_list[3]): return x1, y1
    # if (x1 == x_list[1] and y1 == y_list[3]): return x1, y1
    # if (x1 == x_list[]

    if (theta > 0):
        x1 = centroid[0] - np.cos(theta)*extents[1]/2 - np.sin(theta)*extents[0]/2
        y1 = centroid[1] - np.sin(theta)*extents[1]/2 + np.cos(theta)*extents[0]/2
    
    else:
        x1 = centroid[0] + np.cos(theta)*extents[1]/2 + np.sin(theta)*extents[0]/2
        y1 = centroid[1] + np.sin(theta)*extents[1]/2 - np.cos(theta)*extents[0]/2
    



    # if abs(theta) > np.pi/2:
    # # tried - - - +, 
    #     x1 = centroid[0] - np.cos(theta)*extents[1]/2 - np.sin(theta)*extents[0]/2
    #     y1 = centroid[1] - np.sin(theta)*extents[1]/2 + np.cos(theta)*extents[0]/2
    
    # else:


    return x1, y1



# nms helpers
def createImage(width=800, height=800, depth=3):
    """ Return a black image with an optional scale on the edge

    :param width: width of the returned image
    :type width: int
    :param height: height of the returned image
    :type height: int
    :param depth: either 3 (rgb/bgr) or 1 (mono).  If 1, no scale is drawn
    :type depth: int
    :return: A zero'd out matrix/black image of size (width, height)
    :rtype: :class:`numpy.ndarray`
    """
    # create a black image and put a scale on the edge

    assert depth == 3 or depth == 1
    assert width > 0
    assert height > 0

    hashDistance = 50
    hashLength = 20

    img = np.zeros((int(height), int(width), depth), np.uint8)

    if(depth == 3):
        for x in range(0, int(width / hashDistance)):
            cv2.line(img, (x * hashDistance, 0), (x * hashDistance, hashLength), (0,0,255), 1)

        for y in range(0, int(width / hashDistance)):
            cv2.line(img, (0, y * hashDistance), (hashLength, y * hashDistance), (0,0,255), 1)

    return img

def polygon_intersection_area(polygons):
    """ Compute the area of intersection of an array of polygons

    :param polygons: a list of polygons
    :type polygons: list
    :return: the area of intersection of the polygons
    :rtype: int
    """
    if len(polygons) == 0:
        return 0

    dx = 0
    dy = 0

    maxx = np.amax(np.array(polygons)[...,0])
    minx = np.amin(np.array(polygons)[...,0])
    maxy = np.amax(np.array(polygons)[...,1])
    miny = np.amin(np.array(polygons)[...,1])

    if minx < 0:
        dx = -int(minx)
        maxx = maxx + dx
    if miny < 0:
        dy = -int(miny)
        maxy = maxy + dy
    # (dx, dy) is used as an offset in fillPoly

    for i, polypoints in enumerate(polygons):

        newImage = createImage(maxx, maxy, 1)

        polypoints = np.array(polypoints, np.int32)
        polypoints = polypoints.reshape(-1, 1, 2)

        cv2.fillPoly(newImage, [polypoints], (255, 255, 255), cv2.LINE_8, 0, (dx, dy))

        if(i == 0):
            compositeImage = newImage
        else:
            compositeImage = cv2.bitwise_and(compositeImage, newImage)

        area = cv2.countNonZero(compositeImage)

    return area


def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    """ Get the max scores with corresponding indicies

    Adapted from the OpenCV c++ source in `nms.inl.hpp <https://github.com/opencv/opencv/blob/ee1e1ce377aa61ddea47a6c2114f99951153bb4f/modules/dnn/src/nms.inl.hpp#L33>`__

    :param scores: a list of scores
    :type scores: list
    :param threshold: consider scores higher than this threshold
    :type threshold: float
    :param top_k: return at most top_k scores; if 0, keep all
    :type top_k: int
    :param descending: if True, list is returened in descending order, else ascending
    :returns: a  sorted by score list  of [score, index]
    """
    score_index = []

    # Generate index score pairs
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
        else:
            score_index.append([score, i])

    # Sort the score pair according to the scores in descending order
    npscores = np.array(score_index)

    if descending:
        npscores = npscores[npscores[:,0].argsort()[::-1]] #descending order
    else:
        npscores = npscores[npscores[:,0].argsort()] # ascending order

    if top_k > 0:
        npscores = npscores[0:top_k]

    return npscores.tolist()


def poly_areas(polys):
    """Calculate the area of the list of polygons

    :param polys: a list of polygons, each specified by a list of its verticies
    :type polys: list
    :return: numpy array of areas of the polygons
    :rtype: :class:`numpy.ndarray`
    """
    areas = []
    for poly in polys:
        areas.append(cv2.contourArea(np.array(poly, np.int32)))
    return np.array(areas)

def poly_compare(poly1, polygons, area):
    """Calculate the intersection of poly1 to polygons divided by area

    :param poly1: a polygon specified by a list of its verticies
    :type poly1: list
    :param polygons: a list of polygons, each specified a list of its verticies
    :type polygons: list
    :param area: a list of areas of the corresponding polygons
    :type area: list
    :return: a numpy array of the ratio of overlap of poly1 to each of polygons to the corresponding area.  e.g. overlap(poly1, polygons[n])/area[n]
    :rtype: :class:`numpy.ndarray`
    """
    # return intersection of poly1 with polys[i]/area[i]
    overlap = []
    for i,poly2 in enumerate(polygons):
        intersection_area = polygon_intersection_area([poly1, poly2])
        overlap.append(intersection_area/area[i])

    return np.array(overlap)

def nms(rrects, scores, **kwargs):
    polys = []
    for rrect in rrects:
        r = cv2.boxPoints(rrect)
        # print(r)
        polys.append(r)
    
    kwargs['area_function'] = poly_areas
    kwargs['compare_function'] = poly_compare

    boxes = polys

    if 'top_k' in kwargs:
        top_k = kwargs['top_k']
    else:
        top_k = 0
    assert 0 <= top_k

    if 'score_threshold' in kwargs:
        score_threshold = kwargs['score_threshold']
    else:
        score_threshold = 0.3
    assert 0 < score_threshold

    if 'nms_threshold' in kwargs:
        nms_threshold = kwargs['nms_threshold']
    else:
        nms_threshold = 0.4
    assert 0 < nms_threshold < 1

    if 'compare_function' in kwargs:
        compare_function = kwargs['compare_function']
    else:
        compare_function = None
    assert compare_function is not None

    if 'area_function' in kwargs:
        area_function = kwargs['area_function']
    else:
        area_function = None
    assert area_function is not None

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    if scores is not None:
        assert len(scores) == len(boxes)

    boxes = np.array(boxes)

    # if compare_function == rect_compare:
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        # if boxes.dtype.kind == "i":
            # boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = area_function(boxes) #(x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the boxes by score or the bottom-right y-coordinate of the bounding box
    if scores is not None:
        # sort the bounding boxes by the associated scores
        scores = get_max_score_index(scores, score_threshold, top_k, False)
        idxs = np.array(scores, np.int32)[:, 1]
        # idxs = np.argsort(scores)
    else:
        # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
        y2 = boxes[:3]
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    #boxes = np.array(boxes)
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compute the ratio of overlap
        overlap = compare_function(boxes[i], boxes[idxs[:last]], areas[idxs[:last]])
        #(w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > nms_threshold)[0])))

    # return the indicies of the picked bounding boxes that were picked
    return pick


def get_nusc_map(nusc, scene):
    # Get scene location
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    # Get nusc map
    nusc_map = NuScenesMap(dataroot=INPUT_PATH, map_name=location)

    return nusc_map


def backproject_points(points, depth, view, normalized=True):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 2

    # print("points to backproject", points)

    nbr_points = points.shape[1]

    points = np.concatenate([points, np.ones(nbr_points)[None]])
    if normalized:
        points = np.multiply(points, depth.repeat(3, 0).reshape(nbr_points, 3).transpose())
    
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))

    viewpad_inv = np.linalg.inv(viewpad)
    points = np.dot(viewpad_inv, points)
    # print(points.shape)
    points = points[:3, :]

    return points


def remove_pixels_in_exterior(pixels, inds_x, inds_y, num_virtual):
    new_inds_x = []
    new_inds_y = []
    
    for i in range(num_virtual):
        if inds_x[i] in pixels[0] and inds_y[i] in pixels[1]:
            new_inds_x.append(inds_x[i])
            new_inds_y.append(inds_y[i])
            continue
        num_virtual -= 1

    return np.array(new_inds_x), np.array(new_inds_y), num_virtual


def add_virtual_points(pixels, masked_pc_pixels, pixel_depths, num_virtual=50):
    # pixels is a 2-tuple of size N np.ndarray.
    num_virtual = len(pixels[0]) if len(pixels[0]) < num_virtual else num_virtual

    min_x = np.min(pixels[0])
    min_y = np.min(pixels[1])
    max_x = np.max(pixels[0])
    max_y = np.max(pixels[1])

    mid_x = (max_x + min_x) / 2
    mid_y = (max_y + min_y) / 2

    gauss_inds_x = np.floor(np.random.normal(loc=mid_x, scale=1, size=num_virtual))
    gauss_inds_y = np.floor(np.random.normal(loc=mid_y, scale=1, size=num_virtual))

    inds_x, inds_y, num_virtual = remove_pixels_in_exterior(pixels, gauss_inds_x, gauss_inds_y, num_virtual)
    if num_virtual == 0:
        return torch.Tensor([[], [], []])

    print('i', inds_x.shape, inds_y.shape)
    selected_pixels = torch.vstack([torch.from_numpy(inds_x),
                                    torch.from_numpy(inds_y)])

    selected_pixels = selected_pixels.to(dtype=torch.float32, device=DEVICE).reshape(-1, num_virtual)
    selected_pixels = torch.transpose(selected_pixels, 0, 1)


    # if len(selected_indices) < num_virtual:
    #     selected_indices = torch.cat([selected_indices, selected_indices[
    #         selected_indices.new_zeros(num_virtual - len(selected_indices))]])
    
    # print(selected_indices)
    # print(pixels[0])

    """"""
    # selected_indices = torch.randperm(len(pixels[0]))[:num_virtual]

    # selected_pixels = torch.vstack([torch.from_numpy(pixels[0][selected_indices]),
    #                                 torch.from_numpy(pixels[1][selected_indices])])

    # selected_pixels = selected_pixels.to(dtype=torch.float32, device=DEVICE).reshape(-1, num_virtual)
    # selected_pixels = torch.transpose(selected_pixels, 0, 1)
    """"""

    depths = torch.from_numpy(pixel_depths).to(dtype=torch.float32, device=DEVICE)
    masked_pc_pixels = torch.from_numpy(masked_pc_pixels[:2, :]).to(dtype=torch.float32, device=DEVICE).reshape(-1, 2)

    # print(selected_pixels)
    # print(masked_pc_pixels)

    dist = torch.cdist(selected_pixels, masked_pc_pixels, p=2)

    # print(dist.shape)

    nearest_dist, nearest_indices = torch.min(dist, dim=1)

    # print(len(nearest_dist), len(nearest_indices))

    virtual_depths = depths[nearest_indices]

    # print(virtual_depths)

    coords = torch.hstack([torch.ones(num_virtual, device=DEVICE).unsqueeze(1)*1024,
                           torch.ones(num_virtual, device=DEVICE).unsqueeze(1)*576]) - selected_pixels
    # print(selected_pixels, coords)

    virtual_points = torch.vstack([torch.transpose(selected_pixels, 0, 1).cpu(),
                                   virtual_depths.cpu()])
    # print(virtual_points.shape)

    return virtual_points

    # return virtual_depths, nearest_indices



def get_all_lane_points_in_scene(nusc_map):
    lane_records = nusc_map.lane + nusc_map.lane_connector
    print("num_lanes", len(lane_records))

    lane_tokens = [lane["token"] for lane in lane_records]

    # print([lane["token"] for lane in lane_records])

    lane_pt_dict = nusc_map.discretize_lanes(lane_tokens, 0.5)
    # print(lane_pt_dict.values())

    all_lane_pts = []
    for lane_pts in lane_pt_dict.values():
        for lane_pt in lane_pts:
            all_lane_pts.append(lane_pt)
    print(len(all_lane_pts))

    # lane_polygons = []
    # for record in lane_records:
    #     # print(record)
    #     polygons = [nusc_map.extract_polygon(record["polygon_token"])]
    #     lane_polygons.extend(polygons)
    # # print(lane_polygons)
    
    return lane_pt_dict, all_lane_pts

    
def distance_matrix_lanes(A, B, squared=False):
    A = A.to(device=DEVICE)
    B = B.to(device=DEVICE)

    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = torch.mul(
                torch.mul(A, A).sum(dim=1).reshape((M,1)),
                torch.Tensor(np.ones(shape=(1,N))).to(device=DEVICE)
            )
    B_dots = torch.mul(
                torch.mul(B, B).sum(dim=1),
                torch.Tensor(np.ones(shape=(M,1))).to(device=DEVICE)
            )
    D_squared =  torch.sub(
                    torch.add(A_dots, B_dots),
                    torch.mul(
                        2, torch.mm(
                            A, B.transpose(0, 1)
                        )
                    )
                )

    if squared == False:
        zero_mask = torch.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return torch.sqrt(D_squared)

    print(A[1, :], B[1, :], D_squared[1, 1])
    return D_squared

def lane_yaws_distances_and_coords(all_centroids, all_lane_pts):
    all_lane_pts = torch.Tensor(all_lane_pts).to(device='cpu')
    all_centroids = torch.Tensor(all_centroids).to(device='cpu')
    print(all_lane_pts, all_centroids)
    start = time.time()
    # DistMat = distance_matrix_lanes(all_centroids[:, :2], all_lane_pts[:, :2])
    # DistMat = distance_matrix(all_centroids[:, :2], all_lane_pts[:, :2])
    DistMat = scipy.spatial.distance.cdist(all_centroids[:, :2], all_lane_pts[:, :2])
    
    min_lane_indices = np.argmin(DistMat, axis=1)
    print(min_lane_indices)
    distances = np.min(DistMat, axis=1)

    all_lane_pts = np.array(all_lane_pts)
    min_lanes = np.array([all_lane_pts[min_lane_indices[0]]])
    for idx in min_lane_indices:
        # print(idx)
        min_lanes = np.vstack([min_lanes, all_lane_pts[idx, :]])
    
    # print(min_lanes.shape)

    yaws = min_lanes[1:, 2]
    coords = min_lanes[1:, :2]

    print(distances.shape, yaws.shape, coords.shape)
    end = time.time()

    print(f"Closest lane took {end - start} seconds.")
    timer['closest lane'] += end - start

    return yaws, distances, coords


# Thanks to Tianwei Yin for the following function, available at https://github.com/tianweiy/CenterPoint/blob/master/det3d/core/utils/circle_nms_jit.py
import numba

# @numba.jit(nopython=True)
def circle_nms(dets, det_labels, threshs_by_label):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i]-x1[j])**2 + (y1[i]-y1[j])**2

            # ovr = inter / areas[j]
            if dist <= threshs_by_label[det_labels[j]] and det_labels[j] == det_labels[i]:
                suppressed[j] = 1
    return keep


def get_depth_bbox(pts3d):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts3d)

    obb = point_cloud.get_oriented_bounding_box()
    
    x_size = pts3d[:, 0].max() - pts3d[:, 0].min()
    y_size = pts3d[:, 1].max() - pts3d[:, 1].min()
    z_size = pts3d[:, 2].max() - pts3d[:, 2].min()
    
    axis = [ax[1] for ax in sorted([(x_size, "x"), (y_size, "y"), (z_size, "z")], key = lambda x : x[0])]

    center = obb.center.tolist()
    wlh = obb.extent.tolist()
    R = obb.R
    
    wlh = [wlh[axis.index("x")], wlh[axis.index("y")], wlh[axis.index("z")]]
    center = [center[0], center[1], center[2]]

    R = np.stack([R[:, axis.index("z")], R[:, axis.index("y")], R[:, axis.index("x")]], axis=1)
    
    return center, wlh, R


def save_pred(pred_path, object_type, ltrb, wlh, xyz, yaw, conf, truncation=-1, occlusion=-1, alpha=-10):
    if conf == None:
        with open(pred_path, 'a') as f:
            f.write(f"{object_type} {truncation} {occlusion} {alpha} {ltrb[0]} {ltrb[1]} {ltrb[2]} {ltrb[3]} {wlh[0]} {wlh[1]} {wlh[2]} {xyz[0]} {xyz[1]} {xyz[2]} {yaw}\n")
    else:
        with open(pred_path, 'a') as f:
            f.write(f"{object_type} {truncation} {occlusion} {alpha} {ltrb[0]} {ltrb[1]} {ltrb[2]} {ltrb[3]} {wlh[0]} {wlh[1]} {wlh[2]} {xyz[0]} {xyz[1]} {xyz[2]} {yaw} {conf}\n")










if __name__ == "__main__":
    total_start = time.time()
    camera_channel="CAM_BACK"
    pointsensor_channel="LIDAR_TOP"
    out_path=os.path.join(OUTPUT_DIR, "lidar_2.png")
    dot_size=0.5
    min_dist= 2.3
    render_intensity= False
    show_lidarseg = False
    filter_lidarseg_labels = None
    lidarseg_preds_bin_path = None
    show_panoptic = False
    ax = None
    floor_thresh = 0.6
    # frame_num = 0

    predictions = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": True,
            "use_external": False,
        },
        "results": {}
    }

    timer = {
        "io": 0,
        "points in mask": 0,
        "mvp": 0,
        "medoid": 0,
        "drivable": 0,
        "closest lane": 0,
        "lane pose": 0,
        "nms": 0,
        "total": 0,
    }

    # c = 3
    # nusc = NuScenes(VER_NAME, INPUT_PATH, True)
    kitti = kitti_object(os.path.join(INPUT_PATH))

    # shape_priors = json.load(open("shape_priors.json"))
    shape_priors = json.load(open("cfg/shape_priors_chatgpt.json"))

    # progress_bar_main = tqdm.tqdm(enumerate(train[:350]))
    # progress_bar_main = tqdm.tqdm(enumerate(val))
    progress_bar = tqdm.tqdm(range(len(kitti)))
    # for scene_name in mini_train:
    # for scene_name in mini_val:
    # for scene_name in train_detect[:50]:
    # for scene_name in train_detect[175:200]:
    # for scene_num, scene_name in progress_bar_main:
    # # for scene_num, scene_name in enumerate(val[25:]):
    #     progress_bar_main.set_description(f"Scene Progress:")

    # scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    # scene = nusc.get('scene', scene_token)

    # Get map
    # nusc_map = get_nusc_map(nusc, scene)

    # drivable_records = nusc_map.drivable_area

    # drivable_polygons = []
    # for record in drivable_records:
        # print(record)
        # polygons = [nusc_map.extract_polygon(token) for token in record["polygon_tokens"]]
        # drivable_polygons.extend(polygons)
    
    # lane_records = nusc_map.lane + nusc_map.lane_connector
    # print("num_lanes", len(lane_records))
    # lane_polygons = []
    # for record in lane_records:
    #     # print(record)
    #     polygons = [nusc_map.extract_polygon(record["polygon_token"])]
    #     lane_polygons.extend(polygons)
    # print(lane_polygons)

    # lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)
    # print(lane_pt_list)
    # print(lane_pt_dict)
    # print(lane_pt_list[:10], len(lane_pt_list))
    all_centroids_list = []
    centroid_ids = []
    id_offset = -1
    id_offset_list1 = []

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(PSEUDO_DIR, exist_ok=True)

    num_frames = len(kitti)
    # progress_bar = tqdm.tqdm(range(num_frames))
    for frame_num in progress_bar:
        # progress_bar.set_description(f"Processing {scene_name} ({scene_num})")

        # TODO: Switch to reading the masks.pkl files, set coco_decode_in as {'size': (1024, 576), 'counts': mask}
        # print(kitti.get_image(0).shape); exit()
        image_size = [370, 1224]
        ratio = 0.8366
        image_size = [int(i * ratio) for i in image_size]
        io_start = time.time()
        # depth_images_compressed = np.load(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_depth.npy"))
        # depth_images = np.unpackbits(depth_images_compressed)
        masks_compressed = pickle.load(open(os.path.join(INPUT_DIR, f"{frame_num}_masks.pkl"), 'rb'))
        data = json.load(open(os.path.join(INPUT_DIR, f"{frame_num}_data.json")))
        
        depth_images = np.array(pycocotools.mask.decode(masks_compressed))
        
        # depth_images = np.array(depth_images)
        # print(depth_images.shape)
        depth_images = depth_images.transpose([2, 1, 0])
        # print(depth_images.shape)
        
        """ Comment this out after the next setup.py run """
        # print(depth_images.shape)
        # depth_images = depth_images.reshape([len(data["labels"]), image_size[0], image_size[1], 4])
        # print(depth_images.shape)
        # depth_images = depth_images[:, :, :, 3]
        


        # pointsensor_token = sample['data'][pointsensor_channel]
        # pointsensor = nusc.get('sample_data', pointsensor_token)
        
        aggr_set = []
        # pointsensor_next = nusc.get('sample_data', pointsensor_token)

        pred_path = os.path.join(PRED_DIR, f"{frame_num:06}.txt")
        pseudo_path = os.path.join(PSEUDO_DIR, f"{frame_num:06}.txt")
        if os.path.exists(pred_path):
            os.remove(pred_path)
        # TODO: Need to get AGGREGATION WORKING
        # Loop for LiDAR pcd aggregation

        if os.path.exists(pseudo_path):
            os.remove(pseudo_path)
        
        open(pseudo_path, 'a').close()
        open(pred_path, 'a').close()

        for i in range(1):
            # pcl_path = os.path.join(nusc.dataroot, pointsensor_next['filename'])
            # pc = LidarPointCloud.from_file(pcl_path, DEVICE)
            
            """ FIXME: kitti """
            # velo = torch.from_numpy(kitti.get_lidar(frame_num).T).to(device=DEVICE, dtype=torch.float32)
            # print(velo[:, :10])
            # pc = LidarPointCloud(velo)

            # lidar_points = pc.points
            # mask = torch.ones(lidar_points.shape[1]).to(device=DEVICE)
            # mask = torch.logical_and(mask, torch.abs(lidar_points[0, :]) < np.sqrt(min_dist))
            # mask = torch.logical_and(mask, torch.abs(lidar_points[1, :]) < np.sqrt(min_dist))
            # lidar_points = lidar_points[:, ~mask]
            # # pc = LidarPointCloud(lidar_points)
            """ """

            # # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            # cs_record = nusc.get('calibrated_sensor', pointsensor_next['calibrated_sensor_token'])
            # pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
            # pc.translate(torch.from_numpy(np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))

            # # Second step: transform from ego to the global frame.
            # poserecord = nusc.get('ego_pose', pointsensor_next['ego_pose_token'])
            # pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
            # pc.translate(torch.from_numpy(np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))

            velo = kitti.get_lidar(frame_num)
            calib = kitti.get_calibration(frame_num, device=DEVICE)
            # print(velo[:, :10])
            # project to global frame

            velo = torch.from_numpy(velo).to(device=DEVICE, dtype=torch.float32)

            pc = calib.project_velo_to_ref(velo[:, :3])
            # print(pc[:, :10])
            # print(pc.shape)

            aggr_set.append(pc)
            # try:
            #     pointsensor_next = nusc.get('sample_data', pointsensor_next['next'])
            # except KeyError:
            #     break
        
        aggr_pc_points = torch.hstack(tuple([pcd for pcd in aggr_set]))
            
        # print("aggregate shape:", aggr_pc_points.shape)

        """ Commentable code for visualization"""
        # # Load all RGB images for the current frame
        # IMAGE_LIST = []
        # for c in range(len(CAM_LIST)):
        #     camera_token = sample['data'][CAM_LIST[c]]
        #     cam = nusc.get('sample_data', camera_token)

        #     im = Image.open(os.path.join(nusc.dataroot, cam['filename']))
        #     sz_init = im.size
        #     im.thumbnail([1024, 1024])
        #     sz = im.size
        #     ratio = sz[0]/sz_init[0]

        #     IMAGE_LIST.append(im)
        """ Commentable code ends """

        ratio = 0.8366


        # # Storing the camera intrinsic and extrinsic matrices for all cameras.
        # # To store: camera_intrinsic matrix, 
        # cam_data_list = []
        # for camera in CAM_LIST:
        #     camera_token = sample['data'][camera]

        #     # Here we just grab the front camera
        #     cam_data = nusc.get('sample_data', camera_token)
        #     poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
        #     cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        #     cam_data_dict = {
        #         "ego_pose": poserecord,
        #         "calibrated_sensor": cs_record,
        #     }

        #     cam_data_list.append(cam_data_dict)

        io_end = time.time()
        timer["io"] += io_end - io_start
        
        # num_masks_for_frames x h x w
        # Loop on each depth mask
        for i, (label, score) in enumerate(zip(data["labels"], data["detection_scores"])):
            # print(f"Creating box for mask {i} [{label}, {score}] from {scene_name} ({scene_num}) Frame {frame_num}, {CAM_LIST[c]}...")
            id_offset += 1

            pim_start = time.time()
            # cam_data = cam_data_list[c]

            # (path, nusc_boxes, camera_intrinsic) = nusc.get_sample_data(sample['data'][CAM_LIST[c]])

            maskarr_1 = depth_images[i]
            mask_px_count = np.count_nonzero(maskarr_1)
            # print(mask_px_count)

            """ Erosion code """
            # if mask_px_count < 100:
                # pass
            # elif mask_px_count < 500:
            # else:

            """ Commentable code for erosion """
            kernel = np.ones((3, 3), np.uint8)
            maskarr_1 = cv2.erode(maskarr_1, kernel)
            """ """

            # else:
            # # if mask_px_count > 400:
            #     kernel = torch.ones((5, 5), torch.uint8)
            #     maskarr_1 = cv2.erode(maskarr_1, kernel)
            # elif mask_px_count > 1000:
            #     kernel = torch.ones((7, 7), torch.uint8)
            #     maskarr_1 = cv2.erode(maskarr_1, kernel)
            # else:
            #     kernel = torch.ones((9, 9), torch.uint8)
            #     maskarr_1 = cv2.erode(maskarr_1, kernel)
            """ """
            mask_1 = Image.fromarray(maskarr_1)
            maskarr_1 = maskarr_1[:, :].astype(bool)
            maskarr_1 = torch.transpose(torch.from_numpy(maskarr_1).to(device=DEVICE, dtype=bool), 1, 0)

            # # Here we just grab the front camera
            # camera_token = sample['data'][CAM_LIST[c]]
            # cam = nusc.get('sample_data', camera_token)
        
            # # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            # cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
            # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            # pc.translate(np.array(cs_record['translation']))

            # # Second step: transform from ego to the global frame.
            # poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
            # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            # pc.translate(np.array(poserecord['translation']))

            # pc_points = np.copy(aggr_pc_points)
            # Array of indices to keep track of masked points

            # z_mask = np.ones(aggr_pc_points.shape[1], dtype=bool)
            # z_mask = np.logical_and(z_mask, aggr_pc_points[2] > floor_thresh)
            # # pc_points = pc_points[:, z_mask]
            # aggr_pc_points = aggr_pc_points[:, z_mask]
            # track_points = track_points[z_mask]
            # pc_points = pc_points.T
            # cluster_results = clusters_hdbscan(pc_points[:, :3])
            # labels_ = np.expand_dims(cluster_results, axis=-1)
            # cluster_labels = np.ones((pc_points.shape[0], 1)) * -1
            # mask = np.ones(cluster_labels.shape[0], dtype=bool)
            # cluster_labels[mask] = labels_
            # print("created", len(np.unique(cluster_results)), "cluster labels.")
            # pc_points = pc_points.T

            
            # array to track id of masked points
            track_points = np.array(range(aggr_pc_points.shape[0]))
            

            # pass in a copy of the aggregate pointcloud array
            # reset the lidar pointcloud
            # cam_pc = LidarPointCloud(torch.clone(aggr_pc_points))
            cam_pc = aggr_pc_points

            """ Visualize the pointcloud in a trimesh model """
            # # Reshape points
            # scene_points = np.transpose(cam_pc.points[:3, :])

            # # Create trimesh scene
            # trimesh_scene = trimesh.scene.Scene()
            # trimesh_scene.add_geometry(trimesh.PointCloud(scene_points))

            # trimesh_scene.set_camera([3.14159/2, 3.14159/2, 0], 20, trimesh_scene.centroid)
            # trimesh_scene.export(file_obj=f'lidar_{scene_name}_{frame_num}.glb')

            # # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            # # poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
            # poserecord = cam_data['ego_pose']
            # cam_pc.translate(torch.from_numpy(-np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))
            # cam_pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

            # # Fourth step: transform from ego into the camera.
            # # cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            # cs_record = cam_data['calibrated_sensor']
            # cam_pc.translate(torch.from_numpy(-np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))
            # cam_pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

            # ref to rect


            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths_ = calib.project_ref_to_velo(cam_pc)
            cam_pc_pts = calib.project_velo_to_rect(depths_)
            depths = cam_pc_pts[:, 2]
            # print(depths)


            coloring = depths

            # depths = torch.from_numpy(depths).to(device=DEVICE, dtype=torch.float32)

            # colors = {}
            # for cluster in cluster_results:
            #     if cluster not in colors:
            #         colors[cluster] = (random.randint(128, 255)/256, random.randint(128, 255)/256, random.randint(128, 255)/256, 255/256)

            # coloring = np.array([colors[cluster] for cluster in cluster_results])

            # camera_intrinsic = torch.from_numpy(np.array(cs_record["camera_intrinsic"])).to(device=DEVICE, dtype=torch.float32)
            # camera_intrinsic = camera_intrinsic*ratio
            # camera_intrinsic[2, 2] = 1

            camera_intrinsic = [
                [calib.f_u, 0, calib.c_u],
                [0, calib.f_v, calib.c_v],
                [0, 0, 1]
            ]
            camera_intrinsic = torch.Tensor(camera_intrinsic).to(device=DEVICE, dtype=torch.float32)
            camera_intrinsic = camera_intrinsic*ratio
            camera_intrinsic[2, 2] = 1

            cam_pc_pts = cam_pc_pts.T
            cam_pc_pts = cam_pc_pts[:3, :]
            # cam_pc_pts = torch.from_numpy(cam_pc_pts).to(device=DEVICE, dtype=torch.float32)
            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
            points, point_depths = view_points(cam_pc_pts, camera_intrinsic, normalize=True, device=DEVICE)

            # # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            # # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
            # # casing for non-keyframes which are slightly out of sync.
            # mask = np.ones(depths.shape[0], dtype=bool)
            # mask = np.logical_and(mask, depths > min_dist)
            # mask = np.logical_and(mask, points[0, :] > 0)
            # mask = np.logical_and(mask, points[0, :] < maskarr_1.shape[0] - 1)
            # mask = np.logical_and(mask, points[1, :] > 0)
            # mask = np.logical_and(mask, points[1, :] < maskarr_1.shape[1] - 1)
            # points = points[:, mask]
            # track_points = track_points[mask]
            # coloring = coloring[mask]
            # point_depths = point_depths[mask]

            # # Floor points for indexing on mask
            # floored_points = np.floor(points).astype(int)

            # mask1 = maskarr_1
            # mask2 = np.zeros_like(mask1)
            # mask2[floored_points[0, :], floored_points[1, :]] = 1
            # mask3 = mask1 & mask2
            # # indices of image where the mask is present
            # masked_pixel_indices = np.where(mask3)
            # # print("masked_pixel_indices shape:", (len(masked_pixel_indices[0]), len(masked_pixel_indices[1])))

            # # mask1 = np.logical_and(np.floor(points[1, :]).astype('int') == masked_pixel_indices[1], np.floor(points[0, :]).astype('int') == masked_pixel_indices[0])
            # # print(np.where(mask1))
            # # print(np.where(mask2))
            # # mask = mask1 & mask2
            # # print(mask.shape)
            # # print(np.where(mask1))
            # print(len(track_points), floored_points.shape)

            # mask = np.zeros(points.shape[1], dtype=bool)
            # for i in range(len(masked_pixel_indices[0])):
            #     point_mask = np.zeros(points.shape[1], dtype=bool)
            #     point_mask2 = np.logical_or(point_mask, np.floor(points[1, :]) == masked_pixel_indices[1][i])
            #     point_mask3 = np.logical_or(point_mask, np.floor(points[0, :]) == masked_pixel_indices[0][i])
            #     point_mask4 = np.logical_and(point_mask2, point_mask3)
            #     mask = np.logical_or(mask, point_mask4)
            
            # track_points = track_points[mask]
            # masked_floored_points = floored_points[:, mask]
            # masked_points = points[:, mask]
            # point_depths = point_depths[mask]

            image_mask = maskarr_1 # (W, H)
            # Create a boolean mask where True corresponds to masked pixels
            masked_pixels = (image_mask == 1) # (W, H)
            # print(maskarr_1.shape, points.shape, depths.shape) # points (3, N)
            # exit()
            # print(depths.shape, points.shape, masked_pixels.shape)
            # Use np.logical_and to find points within masked pixels
            points_within_image = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(
                depths > min_dist,                      # depths (N)
                points[0] > 0),                          # points (3, N) -> points[0, :] (1, N)
                points[0] < image_mask.shape[0] - 1),    # ^
                points[1] > 0),                          # ^
                points[1] < image_mask.shape[1] - 1     # ^
            )

            # print(points_within_image.shape)

            # floored_points_within_image = np.floor(points_within_image).astype(int)
            floored_points = torch.floor(points[:, points_within_image]).to(dtype=int) # (N_masked,)
            track_points = track_points[points_within_image.cpu()]
            # print(floored_points.shape, "fp")
            # print(masked_pixels[floored_points[0], floored_points[1]].shape, "mp")
            # print()

            # print(floored_points.shape)

            points_within_mask = torch.logical_and(
                floored_points,
                masked_pixels[floored_points[0], floored_points[1]]
            )

            indices_within_mask = torch.where(torch.logical_and(torch.logical_and(points_within_mask[0, :], points_within_mask[1, :]), points_within_mask[2, :]))[0]
            masked_points_pixels = torch.where(points_within_mask)
            # print(masked_points_pixels)


            # Get the indices of points within masked pixels
            # indices_within_mask = np.where(points_within_mask)[0]
            # indices_within_mask = np.where(points_within_mask)[0]
            # print(len(indices_within_mask), "ind")

            # Now, indices_within_mask contains the indices of points within the masked pixels
            # print(indices_within_mask.shape)
            track_points = track_points[indices_within_mask.cpu()]

            # print(aggr_pc_points.shape)
            # print(track_points)
            # print(aggr_pc_points.shape, track_points.shape)
            # print(aggr_pc_points.shape, track_points.shape)
            global_masked_points = aggr_pc_points[track_points, :]


            # print(global_masked_points)
            # exit()
            # print(global_masked_points)
            pim_end = time.time()
            timer["points in mask"] += pim_end - pim_start
            # print(global_masked_points.shape)
            # print(global_masked_points.shape)
            # if len(global_masked_points) == 0 or global_masked_points.shape[0] == 0 or global_masked_points.size == 0:
            if global_masked_points.numel() == 0:
                # print(">>>>>>>>>>>>>No LiDAR points found within mask.<<<<<<<<<<<<")
                # image_pil = Image.open(path)
                # size_init = image_pil.size

                # image_pil.thumbnail((1024, 1024))
                # size = image_pil.size # w, h
                # ratio = np.array(size)/np.array(size_init)
                # ratio = ratio[0]

                # # Load image
                # image_pil_rgb = image_pil.convert("RGB")
                
                # print("ax im: ", IMAGE_LIST[c].size)
                # plt.figure(figsize=(16, 9))
                # plt.imshow(IMAGE_LIST[c])
                # plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
                # plt.axis('off')
                # plt.savefig(os.path.join(OUTPUT_DIR, f"lidar_2_{frame_num}_{c}"), bbox_inches='tight', pad_inches=0, dpi=200)
                
                # lidar_img = Image.open(os.path.join(OUTPUT_DIR, f"lidar_2_{frame_num}_{c}.png"))
                # lidar_img.thumbnail([1024, 1024])

                # mask_image_1 = Image.new('RGBA', mask_1.size, color=(0, 0, 0, 0))
                # mask_draw_1 = ImageDraw.Draw(mask_image_1)
                # draw_mask(maskarr_1.T, mask_draw_1, random_color=True)

                # lidar_img.alpha_composite(mask_image_1)

                # lidar_img.save(os.path.join(OUTPUT_DIR, f"lidar_2_masked_{frame_num}_{c}.png"))
                # # Run ZoeDepth

                # time.sleep(5)
                continue
            

            

            id_offset_list1.append(id_offset)
            """ Multimodal virtual points """
            mvp_start = time.time()
            """Commentable Code"""
            # # print("global_masked_points:", global_masked_points.shape)

            # # print("points", track_points.shape)

            # # print("cam_pc.points")
            # # virtual_depths, nearest_indices = add_virtual_points(masked_pixel_indices, masked_points, point_depths)
            
            # virtual_points = add_virtual_points(np.where(maskarr_1), masked_points, point_depths)

            # cam_virtual_points = backproject_points(virtual_points[:2, :], np.copy(virtual_points[2, :]), camera_intrinsic, normalized=True)

            # cam_virtual_points = np.concatenate([cam_virtual_points,
            #                                      np.ones(cam_virtual_points.shape[1])[None]])
            # # print(cam_virtual_points)
            # cam_v_pc = LidarPointCloud(np.copy(cam_virtual_points))

            # cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            # cam_v_pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            # cam_v_pc.translate(np.array(cs_record['translation']))

            # poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
            # cam_v_pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            # cam_v_pc.translate(np.array(poserecord['translation']))
            
            # global_masked_virtual_points = cam_v_pc.points
            
            # # print(global_masked_virtual_points.shape, end=" ")

            # # print(global_masked_points.shape)

            # global_masked_points = np.hstack([global_masked_points, global_masked_virtual_points])
            # # print(global_masked_points.shape)
            """Commentable Code"""
            mvp_end = time.time()
            timer["mvp"] += mvp_end - mvp_start

            """ Centroid using mean """
            # global_centroid = np.array([np.mean(global_masked_points[0, :]),
            #                         np.mean(global_masked_points[1, :]),
            #                         np.mean(global_masked_points[2, :]),
            #                         np.mean(global_masked_points[3, :])])

            # mask_pc = LidarPointCloud(global_centroid[None].T)

            """ Centroid using medoid """
            medoid_start = time.time()
            # print("points for medoid calculation:", global_masked_points.shape)
            # TODO: change this to float64?
            
            if len(global_masked_points.shape) == 1:
                global_masked_points = torch.unsqueeze(global_masked_points, 0)
            
            global_masked_points = global_masked_points.T

            # print(global_masked_points.shape)
            pts3d = global_masked_points[:3, :].cpu().numpy().T
            # print(pts3d.shape)
            if pts3d.shape[0] <= 3:
                continue
            try:
                bbox = get_depth_bbox(pts3d)
            except:
                bbox = [pts3d[0], np.array([1, 1, 1]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]

            

            """ MEDOID COMPUTATION """
            global_masked_points = torch.Tensor(global_masked_points).to(DEVICE, dtype=torch.float32)
            global_centroid = get_medoid(global_masked_points[:3, :].to(dtype=torch.float32, device=DEVICE))
            """ """

            

            # # Create trimesh scene
            # scene_points = np.hstack([aggr_pc_points, global_masked_virtual_points])
            # print(aggr_pc_points.shape, global_masked_virtual_points.shape)
            # # scene_points = aggr_pc_points
            # trimesh_scene = trimesh.scene.Scene()
            # trimesh_scene.add_geometry(trimesh.PointCloud(scene_points.transpose()[:, :3]))

            # trimesh_scene.set_camera([3.14159/2, 3.14159/2, 0], 20, trimesh_scene.centroid)
            # trimesh_scene.export(file_obj=f'lidar_{scene_name}_{frame_num}.glb')

            # time.sleep(10)

            """ Commentable code """
            # mask_pc = LidarPointCloud(global_masked_points[:, global_centroid][None].T)
            # cam_to_ego = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            # mask_pc.rotate(Quaternion(cam_to_ego['rotation']).rotation_matrix)
            # mask_pc.translate(np.array(cam_to_ego['translation']))
            
            # ego_to_global = nusc.get('ego_pose', cam['ego_pose_token'])
            # mask_pc.rotate(Quaternion(ego_to_global['rotation']).rotation_matrix)
            # mask_pc.translate(np.array(ego_to_global['translation']))
            """ Commentable code ends """
            # centroid = view_points(mask_pc.points[:3, :], camera_intrinsic, normalize=True)
            # centroid = mask_pc.points

            centroid = global_masked_points[:, global_centroid][None].T[:3]
            # print("CENTORID:", centroid)

            detection_name = get_detection_name(label)
            yaw = R.from_matrix(bbox[2]).as_euler('zyx')[0]
            wlh = bbox[1]
            # center = bbox[0]
            center = centroid.cpu().numpy().squeeze().tolist()
            print(center, wlh, yaw); exit()

            wlh = get_shape_prior(shape_priors, label)
            wlh = [wlh[2], wlh[0], wlh[1]]

            center = [center[0], center[1] + wlh[0]/2, center[2]]

            save_pred(pred_path, detection_name, [0, 0, 0, 0], wlh, center, yaw, score)
            save_pred(pseudo_path, detection_name, [0, 0, 0, 0], wlh, center, yaw, None)

            all_centroids_list.append(torch.Tensor(centroid).to(DEVICE, dtype=torch.float32))
            centroid_ids.append(id_offset)
            medoid_end = time.time()
            timer["medoid"] += medoid_end - medoid_start
            final_id_offset = id_offset
            

            """ Commentable code for visualization """
            # masked_points_color = np.array([[0, 1, 0, 1]]*masked_points_pixels[0].shape[0])
            # print("ax im: ", IMAGE_LIST[c].size)
            # plt.figure(figsize=(16, 9))
            # plt.imshow(IMAGE_LIST[c])
            # # plt.scatter(masked_points_pixels[0][:], masked_points_pixels[1][:], c=masked_points_color, s=4)
            # # plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
            # # plt.scatter(virtual_points[0, :], virtual_points[1, :], c=[[1, 0, 0, 1]], s=2)
            # # plt.scatter(centroid[0], centroid[1], c=[[1, 0, 0, 1]], s=6)
            # plt.axis('off')
            # plt.savefig(os.path.join(OUTPUT_DIR, f"lidar_2tnew_{frame_num}_{c}"), bbox_inches='tight', pad_inches=0, dpi=200)

            # lidar_img = Image.open(os.path.join(OUTPUT_DIR, f"lidar_2tnew_{frame_num}_{c}.png"))
            # lidar_img.thumbnail([1024, 1024])

            # mask_image_1 = Image.new('RGBA', mask_1.size, color=(0, 0, 0, 0))
            # mask_draw_1 = ImageDraw.Draw(mask_image_1)
            # draw_mask(maskarr_1.T, mask_draw_1, random_color=True)

            # lidar_img.alpha_composite(mask_image_1)

            # lidar_img.save(os.path.join(OUTPUT_DIR, f"lidar_2tnew_masked_{frame_num}_{c}.png"))
            """ Commentable code ends """

        # if sample['next'] != "":
        #     sample = nusc.get('sample', sample['next'])
    """ End of object centroids loop """

            
    print(len(all_centroids_list))
    all_centroids_list = torch.stack(all_centroids_list)
    all_centroids_list = torch.squeeze(all_centroids_list)
    print(all_centroids_list.shape)
    
    # np.save('centroids.npy', np.array(all_centroids_list.to(device='cpu')))

    # yaw_list, min_distance_list, lane_pt_coords_list = lane_yaws_distances_and_coords(
    #     all_centroids_list, lane_pt_list
    # )

    # print(all_centroids_list)

    # check if vehicles are far from any lanes, remove these
    # ? change the centroid x and y coordinates to lane coordinates
    # change the rotation matrix/pyquaternion based on lane yaw

    # time.sleep(1000)
    print(centroid_ids)
    # scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    # scene = nusc.get('scene', scene_token)
    # sample = nusc.get('sample', scene['first_sample_token'])
    

    # Get map
    # nusc_map = get_nusc_map(nusc, scene)

    # drivable_records = nusc_map.drivable_area

    # drivable_polygons = []
    # for record in drivable_records:
    #     # print(record)
    #     polygons = [nusc_map.extract_polygon(token) for token in record["polygon_tokens"]]
    #     drivable_polygons.extend(polygons)
    
    # lane_records = nusc_map.lane + nusc_map.lane_connector
    # print("num_lanes", len(lane_records))
    # lane_polygons = []
    # for record in lane_records:
    #     # print(record)
    #     polygons = [nusc_map.extract_polygon(record["polygon_token"])]
    #     lane_polygons.extend(polygons)
    # print(lane_polygons)

    # lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)

    id_offset = -1
    id_offset_list2 = []

    for frame_num in range(num_frames):
        data = json.load(open(os.path.join(INPUT_DIR, f"{frame_num}_data.json")))
        predictions["results"][frame_num] = []
        for i, (label, score) in enumerate(zip(data["labels"], data["detection_scores"])):
            id_offset += 1
            if id_offset not in centroid_ids:
                continue
            else:
                # print("id_offset:", id_offset)
                id = centroid_ids.index(id_offset)
                final_id_offset2 = id_offset
            
            id_offset_list2.append(id_offset)
            detection_name = get_detection_name(label)
            centroid = np.squeeze(np.array(all_centroids_list[id, :].to(device='cpu')))
            # obj_rot_matrix_ego = im_to_ego_mats[c] @ corrective_mat @ im_mat
            # obj_rot_matrix = ego_to_global_mat @ obj_rot_matrix_ego
            # rot_quaternion = pQuaternion(matrix=obj_rot_matrix[:3, :3])
            cl_start = time.time()
            m_x, m_y, m_z = [float(i) for i in centroid]
            # print(m_x, m_y, m_z)
            # print(nusc_map.lookup_polygon_layers)
            """ Lane query """
            # radius=20
            # patch_for_lane = (m_x - radius, m_y - radius, m_x + radius, m_y + radius)
            # layer_names = ['lane', 'lane_connector']

            # print(patch_for_lane)
            # records_in_patch = dict()
            # for layer_name in layer_names:
            #     layer_records = []

            #     for record in lane_records:
            #         token = record['polygon_token']
            #         # xmin, ymin, xmax, ymax = patch_for_lane
            #         # rectangular_patch = box(xmin, ymin, xmax, ymax)
            #         rectangular_patch = box(*patch_for_lane)
            #         poly = nusc_map.extract_polygon(token)
            #         # print("polygon:", poly)
            #         if poly.intersects(rectangular_patch):
            #             layer_records.append(token)

            #     records_in_patch.update({layer_name: layer_records})

            # lanes = records_in_patch["lane"] + records_in_patch["lane_connector"]
            # print(records_in_patch)

            # discrete_points = nusc_map.discretize_lanes(lanes, 0.5)
            # print(discrete_points)
            
            #     # print("no closest lanes found.>?>?>?>?>?>?>?>?>?>?>?>?>")
            #     # continue

            # current_min = np.inf

            # min_id = ""
            # for lane_id, points in discrete_points.items():

            #     distance = np.linalg.norm(np.array(points)[:, :2] - [m_x, m_y], axis=1).min()
            #     if distance <= current_min:
            #         current_min = distance
            #         min_id = lane_id

            """ Old Lane Query """
            # closest_lane = nusc_map.get_closest_lane(m_x, m_y, radius=20)
            # cl_end = time.time()
            # timer["closest lane"] += cl_end - cl_start

            # if closest_lane == "":
            #     print("no closest lanes found.>?>?>?>?>?>?>?>?>?>?>?>?>")
            #     continue
            """ """

            # dist_from_lane = min_distance_list[id]
            # lane_yaw = yaw_list[id]
            # coord_x, coord_y = lane_pt_coords_list[id, :]

            """ Object lane thresh"""
            # if dist_from_lane > 20:
            #     print("no closest lanes found.>?>?>?>?>?>?>?>?>?>?>?>?>")
            #     continue
            """ """

            extents = get_shape_prior(shape_priors, detection_name)

            """ Commented out code """
            if detection_name in ["car", "truck", "bus", "construction_vehicle", "trailer", "barrier"]:
                # TODO: take polygons out of the loop
                # point = Point(m_x, m_y)
                # drivable_start = time.time()
                # is_drivable = False
                # for polygon in drivable_polygons:
                #     if point.within(polygon):
                #         is_drivable = True
                #         break
                # drivable_end = time.time()
                # timer["drivable"] += drivable_end - drivable_start
                
                """ Comment in for drivable filtering """
                # if not is_drivable and not detection_name in ["construction_vehicle", "trailer", "barrier"]:
                #     # print(f"[{detection_name}] vehicle not on road. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #     continue
                #     # ignore objects that are not in the drivable region
                """ """

                """ Old Lane Query """
                # # get lane yaw to align the bounding box
                # cl_start = time.time()
                # closest_lane = nusc_map.get_closest_lane(m_x, m_y, radius=4)
                # cl_end = time.time()
                # timer["closest lane"] += cl_end - cl_start
                # # if nusc_map.layers_on_point(m_x, m_y)["drivable_area"] == "":
                # #     print("vehicle not on road. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # #     continue
                # if closest_lane == "":
                #     print("vehicle far from any lanes ###############@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                #     continue
                #     # TODO: This could remove cars in parking lots
                
                # cl_start = time.time()
                # lane_record = nusc_map.get_arcline_path(closest_lane)
                # closest_pose_on_lane, distance_along_lane = arcline_path_utils.project_pose_to_lane((m_x, m_y, 0), lane_record)
                # global_x, global_y, lane_yaw = closest_pose_on_lane
                # cl_end = time.time()
                # timer["lane pose"] += cl_end - cl_start
                """ """

                """ Vehicle lane thresh """
                # if dist_from_lane > 4:
                #     print("vehicle far from any lanes ###############@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                #     continue
                #     # TODO: This could remove cars in parking lots
                """ """

                # lane_mat = np.eye(4)
                # lane_mat[0:2, 3] = [global_x, global_y]

                # global_to_ego_mat = np.linalg.inv(ego_to_global_mat)

                # # transform the lane matrix from global to ego frame
                # ego_lane_mat = global_to_ego_mat @ lane_mat

                # # transform the lane matrix to from ego to image frame
                # ego_to_im_mat = np.linalg.inv(im_to_ego_mats[cam_nums[m]])
                # im_lane_mat = ego_to_im_mat @ ego_lane_mat

                # # store image-frame x and y coordinates of the centroid

                # # center[0] = im_lane_mat[0, 3]
                # # center[1] = im_lane_mat[1, 3]
                # # im_mat[0, 3] = center[0]
                # # im_mat[1, 3] = center[1]

                # # calculate ego yaw, subtract from lane yaw (change yaw from global frame to ego frame)
                # angle = R.from_quat(pose_record["rotation"]).as_euler('xyz', degrees=True)
                # ego_yaw = -angle[1]
                # yaw = lane_yaw - ego_yaw

                # align_mat = np.eye(4)
                # align_mat[0:2, 0:2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]

                # # transform the align_mat from ego to image-frame
                # align_mat_im = ego_to_im_mat @ align_mat

                # # inv_T_mat = np.linalg.inv(im_to_ego_mats[cam_nums[m]])
                # # rot_inv_T_mat = np.eye(4)
                # # rot_inv_T_mat[:3, :3] = inv_T_mat[:3, :3]
                
                # # align_mat_rgb = rot_inv_T_mat @ align_mat

                # # im_mat = im_mat @ align_mat_rgb

                # # align_mat[0:1, 3] = 

                # im_mat = im_mat @ align_mat_im

                align_mat = np.eye(3)
                align_mat[0:2, 0:2] = [[np.cos(lane_yaw), -np.sin(lane_yaw)], [np.sin(lane_yaw), np.cos(lane_yaw)]]

                """ Pushback code """
                # Push centroid back for large objects
                # pointsensor_token = sample['data'][pointsensor_channel]
                # pointsensor = nusc.get('sample_data', pointsensor_token)
                # poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
                pushed_centroid = push_centroid(centroid, extents, Quaternion(matrix=align_mat), poserecord)
                # pushed_centroid = centroid
                """ """
            
            else:
                """ Commented out code ends """
                align_mat = np.eye(3)
                pushed_centroid = centroid

            # extents = [extents[2], extents[0], extents[1]]

            rot_quaternion = Quaternion(matrix=align_mat)

            box_dict = {
                    "sample_token": sample["token"],
                    "translation": [float(i) for i in pushed_centroid],
                    "size": list(extents),
                    "rotation": list(rot_quaternion),
                    "velocity": [0, 0],
                    "detection_name": detection_name,
                    "detection_score": score,
                    "attribute_name": ATTRIBUTE_NAMES[detection_name]
                }

            
            assert sample["token"] in predictions["results"]
                
            predictions["results"][sample["token"]].append(box_dict)
            # print(f"[{detection_name}] created prediction {id}")

        if sample['next'] != "":
            sample = nusc.get('sample', sample['next'])

        print()
    
    print(len(set(id_offset_list1).intersection(set(id_offset_list2))))
    print(len(set(id_offset_list1).union(set(id_offset_list2))))
    print(len(set(id_offset_list1) - set(id_offset_list2)))
    print(len(set(id_offset_list2) - set(id_offset_list1)))
    print(len(set(centroid_ids) - set(id_offset_list1)), len(set(id_offset_list1) - set(centroid_ids)))
    print(final_id_offset2, final_id_offset)
    print(len(set(centroid_ids).intersection(set(id_offset_list2))))
    print(len(set(centroid_ids).union(set(id_offset_list2))))
    print(len(set(centroid_ids) - set(id_offset_list2)))
    print(len(set(id_offset_list2) - set(centroid_ids)))

    print(len([box for sample_boxes in predictions["results"].values() for box in sample_boxes]))

    # time.sleep(1000)

    print("\nRunning NMS on the predictions.\n")
    nms_start = time.time()
    
    final_predictions = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": True,
            "use_external": False,
        },
        "results": {}
    }

    for sample in predictions["results"]:
        # class_to_num = {
        #     "bicycle" : 0,
        #     "car" : 1,
        #     "pedestrian" : 2,
        #     "truck" : 3,
        #     "bus" : 4,
        #     "construction_vehicle" : 5,

        # }

        final_predictions["results"][sample] = []
        # dets_by_label = {
        #     "barrier": [],
        #     "traffic_cone": [],
        #     "bicycle": [],
        #     "motorcycle": [],
        #     "pedestrian": [],
        #     "car": [],
        #     "bus": [],
        #     "construction_vehicle": [],
        #     "trailer": [],
        #     "truck": [],
        # }
        dets = []
        det_labels = []
        # TODO: threshs borrows from centerpoint
        threshs_by_label = {
            "barrier": 1,
            "traffic_cone": 0.175,
            "bicycle": 0.85,
            "motorcycle": 0.85,
            "pedestrian": 0.175,
            "car": 4,
            "bus": 10,
            "construction_vehicle": 12,
            "trailer": 10,
            "truck": 12,
        }

        # [4, 12, 10, 1, 0.85, 0.175],
        # tasks = [
        # dict(num_class=1, class_names=["car"]),
        # dict(num_class=2, class_names=["truck", "construction_vehicle"]),
        # dict(num_class=2, class_names=["bus", "trailer"]),
        # dict(num_class=1, class_names=["barrier"]),
        # dict(num_class=2, class_names=["motorcycle", "bicycle"]),
        # dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        # ]

        centroids_list = []
        extents_list = []
        rot_list = []
        det_names_list = []
        attr_names_list = []
        vertices_list = []
        scores = []

        for box_dict in predictions["results"][sample]:
            centroid = box_dict["translation"]
            extents = box_dict["size"]
            score = box_dict["detection_score"]
            rot = box_dict["rotation"]
            detection_name = box_dict["detection_name"]
            attr_name = box_dict["attribute_name"]
            rot_quaternion = Quaternion(rot)

            dets.append(np.array([centroid[0], centroid[1], score]))
            det_labels.append(detection_name)

            centroids_list.append(centroid)
            extents_list.append(extents)
            rot_list.append(rot)
            det_names_list.append(detection_name)
            attr_names_list.append(attr_name)
            scores.append(score)

        # keep_by_label = {
        #     "barrier": [],
        #     "traffic_cone": [],
        #     "bicycle": [],
        #     "motorcycle": [],
        #     "pedestrian": [],
        #     "car": [],
        #     "bus": [],
        #     "construction_vehicle": [],
        #     "trailer": [],
        #     "truck": [],
        # }
        # keep_indices = []

        # for _label in threshs_by_label.keys():
        dets = np.array(dets)
        print(len(det_labels), end=" ")
        if len(det_labels) > 0:
            keep_indices = list(circle_nms(dets, det_labels, threshs_by_label))
        else:
            # Skip this sample if we dont have any predictions in it
            continue

        print(len(keep_indices))

        """ Commentable code """
        centroids_list = [centroids_list[c] for c in range(len(centroids_list)) if c in keep_indices]
        extents_list = [extents_list[c] for c in range(len(extents_list)) if c in keep_indices]
        rot_list = [rot_list[c] for c in range(len(rot_list)) if c in keep_indices]
        det_names_list = [det_names_list[c] for c in range(len(det_names_list)) if c in keep_indices]
        attr_names_list = [attr_names_list[c] for c in range(len(attr_names_list)) if c in keep_indices]
        scores = [scores[c] for c in range(len(scores)) if c in keep_indices]
        """ Commentable code """

        for i, (centroid, extents, rot, det_name, attr_name, score) in enumerate(zip(
            centroids_list, extents_list, rot_list, det_names_list, attr_names_list, scores
        )):
            # print(i, end='')
            box_dict = {
                "sample_token": sample,
                "translation": centroid,
                "size": extents,
                "rotation": rot,
                "velocity": [0, 0],
                "detection_name": det_name,
                "detection_score": score,
                "attribute_name": attr_name,
            }

            final_predictions["results"][sample].append(box_dict)


        # boxes = []
        # scores = []

        # centroids_list = []
        # extents_list = []
        # rot_list = []
        # det_names_list = []
        # attr_names_list = []
        # vertices_list = []
        
        # for box_dict in predictions["results"][sample]:
        #     centroid = box_dict["translation"]
        #     extents = box_dict["size"]
        #     score = box_dict["detection_score"]
        #     rot = box_dict["rotation"]
        #     detection_name = box_dict["detection_name"]
        #     attr_name = box_dict["attribute_name"]
        #     rot_quaternion = Quaternion(rot)

        #     unit_z = np.array([0, 0, 1])
        #     rot_mat = rot_quaternion.rotation_matrix
        #     # print(list(rot_quaternion))
        #     angle = R.from_quat(list(rot_quaternion)).as_euler('xyz', degrees=False)
        #     # print(angle)
        #     theta = -angle[0]

        #     top_left_x, top_left_y = get_top_left_vertex(centroid, extents, theta)
        #     box_i = [[top_left_x, top_left_y], [extents[0], extents[1]], theta]

        #     # print(box, score)
        #     """ """
        #     boxes.append(box_i)
        #     scores.append(score)
        #     """ """

        #     # score_tensor = torch.Tensor([0, 0, 0, 0, 0, 0])
        #     # score_tensor[class_to_num(detection_name)] = score
        #     # scores.append(score_tensor)
        #     # boxes.append(torch.Tensor(box_i))



        #     vertex_dict = {
        #                 "sample_token": sample,
        #                 "translation": [top_left_x, top_left_y, centroid[2]],
        #                 "size": get_shape_prior(shape_priors, "traffic_cone"),
        #                 "rotation": list(rot_quaternion),
        #                 "velocity": [0, 0],
        #                 "detection_name": "traffic_cone",
        #                 "detection_score": score,
        #                 "attribute_name": ""
        #             }

        #     vertices_list.append(vertex_dict)

        #     centroids_list.append(centroid)
        #     extents_list.append(extents)
        #     rot_list.append(rot)
        #     det_names_list.append(detection_name)
        #     attr_names_list.append(attr_name)
        
        # # print(torch.vstack(boxes), torch.Tensor(scores))
        
        # # indices = torchvision.ops.nms(
        # #     torch.vstack(boxes),
        # #     torch.Tensor(scores),
        # #     iou_threshold=0.02
        # # )

        # # predictions["results"][sample].extend(vertices_list)

        # # indices = nms(boxes, scores, nms_threshold=0.3)
        # indices = rboxes(boxes, scores, nms_algorithm=malisiewicz.nms, nms_threshold=0.3, score_threshold=0.2)

        # # print(len(centroids_list), len(extents_list), len(rot_list), len(det_names_list), len(attr_names_list), len(scores))
        # # print(len(predictions["results"][sample]), end=' ')
        
        # """ Commentable code """
        # # centroids_list = [centroids_list[c] for c in range(len(centroids_list)) if c in indices]
        # # extents_list = [extents_list[c] for c in range(len(extents_list)) if c in indices]
        # # rot_list = [rot_list[c] for c in range(len(rot_list)) if c in indices]
        # # det_names_list = [det_names_list[c] for c in range(len(det_names_list)) if c in indices]
        # # attr_names_list = [attr_names_list[c] for c in range(len(attr_names_list)) if c in indices]
        # # scores = [scores[c] for c in range(len(scores)) if c in indices]
        # """ Commentable code """

        # indices = sorted(indices, reverse=True)
        # # print(indices, end=' ')
        # # for i in indices:
        # #     del predictions["results"][sample][i]

        # # TODO: try deleting the elements directly from the list in the dictionary.
        # # print(len(predictions["results"][sample]))
        # print(len(centroids_list), len(extents_list), len(rot_list), len(det_names_list), len(attr_names_list), len(scores))
        
        # for i, (centroid, extents, rot, det_name, attr_name, score) in enumerate(zip(
        #     centroids_list, extents_list, rot_list, det_names_list, attr_names_list, scores
        # )):
        #     print(i, end='')
        #     box_dict = {
        #         "sample_token": sample,
        #         "translation": centroid,
        #         "size": extents,
        #         "rotation": rot,
        #         "velocity": [0, 0],
        #         "detection_name": det_name,
        #         "detection_score": score,
        #         "attribute_name": attr_name,
        #     }

        #     final_predictions["results"][sample].append(box_dict)
        
        # """ Commentable code """
        # # for vertex in vertices_list:
        # #     final_predictions["results"][sample].append(vertex)
        # """ Commentable code """

    nms_end = time.time()
    timer["nms"] += nms_end - nms_start

    # with open("predictions_lidar_train.json", "w") as f:
    # with open("test_predictions_lidar_569_700.json", "w") as f:
    # with open(os.path.join(OUTPUT_DIR, "test_rle.json"), "w") as f:
    with open(os.path.join(OUTPUT_DIR, "visualize_cpb.json"), "w") as f:
    # with open("predictions_lidar_traindetect.json", "w") as f:
    # with open("predictions_lidar_traindetect25_rare.json", "w") as f:
        json.dump(final_predictions, f)

    print(f"wrote {len(final_predictions['results'])} samples.")
    #     json.dump(predictions, f)
    # print(f"wrote {len(predictions['results'])} samples.")

    total_end = time.time()
    timer["total"] += total_end - total_start

    for operation in timer:
        print(operation, ":\t\t", timer[operation])