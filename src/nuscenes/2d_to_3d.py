import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import copy
import multiprocessing as mp
import random
import argparse
import json
import pickle
import pycocotools
import hdbscan
from inspect import getmembers, isfunction
from pyquaternion import Quaternion as pQuaternion

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image, ImageDraw, ImageFont
from os.path import join

import sys
from shapely.geometry import Point, box

from functools import partial

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Quaternion as nQuaternion
from utils.pcd import LidarPointCloud, view_points
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.utils.splits import mini_val, mini_train, train_detect, train, val


from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import scipy
import time
import tqdm

# from mmdetection3d.mmdet3d.models.layers import box3d_multiclass_nms


VER_NAME = "v1.0-trainval"
INPUT_PATH = "../../data/nuScenes/"

OUTPUT_DIR = "../../outputs/nuscenes/"
INPUT_DIR = "../../mask_outputs/nuscenes-detic/"


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


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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



def get_medoid(points):
    dist_matrix = torch.cdist(points.T, points.T, p=2)

    return torch.argmin(dist_matrix.sum(axis=0))


def get_detection_name(name):
    if name not in ["trafficcone", "constructionvehicle", "human"]:
        detection_name = name
    elif name == "trafficcone":
        detection_name = "traffic_cone"
    elif name == "constructionvehicle":
        detection_name = "construction_vehicle"
    elif name == "human":
        detection_name = "pedestrian"
    
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


def push_centroid(centroid, extents, rot_quaternion, poserecord):
    centroid = np.squeeze(centroid)
    av_centroid = poserecord["translation"]
    ego_centroid = centroid - av_centroid

    l = extents[0]
    w = extents[1]

    
    angle = R.from_quat(list(rot_quaternion)).as_euler('xyz', degrees=False)

    theta = -angle[0]

    if np.isnan(theta):
        theta = 0.5 * np.pi
    
    alpha = np.arctan(np.abs(ego_centroid[1]) / np.abs(ego_centroid[0]))

    if ego_centroid[0] < 0:
        if ego_centroid[1] < 0:
            alpha = -np.pi + alpha
        else:
            alpha = np.pi - alpha
    else:
        if ego_centroid[1] < 0:
            alpha = -alpha

    offset = np.min( [np.abs(w / (2*np.sin(theta - alpha))), np.abs(l / (2*np.cos(theta - alpha)))] )

    x_dash = centroid[0] + offset * np.cos(alpha)
    y_dash = centroid[1] + offset * np.sin(alpha)

    pushed_centroid = np.array([x_dash, y_dash, centroid[2]])

    return pushed_centroid


def get_top_left_vertex(centroid, extents, theta):

    if (theta > 0):
        x1 = centroid[0] - np.cos(theta)*extents[1]/2 - np.sin(theta)*extents[0]/2
        y1 = centroid[1] - np.sin(theta)*extents[1]/2 + np.cos(theta)*extents[0]/2
    
    else:
        x1 = centroid[0] + np.cos(theta)*extents[1]/2 + np.sin(theta)*extents[0]/2
        y1 = centroid[1] + np.sin(theta)*extents[1]/2 - np.cos(theta)*extents[0]/2
    
    return x1, y1




def get_nusc_map(nusc, scene):
    # Get scene location
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    # Get nusc map
    nusc_map = NuScenesMap(dataroot=INPUT_PATH, map_name=location)

    return nusc_map



def get_all_lane_points_in_scene(nusc_map):
    lane_records = nusc_map.lane + nusc_map.lane_connector

    lane_tokens = [lane["token"] for lane in lane_records]

    lane_pt_dict = nusc_map.discretize_lanes(lane_tokens, 0.5)

    all_lane_pts = []
    for lane_pts in lane_pt_dict.values():
        for lane_pt in lane_pts:
            all_lane_pts.append(lane_pt)
    
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

    return D_squared

def lane_yaws_distances_and_coords(all_centroids, all_lane_pts):
    all_lane_pts = torch.Tensor(all_lane_pts).to(device='cpu')
    all_centroids = torch.Tensor(all_centroids).to(device='cpu')

    start = time.time()

    DistMat = scipy.spatial.distance.cdist(all_centroids[:, :2], all_lane_pts[:, :2])
    
    min_lane_indices = np.argmin(DistMat, axis=1)

    distances = np.min(DistMat, axis=1)

    all_lane_pts = np.array(all_lane_pts)
    min_lanes = np.array([all_lane_pts[min_lane_indices[0]]])
    for idx in min_lane_indices:
        min_lanes = np.vstack([min_lanes, all_lane_pts[idx, :]])
    

    yaws = min_lanes[1:, 2]
    coords = min_lanes[1:, :2]

    end = time.time()

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










if __name__ == "__main__":
    total_start = time.time()
    camera_channel="CAM_BACK"
    pointsensor_channel="LIDAR_TOP"
    dot_size=0.5
    min_dist= 2.3
    render_intensity= False
    show_lidarseg = False
    filter_lidarseg_labels = None
    lidarseg_preds_bin_path = None
    show_panoptic = False
    ax = None
    floor_thresh = 0.6

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

    c = 3

    nusc = NuScenes(VER_NAME, INPUT_PATH, True)

    # shape_priors = json.load(open("shape_priors.json"))
    shape_priors = json.load(open("cfg/shape_priors_chatgpt.json"))

    progress_bar_main = tqdm.tqdm(enumerate(mini_val))
    for scene_num, scene_name in progress_bar_main:
        progress_bar_main.set_description(f"Scene Progress:")

        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
        scene = nusc.get('scene', scene_token)
        sample = nusc.get('sample', scene['first_sample_token'])

        # Get map
        nusc_map = get_nusc_map(nusc, scene)

        drivable_records = nusc_map.drivable_area

        drivable_polygons = []
        for record in drivable_records:
            # print(record)
            polygons = [nusc_map.extract_polygon(token) for token in record["polygon_tokens"]]
            drivable_polygons.extend(polygons)

        lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)

        all_centroids_list = []
        centroid_ids = []
        id_offset = -1
        id_offset_list1 = []

        num_frames = count_frames(nusc, sample)
        progress_bar = tqdm.tqdm(range(num_frames))
        for frame_num in progress_bar:
            progress_bar.set_description(f"Processing {scene_name} ({scene_num})")

            image_size = [900, 1600]
            ratio = 0.64
            image_size = [int(i * ratio) for i in image_size]
            io_start = time.time()
            masks_compressed = pickle.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_masks.pkl"), 'rb'))
            data = json.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_data.json")))
            
            depth_images = np.array(pycocotools.mask.decode(masks_compressed))
            

            depth_images = depth_images.transpose([2, 1, 0]) # num_masks_for_frames x h x w

            pointsensor_token = sample['data'][pointsensor_channel]
            pointsensor = nusc.get('sample_data', pointsensor_token)
            
            aggr_set = []
            pointsensor_next = nusc.get('sample_data', pointsensor_token)

            # Loop for LiDAR pcd aggregation
            for i in range(3):
                pcl_path = os.path.join(nusc.dataroot, pointsensor_next['filename'])
                pc = LidarPointCloud.from_file(pcl_path, DEVICE)

                lidar_points = pc.points
                mask = torch.ones(lidar_points.shape[1]).to(device=DEVICE)
                mask = torch.logical_and(mask, torch.abs(lidar_points[0, :]) < np.sqrt(min_dist))
                mask = torch.logical_and(mask, torch.abs(lidar_points[1, :]) < np.sqrt(min_dist))
                lidar_points = lidar_points[:, ~mask]
                pc = LidarPointCloud(lidar_points)

                # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
                # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                cs_record = nusc.get('calibrated_sensor', pointsensor_next['calibrated_sensor_token'])
                pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))

                # Second step: transform from ego to the global frame.
                poserecord = nusc.get('ego_pose', pointsensor_next['ego_pose_token'])
                pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))

                aggr_set.append(pc.points)
                try:
                    pointsensor_next = nusc.get('sample_data', pointsensor_next['next'])
                except KeyError:
                    break
            
            aggr_pc_points = torch.hstack(tuple([pcd for pcd in aggr_set]))


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

            ratio = 0.64


            # Storing the camera intrinsic and extrinsic matrices for all cameras.
            # To store: camera_intrinsic matrix, 
            cam_data_list = []
            for camera in CAM_LIST:
                camera_token = sample['data'][camera]

                # Here we just grab the front camera
                cam_data = nusc.get('sample_data', camera_token)
                poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
                cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

                cam_data_dict = {
                    "ego_pose": poserecord,
                    "calibrated_sensor": cs_record,
                }

                cam_data_list.append(cam_data_dict)

            io_end = time.time()
            timer["io"] += io_end - io_start
            
            
            # Loop on each depth mask
            for i, (label, score, c) in enumerate(zip(data["labels"], data["detection_scores"], data["cam_nums"])):
                id_offset += 1

                pim_start = time.time()
                cam_data = cam_data_list[c]

                maskarr_1 = depth_images[i]
                mask_px_count = np.count_nonzero(maskarr_1)

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
                """ Erosion code ends """

                mask_1 = Image.fromarray(maskarr_1)
                maskarr_1 = maskarr_1[:, :].astype(bool)
                maskarr_1 = torch.transpose(torch.from_numpy(maskarr_1).to(device=DEVICE, dtype=bool), 1, 0)

                
                # array to track id of masked points
                track_points = np.array(range(aggr_pc_points.shape[1]))
                

                # pass in a copy of the aggregate pointcloud array
                # reset the lidar pointcloud
                cam_pc = LidarPointCloud(torch.clone(aggr_pc_points))

                """ Visualize the pointcloud in a trimesh model """
                # # Reshape points
                # scene_points = np.transpose(cam_pc.points[:3, :])

                # # Create trimesh scene
                # trimesh_scene = trimesh.scene.Scene()
                # trimesh_scene.add_geometry(trimesh.PointCloud(scene_points))

                # trimesh_scene.set_camera([3.14159/2, 3.14159/2, 0], 20, trimesh_scene.centroid)
                # trimesh_scene.export(file_obj=f'lidar_{scene_name}_{frame_num}.glb')
                """ """

                # transform from global into the ego vehicle frame for the timestamp of the image.
                # poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
                poserecord = cam_data['ego_pose']
                cam_pc.translate(torch.from_numpy(-np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))
                cam_pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

                # transform from ego into the camera.
                # cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                cs_record = cam_data['calibrated_sensor']
                cam_pc.translate(torch.from_numpy(-np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))
                cam_pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

                # actually take a "picture" of the point cloud.
                # Grab the depths (camera frame z axis points away from the camera).
                depths = cam_pc.points[2, :]

                coloring = depths

                camera_intrinsic = torch.from_numpy(np.array(cs_record["camera_intrinsic"])).to(device=DEVICE, dtype=torch.float32)
                camera_intrinsic = camera_intrinsic*ratio
                camera_intrinsic[2, 2] = 1

                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points, point_depths = view_points(cam_pc.points[:3, :], camera_intrinsic, normalize=True, device=DEVICE)

                image_mask = maskarr_1 # (W, H)
                # Create a boolean mask where True corresponds to masked pixels
                masked_pixels = (image_mask == 1) # (W, H)

                # Use np.logical_and to find points within masked pixels
                points_within_image = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(
                    depths > min_dist,                      # depths (N)
                    points[0] > 0),                          # points (3, N) -> points[0, :] (1, N)
                    points[0] < image_mask.shape[0] - 1),    # ^
                    points[1] > 0),                          # ^
                    points[1] < image_mask.shape[1] - 1     # ^
                )

                floored_points = torch.floor(points[:, points_within_image]).to(dtype=int) # (N_masked,)
                track_points = track_points[points_within_image.cpu()]

                points_within_mask = torch.logical_and(
                    floored_points,
                    masked_pixels[floored_points[0], floored_points[1]]
                )

                indices_within_mask = torch.where(torch.logical_and(torch.logical_and(points_within_mask[0, :], points_within_mask[1, :]), points_within_mask[2, :]))[0]
                masked_points_pixels = torch.where(points_within_mask)

                # Now, indices_within_mask contains the indices of points within the masked pixels
                track_points = track_points[indices_within_mask.cpu()]

                
                global_masked_points = aggr_pc_points[:, track_points]

                pim_end = time.time()
                timer["points in mask"] += pim_end - pim_start

                
                if global_masked_points.numel() == 0:
                    # No lidar points in the mask
                    continue

                id_offset_list1.append(id_offset)

                """ Centroid using mean """
                # global_centroid = np.array([np.mean(global_masked_points[0, :]),
                #                         np.mean(global_masked_points[1, :]),
                #                         np.mean(global_masked_points[2, :]),
                #                         np.mean(global_masked_points[3, :])])

                # mask_pc = LidarPointCloud(global_centroid[None].T)

                """ Centroid using medoid """
                medoid_start = time.time()
                
                if len(global_masked_points.shape) == 1:
                    global_masked_points = torch.unsqueeze(global_masked_points, 1)
                global_centroid = get_medoid(global_masked_points[:3, :].to(dtype=torch.float32, device=DEVICE))
                
                mask_pc = LidarPointCloud(global_masked_points[:, global_centroid][None].T)

                """ Commentable code for visualization """
                # cam_to_ego = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                # mask_pc.rotate(Quaternion(cam_to_ego['rotation']).rotation_matrix)
                # mask_pc.translate(np.array(cam_to_ego['translation']))
                
                # ego_to_global = nusc.get('ego_pose', cam['ego_pose_token'])
                # mask_pc.rotate(Quaternion(ego_to_global['rotation']).rotation_matrix)
                # mask_pc.translate(np.array(ego_to_global['translation']))
                """ Commentable code ends """

                # centroid = view_points(mask_pc.points[:3, :], camera_intrinsic, normalize=True)
                centroid = mask_pc.points[:3]

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

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])
        """ End of object centroids loop """

                
        print(len(all_centroids_list))
        all_centroids_list = torch.stack(all_centroids_list)
        all_centroids_list = torch.squeeze(all_centroids_list)
        print(all_centroids_list.shape)
        

        yaw_list, min_distance_list, lane_pt_coords_list = lane_yaws_distances_and_coords(
            all_centroids_list, lane_pt_list
        )

        # print(all_centroids_list)

        # check if vehicles are far from any lanes, remove these
        # ? change the centroid x and y coordinates to lane coordinates
        # change the rotation matrix/pyquaternion based on lane yaw

        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
        scene = nusc.get('scene', scene_token)
        sample = nusc.get('sample', scene['first_sample_token'])

        # Get map
        nusc_map = get_nusc_map(nusc, scene)

        drivable_records = nusc_map.drivable_area

        drivable_polygons = []
        for record in drivable_records:
            polygons = [nusc_map.extract_polygon(token) for token in record["polygon_tokens"]]
            drivable_polygons.extend(polygons)

        lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)

        id_offset = -1
        id_offset_list2 = []

        for frame_num in range(num_frames):
            data = json.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_data.json")))
            predictions["results"][sample["token"]] = []
            for i, (label, score, c) in enumerate(zip(data["labels"], data["detection_scores"], data["cam_nums"])):
                id_offset += 1
                if id_offset not in centroid_ids:
                    continue
                else:
                    id = centroid_ids.index(id_offset)
                    final_id_offset2 = id_offset
                
                id_offset_list2.append(id_offset)
                detection_name = get_detection_name(label)
                centroid = np.squeeze(np.array(all_centroids_list[id, :].to(device='cpu')))

                cl_start = time.time()
                m_x, m_y, m_z = [float(i) for i in centroid]

                dist_from_lane = min_distance_list[id]
                lane_yaw = yaw_list[id]

                """ Object lane thresh """
                # if dist_from_lane > 20:
                #     print("no closest lanes found.>?>?>?>?>?>?>?>?>?>?>?>?>")
                #     continue
                """ """

                extents = get_shape_prior(shape_priors, detection_name)


                if detection_name in ["car", "truck", "bus", "construction_vehicle", "trailer", "barrier"]:
                    point = Point(m_x, m_y)
                    drivable_start = time.time()
                    is_drivable = False
                    for polygon in drivable_polygons:
                        if point.within(polygon):
                            is_drivable = True
                            break
                    drivable_end = time.time()
                    timer["drivable"] += drivable_end - drivable_start
                    
                    """ Uncomment for drivable filtering """
                    # if not is_drivable and not detection_name in ["construction_vehicle", "trailer", "barrier"]:
                    #     # print(f"[{detection_name}] vehicle not on road. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    #     continue
                    #     # ignore objects that are not in the drivable region
                    """ """

                    """ Vehicle lane thresh """
                    # if dist_from_lane > 4:
                    #     print("vehicle far from any lanes ###############@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    #     continue
                    #     # TODO: This could remove cars in parking lots
                    """ """

                    align_mat = np.eye(3)
                    align_mat[0:2, 0:2] = [[np.cos(lane_yaw), -np.sin(lane_yaw)], [np.sin(lane_yaw), np.cos(lane_yaw)]]

                    """ Pushback code """
                    # Push centroid back for large objects
                    pointsensor_token = sample['data'][pointsensor_channel]
                    pointsensor = nusc.get('sample_data', pointsensor_token)
                    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
                    pushed_centroid = push_centroid(centroid, extents, Quaternion(matrix=align_mat), poserecord)
                    # pushed_centroid = centroid
                    """ """
                
                else:
                    """ Commented out code ends """
                    align_mat = np.eye(3)
                    pushed_centroid = centroid


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

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])

            print()
        

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
        final_predictions["results"][sample] = []

        dets = []
        det_labels = []
        # threshs borrowed from centerpoint
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

        dets = np.array(dets)
        print(len(det_labels), end=" ")
        if len(det_labels) > 0:
            keep_indices = list(circle_nms(dets, det_labels, threshs_by_label))
        else:
            # Skip this sample if we dont have any predictions in it
            continue

        print(len(keep_indices))

        """ Commentable code for NMS """
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

    nms_end = time.time()
    timer["nms"] += nms_end - nms_start

    with open(os.path.join(OUTPUT_DIR, "pseudolabels_minival.json"), "w") as f:
        json.dump(final_predictions, f)

    print(f"wrote {len(final_predictions['results'])} samples.")

    total_end = time.time()
    timer["total"] += total_end - total_start

    for operation in timer:
        print(operation, ":\t\t", timer[operation])