import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import open3d as o3d
import torch
import torchvision
import copy
import multiprocessing as mp
# import pointops
import random
import argparse
import json
import pickle
from pycocotools import mask as pymask
# import hdbscan
from inspect import getmembers, isfunction


from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image, ImageDraw, ImageFont
from os.path import join

import trimesh
import sys

from functools import partial
# import tempfile

# from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Quaternion as nQuaternion
from utils.pcd import LidarPointCloud, view_points

from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import scipy
import time


from cfg.prompt_cfg import TEXT_PROMPT, MAPS, BOX_THRESHOLDS, TEXT_THRESHOLDS, NUSC_TO_WAYMO


INPUT_PATH = "../../data/waymo-v1.4.2/waymo_format/training/"



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


OUTPUT_DIR = "../../outputs/waymo/"
INPUT_DIR = "../../mask_outputs/waymo-detic/"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"



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
    print(nonzero_coords.shape)


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
    dist_matrix = torch.cdist(points.T, points.T)
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
        if name == "vehicle":
            return shape_priors["car"]
        elif name == "pedestrian":
            return shape_priors["pedestrian"]
        elif name == "cyclist":
            return shape_priors["bicycle"]
        return shape_priors[name]


def push_centroid(centroid, extents, rot_quaternion, poserecord=None, ego_frame=False):
    centroid = np.squeeze(centroid)

    if not ego_frame:
        av_centroid = poserecord["translation"]
        ego_centroid = centroid - av_centroid
    else:
        ego_centroid = centroid

    l = extents[0]
    w = extents[1]

    
    angle = R.from_quat(list(rot_quaternion)).as_euler('xyz', degrees=False)
    theta = -angle[0]

    if np.isnan(theta):
        theta = 0.5 * np.pi
    
    alpha = np.arctan(np.abs(ego_centroid[1]) / np.abs(ego_centroid[0]))

    if ego_centroid[0] < 0:
        if ego_centroid[1] < 0:
            alpha = -np.pi + alpha # do something else
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


    return yaws, distances, coords


# Thanks to Tianwei Yin for the following function, available at https://github.com/tianweiy/CenterPoint/blob/master/det3d/core/utils/circle_nms_jit.py
# import numba

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


import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from tqdm import tqdm

# Name of all the RGB cameras
RGB_Name  = {
    0:"UNKNOWN",
    1:'FRONT',
    2:'FRONT_LEFT',
    3:'FRONT_RIGHT',
    4:'SIDE_LEFT',
    5:'SIDE_RIGHT'
  }

CAM_LIST = list(RGB_Name.values())
CAM_LIST.remove("UNKNOWN")

# Name of all LiDAR sensors
Lidar_Name = {
    0:"UNKNOWN",
    1:"TOP",
    2:"FRONT",
    3:"SIDE_LEFT",
    4:"SIDE_RIGHT",
    5:"REAR"
}

def get_yaws_from_lane_coords(lane_list):
    prev_x, prev_y = 0, 0
    lane_list_with_yaw = []
    for xyz in lane_list:
        x = xyz.x; y = xyz.y; z = xyz.z
        # print(x, y, z)
        yaw = np.arctan2(y-prev_y, x-prev_x)
        prev_x, prev_y = x, y
        # print(yaw)
        lane_list_with_yaw.append([x, y, yaw])
    
    if len(lane_list_with_yaw) > 1:
        lane_list_with_yaw[0][2] = lane_list_with_yaw[1][2]
    
    return np.array(lane_list_with_yaw)
    
        



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
    floor_thresh = -0.6

    objects = metrics_pb2.Objects()

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


    scene_list = os.listdir(INPUT_PATH)
    scene_list.sort()
    print(len(scene_list))

    # shape_priors = json.load(open("shape_priors.json"))
    shape_priors = json.load(open("cfg/shape_priors_chatgpt.json"))

    progress_bar_main = tqdm(enumerate(scene_list[680:710]))

    for scene_num, scene_name in progress_bar_main:
        progress_bar_main.set_description(f"Scene Progress:")

        scene_dataset = tf.data.TFRecordDataset(INPUT_PATH+scene_name, compression_type='')

        all_centroids_list = []
        centroid_ids = []
        id_offset = -1
        id_offset_list1 = []
        num_frames = 0
 
        for frame_num, frame_data in tqdm(enumerate(scene_dataset), desc=scene_name+f" ({scene_num})"+": "):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))

            io_start = time.time()

            try:
                masks_compressed = pickle.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_masks.pkl"), 'rb'))
                data = json.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_data.json")))
            except FileNotFoundError:
                num_frames += 1
                continue
            
            depth_images = masks_compressed

            if frame_num == 0:
                map_features = frame.map_features
                lane_pt_list = []
                for feature in map_features:
                    if feature.HasField('lane'):

                        lanes_with_yaws = get_yaws_from_lane_coords(list(feature.lane.polyline))
                        lane_pt_list.append(lanes_with_yaws)
                
                lane_pt_list = np.vstack(lane_pt_list)



            (range_images, camera_projections,
            _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            point_clouds, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose,0,False)


            point_clouds = point_clouds[0]
            ones = np.ones(point_clouds.shape[0]).reshape(point_clouds.shape[0], 1)
            point_clouds = torch.from_numpy(np.hstack([point_clouds, ones]).transpose()).to(device=DEVICE, dtype=torch.float32)

            pc = LidarPointCloud(point_clouds)

            
            aggr_set = [pc.points]
            
            aggr_pc_points = torch.hstack(tuple([pcd for pcd in aggr_set]))

                

            """ Commentable code for visualization"""
            # # Load all RGB images for the current frame
            # IMAGE_LIST = {}
            # for cam_image in frame.images:
            #     im = np.array(tf.image.decode_jpeg(cam_image.image))
            #     im = Image.fromarray(im)
            #     sz_init = im.size
            #     im.thumbnail([1024, 1024])
            #     sz = im.size
            #     ratio = sz[0]/sz_init[0]

            #     IMAGE_LIST[cam_image.name] = im
            """ Commentable code ends """

            io_end = time.time()
            timer["io"] += io_end - io_start

            cam_calibs = frame.context.camera_calibrations

            # Loop on each depth mask
            for i, (label, score, c) in enumerate(zip(data["labels"], data["detection_scores"], data["cam_nums"])):
                id_offset += 1

                for cam_calib in cam_calibs:
                    if cam_calib.name == c+1:
                        break
                else:
                    print("Invalid cam_num")
                    exit()

                maskarr_1 = np.array(pymask.decode(depth_images[i]))
                maskarr_1 = maskarr_1.transpose([1, 0])

                ratio = 1024/1920

                mask_px_count = np.count_nonzero(maskarr_1)



                """ Erosion code """

                kernel = np.ones((3, 3), np.uint8)
                maskarr_1 = cv2.erode(maskarr_1, kernel)

                # else:
                # # if mask_px_count > 400:
                #     kernel = np.ones((5, 5), np.uint8)
                #     maskarr_1 = cv2.erode(maskarr_1, kernel)
                # elif mask_px_count > 1000:
                #     kernel = np.ones((7, 7), np.uint8)
                #     maskarr_1 = cv2.erode(maskarr_1, kernel)
                # else:
                #     kernel = np.ones((9, 9), np.uint8)
                #     maskarr_1 = cv2.erode(maskarr_1, kernel)
                """ """

                mask_1 = Image.fromarray(maskarr_1)
                maskarr_1 = maskarr_1[:, :].astype(bool)
                maskarr_1 = torch.transpose(torch.from_numpy(maskarr_1).to(device=DEVICE, dtype=bool), 1, 0)


                pim_start = time.time()
                track_points = np.array(range(aggr_pc_points.shape[1]))


                # pass in a copy of the aggregate pointcloud array
                # reset the lidar pointcloud
                cam_pc = LidarPointCloud(torch.clone(aggr_pc_points))


                # transform from ego into the camera.
                axes_transformation = torch.Tensor([
                        [0,-1,0,0],
                        [0,0,-1,0],
                        [1,0,0,0],
                        [0,0,0,1]]).to(device=DEVICE, dtype=torch.float32)
                axes_transformation = torch.linalg.inv(axes_transformation)
                transform_matrix = torch.from_numpy(np.array(cam_calib.extrinsic.transform).reshape(4, 4)).to(device=DEVICE, dtype=torch.float32)
                transform_matrix = torch.matmul(transform_matrix, axes_transformation)
                rotation_matrix = transform_matrix[:3, :3]
                quat = R.from_matrix(rotation_matrix.cpu())
                rotation = (quat.as_quat()[3], quat.as_quat()[0], quat.as_quat()[1], quat.as_quat()[2])


                cam_pc.translate(-transform_matrix[:3, 3])
                cam_pc.rotate(torch.from_numpy(Quaternion(rotation).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))



                # actually take a "picture" of the point cloud.
                # Grab the depths (camera frame z axis points away from the camera).
                depths = cam_pc.points[2, :]

                coloring = depths


                matrix = np.array(cam_calib.intrinsic, np.float32).tolist()
                camera_intrinsic = np.array([[matrix[0],0,matrix[2]],[0, matrix[1], matrix[3]],[0,0,1]])
                camera_intrinsic = camera_intrinsic*ratio
                camera_intrinsic[2, 2] = 1


                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points, point_depths = view_points(cam_pc.points[:3, :], torch.from_numpy(camera_intrinsic).to(device=DEVICE, dtype=torch.float32), normalize=True, device=DEVICE)

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

                view_masked_points = floored_points[:, masked_points_pixels[1]]


                # Now, indices_within_mask contains the indices of points within the masked pixels
                # print(track_points.shape)
                track_points = track_points[indices_within_mask.cpu()]


                global_masked_points = aggr_pc_points[:, track_points]

                pim_end = time.time()
                timer["points in mask"] += pim_end - pim_start


                if global_masked_points.numel() == 0:
                    # No points in the mask
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

            

                """ Commentable code for Visualization """
                # cam_to_ego = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                # mask_pc.rotate(Quaternion(cam_to_ego['rotation']).rotation_matrix)
                # mask_pc.translate(np.array(cam_to_ego['translation']))

                # axes_transformation = np.array([
                #         [0,-1,0,0],
                #         [0,0,-1,0],
                #         [1,0,0,0],
                #         [0,0,0,1]])
                # transform_matrix = np.array(cam_calib.extrinsic.transform).reshape(4, 4)
                # transform_matrix = np.matmul(transform_matrix, axes_transformation)
                # rotation_matrix = transform_matrix[:3, :3]
                # quat = R.from_matrix(rotation_matrix)
                # rotation = (quat.as_quat()[3], quat.as_quat()[0], quat.as_quat()[1], quat.as_quat()[2])
                # mask_pc.rotate(-Quaternion(rotation).rotation_matrix)
                # mask_pc.translate(np.array(transform_matrix[:3, 3]))
                
                # ego_to_global = nusc.get('ego_pose', cam['ego_pose_token'])
                # mask_pc.rotate(Quaternion(ego_to_global['rotation']).rotation_matrix)
                # mask_pc.translate(np.array(ego_to_global['translation']))
                """ Commentable code ends """


                
                """ Converting from ego frame to global frame for closest lane assignment """
                # print("EGO CENTROID:", mask_pc.points)
                transform_matrix = torch.from_numpy(np.array(frame.pose.transform, np.float32).reshape(4, 4)).to(DEVICE, dtype=torch.float32)
                # print(transform_matrix)
                rotation_matrix = transform_matrix[:3, :3]
                quat = R.from_matrix(rotation_matrix.cpu())
                rotation = (quat.as_quat()[3], quat.as_quat()[0], quat.as_quat()[1], quat.as_quat()[2])
                mask_pc.rotate(torch.from_numpy(Quaternion(rotation).rotation_matrix).to(DEVICE, dtype=torch.float32))
                mask_pc.translate(transform_matrix[:3, 3])


                # centroid = view_points(mask_pc.points[:3, :], camera_intrinsic, normalize=True)
                centroid = mask_pc.points

                centroid = centroid[:3]

                all_centroids_list.append(torch.Tensor(centroid).to(DEVICE, dtype=torch.float32))
                centroid_ids.append(id_offset)
                medoid_end = time.time()
                timer["medoid"] += medoid_end - medoid_start
                final_id_offset = id_offset
                

                """ Commentable code for Visualization """
                # print(view_masked_points)
                # masked_points_color = np.array([[1, 1, 1, 1]]*masked_pixels.detach().cpu().numpy()[0].shape[0])
                # print("ax im: ", IMAGE_LIST[c+1].size)
                # print("mask size: ", maskarr_1.shape)
                # print("masked_points:", masked_points_pixels[0].shape)
                # plt.figure(figsize=(16, 9))
                # plt.imshow(IMAGE_LIST[c+1])
                # plt.scatter(masked_pixels.detach().cpu().numpy()[0][:], masked_pixels.detach().cpu().numpy()[1][:], c=masked_points_color, s=2)
                # plt.scatter(view_masked_points[0].detach().cpu().numpy(), view_masked_points[1].detach().cpu().numpy(), c=(1, 1, 1, 1), s=4)
                # plt.scatter(view_masked_points[:, global_centroid][0].detach().cpu().numpy(),
                #             view_masked_points[:, global_centroid][1].detach().cpu().numpy(),
                #             c = (1, 0, 0, 1), s=6)
                
                # # plt.scatter(centroid[0], centroid[1], c=[[1, 0, 0, 1]], s=6)
                # plt.axis('off')
                # plt.savefig(os.path.join(OUTPUT_DIR, f"lidar_2n_{frame_num}_{c}"), bbox_inches='tight', pad_inches=0, dpi=200)

                # lidar_img = Image.open(os.path.join(OUTPUT_DIR, f"lidar_2n_{frame_num}_{c}.png"))
                # # lidar_img.thumbnail([1024, 1024])

                # mask_image_1 = Image.new('RGBA', mask_1.size, color=(0, 0, 0, 0))
                # mask_draw_1 = ImageDraw.Draw(mask_image_1)
                # draw_mask(maskarr_1.T.detach().cpu().numpy(), mask_draw_1, random_color=True)

                # lidar_img.alpha_composite(mask_image_1)

                # lidar_img.save(os.path.join(OUTPUT_DIR, f"lidar_2n_masked_{frame_num}_{c}.png"))
                # # time.sleep(3)
                """ Commentable code ends """


            num_frames+=1
        """ End of object centroids loop """

        cl_start = time.time()   
        print(len(all_centroids_list))
        all_centroids_list = torch.stack(all_centroids_list)
        all_centroids_list = torch.squeeze(all_centroids_list)
        print(all_centroids_list.shape)
        
        np.save('centroids.npy', np.array(all_centroids_list.to(device='cpu')))

        yaw_list, min_distance_list, lane_pt_coords_list = lane_yaws_distances_and_coords(
            all_centroids_list, lane_pt_list
        )
        # print(yaw_list)
        # exit()
        timer["closest lane"] += time.time() - cl_start

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

        # lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)


        id_offset = -1
        id_offset_list2 = []

        for frame_num, frame_data in tqdm(enumerate(scene_dataset), desc=scene_name+": "):
            # if frame_num >= 20:
            # if frame_num >= 1:
                # continue
            try: 
                data = json.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_data.json")))
            except FileNotFoundError:
                continue
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))

            

            # predictions["results"][sample["token"]] = []
            for i, (label, score, c) in enumerate(zip(data["labels"], data["detection_scores"], data["cam_nums"])):
                id_offset += 1
                if id_offset not in centroid_ids:
                    continue
                else:
                    print("id_offset:", id_offset)
                    id = centroid_ids.index(id_offset)
                    final_id_offset2 = id_offset
                
                id_offset_list2.append(id_offset)
                detection_name = get_detection_name(label)
                centroid = np.squeeze(np.array(all_centroids_list[id, :].to(device='cpu')))

                transform_matrix = np.array(frame.pose.transform, np.float32).reshape(4, 4)
                transform_matrix = np.linalg.inv(transform_matrix)
                centroid_pc = np.hstack([centroid, [1]])
                centroid_pc = np.dot(transform_matrix, centroid_pc)
                centroid = centroid_pc[:3]
                print(centroid)

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

                dist_from_lane = min_distance_list[id]
                global_lane_yaw = yaw_list[id]
                # coord_x, coord_y = lane_pt_coords_list[id, :]

                """ Object lane thresh"""
                # if dist_from_lane > 20:
                #     print("no closest lanes found.>?>?>?>?>?>?>?>?>?>?>?>?>")
                #     continue
                """ """

                extents = get_shape_prior(shape_priors, detection_name)

                """ Commented out code """
                if detection_name in ["car", "truck", "bus", "construction_vehicle", "trailer", "barrier"]:
                    # # TODO: take polygons out of the loop
                    # point = Point(m_x, m_y)
                    # drivable_start = time.time()
                    # is_drivable = False
                    # for polygon in drivable_polygons:
                    #     if point.within(polygon):
                    #         is_drivable = True
                    #         break
                    # drivable_end = time.time()
                    # timer["drivable"] += drivable_end - drivable_start
                    
                    # if not is_drivable:
                    #     print(f"[{detection_name}] vehicle not on road. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    #     continue
                    #     # ignore objects that are not in the drivable region

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

                    global_align_mat = np.eye(3)
                    global_align_mat[0:2, 0:2] = [[np.cos(global_lane_yaw), -np.sin(global_lane_yaw)], [np.sin(global_lane_yaw), np.cos(global_lane_yaw)]]
                    # print(global_align_mat)
                    euler_angles = R.from_matrix(global_align_mat).as_euler('xyz', degrees=False)
                    # print("GLOBAL_ANGLES:", euler_angles)

                    rot_matrix = transform_matrix[:3, :3]
                    align_mat = np.dot(rot_matrix, global_align_mat)
                    # print(align_mat)
                    # align_mat = global_align_mat
                    

                    """ Pushback code """
                    # Push centroid back for large objects
                    # pointsensor_token = sample['data'][pointsensor_channel]
                    # pointsensor = nusc.get('sample_data', pointsensor_token)
                    # poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
                    pushed_centroid = push_centroid(centroid, extents, Quaternion(matrix=global_align_mat), ego_frame=True)
                    # pushed_centroid = centroid
                    """ """

                    euler_angles = R.from_matrix(align_mat).as_euler('xyz', degrees=False)
                    # heading = euler_angles[1]
                    heading = euler_angles[2]
                    # print("EGO_ANGLES:", euler_angles)
                
                else:
                    """ Commented out code ends """
                    align_mat = np.eye(3)
                    pushed_centroid = centroid

                    euler_angles = R.from_matrix(align_mat).as_euler('xyz', degrees=False)
                    # heading = euler_angles[1]
                    heading = euler_angles[2]
                    # print(euler_angles)

                # extents = [extents[2], extents[0], extents[1]]

                # rot_quaternion = Quaternion(matrix=align_mat)

                # box_dict = {
                #         "sample_token": sample["token"],
                #         "translation": [float(i) for i in pushed_centroid],
                #         "size": list(extents),
                #         "rotation": list(rot_quaternion),
                #         "velocity": [0, 0],
                #         "detection_name": detection_name,
                #         "detection_score": score,
                #         "attribute_name": ATTRIBUTE_NAMES[detection_name]
                #     }

                # print("CENTROID:", pushed_centroid)

                print(detection_name, NUSC_TO_WAYMO[detection_name])
                detection_name = NUSC_TO_WAYMO[detection_name]
                
                o = metrics_pb2.Object()
                o.context_name = frame.context.name
                o.frame_timestamp_micros = frame.timestamp_micros

                waymo_box = label_pb2.Label.Box()
                waymo_box.center_x = float(pushed_centroid[0])
                waymo_box.center_y = float(pushed_centroid[1])
                waymo_box.center_z = float(pushed_centroid[2])
                waymo_box.length = float(extents[1])
                waymo_box.width = float(extents[0])
                waymo_box.height = float(extents[2])
                waymo_box.heading = heading
                # waymo_box.heading = 0
                o.object.box.CopyFrom(waymo_box)

                # print("HEADING:", heading)

                o.score = float(score)
                o.object.id = 'unique object tracking ID'

                if detection_name == "vehicle":
                    o.object.type = label_pb2.Label.TYPE_VEHICLE
                elif detection_name == "cyclist":
                    o.object.type = label_pb2.Label.TYPE_CYCLIST
                elif detection_name == "pedestrian":
                    o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
                else:
                    raise ValueError
                    o.object.type = label_pb2.Label.TYPE_UNKNOWN
                    
                # predictions["results"][sample["token"]].append(box_dict)
                objects.objects.append(o)
                print(f"[{detection_name}] created prediction {id}, score={score}")

            # if sample['next'] != "":
            #     sample = nusc.get('sample', sample['next'])

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

        # print(len([box for sample_boxes in predictions["results"].values() for box in sample_boxes]))

        # time.sleep(1000)

    

    print("\nRunning NMS on the predictions.\n")
    nms_start = time.time()

    
    # final_predictions = {
    #     "meta": {
    #         "use_camera": True,
    #         "use_lidar": False,
    #         "use_radar": False,
    #         "use_map": True,
    #         "use_external": False,
    #     },
    #     "results": {}
    # }


    # This is a dict of lists with keys as timestamps.
    objects_dict = {}
    for obj in objects.objects:
        if obj.frame_timestamp_micros in objects_dict:
            objects_dict[obj.frame_timestamp_micros].append(obj)
        else:
            objects_dict[obj.frame_timestamp_micros] = [obj]

    final_objects_dict = {}
    final_objects = metrics_pb2.Objects()
    
    for timestamp in objects_dict:

    # for sample in predictions["results"]:
    #     # class_to_num = {
    #     #     "bicycle" : 0,
    #     #     "car" : 1,
    #     #     "pedestrian" : 2,
    #     #     "truck" : 3,
    #     #     "bus" : 4,
    #     #     "construction_vehicle" : 5,

    #     # }

    #     final_predictions["results"][sample] = []
    #     # dets_by_label = {
    #     #     "barrier": [],
    #     #     "traffic_cone": [],
    #     #     "bicycle": [],
    #     #     "motorcycle": [],
    #     #     "pedestrian": [],
    #     #     "car": [],
    #     #     "bus": [],
    #     #     "construction_vehicle": [],
    #     #     "trailer": [],
    #     #     "truck": [],
    #     # }

        dets = []
        det_labels = []
        # TODO: define threshs.
        threshs_by_label = {
            label_pb2.Label.TYPE_UNKNOWN: 1,
            label_pb2.Label.TYPE_SIGN: 0.175,
            label_pb2.Label.TYPE_CYCLIST: 0.85,
            # "motorcycle": 0.85,
            label_pb2.Label.TYPE_PEDESTRIAN: 0.175,
            label_pb2.Label.TYPE_VEHICLE: 4,
            # "bus": 10,
            # "construction_vehicle": 12,
            # "trailer": 10,
            # "truck": 12,
        }

    #     # [4, 12, 10, 1, 0.85, 0.175],
    #     # tasks = [
    #     # dict(num_class=1, class_names=["car"]),
    #     # dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    #     # dict(num_class=2, class_names=["bus", "trailer"]),
    #     # dict(num_class=1, class_names=["barrier"]),
    #     # dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    #     # dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
    #     # ]
        
        context_names_list = []
        timestamps_list = []

        centroids_list = []
        extents_list = []
        rot_list = []
        det_names_list = []
        attr_names_list = []
        vertices_list = []
        scores = []

        for o in objects_dict[timestamp]:
            context_name = o.context_name
            ts = o.frame_timestamp_micros
            centroid = [
                o.object.box.center_x,
                o.object.box.center_y,
                o.object.box.center_z,
            ]
            extents = [
                o.object.box.width,
                o.object.box.length,
                o.object.box.height,
            ]
            score = o.score
            rot = o.object.box.heading
            detection_name = o.object.type

            dets.append(np.array([centroid[0], centroid[1], score]))
            det_labels.append(detection_name)

            context_names_list.append(context_name)
            timestamps_list.append(ts)
            centroids_list.append(centroid)
            extents_list.append(extents)
            rot_list.append(rot)
            det_names_list.append(detection_name)
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
        print(len(det_labels), dets.shape, end=" ")
        keep_indices = list(circle_nms(dets, det_labels, threshs_by_label))
        print(len(keep_indices))
        

        """ Commentable code """
        context_names_list = [context_names_list[c] for c in range(len(context_names_list)) if c in keep_indices]
        timestamps_list = [timestamps_list[c] for c in range(len(timestamps_list)) if c in keep_indices]
        centroids_list = [centroids_list[c] for c in range(len(centroids_list)) if c in keep_indices]
        extents_list = [extents_list[c] for c in range(len(extents_list)) if c in keep_indices]
        rot_list = [rot_list[c] for c in range(len(rot_list)) if c in keep_indices]
        det_names_list = [det_names_list[c] for c in range(len(det_names_list)) if c in keep_indices]
        # attr_names_list = [attr_names_list[c] for c in range(len(attr_names_list)) if c in keep_indices]
        scores = [scores[c] for c in range(len(scores)) if c in keep_indices]
        """ Commentable code """

        for i, (context_name, ts, centroid, extents, rot, det_name, score) in enumerate(zip(
            context_names_list, timestamps_list, centroids_list, extents_list, rot_list, det_names_list, scores
        )):
            o = metrics_pb2.Object()
            o.context_name = context_name
            o.frame_timestamp_micros = ts

            waymo_box = label_pb2.Label.Box()
            waymo_box.center_x = float(centroid[0])
            waymo_box.center_y = float(centroid[1])
            waymo_box.center_z = float(centroid[2])
            waymo_box.length = float(extents[1])
            waymo_box.width = float(extents[0])
            waymo_box.height = float(extents[2])
            waymo_box.heading = rot
            o.object.box.CopyFrom(waymo_box)

            # print("HEADING:", heading)

            o.score = float(score)
            o.object.id = 'unique object tracking ID'
            o.object.type = det_name
                
            # predictions["results"][sample["token"]].append(box_dict)
            # objects.objects.append(o)
            # print(f"[{detection_name}] created prediction {i}, score={score}")

            if timestamp in final_objects_dict:
                final_objects_dict[timestamp].append(o)
            else:
                final_objects_dict[timestamp] = [o]

    for timestamp in final_objects_dict:
        for o in final_objects_dict[timestamp]:
            final_objects.objects.append(o)
    
    print(len(final_objects.objects), len(objects.objects))

    nms_end = time.time()
    timer["nms"] += nms_end - nms_start



    # # with open("predictions_lidar_train.json", "w") as f:
    # with open("predictions_lidar.json", "w") as f:
    # # with open("predictions_lidar_traindetect.json", "w") as f:
    # # with open("predictions_lidar_traindetect25_rare.json", "w") as f:
    #     json.dump(final_predictions, f)

    # print(f"wrote {len(final_predictions['results'])} samples.")
    # #     json.dump(predictions, f)
    # # print(f"wrote {len(predictions['results'])} samples.")

    total_end = time.time()
    timer["total"] += total_end - total_start

    for operation in timer:
        print(operation, ":\t\t", timer[operation])

    f = open('../../outputs/waymo/pred_0307_detic_train_680_710.bin', 'wb')
    # Without NMS:
    # f.write(objects.SerializeToString())
    # # With NMS:
    f.write(final_objects.SerializeToString())
    f.close()

    # print(objects)