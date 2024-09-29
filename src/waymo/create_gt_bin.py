from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2
import os
import tqdm
import tensorflow as tf

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

INPUT_PATH = "../../data/waymo-v1.4.2/waymo_format/training/"
selected_waymo_locations = None
load_dir = './'
# tfrecord_pathnames = [
#     '',
#     ''
# ]
scene_list = os.listdir(INPUT_PATH)
scene_list.sort()
tfrecord_pathnames = scene_list[0:10]


bin_save_path = os.path.dirname(load_dir)
print(f'Now create gt.bin, will write to {bin_save_path}/gt.bin')
objects = metrics_pb2.Objects()

# running eval for 2 scenes:
for file_idx in tqdm.tqdm(range(len(tfrecord_pathnames))):
    pathname = tfrecord_pathnames[file_idx]
    dataset = tf.data.TFRecordDataset(INPUT_PATH+pathname, compression_type='')

    for frame_idx, data in enumerate(dataset):
        # if frame_idx > 60:
        #     continue
        print(frame_idx, end=" ")
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if (selected_waymo_locations is not None
                and frame.context.stats.location
                not in selected_waymo_locations):
            continue
    
        # parse labels
        o_list = _create_pd_file(frame)
        for obj in o_list:
            if obj.object.type == 4:
                print(obj.object.type)
        objects.objects.extend(o_list)
        # print(objects)
        # exit()

f = open(f'{bin_save_path}/gt_10.bin', 'wb')
f.write(objects.SerializeToString())
f.close()