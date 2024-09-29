import json
import os
import random
import time

import numpy as np
import torch
import torchvision
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val, train_detect, train, val
from PIL import Image, ImageDraw, ImageFont
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, build_sam
from tqdm import tqdm
import pycocotools
import pickle

from cfg.prompt_cfg import TEXT_PROMPT, MAPS, BOX_THRESHOLDS, TEXT_THRESHOLDS, OLD_MAPS
import sys
sys.path.insert(0, '../../Detic/third_party/CenterNet2/')
sys.path.insert(0, '../../Detic/')
# sys.path.insert(0, '../../../Detic/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("../../Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = "../../Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
# FIXME:
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False # For better visualization purpose. Set to False for all classes.
cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = '../../Detic/datasets/metadata/lvis_v1_train_cat_info.json'
cfg.MODEL.DEVICE='cuda:1' # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)

# Setup the model's vocabulary using build-in datasets
def get_clip_embeddings(vocabulary, prompt='a '):
    from detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

custom_vocabulary = [
    "bus",
    "truck",
    "dumptruck", # added
    "car",
    "pedestrian",
    
    "person",
    "human",
    "bicycle",
    "sedan",
    "pickup_truck",
    # "traffic_cone",
    # "barrier",
    # "road_barrier",
    "trailer",
    "truck_trailer",
    "semi_trailer", # added
    "tank_trailer", # added
    "construction_vehicle",
    "motorcycle",
]
metadata = MetadataCatalog.get("__unused")
metadata.thing_classes = custom_vocabulary
classifier = get_clip_embeddings(metadata.thing_classes)

# vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
# metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
# classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)



INPUT_PATH = "../../data/waymo-v1.4.2/waymo_format/training/"


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
print(CAM_LIST)

Lidar_Name = {
    0:"UNKNOWN",
    1:"TOP",
    2:"FRONT",
    3:"SIDE_LEFT",
    4:"SIDE_RIGHT",
    5:"REAR"
}


SAM_CKPT = "../../zs3d/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

OUTPUT_DIR = "../../mask_outputs/waymo-detic/"
OUTPUT_PIC_DIR = "../../outputs/waymo/"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)

    if label:
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")

        draw.text((box[0], box[1]), label)



def map_class(name):
    # if name in MAPS.keys():
    if name in OLD_MAPS.keys():
        print("[][]", end="")
        return OLD_MAPS[name]
    elif "car" in name or "sedan" in name or "suv" in name or "pickup truck" in name:
        return "car"
    elif "pickup" in name:
        print("this is a pickup,.,.,.,.,<><><><><,.,.,.<<><><><,.,.,><>><><><>,.,.,.<><><><")
        return "car"
    elif "human" in name or "person" in name or "pedestrian" in name:
        return "pedestrian"
    elif "bicycle" in name or "bike" in name:
        return "bicycle"
    elif "truck" in name or "lorry" in name:
        return "truck"
    elif "bus" in name:
        return "bus"
    # elif "ck" in name:
    #     return "truck"
    
    print("couldn't assign class. <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return "car"



import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt

    


if __name__ == "__main__":
    start_time = time.time()
    global blip_processor, blip_model, groundingdino_model, sam_predictor, sam_automask_generator, inpaint_pipeline
    
    scene_list = os.listdir(INPUT_PATH)
    scene_list.sort()

    import matplotlib.patches as patches

    assert SAM_CKPT, "sam_checkpoint is not found!"
    sam = build_sam(checkpoint=SAM_CKPT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    sam_automask_generator = SamAutomaticMaskGenerator(sam)


    # make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_count = 1


    for scene_num, scene_name in enumerate(scene_list[:10]):
        # Lazily read
        scene_dataset = tf.data.TFRecordDataset(INPUT_PATH+scene_name, compression_type='')
        print(scene_dataset)


        for f, data in tqdm(enumerate(scene_dataset), desc=scene_name+": "):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            cam_nums = []
            labels = []
            detection_scores = []
            depth_images = []  # stores masked depth images
            im_to_ego_mats = []
            mask_images = []
            data = {}
            np_mask_images = []

            cam_images = frame.images
            cam_calibs = frame.context.camera_calibrations
            for c in range(len(CAM_LIST)):
                for cam_image in cam_images:
                    if RGB_Name[cam_image.name] == CAM_LIST[c]:
                        break
                else:
                    print(RGB_Name[cam_image.name])
                    print('Unknown camera label.')
                    exit()
                cam_image = np.array(tf.image.decode_jpeg(cam_image.image))
                this_cam_labels = []
                this_cam_scores = []
                print(f"Exporting boxes from Frame {f}, {CAM_LIST[c]} of {scene_name} ({scene_num})...")

                use_token_spans = True
                detic_start = time.time()

                # image_pil, image, ratio = load_image(path)
                image_pil = Image.fromarray(cam_image).convert('RGB')
                size_init = image_pil.size

                # # resize image and save ratio
                image_pil.thumbnail((1024, 1024))
                size = image_pil.size  # w, h
                ratio = np.array(size) / np.array(size_init)
                ratio = ratio[0]

                # load image as an array
                image = np.array(image_pil)

                # Convert to BGR for Detic
                image = image[:, :, ::-1].copy()
                outputs = predictor(image)
                # print(outputs)

                try:
                    boxes_filt = outputs["instances"].pred_boxes.tensor
                    # print(outputs['instances'].scores[0].item())
                    pred_phrases = [
                        f"{custom_vocabulary[c]}({round(outputs['instances'].scores[i].item(), 2)})" for i, c in enumerate(outputs["instances"].pred_classes)
                    ]
                    # print(boxes_filt, pred_phrases)
                except IndexError:
                    print("\n\t\t\t\tNo box found in this image.\n")
                    im_to_ego_mats.append([])
                    continue

                detic_end = time.time()
                print(f"Detic took {detic_end - detic_start} seconds.")

                # visualize pred
                size = image_pil.size
                pred_dict = {
                    "boxes": boxes_filt,
                    "size": [size[1], size[0]],  # H,W
                    "labels": pred_phrases,
                }

                """ VISUALIZATION CODE """
                # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
                # image_with_box.save(os.path.join(OUTPUT_PIC_DIR, "pred_detic.jpg"))
                """ """


                np_image = np.array(image_pil)


                boxes_filt = boxes_filt.cpu()

                # implement class-wise nms on these boxes
                nms_start = time.time()
                for i, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
                    idx = label.find("(")
                    ds = label[idx + 1 : -1]
                    label = label[:idx]

                    this_cam_labels.append(map_class(label.lower()))
                    this_cam_scores.append(float(ds))
                
                
                nms_scores = []
                nms_boxes_filt = torch.Tensor([])
                nms_labels = []
                for cls in set(OLD_MAPS.values()):
                    cls_boxes = boxes_filt[[i for i, l in enumerate(this_cam_labels) if l == cls]]
                    cls_scores = [this_cam_scores[i] for i, l in enumerate(this_cam_labels) if l == cls]
                    if len(cls_boxes) == 0:
                        continue
                    keep = torchvision.ops.nms(cls_boxes, torch.Tensor(cls_scores), 0.75)
                    keep = keep.cpu()
                    cls_boxes = cls_boxes[keep]
                    cls_scores = [cls_scores[i] for i in keep]
                    cls_labels = [cls for _ in range(len(keep))]
                    nms_boxes_filt = torch.cat((nms_boxes_filt, cls_boxes), dim=0)
                    nms_scores.extend(cls_scores)
                    nms_labels.extend(cls_labels)
                

                boxes_filt = nms_boxes_filt
                detection_scores.extend(nms_scores)
                labels.extend(nms_labels)
                nms_end = time.time()
                print("NMS took", nms_end - nms_start, "seconds.")
                    
                sam_start = time.time()
                sam_predictor.set_image(np_image)
                
                
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(DEVICE)
                if transformed_boxes.shape[0] == 0:
                    print("<< No objects found. >>")
                    im_to_ego_mats.append([])
                    continue
                
                # try:
                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                sam_end = time.time()
                print(f"SAM took {sam_end - sam_start} seconds.")


                assert len(boxes_filt) == len(nms_labels) == len(masks)


                """ Get the 3D mesh using ZoeDepth """
                use_zoedepth = False
                image_pil_rgb = image_pil.convert("RGB")
                images_pil_masked = []
                
                if use_zoedepth:
                    zdepth_start = time.time()
                    depth_image = predict_depth(zoe_model, image_pil_rgb)
                    zdepth_end = time.time()
                    print(f"ZoeDepth took {zdepth_end - zdepth_start} seconds.")
                    depth_image_arr = np.array(depth_image)

                    for i, (box, label, mask) in enumerate(zip(boxes_filt, pred_phrases, masks)):
                        mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
                        mask_draw = ImageDraw.Draw(mask_image)
                        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

                        """ Vis """
                        # image_pil_iter = image_pil.copy()
                        # image_draw = ImageDraw.Draw(image_pil_iter)
                        # draw_box(box, image_draw, label)
                        # images_pil_masked.append(image_pil_iter.convert("RGBA"))
                        """ """
                        """ pre-nms code """
                        # idx = label.find("(")
                        # ds = label[idx + 1 : -1]
                        # label = label[:idx]
                        # # label = label.split(" ")[0]

                        # if float(ds) < BOX_THRESHOLDS[map_class(label.lower())]:
                        #     print(f"omitting {label} {i}, with cf value {ds}")
                        #     continue

                        # print(label, "->", map_class(label.lower()))
                        # labels.append(map_class(label.lower()))
                        # detection_scores.append(float(ds))
                        """ """
                        cam_nums.append(c)
                        mask_images.append(mask_image)
                        np_mask_images.append(np.array(mask_image))
                        depth_image_arr_masked = (depth_image_arr * np.array(mask_image)[:, :, 3] / 255)
                        # store the depth image
                        print(np.count_nonzero(depth_image_arr_masked))

                        depth_images.append(depth_image_arr_masked)
                
                else:
                    for i, (box, label, mask) in enumerate(zip(boxes_filt, pred_phrases, masks)):
                        mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
                        mask_draw = ImageDraw.Draw(mask_image)
                        draw_mask(mask[0].cpu().numpy(), mask_draw)

                        """ Vis """
                        # image_pil_iter = image_pil.copy()
                        # image_draw = ImageDraw.Draw(image_pil_iter)
                        # draw_box(box, image_draw, label)
                        # images_pil_masked.append(image_pil_iter.convert("RGBA"))
                        """ """


                        cam_nums.append(c)

                        np_mask_image = np.array(mask_image).astype(np.uint8).transpose([2, 1, 0])[3, :, :]
                        np_mask_image = np.squeeze(np_mask_image)
                        # print(len(np.where(np_mask_image)[0]))
                        # print('where', np_mask_image.shape)
                        # compressed_np_mask_image = np.packbits(np_mask_image[:, :, 3])
                        # print(np_mask_image.shape)
                        compressed_np_mask_image = pycocotools.mask.encode(np.asfortranarray(np_mask_image))
                        # print(compressed_np_mask_image)
                        np_mask_images.append(compressed_np_mask_image)
                        # np_mask_images.append(np_mask_image)
                
                """ Vis """
                # for i in range(0, len(images_pil_masked)):
                #     images_pil_masked[i].alpha_composite(mask_images[i])
                #     images_pil_masked[i].save(os.path.join(OUTPUT_PIC_DIR, f'img_{scene_name}_{c}_{i}.png'))
                #     # mask_images[i].save(os.path.join(OUTPUT_PIC_DIR, f'mask_{scene_name}_{c}_{i}.png'))
                """ """

                print(f"Found {len(masks)} masks.")
                print(f"Got {len(labels)} labels.")
                    
                print()

            if use_zoedepth:
                np_images = np.array(depth_images)
            else:
                np_images = np_mask_images



            if len(labels) == 0:
                continue

            assert len(labels) == len(detection_scores)
            assert len(labels) == len(cam_nums)
            assert len(np_images) == len(labels)

            data["labels"] = labels
            data["detection_scores"] = detection_scores
            data["cam_nums"] = cam_nums
            

            os.makedirs(os.path.join(OUTPUT_DIR, scene_name), exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f"{scene_name}", f"{f}_data.json"), "w") as outfile:
                json.dump(data, outfile)


            pickle.dump(np_images, open(os.path.join(OUTPUT_DIR, f"{scene_name}", f"{f}_masks.pkl"), 'wb'))

            sample_count += 1

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds for {sample_count} samples.")

