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
sys.path.insert(0, '/home/mehark/Detic/third_party/CenterNet2/')
sys.path.insert(0, '/home/mehark/Detic/')
# sys.path.insert(0, '../../../Detic/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

from kitti_object import kitti_object

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("/home/mehark/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = "/home/mehark/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
# FIXME:
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False # For better visualization purpose. Set to False for all classes.
cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = '/home/mehark/Detic/datasets/metadata/lvis_v1_train_cat_info.json'
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
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
    "traffic_cone",
    "barrier",
    "road_barrier",
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

# VER_NAME = "v1.0-mini"
# INPUT_PATH = "/ssd0/mehark/nusc-mini/"
VER_NAME = "v1.0-trainval"
# INPUT_PATH = "/ssd0/mehark/nuScenes/"
INPUT_PATH = "/data2/mehark/kitti/"


CAM_LIST = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

# DINO_CONFIG = "/home/mehark/zs3d/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
SAM_CKPT = "/home/mehark/zs3d/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
# DINO_CKPT = "/home/mehark/zs3d/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

# OUTPUT_DIR = "model_outputs"
# OUTPUT_DIR = "/media/nperi/HDD1/mehar/val_set_outputs_nusc"
# OUTPUT_DIR = "/ssd0/mehark/zs3d_outputs/"
OUTPUT_DIR = "/data2/mehark/zs3d_outputs/kitti_detic_wo_2d_nms/"
OUTPUT_PIC_DIR = "../../outputs/kitti/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# zoe_model = (torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(DEVICE).eval())


# ZoeDepth
def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth


# # GSAM
# def load_image(image_path):
#     # load image
#     image_pil = Image.open(image_path).convert("RGB")  # load image
#     size_init = image_pil.size
#     image_pil.thumbnail((1024, 1024))
#     size = image_pil.size  # w, h
#     ratio = np.array(size) / np.array(size_init)
#     ratio = ratio[0]
#     print(ratio)
#     transform = T.Compose(
#         [
#             T.RandomResize([800], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image, _ = transform(image_pil, None)  # 3, h, w
#     return image_pil, image, ratio

# def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
#     args = SLConfig.fromfile(model_config_path)
#     args.device = "cuda:0" if not cpu_only else "cpu"
#     model = build_model(args)
#     checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#     load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     print(load_res)
#     _ = model.eval()
#     return model


# def transform_image(image_pil):
#     transform = T.Compose(
#         [
#             T.RandomResize([800], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image, _ = transform(image_pil, None)  # 3, h, w
#     return image


# def tokenize_caption(model, caption, cpu_only=False):
#     caption = caption.lower()
#     caption = caption.strip()
#     if not caption.endswith("."):
#         caption = caption + "."
#     device = "cuda:0" if not cpu_only else "cpu"
#     print(device)
#     model = model.to(device)

#     tokenlizer = model.tokenizer
#     tokenized = tokenlizer(caption)

#     return tokenlizer, tokenized


# def get_grounding_output1(
#     model, image, caption, tokenlizer, tokenized, box_threshold, text_threshold, with_logits=True, cpu_only=False,
# ):
#     caption = caption.lower()
#     caption = caption.strip()
#     if not caption.endswith("."):
#         caption = caption + "."
#     device = "cuda:0" if not cpu_only else "cpu"
#     print(device)
#     model = model.to(device)
#     image = image.to(device)
#     with torch.no_grad():
#         outputs = model(image[None], captions=[caption])
#     # print(outputs)
#     logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
#     boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
#     # print("logits", logits)
#     # print(logits.shape, boxes.shape)
#     # print(logits[:, :].max(dim=1))

#     # filter output
#     logits_filt = logits.clone()
#     boxes_filt = boxes.clone()
#     filt_mask = logits_filt.max(dim=1)[0] > box_threshold
#     logits_filt = logits_filt[filt_mask]  # num_filt, 256
#     boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
#     logits_filt.shape[0]

#     # get phrase
#     # tokenize_start = time.time()
#     # tokenlizer = model.tokenizer
#     # tokenized = tokenlizer(caption)
#     # print("Tokenizer took", time.time() - tokenize_start, "seconds.")
#     # build pred
#     pred_phrases = []
#     scores = []
#     for logit, box in zip(logits_filt, boxes_filt):
#         pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
#         if with_logits:
#             pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
#         else:
#             pred_phrases.append(pred_phrase)
#         scores.append(logit.max().item())

# #     return boxes_filt, torch.Tensor(scores), pred_phrases
#     return boxes_filt, pred_phrases


# # def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
# #     print(image.get_device())
# #     assert text_threshold is not None or token_spans is not None, "text_threshold and token_spans should not be None at the same time!"
# #     caption = caption.lower()
# #     caption = caption.strip()
# #     if not caption.endswith("."):
# #         caption = caption + "."
# #     # device = "cuda:0" if not cpu_only else "cpu"
# #     # model = model.to(device)
# #     with torch.no_grad():
# #         outputs = model(image[None], captions=[caption], device=DEVICE)
# #     logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
# #     boxes = outputs["pred_boxes"][0]  # (nq, 4)

# #     print(torch.max(boxes[:, 0] - boxes[:, 2]))

# #     pred_phrases = []
# #     scores = []
# #     boxes_filt=[]
# #     # filter output
# #     if token_spans is None:
# #         # logits_filt = logits.cpu().clone()
# #         # boxes_filt = boxes.cpu().clone()
# #         # filt_mask = logits_filt.max(dim=1)[0] > box_threshold
# #         # logits_filt = logits_filt[filt_mask]  # num_filt, 256
# #         # boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

# #         # # get phrase
# #         # tokenlizer = model.tokenizer
# #         # tokenized = tokenlizer(caption)
# #         # # build pred
# #         # pred_phrases = []
# #         # for logit, box in zip(logits_filt, boxes_filt):
# #         #     pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
# #         #     if with_logits:
# #         #         pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
# #         #     else:
# #         #         pred_phrases.append(pred_phrase)

# #         # filter output
# #         logits_filt = logits.clone()
# #         boxes_filt = boxes.clone()
# #         filt_mask = logits_filt.max(dim=1)[0] > box_threshold
# #         logits_filt = logits_filt[filt_mask]  # num_filt, 256
# #         boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
# #         print(logits_filt.shape[0])

# #         # get phrase
# #         tokenlizer = model.tokenizer
# #         tokenized = tokenlizer(caption)
# #         # build pred
        
# #         for logit, box in zip(logits_filt, boxes_filt):
# #             pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
# #             if with_logits:
# #                 pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
# #             else:
# #                 pred_phrases.append(pred_phrase)
# #             scores.append(logit.max().item())
    
# #     else:
# #         # given-phrase mode
# #         positive_maps = create_positive_map_from_span(
# #             # TODO: check
# #             model.tokenizer(caption),
# #             # model.tokenizer(caption),
# #             token_span=token_spans
# #         ).to(image.device) # n_phrase, 256

# #         logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
# #         print(logits_for_phrases.shape)
# #         all_logits = []
# #         all_phrases = []
# #         all_boxes = []
# #         for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
# #             print([caption[_s:_e] for (_s, _e) in token_span], torch.max(logit_phr))
# #             # get phrase
# #             phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
# #             # get mask
# #             filt_mask = logit_phr > box_threshold
# #             # filt box
# #             all_boxes.append(boxes[filt_mask])
# #             # filt logits
# #             all_logits.append(logit_phr[filt_mask])
# #             if with_logits:
# #                 logit_phr_num = logit_phr[filt_mask]
# #                 all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
# #             else:
# #                 all_phrases.extend([phrase for _ in range(len(filt_mask))])
# #         boxes_filt = torch.cat(all_boxes, dim=0).cpu()
# #         pred_phrases = all_phrases


# #     return boxes_filt, scores, pred_phrases

# # def get_gronunding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
# #     assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
# #     caption = caption.lower()
# #     caption = caption.strip()
# #     if not caption.endswith("."):
# #         caption = caption + "."
# #     device = "cuda:0" if not cpu_only else "cpu"
# #     print(device)
# #     model = model.to(device)
# #     image = image.to(device)
# #     with torch.no_grad():
# #         outputs = model(image[None], captions=[caption])
# #     print(outputs)
# #     logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
# #     boxes = outputs["pred_boxes"][0]  # (nq, 4)
# #     print(torch.max(logits, axis=1))

# #     # given-phrase mode TODO:
# #     positive_maps = create_positive_map_from_span(
# #         model.tokenizer(caption),
# #         token_span=token_spans
# #     ).to(image.device) # n_phrase, 256

# #     logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
# #     all_logits = []
# #     all_phrases = []
# #     all_boxes = []
# #     for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
# #         # get phrase
# #         phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
# #         # get mask
# #         filt_mask = logit_phr > box_threshold
# #         # filt box
# #         all_boxes.append(boxes[filt_mask])
# #         # filt logits
# #         all_logits.append(logit_phr[filt_mask])
# #         if with_logits:
# #             logit_phr_num = logit_phr[filt_mask]
# #             all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
# #         else:
# #             all_phrases.extend([phrase for _ in range(len(filt_mask))])
# #     boxes_filt = torch.cat(all_boxes, dim=0).cpu()
# #     pred_phrases = all_phrases


# #     return boxes_filt, pred_phrases

# """ DO NOT CHANGE THIS FUNCTION """
# def get_grounding_output(model, image, caption, tokenlizer, tokenized, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
#     assert text_threshold is not None or token_spans is not None, "text_threshold and token_spans should not be None at the same time!"
#     caption = caption.lower()
#     caption = caption.strip()
#     if not caption.endswith("."):
#         caption = caption + "."
#     device = "cuda:0" if not cpu_only else "cpu"
#     print(device)
#     model = model.to(device)
#     image = image.to(device)
#     with torch.no_grad():
#         outputs = model(image[None], captions=[caption])
#     logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
#     boxes = outputs["pred_boxes"][0]  # (nq, 4)

#     # filter output
#     if token_spans is None:
#         logits_filt = logits.cpu().clone()
#         boxes_filt = boxes.cpu().clone()
#         filt_mask = logits_filt.max(dim=1)[0] > box_threshold
#         logits_filt = logits_filt[filt_mask]  # num_filt, 256
#         boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

#         # get phrase
#         # tokenlizer = model.tokenizer
#         # tokenized = tokenlizer(caption)
#         # build pred
#         pred_phrases = []
#         for logit, box in zip(logits_filt, boxes_filt):
#             pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
#             if with_logits:
#                 pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
#             else:
#                 pred_phrases.append(pred_phrase)
#     else:
#         # given-phrase mode
#         positive_maps = create_positive_map_from_span(
#             # model.tokenizer(caption),
#             tokenized,
#             token_span=token_spans
#         ).to(image.device) # n_phrase, 256

#         logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
#         all_logits = []
#         all_phrases = []
#         all_boxes = []
#         for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
#             # get phrase
#             phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
#             # get mask
#             filt_mask = logit_phr > box_threshold
#             # filt box
#             all_boxes.append(boxes[filt_mask])
#             # filt logits
#             all_logits.append(logit_phr[filt_mask])
#             if with_logits:
#                 logit_phr_num = logit_phr[filt_mask]
#                 all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
#             else:
#                 all_phrases.extend([phrase for _ in range(len(filt_mask))])
#         boxes_filt = torch.cat(all_boxes, dim=0).cpu()
#         pred_phrases = all_phrases


#     return boxes_filt, pred_phrases


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
        # box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        # box[:2] -= box[2:] / 2
        # box[2:] += box[:2]
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
        # color = (30, 144, 255, 153)
        color = (30, 144, 255, 255)

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


# utils
def count_frames(nusc, sample):
    frame_count = 1

    if sample["next"] != "":
        frame_count += 1

        # Don't want to change where sample['next'] points to since it's used later, so we'll create our own pointer
        sample_counter = nusc.get("sample", sample["next"])

        while sample_counter["next"] != "":
            frame_count += 1
            sample_counter = nusc.get("sample", sample_counter["next"])

    return frame_count

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

# def get_word_wise_token_span(prompt):
#     """ spaces are also considered as delimiters """
#     span = []
#     flag = True
#     span_start = 0
#     span_end = 0
#     for i in range(len(prompt)):
#         if prompt[i] not in [' ', '.']:
#             if flag == False:
#                 flag = True
#                 span_start = i
#         else:
#             if flag == True:
#                 flag = False
#                 span_end = i
#                 span.append([[span_start, span_end]])

#     print(span)
#     return span

# def get_phrase_wise_token_span(prompt):
#     """ only periods are considered as delimiters """
#     span = []
#     flag = True
#     span_start = 0
#     span_end = 0
#     phrase = []
#     for i in range(len(prompt)):
#         if prompt[i] == '.':
#             flag = False
#             span.append(phrase)
#             phrase = []
        
#         elif prompt[i] == ' ':
#             if flag == True:
#                 span_end = i
#                 phrase.append([span_start, span_end])
#                 flag = False
        
#         else:
#             if flag == False:
#                 span_start = i
#                 flag = True
#             if flag == True:
#                 span_end = i
    
#     print(span)
#     return span
        
            
    


if __name__ == "__main__":
    start_time = time.time()
    global blip_processor, blip_model, groundingdino_model, sam_predictor, sam_automask_generator, inpaint_pipeline

    # nusc = NuScenes(version=VER_NAME, dataroot=INPUT_PATH, verbose=True)

    assert SAM_CKPT, "sam_checkpoint is not found!"
    sam = build_sam(checkpoint=SAM_CKPT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    sam_automask_generator = SamAutomaticMaskGenerator(sam)

    # groundingdino_model = load_model(DINO_CONFIG, DINO_CKPT, cpu_only=False)

    # make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_count = 1

    kitti = kitti_object(os.path.join(INPUT_PATH))

    # tokenlizer, tokenized = tokenize_caption(groundingdino_model, TEXT_PROMPT)

    # mini_val = ['scene-0916']
    # for scene_num, scene_name in enumerate(mini_val):
    # for scene_name in mini_train+mini_val:
    # for scene_num, scene_name in enumerate(['scene-0061']):
    # for scene_num, scene_name in enumerate(train_detect[:50]):
    # for scene_num, scene_name in enumerate(train[530:]):
    # for scene_num, scene_name in tqdm(enumerate(val[:25])):
        # scene_token = nusc.field2token("scene", "name", scene_name)[0]
        # scene = nusc.get("scene", scene_token)
        # sample = nusc.get("sample", scene["first_sample_token"])

    # num_frames = count_frames(nusc, sample)
    num_frames = len(kitti)
    for f in tqdm(range(num_frames), desc=": "):
        cam_nums = []
        # meshes = []
        labels = []
        detection_scores = []
        # to_remove = []
        depth_images = []  # stores masked depth images
        im_to_ego_mats = []
        mask_images = []
        data = {}
        np_mask_images = []

        # for c in range(len(CAM_LIST)):
        this_cam_labels = []
        this_cam_scores = []
        print(f"Exporting boxes from Frame {f}...")

        # (path, nusc_boxes, camera_instrinsic) = nusc.get_sample_data(
        #     sample["data"][CAM_LIST[c]]
        # )

        use_token_spans = True
        gdino_start = time.time()

        if not use_token_spans:
            image_pil = Image.fromarray(kitti.get_image(f)[:, :, ::-1]).convert('RGB')
            size_init = image_pil.size

            # # resize image and save ratio
            image_pil.thumbnail((1024, 1024))
            size = image_pil.size  # w, h
            ratio = np.array(size) / np.array(size_init)
            ratio = ratio[0]

            # load image as an array
            image = np.array(image_pil)

            # transform image for groundingdino
            transformed_image = transform_image(image_pil)
            # transformed_image = image_pil.convert("RGB")
            boxes_filt, pred_phrases = get_grounding_output1(
                groundingdino_model,
                transformed_image,
                TEXT_PROMPT,
                tokenlizer,
                tokenized,
                0.15,
                0.15,
            )
        else:
            # image_pil, image, ratio = load_image(path)
            image_pil = Image.fromarray(kitti.get_image(f)[:, :, ::-1]).convert('RGB')
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
            

        gdino_end = time.time()
        print(f"Detic took {gdino_end - gdino_start} seconds.")
        # time.sleep(1000)
        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

        # """ VISUALIZATION CODE """
        # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        # image_with_box.save(os.path.join(OUTPUT_PIC_DIR, "pred_kitti.jpg"))
        # """ """
        # time.sleep(2)
        # exit()

        np_image = np.array(image_pil)

        # # process boxes
        # H, W = size[1], size[0]
        # for i in range(boxes_filt.size(0)):
        #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        #     boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        # implement class-wise nms on these boxes
        nms_start = time.time()
        for i, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
            idx = label.find("(")
            ds = label[idx + 1 : -1]
            label = label[:idx]
            # if float(ds) < BOX_THRESHOLDS[map_class(label.lower())]:
            #     print(f"omitting {label} {i}, with cf value {ds}")
            #     continue
            print(label, map_class(label.lower()), end='\t')
            this_cam_labels.append(map_class(label.lower()))
            this_cam_scores.append(float(ds))
    
        """ Comment out for NMS """
        run_nms = False
        if run_nms:
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
            
            print(boxes_filt.shape)
            print(nms_boxes_filt.shape)

            boxes_filt = nms_boxes_filt
            detection_scores.extend(nms_scores)
            labels.extend(nms_labels)

        else:
            detection_scores.extend(this_cam_scores)
            labels.extend(this_cam_labels)
            nms_labels = this_cam_labels
        """ """

        
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
        # except RuntimeError:
        #     print("<< No objects found. >>")
        #     im_to_ego_mats.append([])
        #     continue

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
                # cam_nums.append(c)
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


                # cam_nums.append(c)

                
                # time.sleep(1000)
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

        # print(len(labels))
        # print(len(detection_scores))
        # print(len(cam_nums))
        # print(np_images.shape[0])

        if len(labels) == 0:
            continue

        assert len(labels) == len(detection_scores)
        # assert len(labels) == len(cam_nums)
        assert len(np_images) == len(labels)

        data["labels"] = labels
        data["detection_scores"] = detection_scores
        # data["cam_nums"] = cam_nums
        

        os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"{f}_data.json"), "w") as outfile:
            json.dump(data, outfile)

        # np.save(os.path.join(OUTPUT_DIR, f"{scene_name}", f"{f}_depth.npy"), np_images)
        # print(np_images)
        pickle.dump(np_images, open(os.path.join(OUTPUT_DIR, f"{f}_masks.pkl"), 'wb'))

        # if sample['next'] != "":
        #     sample = nusc.get('sample', sample['next'])
        #     sample_count += 1

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds for {sample_count} samples.")

