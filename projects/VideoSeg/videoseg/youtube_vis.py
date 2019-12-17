import os
import json
import time
import numpy as np

from pycocotools import mask
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_dicts(path, start_frame=5):
    start = time.time()

    with open(path, "r") as fi:
        labels = json.load(fi)

    # intialize image id and annotation id. Annotation index starts from 1
    img_id, annot_id, final_labels = 0, 0, []

    for i in range(len(labels["videos"])):
        # get all the files for the same video id
        file_names = labels["videos"][i]["file_names"]
        annotation_buffer = []
        # get all the instances for the same video id
        while (annot_id < len(labels["annotations"]) and
                labels["annotations"][annot_id]["video_id"] == i + 1):
            annotation_buffer.append(labels["annotations"][annot_id])
            annot_id += 1
        for j in range(start_frame, len(file_names)):
            record = {}
            record["file_name"] = os.path.join('/'.join(path.split('/')[:-1]),
                                               "JPEGImages", file_names[j])
            record["height"] = labels["videos"][i]['height']
            record["width"] = labels["videos"][i]['width']
            record["image_id"] = img_id
            record["tm1_file_name"] = os.path.join('/'.join(path.split('/')[:-1]),
                                                   "JPEGImages", file_names[j - 1])
            objs = []
            for annotation in annotation_buffer:
                if annotation["bboxes"][j] is not None:
                    bbox = annotation["bboxes"][j]
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        # `category_id` in the original format starts from 1
                        "category_id": annotation["category_id"] - 1,
                        # segmentation in COCO's RLE format
                        "segmentation": annotation["segmentations"][j],
                        }
                    objs.append(obj)

                tm1_mask = np.zeros((record["height"], record["width"]))
                tm1_mask = []
                if annotation["bboxes"][j - 1] is not None:
                    tm1_mask.append(annotation["segmentations"][j - 1])
                    # warning: extremely slow !!
                    # process them as a single-channel class-agnositic mask for simplicity
                    # rle_segment = mask.frPyObjects(rle_segment, rle_segment.get('size')[0],
                    #                               rle_segment.get('size')[1])
                    # clip the mask value to limit it to [0, 1]
                    # tm1_mask = np.clip(tm1_mask + mask.decode(rle_segment), 0, 1)

            record["annotations"] = objs
            record["tm1_mask"] = tm1_mask

            # increment the image id
            img_id += 1
            final_labels.append(record)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Processing time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return final_labels


def get_thing_classes(path):
    with open(path, "r") as fi:
        labels = json.load(fi)
    # obtain `thing_classes`
    thing_classes = [category["name"] for category in labels["categories"]]

    return thing_classes


def register_youtube_vis_from_dicts(path, dataset_name, eval=False):
    def get_youtube_vis_labels():
        return get_dicts(path)
    thing_classes = get_thing_classes(path)
    DatasetCatalog.register(dataset_name, get_youtube_vis_labels)
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
    # use coco evaluator if `eval` mode
    if eval:
        MetadataCatalog.get(dataset_name).set(evaluator_type="coco")
