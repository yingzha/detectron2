import os
import json

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

#DETECTRON2_FORAMT = {'file_name': '',
#                     'height': 0,
#                     'width': 0,
#                     'annotations':
#                        {
#                         "bbox": [0, 0, 0, 0],
#                         "bbox_mode": BoxMode.XYXY_ABS,
#                         "category_id": 0,
#                        }
#                }


def write_to_dict(input_files):
    detectron2_labels = []

    for input_file in input_files:
        with open(input_file, 'r') as fi:
            labels = json.load(fi)

        for i, label in enumerate(labels['samples']):
            record = {}
            record['file_name'] = label['x']['img_path']
            record['height'] = label['x']['img_height']
            record['width'] = label['x']['img_width']
            # `image_id` is used for COCOEvaluator
            record['image_id'] = i

            objs = []
            for bbox in label['y']['label_boxes']:
                bbox = bbox['bounding_box']
                top, left, bottom, right = bbox['top'], bbox['left'], bbox['bottom'], bbox['right']
                obj = {
                   'bbox': [left, top, right, bottom],
                   'bbox_mode': BoxMode.XYXY_ABS,
                   'category_id': 0,
                    }
                objs.append(obj)
            record["annotations"] = objs

        detectron2_labels.append(record)

    return detectron2_labels


def register_tablebank_from_dict(input_files, dataset_name, eval=False):
    def get_detectron2_labels():
        return write_to_dict(input_files)
    DatasetCatalog.register(dataset_name, get_detectron2_labels)
    MetadataCatalog.get(dataset_name).set(thing_classes=["table"])
    if eval:
        MetadataCatalog.get(dataset_name).set(evaluator_type="coco")
