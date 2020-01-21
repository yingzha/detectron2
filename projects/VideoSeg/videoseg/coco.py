"""
Register a partial COCO dataset with the given categories
"""
import copy
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances


def _get_coco_dict(dataset_name, name_list=["person"]):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    categories = [{"id": id, "name": name} for id, name in enumerate(name_list)]
    # build categories mapping dict
    categories_map = {}
    for id, name in enumerate(MetadataCatalog.get(dataset_name).thing_classes):
        if name in name_list:
            categories_map[id] = name_list.index(name)

    updated_dataset_dicts = []

    for image_id, image_dict in enumerate(dataset_dicts):
        image_dict = copy.deepcopy(image_dict)
        anns_per_image = image_dict["annotations"]
        updated_annotation = []
        for annotation in anns_per_image:
            if annotation["category_id"] not in categories_map:
                continue
            updated_annotation.append(annotation)
        # skip the image if there's no valid label
        if not updated_annotation:
            continue
        image_dict["annotations"] = updated_annotation

        updated_dataset_dicts.append(image_dict)

    return updated_dataset_dicts


def register_coco_person_from_dicts(dataset_name, json_file, image_root, set_="train"):
    # Tricky! Register the original coco first
    register_coco_instances("coco_" + set_, {}, json_file, image_root)

    def get_coco_dict():
        return _get_coco_dict("coco_" + set_, name_list=["person"])
    thing_classes = ["person"]
    DatasetCatalog.register(dataset_name, get_coco_dict)
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
    # use coco evaluator if `eval` mode
    if set_ != "train":
        MetadataCatalog.get(dataset_name).set(evaluator_type="coco")
