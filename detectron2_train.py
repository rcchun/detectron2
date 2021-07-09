import detectron2_instance_fpn.detectron2
from detectron2_instance_fpn.detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2_instance_fpn.detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2_instance_fpn.detectron2.engine import DefaultTrainer
from detectron2_instance_fpn.detectron2.evaluation import COCOEvaluator

from sqlalchemy import create_engine, MetaData, select, and_

db = create_engine('postgresql://postgres:postgres@192.168.10.22:5433/AIMaps_2.0')
metadata = MetaData()
metadata.reflect(bind=db)

# some definition for detectron train dataset
def get_balloon_dicts(data_directory, dats_id):
    pf_train_dataset = metadata.tables['pf_train_dataset']
    stmt_dir = select([pf_train_dataset.c.annotation_json]).where(pf_train_dataset.c.tr_dats_id == dats_id)
    conn = db.connect()
    json_file = conn.execute(stmt_dir).fetchone()[0]

    dataset_dicts = []
    for idx, v in enumerate(json_file.values()):
        record = {}
        
        filename = data_directory + '/' + v["filename"]
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []

        for i in range(len(annos)):
            class_ = annos[i]["region_attributes"]
            anno = annos[i]["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "iscrowd" : 0,
                "category_id": int(class_['classification']) - 1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main(status, IMS_PER_BATCH, MAX_ITER, BASE_LR, dats_id, val_split, angle, resolution, NUM_CLASSES,
         classification_list, data_directory):

    # parameter(model / data directory / num_class / class list)
    application_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    conn = db.connect()

    NUM_CLASSES = NUM_CLASSES
    classification_list = classification_list
    print(classification_list)

    # angle 처리
    if not angle == None:
        ang_list = []
        len_ang = len(angle.split(','))
        for i in range(len_ang):
            ang_list.append(int(angle.split(',')[i]))
    else:
        ang_list = []

    # train dataset enrollment(train/test ratio)
    obj = get_balloon_dicts(data_directory, dats_id)
    datasets = {}
    val_split = val_split / 100
    datasets['train'], datasets['test'] = train_test_split(obj, test_size=val_split)
    file_data_train = datasets['train']
    file_data_test = datasets['test']
    with open(data_directory + '/' + 'datasets_train.json', 'w', encoding="utf-8") as make_file_train:
        json.dump(file_data_train, make_file_train, ensure_ascii=False, indent="\t")
    with open(data_directory + '/' + 'datasets_test.json', 'w', encoding="utf-8") as make_file_test:
        json.dump(file_data_test, make_file_test, ensure_ascii=False, indent="\t")

    data_idx = str(status)

    for d in ["train", "test"]:
        DatasetCatalog.register("d2_ins_fpn_" + d + data_idx, lambda d=d: datasets['{}'.format(d)])
        MetadataCatalog.get("d2_ins_fpn_" + d + data_idx).set(thing_classes=classification_list)


    class CocoTrainer(DefaultTrainer):

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):

            if output_folder is None:
                os.makedirs("detectron2_instance_fpn/coco_eval", exist_ok=True)
                output_folder = "detectron2_instance_fpn/coco_eval"

            return COCOEvaluator(status, dataset_name, cfg, False, output_folder)


    # config parameter
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(application_model))
    cfg.DATASETS.TRAIN = ("d2_ins_fpn_train"+ data_idx,)
    cfg.DATASETS.TEST = ("d2_ins_fpn_test"+ data_idx,)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(application_model)
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = float(BASE_LR)
    
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.STEPS = (100, MAX_ITER)
    # cfg.SOLVER.STEPS = []
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.MAX_ITER = MAX_ITER
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.TEST.EVAL_PERIOD = 50

    cfg.INPUT.ANGLE = ang_list  #ang_list 삽입
    cfg.INPUT.MIN_SIZE_TRAIN = resolution #resize 삽입

    # TRAINING PART
    cfg.OUTPUT_DIR = "./detectron2_instance_fpn/output/" + data_idx
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train(status)

    # weight_file 경로 insert
    pf_train = metadata.tables['pf_train']
    weight_dir_upd = pf_train.update().where(pf_train.c.tr_wk_id == status).values(
        weight_path=cfg.OUTPUT_DIR[2:] + '/model_final.pth')
    conn.execute(weight_dir_upd)