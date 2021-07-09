import detectron2_instance_fpn.detectron2
from detectron2_instance_fpn.detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
from random import *
from skimage import io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2_instance_fpn.detectron2.engine import DefaultPredictor
from detectron2_instance_fpn.detectron2.config import get_cfg
from detectron2_instance_fpn.detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2_instance_fpn.detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import build_detection_test_loader
from detectron2_instance_fpn.detectron2.evaluation import COCOEvaluator, inference_on_dataset

from sqlalchemy import create_engine, MetaData, select, and_

db = create_engine('postgresql://postgres:postgres@192.168.10.22:5433/AIMaps_2.0')
metadata = MetaData()
metadata.reflect(bind=db)

# some definition for detectron train dataset
def get_balloon_dicts_train(img_dir):
    json_file = os.path.join(img_dir, "datasets_train.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        filename = v["file_name"]
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["annotations"]
        objs = []
        for i in range(len(annos)):
            obj = {
                "bbox": annos[i]["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": annos[i]["segmentation"],
                "iscrowd": 0,
                "category_id": annos[i]['category_id'],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_balloon_dicts_test(img_dir):
    json_file = os.path.join(img_dir, "datasets_test.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        filename = v["file_name"]
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["annotations"]
        objs = []
        for i in range(len(annos)):
            obj = {
                "bbox": annos[i]["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": annos[i]["segmentation"],
                "iscrowd": 0,
                "category_id": annos[i]['category_id'],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main(ts_wk_id, tr_wk_id, threshold, resolution, NUM_CLASSES, data_directory, classification_list):
    # parameter
    application_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    conn = db.connect()
    NUM_CLASSES = NUM_CLASSES
    classification_list = classification_list
    print(classification_list)

    # train / test dataset enrollment
    data_idx = str(tr_wk_id)
    data_idx_test = str(ts_wk_id+tr_wk_id)

    for d in ["train"]:
        DatasetCatalog.register("d2_ins_fpn_" + d + data_idx_test, lambda d=d: get_balloon_dicts_train(data_directory))
        DatasetCatalog.get("d2_ins_fpn_" + d + data_idx_test)

    for d in ["test"]:
        DatasetCatalog.register("d2_ins_fpn_" + d + data_idx_test, lambda d=d: get_balloon_dicts_test(data_directory))
        MetadataCatalog.get("d2_ins_fpn_" + d + data_idx_test).set(thing_classes=classification_list)

    test_dataset_dicts = DatasetCatalog.get("d2_ins_fpn_test" + data_idx_test)
    microcontroller_metadata = MetadataCatalog.get("d2_ins_fpn_test" + data_idx_test)

    # config parameter
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(application_model))
    cfg.DATASETS.TRAIN = ("d2_ins_fpn_train" + data_idx_test,)
    cfg.DATASETS.TEST = ("d2_ins_fpn_test" + data_idx_test,)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.INPUT.MIN_SIZE_TEST = resolution

    # INFERENCE PART
    cfg.OUTPUT_DIR = "./detectron2_instance_fpn/output/" + data_idx
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    predictor = DefaultPredictor(cfg)
    result_directory = 'detectron2_instance_fpn/data/result/' + data_idx
    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)
    
    start_time = time.time()
    prog_rate = 0
    objs = {}
    for idx, d in enumerate(test_dataset_dicts):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        file_name = d["file_name"].split('/')[-1]
        print(file_name)
        
        v = Visualizer(im[:, :, ::-1],
                       metadata=microcontroller_metadata, 
                       scale=1, 
                       instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels
        )
        v, boxes, masks, labels = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        io.imsave(result_directory + '/' + file_name.split('.')[0] + '_result.jpg',
                  cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

        k = 0
        obj = {}
        for m_id, mask_id in enumerate(masks):
            mask = np.array(mask_id._mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            label = labels[m_id].split(' ')[0]
            confi_score = labels[m_id].split(' ')[1]

            for i in range(len(contours)):
                if len(contours[i]) > 3:
                    x0, y0 = zip(*np.squeeze(contours[i]))
                    # x_0 = ndimage.gaussian_filter(x0, sigma=3.0, order=0)
                    # y_0 = ndimage.gaussian_filter(y0, sigma=3.0, order=0)
                    polygon = []
                    k += 1
                    for m in range(len(x0)):
                        # polygon.append([x0[m], -y0[m]])
                        polygon.append([float(x0[m]), float(y0[m])])
                    area = int(cv2.contourArea(contours[i]))
            obj.update({"{}".format(m_id): {'class': label,
                                            'confidence': float(int(confi_score[:-1]) / 100),
                                            'area': area,
                                            'mask': polygon}})
        objs.update({file_name.split('.')[0] + '_{}'.format(idx): obj})

        # filename[5:] for window version remove [5:] for linux version
        epoch_batches_left = len(test_dataset_dicts) - (idx + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (idx + 1))
        print('ETA : ',time_left)

        prog_rate = int(((idx+1)/len(test_dataset_dicts))*100)
        print(prog_rate)
        pf_test = metadata.tables['pf_test']
        udt = pf_test.update().where(pf_test.c.ts_wk_id==ts_wk_id).values(
                                                progress_rate = prog_rate
                                                )
        stmt_fp = select([pf_test.c.force_pause]).where(and_(pf_test.c.ts_wk_id == ts_wk_id,
                                                            pf_test.c.tr_wk_id == tr_wk_id))
        conn.execute(udt)
        res_fp = conn.execute(stmt_fp).fetchone()
        if res_fp[0]:
            ins = pf_test.update().where(and_(pf_test.c.ts_wk_id == ts_wk_id,
                                              pf_test.c.tr_wk_id == tr_wk_id)).values(
                status=3)
            conn.execute(ins)
            break

    class CocoTrainer(DefaultTrainer):

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):

            if output_folder is None:
                os.makedirs("detectron2_instance_fpn/coco_eval", exist_ok=True)
                output_folder = "detectron2_instance_fpn/coco_eval"

            return COCOEvaluator(tr_wk_id, dataset_name, cfg, False, output_folder)

    with open(result_directory + '/' + 'output.json', 'w', encoding="utf-8") as make_file:
        json.dump(objs, make_file, ensure_ascii=False, indent="\t")

#   EVALUATION PART
    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load()
    evaluator = COCOEvaluator(tr_wk_id, "d2_ins_fpn_test" + data_idx_test,
                              cfg, False, output_dir="detectron2_instance_fpn/coco_eval")
    val_loader = build_detection_test_loader(cfg, "d2_ins_fpn_test" + data_idx_test)
    inference_on_dataset(trainer.model, val_loader, evaluator)

    return objs, result_directory
