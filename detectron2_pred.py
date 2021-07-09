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
import json
from collections import OrderedDict
import shapefile
import scipy.ndimage as ndimage
from rdp import rdp
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2_instance_fpn.detectron2.engine import DefaultPredictor
from detectron2_instance_fpn.detectron2.config import get_cfg
from detectron2_instance_fpn.detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2_instance_fpn.detectron2.engine import DefaultTrainer
from detectron2_instance_fpn.detectron2.utils.visualizer import ColorMode
from detectron2.data import build_detection_test_loader
from detectron2_instance_fpn.detectron2.evaluation import COCOEvaluator, inference_on_dataset

from sqlalchemy import create_engine, MetaData, select, and_

db = create_engine('postgresql://postgres:postgres@192.168.10.22:5433/AIMaps_2.0')
metadata = MetaData()
metadata.reflect(bind=db)

# download coordinate system (prj file)
def getWKT_PRJ (epsg_code):
    import urllib.request
    wkt = urllib.request.urlopen("http://spatialreference.org/ref/sr-org/{0}/prj/".format(epsg_code))
    remove_spaces = wkt.read().decode("utf-8").replace(" ","")
    output = remove_spaces.replace("\n", "")
    return output


def main(pr_wk_id, tr_wk_id, inp, file_list, threshold, len_list, NUM_CLASSES, classification_list):
    # parameter
    application_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    conn = db.connect()

    NUM_CLASSES = NUM_CLASSES
    classification_list = classification_list
    print('class_list : {}'.format(classification_list))

    data_idx = str(tr_wk_id)
    test_dataset_dicts = inp

    data_idx_test = str(pr_wk_id + tr_wk_id)
    microcontroller_metadata = MetadataCatalog.get("d2_ins_fpn_pred" + data_idx_test).set(
        thing_classes=classification_list)
    # config parameter
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(application_model))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    # INFERENCE PART
    cfg.OUTPUT_DIR = "./detectron2_instance_fpn/output/" + data_idx
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    predictor = DefaultPredictor(cfg)
    result_directory = 'detectron2_instance_fpn/output/result/' + data_idx
    if not os.path.isdir('detectron2_instance_fpn/output/result'):
        os.mkdir('detectron2_instance_fpn/output/result')
    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)

    start_time = time.time()
    prog_rate = 0
    objs = {}
    for idx, d in enumerate(test_dataset_dicts):
        ii = -1
        for ix, r in enumerate(d):
            im = r
            outputs = predictor(im)
            file_name = file_list[idx]
            print(file_name)
            v = Visualizer(im[:, :, ::-1],
                           metadata=microcontroller_metadata,
                           scale=1,
                           instance_mode=ColorMode.SEGMENTATION,
                           )
            v, boxes, masks, labels = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            if ix % (len_list[idx]+1) == 0:
                ii += 1
            jj = ix % (len_list[idx]+1)
            io.imsave(result_directory + '/' + file_name.split('.')[0] + '_{}_{}_{}_result.jpg'.format(idx, ii, jj),
                      cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            obj = {}
            for m_id, mask_id in enumerate(masks):
                mask = np.array(mask_id._mask)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                label = labels[m_id].split(' ')[0]
                confi_score = labels[m_id].split(' ')[1]

                for i in range(len(contours)):
                    epsilon = 0.02 * cv2.arcLength(contours[i], True)
                    approx = cv2.approxPolyDP(contours[i], epsilon, True)
                    if len(approx) > 3:
                        x0, y0 = zip(*np.squeeze(approx))
                        # x0, y0 = zip(*np.squeeze(contours[i]))
                        # x_0 = ndimage.gaussian_filter(x0, sigma=1.0, order=0)
                        # y_0 = ndimage.gaussian_filter(y0, sigma=1.0, order=0)
                        polygon = []
                        for m in range(len(x0)):
                            # polygon.append([x0[m], -y0[m]])
                            polygon.append([float(x0[m]), float(y0[m])])
                        # simple_polygon = rdp(polygon, epsilon=0.5)
                        area = int(cv2.contourArea(contours[i]))
                obj.update({"{}".format(m_id): {'class': label,
                                                'confidence': float(int(confi_score[:-1]) / 100),
                                                'area': area,
                                                'mask': polygon}})
            objs.update({file_name + '_{}_{}_{}'.format(idx, ii, jj): obj})
            # writing coordinate system(generating project file)
            # prj = open(result_directory + '/' + file_name.split('.')[0] + '_{}_{}_{}_mask.prj'.format(idx, ii, jj), "w")
            # epsg = getWKT_PRJ("8862")
            # prj.write(epsg)
            # prj.close()

            epoch_batches_left = len(test_dataset_dicts) - (idx + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (idx + 1))
            print('ETA : ', time_left)

            prog_rate = ((idx / len(test_dataset_dicts)) + ((1 / len(test_dataset_dicts)) * ((ix+1) / len(d)))) * 100
            print(int(prog_rate))
            pf_pred = metadata.tables['pf_pred']
            udt = pf_pred.update().where(pf_pred.c.pr_wk_id == pr_wk_id).values(progress_rate=prog_rate)
            stmt_fp = select([pf_pred.c.force_pause]).where(and_(pf_pred.c.pr_wk_id == pr_wk_id,
                                                                 pf_pred.c.tr_wk_id == tr_wk_id))
            conn.execute(udt)
            res_fp = conn.execute(stmt_fp).fetchone()
            if res_fp[0]:
                ins = pf_pred.update().where(and_(pf_pred.c.pr_wk_id == pr_wk_id,
                                                  pf_pred.c.tr_wk_id == tr_wk_id)).values(
                    status=3)
                conn = db.connect()
                conn.execute(ins)
                break

    with open(result_directory + '/' + 'output.json', 'w', encoding="utf-8") as make_file:
        json.dump(objs, make_file, ensure_ascii=False, indent="\t")
    return objs, result_directory
