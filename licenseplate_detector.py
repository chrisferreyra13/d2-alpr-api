import argparse
import cv2

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.engine import DefaultPredictor



class LicenseplateDetector():
    def __init__(self):
        self.split_test='test' 

        self.args=argparse.Namespace(
                    input='assets/datasets/licenseplates/images/04ow1.jpg',
                    config_file='configs/lp_faster_rcnn_R_50_FPN_3x.yaml',
                    samples=1,
                    confidence_threshold=0.85,
                    opts=['MODEL.WEIGHTS', 'output/model_final.pth']
                    )

    def detect(self, img: dict):

        if self.args.confidence_threshold is not None:
            # Set score_threshold for builtin models
            self.args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
            self.args.opts.append(str(self.args.confidence_threshold))
            self.args.opts.append('MODEL.RETINANET.SCORE_THRESH_TEST')
            self.args.opts.append(str(self.args.confidence_threshold))
        
        
        cfg = get_cfg()
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        cfg.freeze()
        default_setup(cfg, self.args)
        
        predictor = DefaultPredictor(cfg)
        img = cv2.imread(self.input)
        prediction = predictor(img)
                   
        pred_boxes=[]
        output_pred_boxes=prediction["prediction"]["instances"].pred_boxes

        for box in output_pred_boxes:
            pred_boxes.append(box.cpu().numpy().tolist())
        
        return {
            "prediction":{
                "pred_boxes":pred_boxes
                }
            }

