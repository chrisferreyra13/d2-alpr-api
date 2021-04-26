import argparse

from transformers.detectron2.load_test_images import LoadTestImages
from transformers.detectron2.set_config import SetConfig
from transformers.detectron2.samples_from_folder import SamplesFromFolder
from transformers.detectron2.predictor import Predictor


class LicenseplateDetector():
    def __init__(self):
        self.split_test='test' 

        self.args=argparse.Namespace(
                    input='assets/datasets/licenseplates',
                    config_file='configs/lp_faster_rcnn_R_50_FPN_3x.yaml',
                    samples=1,
                    confidence_threshold=0.85,
                    opts=['MODEL.WEIGHTS', 'output/model_final.pth']
                    )

    def detect(self, features: dict):

        if self.args.confidence_threshold is not None:
            # Set score_threshold for builtin models
            self.args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
            self.args.opts.append(str(self.args.confidence_threshold))
            self.args.opts.append('MODEL.RETINANET.SCORE_THRESH_TEST')
            self.args.opts.append(str(self.args.confidence_threshold))
        
        # Create pipeline steps
        load_test_images=LoadTestImages(self.args.input,self.split_test)    # TODO: Tener automatizado para, mediante un arg, saber que Loader usar (VOC, COCO, etc...)
        
        samples_from_folder = SamplesFromFolder(self.args.samples)
        set_config=SetConfig(self.args)
        predictor=Predictor()

        # Create licenseplate detection pipeline

        prediction_pipeline = (
            load_test_images  |
            samples_from_folder   | #TODO: Mejorar el orden
            set_config  |
            predictor

        )
        result={}
        try:
            for r in prediction_pipeline:
                result=r
                
        except StopIteration:
            print("Error in prediction")
        
        
        
        pred_boxes=[]
        output_pred_boxes=result["prediction"]["instances"].pred_boxes

        for box in output_pred_boxes:
            pred_boxes.append(box.cpu().numpy().tolist())
        
        return {
            "prediction":{
                "pred_boxes":pred_boxes
                }
            }

