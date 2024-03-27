import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load a pre-configured model and weights
cfg = get_cfg()
cfg.merge_from_file("path/to/config.yaml")
cfg.MODEL.WEIGHTS = "path/to/model_weights.pth"
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
predictor = DefaultPredictor(cfg)

# Perform panoptic segmentation on an image
outputs = predictor(image)
panoptic_seg, segments_info = outputs["panoptic_seg"]
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
