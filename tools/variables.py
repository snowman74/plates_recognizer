import torch

MODEL_PLATES_PATH = r'tools/plates_weigths.pt'
MODEL_CHARS_PATH = r'tools/chars_weights.pt'
YOLO_PATH = r'tools/'
FONT_PATH = r'tools/Times New Roman.ttf'


MODEL_PLATES = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PLATES_PATH, source='local')
MODEL_PLATES.conf = 0.5  # NMS confidence threshold
MODEL_PLATES.iou = 0.5  # NMS IoU threshold

MODEL_CHARS = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_CHARS_PATH, source='local')
MODEL_CHARS.conf = 0.7  # NMS confidence threshold
MODEL_CHARS.iou = 0.7  # NMS IoU threshold

SAVE_PATH = r'predicted_images/'
FILENAME = 'result_image.png'
DIR_IMAGES = r'data/images'
