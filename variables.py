import torch

class_names = list('ABEKMHOPCTYX0123456789')
model_plates_path = r'70_epochs_8_batch_plates.pt'
model_chars_path = r'230_280_epochs_16_batch.pt'
yolo_path = ''
font_path = 'Times New Roman.ttf'


model_plates = torch.hub.load(yolo_path, 'custom', path=model_plates_path, source='local')
model_plates.conf = 0.5  # NMS confidence threshold
model_plates.iou = 0.5  # NMS IoU threshold

model_chars = torch.hub.load(yolo_path, 'custom', path=model_chars_path, source='local')
model_chars.conf = 0.7  # NMS confidence threshold
model_chars.iou = 0.7  # NMS IoU threshold

save_path = r'predicted_images/'
filename = 'result_image.png'
dir_images = r'data/images'
