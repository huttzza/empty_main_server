from main import *

cfg_file = CFG_FILE     # ".yaml"
model_path = MODEL_PATH  # ".pth"
categories = CATEGORIES  # list
image_path = "static/pklots.png"

# -----------------------------------------------

'''
Predictor parameter & default value
    cfg,
    confidence_threshold=0.7,
    show_mask_heatmaps=False,
    masks_per_dim=2,
    min_image_size=224,
    weight_loading=None
'''
predictor = initial_setting(cfg_file, model_path, categories)

image = read_image(image_path, 'local')

result_img, prediction = predictor.run_on_opencv_image(image)

cv2.imwrite('static/detect.jpg', result_img)
print("successfully predict end")

scores = prediction.get_field("scores").tolist()
labels = prediction.get_field("labels").tolist()
labels_name = get_label_name(labels, categories)
bbox = prediction.bbox.tolist()

response = {
    "scores": scores,
    "labels": labels,
    "labels_name": labels_name,
    "bbox": bbox
}

# print(response)
