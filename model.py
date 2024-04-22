import numpy as np
from ultralytics import YOLO
from ultralytics import FastSAM
import cv2
from ultralytics.utils.plotting import Annotator
from ultralytics.models.fastsam import FastSAMPrompt
import matplotlib.pyplot as plt
import ultralytics.engine.results

MIN_CONF = 0.5
A3 = 3
A35 = 3.5
A4 = 4

# you can turn this option on to check for different issues with the pretrained model, but it takes some time
# ultralytics.checks()

def load_image(photo_encoded):
    image = cv2.imdecode(photo_encoded, cv2.IMREAD_COLOR)
    return image


def apply_mask(image, mask):
    binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    contours = [np.array(mask, dtype=np.int32)]
    cv2.fillPoly(binary_mask, contours, 255)
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
    return masked_image

def determine_tooth_color(image):
    # image = load_image(image[0]) # need to be turned on when sill be connection with API
    array_of_teeth = []
    model_detect = YOLO('weights/detect/best.pt')
    model_segment = FastSAM('weights/segment/FastSAM-x.pt')
    results = model_detect.predict(source=image, save=True)  # you can set "save = True" to see the exact detection
    for result in results:
        boxes = result.boxes
        confidences = result.boxes.conf.numpy()
        for i in range(len(confidences)):
            if confidences[i] > MIN_CONF:
                left, top, right, bottom = map(float, boxes[i].xyxy[0])
                everything_results = model_segment(source=image, device='cpu', retina_masks=True, imgsz=image.shape[0], conf=0.4, iou=0.9)
                prompt_process = FastSAMPrompt(image, everything_results, device='cpu')
                ann = prompt_process.box_prompt(bbox=[left, top, right, bottom])
                prompt_process.plot(annotations=ann, output=f'./result/segmented_tooth{i}') # debug
                mask_of_tooth = ann[0].masks.xy[0]
                masked_image = apply_mask(image,mask_of_tooth)
                array_of_teeth.append(masked_image[int(top):int(bottom), int(left):int(right)])
                # break # debug with one tooth
    return array_of_teeth


def demo_determine_tooth_color(image):
    # image = load_image(image[0])

    model = YOLO(f'weights/detect/demo.pt')
    results = model.predict(source=image, save=True)  # you can set "save = True" to see the exact detection
    mean_prob = 0
    total_num = 0
    for result in results:
        names = result.names
        confidences = result.boxes.conf.numpy()
        labels = result.boxes.cls.numpy()
        for i in range(len(labels)):
            if confidences[i] > MIN_CONF:
                total_num += 1
                float_colour = float(names[labels[i]][1:])
                mean_prob += float_colour
    ans = mean_prob / total_num
    dist_1 = abs(A35 - ans)
    dist_2 = abs(A3 - ans)
    dist_3 = abs(A4 - ans)
    if dist_1 < dist_2 and dist_1 < dist_3:
        ans = A35
    if ans != A35:
        ans = round(ans)
    return "A" + str(ans)

