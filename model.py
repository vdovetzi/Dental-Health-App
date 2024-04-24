import numpy as np
import cv2
from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import os
import shutil

MIN_CONF = 0.5
CLASSES = ['C Lower', 'C Upper', 'CI Lower', 'CI Upper', 'LI Lower', 'LI Upper', 'M1 Lower', 'M1 Upper', 'M2 Lower',
           'M2 Upper', 'P1 Lower', 'P1 Upper', 'P2 Lower', 'P2 Upper']
NUM_CPUS = 5


class MouthSegmentationModel:
    """
    Mouth Segmentation Model for detecting and segmenting teeth from photos.

    Attributes:
        model_detect: YOLO model for tooth detection.
        model_segment: FastSAM model for tooth segmentation.
    """

    def __init__(self):
        """
        Initializes the MouthSegmentationModel.
        """
        self.model_detect = YOLO('weights/detect/best.pt')
        # Select FastSAM model based on system cores count
        if cv2.getNumberOfCPUs() > NUM_CPUS:
            self.model_segment = FastSAM('weights/segment/FastSAM-x.pt')
        else:
            self.model_segment = FastSAM('weights/segment/FastSAM-s.pt')

    def get_teeth_from_photo(self, result_of_detection, image, save_segmentation_result=False, path=None):
        """
        Extracts teeth from the detected bounding boxes in the image.

        Args:
            result_of_detection: Detection result containing bounding box information.
            image: Input image containing teeth.
            save_segmentation_result (bool): Whether to save the segmentation result images.
            path (str): The path to save the segmentation result images.

        Returns:
            dict: A dictionary containing segmented teeth images.
        """
        teeth = dict()
        i = 0
        for result in result_of_detection:
            boxes = result.boxes
            conf_and_tooth = boxes.data[:, 4:6].tolist()
            if conf_and_tooth[0][0] > MIN_CONF:
                left, top, right, bottom = map(float, boxes[0].xyxy[0])
                everything_results = self.model_segment(source=image, device='cpu', retina_masks=True,
                                                        imgsz=image.shape[0], conf=0.4, iou=0.9)
                prompt_process = FastSAMPrompt(image, everything_results, device='cpu')
                ann = prompt_process.box_prompt(bbox=[left, top, right, bottom])
                mask_of_tooth = ann[0].masks.xy[0]
                masked_image = self.apply_mask(image, mask_of_tooth)
                teeth[result.names[int(conf_and_tooth[0][1])]] = masked_image[int(top):int(bottom),
                                                                 int(left):int(right)]
                if save_segmentation_result:
                    prompt_process.plot(annotations=ann, output=f'{path}/tooth{i + 1}')
                    i += 1
        return teeth

    @staticmethod
    def apply_mask(image, mask):
        """
        Applies a mask to the image.

        Args:
            image: Input image.
            mask: Mask to be applied.

        Returns:
            numpy.ndarray: Masked image.
        """
        binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        contours = [np.array(mask, dtype=np.int32)]
        cv2.fillPoly(binary_mask, contours, 255)
        masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
        return masked_image

    def predict(self, source, save_detection_result=False, save_segmentation_result=False, path=None):
        """
        Predicts and processes input images.

        Args:
            source (list): List of input images.
            save_detection_result (bool): Whether to save the detection result images.
            save_segmentation_result (bool): Whether to save the segmentation result images.
            path (str): The path to save the segmentation result images.

        Returns:
            list: Segmented teeth images from each input image.
        """
        try:
            if len(source) < 2:
                raise ValueError("Too few arguments provided. Expected at least 2 images.")
            elif save_segmentation_result and path is None:
                raise ValueError("path variable cannot be set as None, while save_segmentation_result equals True "
                                 "-> you must specify the path variable")
            results = self.model_detect.predict(source=source, save=save_detection_result)
            teeth1 = self.get_teeth_from_photo(results[0], source[0], save_segmentation_result, path)
            teeth2 = self.get_teeth_from_photo(results[1], source[1], save_segmentation_result, path)
            for class_name in CLASSES:
                if class_name not in teeth1 and class_name in teeth2:
                    del teeth2[class_name]
                elif class_name in teeth1 and class_name not in teeth2:
                    del teeth1[class_name]

            if len(teeth1) != len(teeth2):
                raise AssertionError("teeth1.size() == teeth2.size()")

            teeth = [list(teeth1.values()), list(teeth2.values())]

            if save_segmentation_result:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"There is no such folder as {path}")
                for i in range(len(teeth[0])):
                    if teeth[0][i] is not None:
                        cv2.imwrite(f'{path}/tooth1_{i + 1}.jpg', teeth[0][i])
                for i in range(len(teeth[1])):
                    if teeth[1][i] is not None:
                        cv2.imwrite(f'{path}/tooth2_{i + 1}.jpg', teeth[1][i])
            return teeth
        except Exception as e:
            print(f"An error occurred while prediction: {e}")

    def __call__(self, *args, **kwargs):
        """
        Invokes the MouthSegmentationModel to process input images.

        Args:
            *args: Input images.

        Returns:
            list: Segmented teeth images from each input image.
        """
        try:
            if len(args) < 2:
                raise ValueError("Too few arguments provided. Expected at least 2 images.")
            results = self.model_detect.predict(source=args, save=False)
            teeth1 = self.get_teeth_from_photo(results[0], args[0])
            teeth2 = self.get_teeth_from_photo(results[1], args[1])
            for class_name in CLASSES:
                if class_name not in teeth1 and class_name in teeth2:
                    del teeth2[class_name]
                elif class_name in teeth1 and class_name not in teeth2:
                    del teeth1[class_name]

            if len(teeth1) != len(teeth2):
                raise AssertionError("teeth1.size() == teeth2.size()")

            return [list(teeth1.values()), list(teeth2.values())]
        except Exception as e:
            print(f"An error occurred while prediction: {e}")

    @staticmethod
    def clean_save_directories(path):
        """
        Removes the specified folder and its contents.

        Args:
            path (str): The path to the folder to be removed.
        """
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Folder '{path}' and its contents removed successfully.")
            if os.path.exists('./runs'):
                shutil.rmtree('./runs')
        except Exception as e:
            print(f"An error occurred while removing directories: {e}")
