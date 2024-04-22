from model import determine_tooth_color
from model import demo_determine_tooth_color
import cv2

image = cv2.imread("teeth_example1.jpeg", cv2.IMREAD_COLOR)


#demo
# teeth = demo_determine_tooth_color(image)
# print(teeth)

#future
teeth = determine_tooth_color(image)
for i in range(len(teeth)):
    cv2.imwrite(f'./tooth{i+1}.jpg', teeth[i])


















