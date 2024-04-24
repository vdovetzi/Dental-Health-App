from model import MouthSegmentationModel

import cv2

image1 = cv2.imread("dark_photo.jpg", cv2.IMREAD_COLOR)
image2 = cv2.imread("dark_photo.jpg", cv2.IMREAD_COLOR)

resized_image1 = cv2.resize(image1, (320, 320))
resized_image2 = cv2.resize(image2, (320, 320))
model = MouthSegmentationModel()
# arrays_of_teeth = model.predict([resized_image1,resized_image2],True,True, './teeth')

# a = arrays_of_teeth[0]
# b = arrays_of_teeth[1]
# # arrays_of_teeth = determine_tooth_color(image2)
# # print(len(arrays_of_teeth), len(arrays_of_teeth[0]))
#
# # for i in range(len(arrays_of_teeth)):
# #     for j in range(len(arrays_of_teeth[i])):
# #         cv2.imwrite(f'./teeth/tooth{i+1}.jpg', arrays_of_teeth[i][j])
# #
# for i in range(len(a)):
#     if not a[i] is None:
#         cv2.imwrite(f'./teeth/tooth1_{i + 1}.jpg', a[i])
# for i in range(len(b)):
#     if not b[i] is None:
#         cv2.imwrite(f'./teeth/tooth2_{i + 1}.jpg', b[i])

model.clean_save_directories('./teeth')
