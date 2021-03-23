import cv2.cv2 as cv2
import numpy as np
import time


start = time.time()
src_image = cv2.imread("./R3104.jpg", cv2.IMREAD_GRAYSCALE)
end = time.time()
print(end - start)

start = time.time()
cv2.imwrite("./R3105.jpg", src_image)
end = time.time()

print(end - start)
#
# edge = cv2.Canny(src_image, 30, 250)
# cv2.imshow("edge", edge)
#
# s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# d = cv2.dilate(edge, s, iterations=2)
# cv2.imshow('dilate', d)
#
# lines = cv2.HoughLinesP(d, 1, np.pi/180, 50)
# out = cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
# print(lines.shape)
# for line in lines:
#     line = line[0]
#     print(line)
#     cv2.line(out, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1)
# cv2.imshow("out", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # accumulator, accuDict =