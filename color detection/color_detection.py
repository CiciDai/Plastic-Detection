import numpy as np
import cv2


# font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, 0, 2, 8)



blue = np.uint8([[[255, 0, 0]]])
green = np.uint8([[[0, 255, 0]]])
red = np.uint8([[[0, 0, 255]]])
yellow = np.uint8([[[0, 255, 255]]])
white = np.uint8([[[255, 255, 255]]])
black = np.uint8([[[0, 0, 0]]])
colors = [yellow, blue, green, red]

boundaries = []
for color in colors:
    hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    print(hsvColor)
    # print(hsvColor)
    lower = [hsvColor[0][0][0] - 25, 60, 0]
    upper = [hsvColor[0][0][0] + 25, 255, 255]
    boundaries.append([lower, upper])



#
# print(boundaries)
# def dominant_color(img):
#
# 	pixels = np.float32(img.reshape(-1, 3))
#
# 	n_colors = 3
# 	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
# 	flags = cv2.KMEANS_RANDOM_CENTERS
#
# 	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
# 	_, counts = np.unique(labels, return_counts=True)
#
# 	return palette[np.argmax(counts)]

image = cv2.imread("test_7.png")
fh = image.shape[0]
fw = image.shape[1]
image[np.where((image ==[255, 255, 255]).all(axis=2))] = [255, 0, 0]
# cv2.imshow("orginal)



image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(image_hsv)
#
nonSat = s < 180
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
nonSat = cv2.erode(nonSat.astype(np.uint8), disk)




v2 = v.copy()
v2[nonSat == 0] = 0
glare = v2 > 200
glare = cv2.dilate(glare.astype(np.uint8), disk)
glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, disk)
glare = cv2.morphologyEx(glare, cv2.MORPH_CLOSE, disk)
glare = cv2.dilate(glare.astype(np.uint8), disk)
corrected_hsv = cv2.inpaint(image_hsv, glare, 3, cv2.INPAINT_NS)
corrected_bgr = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("img", cv2.resize(image, (480, int(480/fw*fh))))
cv2.imshow("glare",cv2.resize(255*glare, (480, int(480/fw*fh))))
cv2.imshow("corrected", cv2.resize(corrected_bgr, (480, int(480/fw*fh))))

# target_bgr = dominant_color(image)
# b = int(target_bgr[0])
# g = int(target_bgr[1])
# r = int(target_bgr[2])
# print(target_bgr)
# boundaries = [
# 	([max(0, b-15), max(0, g-15), max(0,r-15)], [min(b+15,255), min(g+15,255), min(r+15,255)])
# ]


mask = np.zeros((image.shape[0], image.shape[1]), dtype = bool)
color_masks = []
for (lower, upper) in boundaries:
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask_color = cv2.inRange(corrected_hsv, lower, upper)
    color_masks.append(mask_color)
    mask = (mask_color == 255) | (mask)

i = 0
for img in color_masks:
    i = i + 1
    cv2.imshow("mask " + str(i), cv2.resize(img, (480, int(480/fw*fh))))


mask = mask.astype(np.uint8)
ones = np.sum(mask)
total = mask.size
print(ones/total)
print(mask.size)
# print(ones)
c = 255*mask
output = cv2.bitwise_and(image, image, mask=c)
# cv2.imshow("final", cv2.resize(c, (480, int(480/fw*fh))))


cv2.imshow("final", cv2.resize(output, (480, int(480/fw*fh))))
cv2.waitKey(0)
#
# while True:
# 	cv2.Smooth(src, src, cv2.CV_BLUR, 3)
# 	hsv = cv2.CreateImage(cv2.GetSize(src), 8, 3)
# 	thr = cv2.CreateImage(cv2.GetSize(src), 8, 1)
# 	cv2.CvtColor(src, hsv, cv2.CV_BGR2HSV)
# 	cv2.SetMouseCallback("camera", on_mouse, 0)
# 	s=cv2.Get2D(hsv, y_co, x_co)
# 	print("H:",s[0],"  	 S:",s[1],"       V:",s[2])
# 	# cv2.PutText(src,str(s[0])+","+str(s[1])+","+str(s[2]), (x_co,y_co),font, (55,25,255))
# 	# cv2.ShowImage("camera", src)
# 	if cv2.WaitKey(10) == 27:
# 		break
