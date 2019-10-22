from __future__ import print_function
import cv2
import numpy as np

backSub = cv2.createBackgroundSubtractorMOG2()
tracker = cv2.MultiTracker_create()
# backSub = cv2.createBackgroundSubtractorKNN()
capture = cv2.VideoCapture('newv.mov')
fw = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(capture.get( cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = 480
scale = frame_width/fw
count = 0

if not capture.isOpened:
    print('Unable to open file')
    exit(0)
while True:
    ret, frame = capture.read()

    if frame is None:
        break
    bs_frame = frame[:,0:int(fw/2)]
    # print(bs_frame.shape)
    # print(frame.shape)
    # update background
    fgMask = backSub.apply(bs_frame, learningRate=-2)

    # method 1
    blur = cv2.GaussianBlur(fgMask, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
    # dilated = cv2.dilate(thresh, None, iterations=3)
    stack1 = np.vstack((fgMask, blur, thresh))
    cv2.imshow('mask, blur, thresh', cv2.resize(stack1, (frame_width, int(scale * stack1.shape[0]))))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dil = np.ones((5, 5), np.uint8)
    # blur = cv2.GaussianBlur(fgMask, (10, 10), 0)
    erosion = cv2.erode(thresh, kernel_dil, iterations=1)
    fgMask1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # fgMask2 = cv2.morphologyEx(fgMask1, cv2.MORPH_CLOSE, kernel)


    # _, thresh = cv2.threshold(fgMask, 120, 255, cv2.THRESH_BINARY)
    # lower = np.array([255])
    # upper = np.array([255])
    # fgMask = cv2.inRange(fgMask1, lower, upper)



    dilated = cv2.dilate(fgMask1, kernel_dil, iterations=2)
    stack2 = np.vstack((erosion, fgMask1, dilated))

    cv2.imshow('erosion, open, corrected', cv2.resize(stack2, (frame_width, int(scale * stack2.shape[0]))))


    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    # mask =
    # blob = cv2.bitwise_and(frame, frame, mask=mask)
    _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 500:
            continue
        # tracking all boxes in left half of the frame
        if(x < int(fw/2)):
            bounding_boxes.append((x, y, w, h))
            # print(boxes[id])
        # if area < 5% of frame area, do not draw contour
        # if(w*h < 0.05*frame_width*frame_height):
        #     continue
        #     show bs boxes in left half of img
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame, "Status: {}".fomat('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 0, 255), 3)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    # print(boxes.size)
    # print(boxes)

    for box in bounding_boxes:
        if(box[0]>(fw/2-100)):
            _ = tracker.add(cv2.TrackerMIL_create(), frame, box)

    # get (x, y, w, h) for each tracking object
    # objects = tracker.getObjects()
    # # sorted_objects = sorted(objects, key = getKey)
    # sorted_objects = sorted(objects,key = lambda tup: tup[0])
    #
    #
    #
    #
    # print("new bounding boxes: ")
    # print(bounding_boxes)
    # i = 0
    #
    # new_object_list = []
    # while i <= len(sorted_objects)-1:
    #     if (len(new_object_list) != 0):
    #         (xA, yA, wA, hA) = new_object_list[-1]
    #     else:
    #         (xA, yA, wA, hA) = sorted_objects[i]
    #         new_object_list.append((xA, yA, wA, hA))
    #         i = i+1
    #         if(i == len(sorted_objects)):
    #             continue
    #
    #     (xB, yB, wB, hB) = sorted_objects[i]
    #
    #     width = min(xA+wA, xB + wB) - xB
    #     height = min(yA+hA, yB + hB) - max(yA, yB)
    #
    #     # no intersection
    #     if(width <=0 or height <=0):
    #         i = i+1
    #         new_object_list.append((xB, yB, wB, hB))
    #         continue
    #
    #     interArea = width * height
    #     area_combined = wA * hA + wB * hB - interArea
    #     iou = interArea/(area_combined + 0.1)
    #     # delete boxA from tracking list if iou >= 0.5
    #     if(iou < 0.1):
    #         new_object_list.append((xB, yB, wB, hB))
    #         print(new_object_list)
    #     else:
    #         new_object_list[-1] = (xB, yB, wB, hB)
    #
    #     i = i+1
    #
    #
    # print("previous tracking boxes:")
    # print(sorted_objects)
    # tracker.objects = new_object_list
    # # tracker = cv2.MultiTracker_create()
    #
    # print("after clear")
    # print(tracker.getObjects())
    # print("new list:")
    # print(new_object_list)

    # for box in new_object_list:
    #     _ = tracker.add(cv2.TrackerMIL_create(), frame, box)
    # tracker.add(cv2.TrackerMIL_create(), frame, new_object_list)
    _, boxes = tracker.update(frame)

    for newbox in boxes:
        (x, y, w, h) = newbox
        if (x >= int(fw / 2) and x<(fw-100)):
            print(newbox)
            count = count + 1
            print("here")
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            # cv2.putText(frame, count, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # middle line
    cv2. line(frame, (int(fw/2), 0), (int(fw/2), fh), (0, 200, 200), 2)

    # finish line
    cv2.line(frame, (int(0.9*fw), 0), (int(0.9*fw), fh), (0, 200, 200), 2)
    objects = tracker.getObjects()
    count = 0
    for object in objects:
        if(object[0]>=0.9*fw):
            count = count + 1
    cv2.putText(frame, "Detecting", (int(0.2*fw), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Tracking", (int(0.65 * fw), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Count: " + str(count), (int(0.9 * fw + 30), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)
    cv2.imshow('Frame', cv2.resize(frame, (720, int(720 / fw * fh))))

    # cv2.imshow('FG Mask', fgMask)
    # cv2.imshow('corrected FG Mask', cv2.resize(fgMask, (width, int(scale*fh))))
    # cv2.imshow('erosion', cv2.resize(erosion, (width, int(scale * fh))))
    # cv2.imshow('dilated', cv2.resize(dilated, (width, int(scale * fh))))


    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break