from __future__ import print_function
import cv2
import numpy as np

# backSub frame percentage of orginal frame
bs_factor = 0.65
tk_factor = 0.5
interval_factor = 0.15
finish_factor = 0.9
def ioU(box1, box2):
    (xA, yA, wA, hA) = box1
    (xB, yB, wB, hB) = box2
    width = min(xA + wA, xB + wB) - xB
    height = min(yA + hA, yB + hB) - max(yA, yB)

    # no intersection
    if (width <= 0 or height <= 0):
        return 0

    interArea = width * height
    area_combined = wA * hA + wB * hB - interArea
    iou = interArea / (area_combined + 0.1)
    overlap = max(interArea/(wA*hA), interArea/(wB*hB))
    return iou


def findUniqueObjects(object_list):
    target_list = [];
    for box in object_list:
        if(box[0] > (finish_factor-tk_factor)*fw):
            target_list.append(box)

    sorted_list = sorted(target_list, key=lambda tup: tup[0])
    print(len(target_list))
    i = 0
    unique_object_list = []
    while i <= len(sorted_list)-1:
        if (len(unique_object_list) != 0):
            (xA, yA, wA, hA) = unique_object_list[-1]

        else:
            (xA, yA, wA, hA) = sorted_list[i]
            unique_object_list.append((xA, yA, wA, hA))
            i = i+1
            if(i == len(sorted_list)):
                break

        iou = ioU(unique_object_list[-1], sorted_list[i])
        # consider as bouding box for same target when iou > 0.9
        if(iou <0.8):
            unique_object_list.append(sorted_list[i])
        i = i+1
    return unique_object_list

def isOverlapped(newbox, list):
    for box in list:
        if(ioU(newbox, box) > 0.6):
            return True
    return False

backSub = cv2.createBackgroundSubtractorMOG2()
tracker = cv2.MultiTracker_create()
capture = cv2.VideoCapture('newv.mov')
if not capture.isOpened:
    print('Unable to open file')
    exit(0)
# frame width and height
fw = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(capture.get( cv2.CAP_PROP_FRAME_HEIGHT))

# output frame width
frame_width = 480
scale = frame_width/fw



result = 0;

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_dil = np.ones((5, 5), np.uint8)

while True:
    ret, frame = capture.read()

    if frame is None:
        break
    bs_frame = frame[:, 0:int(bs_factor * fw)]
    tk_frame = frame[:, int(tk_factor * fw):]

    #cv2.imshow('Frame_tf', cv2.resize(tk_frame, (720, int(720 / fw * fh))))
    
    # update background
    fgMask = backSub.apply(bs_frame, learningRate=-2)

    # processing fgMask
    blur = cv2.GaussianBlur(fgMask, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
    #stack1 = np.vstack((fgMask, blur, thresh))
    #cv2.imshow('mask, blur, thresh', cv2.resize(stack1, (frame_width, int(scale * stack1.shape[0]))))


    # blur = cv2.GaussianBlur(fgMask, (10, 10), 0)
    # erosion = cv2.erode(thresh, kernel_dil, iterations=1)
    thresh_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # fgMask2 = cv2.morphologyEx(fgMask1, cv2.MORPH_CLOSE, kernel)

    dilated = cv2.dilate(thresh_img, kernel_dil, iterations=1)
    #stack2 = np.vstack((erosion, fgMask1, dilated))

    #cv2.imshow('erosion, open, corrected', cv2.resize(stack2, (frame_width, int(scale * stack2.shape[0]))))


    # cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    # cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        newbox = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 550:
            continue
        (x, y, w, h) = newbox
        # only include boxes in overlapping area
        if(x > tk_factor*fw):
            object_list = tracker.getObjects()
            if(isOverlapped(newbox, object_list) == False):
                x = int(x - tk_factor*fw)
                _ = tracker.add(cv2.TrackerMIL_create(), tk_frame, (x, y, w, h))
                print("Now tracking" + str(len(tracker.getObjects())) + " objects")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    _, boxes = tracker.update(tk_frame)

    # eliminate bounding boxes for the same target
    final_object_list = findUniqueObjects(tracker.getObjects())
    count = len(final_object_list)
    if(len(tracker.getObjects())>10):
        res = res + count
        tracker = cv2.MultiTracker_create()
    # display bounding box for objects in tracking area
    for newbox in boxes:
        (x, y, w, h) = newbox
        x = x + tk_factor*fw  #convert back to coordinates on original frame
        if (x > tk_factor*fw):
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)


    # show region lines and text
    cv2. line(frame, (int(fw*tk_factor), 0), (int(fw*tk_factor), fh), (0, 200, 200), 2)
    cv2. line(frame, (int(fw*bs_factor), 0), (int(fw*bs_factor), fh), (0, 200, 200), 2)
    cv2.line(frame, (int(finish_factor*fw), 0), (int(finish_factor*fw), fh), (0, 200, 200), 2)

    cv2.putText(frame, "Detecting", (int((bs_factor-interval_factor)*fw*0.4), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Tracking", (int((tk_factor-interval_factor)*fw*0.4 + bs_factor*fw ), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Interval", (int(tk_factor*fw+50), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Count: " + str(count), (int(finish_factor*fw + 30), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)

    cv2.imshow('Frame', cv2.resize(frame, (720, int(720 / fw * fh))))

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break