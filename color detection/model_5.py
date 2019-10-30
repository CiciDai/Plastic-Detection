from __future__ import print_function
import cv2
import numpy as np

def CalcRatio(box1, box2, type = 'overlap'):
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
    if type == 'iou':
        return iou
    return overlap


class countPlastic:
    def __init__(self, capacity = 4):
        # backSub, tracking, and finish region factors
        self.bs_factor = 0.7
        self.tk_factor = 0.55
        self.interval_factor = self.bs_factor - self.tk_factor
        self.finish_factor = 0.85
        # background subtraction object
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        # tracker object
        self.tracker = cv2.MultiTracker_create()
        # result - count since initiation of model
        # count - count since initiation of tracker object
        self.result = 0
        self.count = 0
        # frame width and height
        self.fw = 0
        self.fh = 0
        self.capacity = capacity

    def update(self, frame):
        if frame is None:
            return self.result, []

        self.fh = frame.shape[0]
        self.fw = frame.shape[1]
        # print(frame[0:10, 0:10])
        frame_copy = frame.copy()
        frame_hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
        bs_frame = frame_hsv[:, 0:int(self.bs_factor * self.fw)+30]
        tk_frame = frame_hsv[:, int(self.tk_factor * self.fw):]

        # update backSub Model
        fgMask = self.backSub.apply(bs_frame, learningRate=-1.2)

        # processing fgMask
        blur = cv2.GaussianBlur(fgMask, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_dil = np.ones((5, 5), np.uint8)
        thresh_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(thresh_img, kernel_dil, iterations=1)
        # cv2.imshow("dialted", cv2.resize(dilated, (480, 480/self.fw*self.fh)))
        # cv2.waitKey(30)

        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            newbox = cv2.boundingRect(contour)

            # ignore contour area < 900
            if cv2.contourArea(contour) < 150:
                continue

            (x, y, w, h) = newbox

            # do not include extra long object
            if h > 0.5*self.fh:
                continue

            # only include boxes in Interval area, when box right corner pass interval area
            if (x+w) > self.bs_factor * self.fw:
                object_list = self.tracker.getObjects()
                # convert x to tracker region coordinates
                x0 = x
                x = int(x - self.tk_factor * self.fw)

                # for objects wider than Interval area, do not add
                if x < 0:
                    continue
                
                # calculate artificial color area ratio
                cropped = frame[y:y+h, x0:x0+w]
                mask, ratio = self.color_detection(cropped)

                # if color detected is < 10% of bounding area, assume no plastic is detected
                if ratio < 0.05:
                    continue
                print("ratio: ")
                print(ratio)

                # output = cv2.bitwise_and(cropped, cropped, mask=mask)
                # cv2.imshow("output", output)
                # cv2.waitKey(0)
                # Assume when two bounding boxes overlaps > 60%, they are tracking the same object
                if not self.isOverlapped((x, y, w, h), object_list):
                    _ = self.tracker.add(cv2.TrackerMIL_create(), tk_frame, (x, y, w, h))
                    print("Now tracking " + str(len(self.tracker.getObjects())) + " objects")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # update tracking model
        _, boxes = self.tracker.update(tk_frame)
        # print(boxes)

        # remove boxes due to random moving object
        valid_box = []
        for box in boxes:
            if box[0] + box[2] > self.interval_factor*self.fw:
                valid_box.append(box)
        
        if len(valid_box) != len(boxes):
            self.tracker = cv2.MultiTracker_create()
            for box in valid_box:
                    (x, y, w, h) = box
                    # only include bounding box not in finish line
#                     if x+w < (self.finish_factor - self.tk_factor) * self.fw:
                    _ = self.tracker.add(cv2.TrackerMIL_create(), tk_frame, (x, y, w, h))

        # count target in finish zone
        self.findUniqueObjects(self.tracker.getObjects())

        # remove tracking boxes if tracking number > capacity or object exists in finish zone
        if len(self.tracker.getObjects()) > self.capacity or self.count > 4:
            curr_objects = self.tracker.getObjects()
            self.tracker = cv2.MultiTracker_create()

            if self.count == 0:
                # all bounding boxes are in tracking zone
                # take out the box that is closest to finish zone and add to result
                points = []
                for box in curr_objects:
                    points += [box[0] + box[2]]
                points = np.array(sorted(points))
                for box in curr_objects:
                    (x, y, w, h) = box
                    # keep capacity - 1 objects in tracker
                    if x + w < points[self.capacity-1]:
                        _ = self.tracker.add(cv2.TrackerMIL_create(), tk_frame, (x, y, w, h))
                self.result += points.size - len(self.tracker.getObjects())
                print("new objects number")
                print(len(self.tracker.getObjects()))
            else:
                for box in curr_objects:
                    (x, y, w, h) = box
                    # only include bounding box not in finish line
                    if x+w < (self.finish_factor - self.tk_factor) * self.fw:
                        _ = self.tracker.add(cv2.TrackerMIL_create(), tk_frame, (x, y, w, h))
                self.count = 0
            #_, boxes = self.tracker.update(tk_frame)

        return self.result, boxes

    def preprocess_img(self, image):
        if image is None:
            return []
        # convert white and black pixels to blue
        img_copy = image.copy()
        # img_copy[np.where((img_copy > 245).all(axis=2))] = [255, 0, 0]
        # img_copy[np.where((img_copy < 10).all(axis=2))] = [255, 0, 0]

        image_hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        white = s < 15
        black = v < 15
        img_copy[white == 1] = [255, 0, 0]
        img_copy[black == 1] = [255, 0, 0]

        image_hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        # identify glare region
        nonSat = s < 100
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nonSat = cv2.erode(nonSat.astype(np.uint8), kernel)
        v2 = v.copy()
        v2[nonSat == 0] = 0
        glare = v2 > 200
        glare = cv2.dilate(glare.astype(np.uint8), kernel)
        glare = cv2.dilate(glare.astype(np.uint8), kernel)
        glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, kernel)
        glare = cv2.morphologyEx(glare, cv2.MORPH_CLOSE, kernel)
        corrected_hsv = cv2.inpaint(image_hsv, glare, 5, cv2.INPAINT_NS)
        return corrected_hsv

    def color_detection(self, image):
        # pre-process and convert image to hsv
        corrected_hsv = self.preprocess_img(image)
        # consider less saturated color as natural color
        lower = np.array([0, 65, 65])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(corrected_hsv, lower, upper)
        ratio = np.sum(mask/255)/mask.size
        return mask, ratio

    def displayResults(self, frame, count, boxes):
        if frame is None:
            return
        for newbox in boxes:
            (x, y, w, h) = newbox
            x = x + self.tk_factor * self.fw  # convert back to coordinates on original frame
            if (x > self.tk_factor * self.fw):
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 1)

        # show region lines and text
        cv2.line(frame, (int(self.fw * self.tk_factor), 0), (int(self.fw * self.tk_factor), self.fh), (0, 200, 200), 2)
        cv2.line(frame, (int(self.fw * self.bs_factor), 0), (int(self.fw * self.bs_factor), self.fh), (0, 200, 200), 2)
        cv2.line(frame, (int(self.finish_factor * self.fw), 0), (int(self.finish_factor * self.fw), self.fh), (0, 200, 200),
                 2)

        cv2.putText(frame, "Detecting", (int(self.tk_factor * self.fw * 0.4), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Tracking", (int(30 + self.bs_factor * self.fw), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Interval", (int(self.tk_factor * self.fw + 30), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, "Count: " + str(count), (int(self.finish_factor * self.fw + 5), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 0, 0), 1,
                    cv2.LINE_AA)

        cv2.imshow('Frame', cv2.resize(frame, (720, int(720 / self.fw * self.fh))))
        cv2.waitKey(30)

    def findUniqueObjects(self, object_list):
        target_list = []
        for box in object_list:
            if (box[0] + box[2]>= (self.finish_factor - self.tk_factor) * self.fw):
                target_list.append(box)
        sorted_list = sorted(target_list, key=lambda tup: tup[0])
        i = 0
        unique_object_list = []

        while i <= len(sorted_list) - 1:
            if (len(unique_object_list) != 0):
                (xA, yA, wA, hA) = unique_object_list[-1]
            else:
                (xA, yA, wA, hA) = sorted_list[i]
                unique_object_list.append((xA, yA, wA, hA))
                if (len(sorted_list) == 1):
                    break
                i = i + 1

            # to avoid double counting on one object with multiple bounding boxes
            iou = CalcRatio(unique_object_list[-1], sorted_list[i], 'iou')
            if (iou<0.8):
                unique_object_list.append(sorted_list[i])
            i = i + 1

        # current count of objects in finish zone
        count = len(unique_object_list)
        # update results by adding count changes
        self.result = self.result + count - self.count
        self.count = count

    def isOverlapped(self, newbox, list):
        for box in list:
            overlap = CalcRatio(newbox, box)
            if (overlap > 0.3):
                return True
        return False

