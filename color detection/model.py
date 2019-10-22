from __future__ import print_function
import cv2
import numpy as np
import time

def IoU(box1, box2):
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
    return overlap



class countPlastic:
    def __init__(self):
        # backSub, tracking, and finish region factors
        self.bs_factor = 0.7
        self.tk_factor = 0.55
        self.interval_factor = 0.15
        self.finish_factor = 0.9
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.tracker = cv2.MultiTracker_create()
        self.result = 0
        self.count = 0
        self.fw = 0
        self.fh = 0

    def update(self, frame):
        if frame is None:
            return self.result, []
        self.fh = frame.shape[0]
        self.fw = frame.shape[1]

        bs_frame = frame[:, 0:int((self.bs_factor) * self.fw)+30]
        tk_frame = frame[:, int(self.tk_factor * self.fw):]

        # update backSub Model
        fgMask = self.backSub.apply(bs_frame, learningRate=-2)

        # processing fgMask
        blur = cv2.GaussianBlur(fgMask, (1, 1), 0)
        _, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_dil = np.ones((5, 5), np.uint8)
        thresh_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(thresh_img, kernel_dil, iterations=1)
        # cv2.imshow("dialted", cv2.resize(dilated, (480, 480/self.fw*self.fh)))
        # cv2.waitKey(30)

        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("color mask", 255*mask)
        # cv2.waitKey(30)
        for contour in contours:
            newbox = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 900:
                continue
            (x, y, w, h) = newbox
            # only include boxes in overlapping area
            if ((x+w) > self.bs_factor * self.fw):
                cropped = frame[y:y+h, x:x+w]
                mask, ratio = self.color_detection(cropped)
                print("ratio: ")
                print(ratio)
                # output = cv2.bitwise_and(cropped, cropped, mask=mask*255)
                # cv2.imshow("output", output)
                # cv2.waitKey(30)
                # print(newbox)
                # print(frame.shape)

                # if color detected is < 20% of bounding area, no plastic is detected
                if(ratio < 0.05):
                    continue

                object_list = self.tracker.getObjects()
                x = int(x - self.tk_factor * self.fw)
                if(x<0):
                    w = w+x
                    x = 0
                if (self.isOverlapped((x, y, w, h), object_list) == False):
                    _ = self.tracker.add(cv2.TrackerMIL_create(), tk_frame, (x, y, w, h))
                    print("Now tracking" + str(len(self.tracker.getObjects())) + " objects")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #update tracking model
        _, boxes = self.tracker.update(tk_frame)

        # count target in finish zone
        self.findUniqueObjects(self.tracker.getObjects())

        if (len(self.tracker.getObjects()) > 10):
            self.tracker = cv2.MultiTracker_create()
            self.count = 0

        return self.result, boxes

    def preprocess_img(self, image):

        # convert white objects to blue
        image[np.where((image == [255, 255, 255]).all(axis=2))] = [255, 0, 0]

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        nonSat = s < 180
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nonSat = cv2.erode(nonSat.astype(np.uint8), disk)

        v2 = v.copy()
        v2[nonSat == 0] = 0
        glare = v2 > 200
        glare = cv2.dilate(glare.astype(np.uint8), disk)
        glare = cv2.dilate(glare.astype(np.uint8), disk)
        glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, disk)
        glare = cv2.morphologyEx(glare, cv2.MORPH_CLOSE, disk)
        corrected_hsv = cv2.inpaint(image_hsv, glare, 5, cv2.INPAINT_NS)
        # corrected_bgr = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
        return corrected_hsv

    def color_detection(self, image):

        # preprocess and convert image to hsv
        corrected_hsv = self.preprocess_img(image)

        # define the list of boundaries
        blue = np.uint8([[[255, 0, 0]]])
        green = np.uint8([[[0, 255, 0]]])
        red = np.uint8([[[0, 0, 255]]])

        colors = [blue, green, red]

        boundaries = []
        for color in colors:
            hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            lower = [hsvColor[0][0][0] - 25, 60, 0]
            upper = [hsvColor[0][0][0] + 25, 255, 255]
            boundaries.append([lower, upper])

        mask = np.zeros((image.shape[0], image.shape[1]), dtype = bool)
        # loop over the boundaries
        color_masks = []
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # find the colors within the specified boundaries and apply
            mask_color = cv2.inRange(corrected_hsv, lower, upper)
            color_masks.append(mask_color)
            mask = (mask_color == 255) | (mask)

        mask = mask.astype(np.uint8)
        # print(mask.size)
        # print(np.sum(mask))
        ratio = np.sum(mask)/mask.size
        # mask in int(0,1) format
        return mask, ratio

    def displayResults(self, frame, count, boxes):
        for newbox in boxes:
            (x, y, w, h) = newbox
            x = x + self.tk_factor * self.fw  # convert back to coordinates on original frame
            if (x > self.tk_factor * self.fw):
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

        # show region lines and text
        cv2.line(frame, (int(self.fw * self.tk_factor), 0), (int(self.fw * self.tk_factor), self.fh), (0, 200, 200), 2)
        cv2.line(frame, (int(self.fw * self.bs_factor), 0), (int(self.fw * self.bs_factor), self.fh), (0, 200, 200), 2)
        cv2.line(frame, (int(self.finish_factor * self.fw), 0), (int(self.finish_factor * self.fw), self.fh), (0, 200, 200),
                 2)

        cv2.putText(frame, "Detecting", (int(self.tk_factor * self.fw * 0.4), 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Tracking", (int(30 + self.bs_factor * self.fw), 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Interval", (int(self.tk_factor * self.fw + 30), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, "Count: " + str(count), (int(self.finish_factor * self.fw + 10), 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2,
                    cv2.LINE_AA)

        cv2.imshow('Frame', cv2.resize(frame, (720, int(720 / self.fw * self.fh))))

        cv2.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     break


    def findUniqueObjects(self, object_list):
        target_list = []
        for box in object_list:
            if (box[0] + box[2]> (self.finish_factor - self.tk_factor) * self.fw):
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
                i = i + 1
                if (i == len(sorted_list)):
                    self.result = 1
                    self.count = 1
                    break

            overlap = IoU(unique_object_list[-1], sorted_list[i])

            # consider as bouding box for same target when iou > 0.9
            if (overlap<0.6):
                unique_object_list.append(sorted_list[i])

            count = len(unique_object_list)
            if(self.count == 0):
                self.count = count
                self.result = self.result + count
            elif(count > self.count):
                self.result = self.result + count - self.count
                self.count = count

            i = i + 1

    def isOverlapped(self, newbox, list):
        for box in list:
            overlap = IoU(newbox, box)
            if (overlap > 0.6):
                return True
        return False




if __name__ =="__main__":
    model = countPlastic()
    capture = cv2.VideoCapture('test_5.3.mp4')
    if not capture.isOpened:
        print('Unable to open file')
        exit(0)
    while True:
        time1 = time.time()
        ret, frame = capture.read()
        if frame is None:
            break
        ret, frame = capture.read()
        if frame is None:
            break
        count, boxes = model.update(frame)

        # model.color_detection(frame)
        # display bounding box for objects in tracking area
        model.displayResults(frame, count, boxes)
        time2 = time.time()
        delta = time2 - time1
        print(1/delta)


