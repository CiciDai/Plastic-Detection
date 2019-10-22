import cv2
import numpy as np

cap = cv2.VideoCapture('river_video.mov')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (frame_width,frame_height))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", diff)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)


    # draw contours
    _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #preprocessing points for contours;
    # if x1, y1 and x2, y2 are close. find the threshold for distance different
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # if area < 5% of frame area, do not draw contour
        if(w*h < 0.05*frame_width*frame_height):
            continue
        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    image = frame1;
    out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()