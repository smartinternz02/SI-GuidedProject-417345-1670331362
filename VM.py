import cv2
import numpy as np
import autopy
import time
import HandTrackModule as htm

cam_width, cam_height = 640, 480
screen_width, screen_height = autopy.screen.size()
fr = 100 # Frame reduction
smooth = 7
plocx, plocy = 0, 0  # previous location
clocx, clocy = 0, 0  # current location

previous_time = 0
detector = htm.HandDetector(max_hands=1)

capture = cv2.VideoCapture(0)
capture.set(3, cam_width)
capture.set(4, cam_height)

while True:
    status, image = capture.read()

    # Step-1: Finding the hand landmarks
    image = detector.FindHands(image)
    lmlist = detector.FindPosition(image) # lmlist is list of landmarks

    # Step-2: Getting the tip of the index and middle finger
    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

        # Step-3: Find the fingers which are opened
        fingers = detector.FingersUp()
        #cv2.rectangle(image, (100, 100), (cam_width - 100, screen_height - 100),
                      #(255, 0, 255), 2)
        #print(fingers)
        cv2.rectangle(image, (fr, fr), (cam_width - fr, cam_height - fr), (255, 0, 255), 2)

        # Step-4: Moving mode [if index finger is up then it is moving mode]
        if fingers[1] == 1 and fingers[2] == 0:

            # Step-5: Convert coordinates
            x3 = np.interp(x1, (fr,cam_width-fr), (0,screen_width))
            y3 = np.interp(y1, (fr,cam_height-fr), (0,screen_height))

            # Step-6: Smoothening values
            clocx = plocx + (x3 - plocx) / smooth
            clocy = plocy + (y3 - plocy) / smooth


            # Step-7: Moving mouse
            autopy.mouse.move(screen_width - clocx, clocy)
            cv2.circle(image, (x1,y1), 10, (255,0,255),cv2.FILLED)
            plocx, plocy = clocx, clocy

        # Step-8: Clicking mode: Both index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            # Step-9: Finding the distance between fingers
            length,image,info = detector.FindDistance(8,12,image)
            #print(length)
            # Step-10: Virtual clicking
            if length < 40:
                autopy.mouse.click()

    # Step-11: Visualizing Frame rate
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(image, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    # Step-12: Display
    cv2.imshow('img', image)
    cv2.waitKey(1)






