import cv2
import time
import mediapipe as mp
import math



class HandDetector():   # Initializing
    def __init__(self, mode=False, max_hands=2, min_detection_confidence=1, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hand = mp.solutions.hands   # mandatory for using this module.
        self.hands = self.mp_hand.Hands(self.mode, self.max_hands, self.min_detection_confidence, self.min_tracking_confidence)
        # creating a hands variable with default Hands() object parameters. To view parameter press ctrl + object.
        # This hands object reads only RGB coloured images.
        self.mp_draw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def FindHands(self, image):
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting BGR to RGB.
        self.result = self.hands.process(image_RGB)  # hands object processing the image.
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand, self.mp_hand.HAND_CONNECTIONS)
        return image

    def FindPosition(self,image,hand_no=0):
        self.lmlist=[]
        if self.result.multi_hand_landmarks:
            one_hand = self.result.multi_hand_landmarks[hand_no]
            for id, landmarks in enumerate(one_hand.landmark):
                #print(id, landmarks)
                height, width, channel = image.shape
                cx, cy = int(landmarks.x * width), int(landmarks.y * height)
                self.lmlist.append([id,cx,cy])
                cv2.circle(image,(cx,cy), 4, (0,0,255), cv2.FILLED)
        return self.lmlist
            #print(id, cx, cy)

    def FingersUp(self):
        fingers = []
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def FindDistance(self,p1,p2,image,draw=True,r=10,t=3):
        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        cx,cy = (x1+x2) // 2 ,(y1+y2) //2

        if draw:
            cv2.line(image, (x1,y1), (x2,y2), (255,0,0), t)
            cv2.circle(image, (x1,y1), r, (0,0,255), cv2.FILLED)
            cv2.circle(image, (x2,y2), r, (0,0,255), cv2.FILLED)
            cv2.circle(image, (cx,cy), r, (255,0,0), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, image, [x1,y1,x2,y2,cx,cy]


def main():
    previous_time = 0
    current_time = 0
    capture = cv2.VideoCapture(0)  # used for opening webcam
    detector = HandDetector()
    while True:
        status, image = capture.read()
        # status variable have the status of the loop(True or False).
        # Image have the captured frames.
        image = detector.FindHands(image)
        lmlist = detector.FindPosition(image)



        if len(lmlist) != 0:
            #print(lmlist[4])
            fingers = detector.FingersUp()
            print(fingers)
            dis = detector.FindDistance(8,12,image)
            print(dis)
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('img', image)
        cv2.waitKey(1)

if __name__ == '__init__':
    main()
