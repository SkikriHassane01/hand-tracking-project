#=================>
# Table of content
#=================>
"""
1. Importing libraries
2. create the __init__ for the hand Detector
3. Find the Hand
4. Find Positions
5. Finding Distance Between Two Points
"""

#=======================>
# 1. Importing Libraries
#=======================>
import cv2
import mediapipe as mp
import time 
import math 


class HandDetector():
    #=======================>
    # 2. Create the __init__
    #=======================>
    """
    Args:
      - static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream. See details in
        https://solutions.mediapipe.dev/hands#static_image_mode.

      - max_num_hands: Maximum number of hands to detect. See details in
        https://solutions.mediapipe.dev/hands#max_num_hands.

      - min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/hands#min_detection_confidence.

      - min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
        hand landmarks to be considered tracked successfully. See details in
        https://solutions.mediapipe.dev/hands#min_tracking_confidence.
    """
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,tracCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.tracCon = tracCon

        self.mpHands = mp.solutions.hands #Initializes the MediaPipe Hands object.
        self.hands = self.mpHands.Hands(  #Configures the MediaPipe Hands object
                            static_image_mode=self.mode,
                            max_num_hands=self.maxHands,
                            min_detection_confidence=self.detectionCon,
                            min_tracking_confidence=self.tracCon)
        self.mpDraw = mp.solutions.drawing_utils #Utility for drawing the hand landmarks on the image.

    #=======================>
    # 3. Find the hands
    #=======================>

    def findHands(self,img,draw=True):
        imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #required by MediaPipe
        self.results = self.hands.process(imageRGB)

        # check for hands
        hands = self.results.multi_hand_landmarks
        if self.results.multi_hand_landmarks:
            for handmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handmark,self.mpHands.HAND_CONNECTIONS)
        return img,hands


    #=====================================>
    # 4. Finding Hand Positions (Landmarks)
    #=====================================>

    def findPosition(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            
            for id,lm in enumerate(myhand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
        
        return lmList
    

    #=======================================>
    # 5. Finding Distance Between Two Points
    #=======================================>

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        distance = math.hypot(x2 - x1, y2 - y1) #Euclidean distance formula
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
        return distance
    
#_______________________________________________________________________________________

def main():
    # we will use the time module to calculate the frame rate of the video
    pTime = 0
    cTime = 0
    # cap is the object of the video capture class and 0 is the index of the camera
    cap = cv2.VideoCapture(0)   
    
    detector = HandDetector()

    while True:
        success,img =cap.read()
        img,_ = detector.findHands(img)
        img = cv2.flip(img,1) #flip horizontally

        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[0],lmList[4])

        # calculating the frame rate of the video

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime 
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,222),3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

    """The 21 hand landmarks.
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    """