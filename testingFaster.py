from cmu_112_graphics import *
from threading import Thread
import numpy as np
import math, random, sys
import cv2

# CITATION: Improving FPS from https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return

            self.ret, self.frame = self.stream.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.positions = []

    def getCoords(self):
        return (self.x, self.y)

    def updateCoords(self, newX, newY):
        self.x = newX
        self.y = newY

class Ball(Point):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = "lightGreen"
        self.radius = 10
        self.score = 0

class FingerDetect(App):
    vs = WebcamVideoStream(src=0).start()
    FingerMouse = Point(0, 0)
    Player = Ball(0, 0)

    def appStarted(self):  
        # Imported images
        # image1 CITATION: https://images.app.goo.gl/pMDNw7xBT5sGrX7D6
        self.image1 = self.loadImage('learningBlocks.png')
        self.learningBlocks = self.scaleImage(self.image1, 1/19)

        # image2 CITATION: https://images.app.goo.gl/he8JT9oR4Sk2T44N8
        self.image2 = self.loadImage('rightArr.png')
        self.rightArr = self.scaleImage(self.image2, 1/6.88)

        # image3 CITATION: https://images.app.goo.gl/45MvqzXmMTMPGp369
        self.image3 = self.loadImage('openEye.png')
        self.openEye = self.scaleImage(self.image3, 1/19)

        # image4 CITATION: https://images.app.goo.gl/jUcdwZvNwHwuyUwR9
        self.image4 = self.loadImage('closedEye.png')
        self.closedEye = self.scaleImage(self.image4, 1/8)
        
        # Personal images (no citation necessary)
        self.image5 = self.loadImage('Tutorial_Img.png')
        self.tutorImg = self.scaleImage(self.image5, 1/3.2)

        self.image6 = self.loadImage('interactive_buttons.png')
        self.buttonsEx = self.scaleImage(self.image6, 1)

        # BOOLEANS
        self.isFirstPage = True
        self.isStartPressed = False
        self.mouseHoverStart = False
        self.introductionPage = False
        self.onlyPoints = False
        self.thresholdDetect = False
        self.showEye = False
        self.hideEye = False
        self.showErrorMsg = False
        self.mouseHoverLearn = False
        self.playChallenge = False
        self.learnPage1 = False

        # Miscellaneous
        self.specConts = []
        self.finalImg = self.createFrames()

    def createFrames(self):
        self.frame = FingerDetect.vs.read()
        self.frame = cv2.flip(self.frame, 1)
        self.frame2 = cv2.Canny(self.frame, 100, 200)

        self.specConts = []

        self.contours, self.hierarchy = cv2.findContours(self.frame2,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # initialize local var
        temp = []
        (sx, sy) = (0, 0)

        pinkyIndex = (-1000, -1000)

        thumbIndex = (1000, 1000)

        pointerIndex = (1000, 1000)
        minDistPointer = 10000

        ringIndex = (-1000, -1000)
        minDistRing = 10000

        middleIndex = (-1000, -1000)
        minDistMiddle = 10000

        leftPalmPt = (1000, 1000)
        minDistToLeft = 10000

        rightPalmPt = (-1000, -1000)
        minDistToRight = 10000
        
        midPalmPt = (1065, 450)

        # self.contours is a non-rectangular array, each dimension being a new contour level
        # following code selects contours specifically within the purple box
        level1Len = len(self.contours)
        for level1 in range(level1Len):
            level2Len = len(self.contours[level1])
            for level2 in range(level2Len):
                [[x, y]] = self.contours[level1][level2]
                if ((950 <= x <= 1180) and (220 <= y <= 450)):
                    temp.append([x, y])
            if temp != []:
                newArr = np.array(temp, dtype=np.int32)
                self.specConts.append(newArr)
            temp = []

        for layer in range(len(self.specConts)):
            contLen = len(self.specConts[layer])
            for cont in range(contLen):
                (sx, sy) = self.specConts[layer][cont]

                # Pinky Finger Detection
                if ((sx, sy) > pinkyIndex):
                    pinkyIndex = (sx, sy)

                # Thumb Detection
                if ((sx, sy) < thumbIndex):
                    thumbIndex = (sx, sy)

                # Pointer Finger Detection
                #       i.e. minimizes distance between contour coord and top left corner of box
                if (self.distance(950, 220, sx, sy) < minDistPointer and sx < 1065 and sy < 335):
                    pointerIndex = (sx, sy)
                    minDistPointer = self.distance(950, 220, sx, sy)
                
                # Ring Finger Detection
                #       i.e. minimizes distance between contour coord and top right corner of box
                if (self.distance(1180, 220, sx, sy) < minDistRing and sx > 1065 and sy < 335):
                    ringIndex = (sx, sy)
                    minDistRing = self.distance(1180, 220, sx, sy)
                
                # Middle Finger Detection
                #       i.e. approximate center point between Pointer and Ring fingers and minimize to top of box
                distPointerToRing = self.distance(pointerIndex[0], pointerIndex[1], ringIndex[0], ringIndex[1])
                approxMidPointX = pointerIndex[0] + distPointerToRing / 2
                if ((approxMidPointX - 7 <= sx <= approxMidPointX + 7) and (self.distance(1065, 220, sx, sy) < minDistMiddle)):
                    middleIndex = (sx, sy)
                    minDistMiddle = self.distance(1065, 220, sx, sy)

                # Find intersection of left side of palm with bottom of the purple box
                d1 = self.distance(sx, sy, thumbIndex[0], thumbIndex[1])    # distance from left of palm to thumbIndex
                d2 = self.distance(sx, sy, pinkyIndex[0], pinkyIndex[1])    # distance from left of palm to pinkyIndex

                if ((sy == 450) and (self.distance(950, 450, thumbIndex[0], thumbIndex[1]) < minDistToLeft)
                                and (d1 < d2)):
                    leftPalmPt = (sx, sy)
                    minDistToLeft = self.distance(950, 450, sx, sy)

                # Find intersection of the right of palm with bottom of the purple box
                if ((sy == 450) and (self.distance(1180, 450, pinkyIndex[0], pinkyIndex[1]) < minDistToRight)
                                and (d2 < d1)):
                    rightPalmPt = (sx, sy)
                    minDistToRight = self.distance(1180, 450, sx, sy)

                # Find midpoint of both ends of the palm
                midPalmPt = self.midPoint(leftPalmPt[0], 450, rightPalmPt[0], 450)
                if (midPalmPt[0] < 1060 or midPalmPt[0] > 1070):
                    midPalmPt = (1065, midPalmPt[1])
       
        if self.isStartPressed:
            if self.onlyPoints:
                # draw circles on fingertips
                self.frame = self.drawCircleFingers(pinkyIndex, thumbIndex, pointerIndex, ringIndex, middleIndex)
                self.frame = cv2.circle(self.frame, midPalmPt, 6, (255,0,0), -1)

                # draw skeleton lines
                if midPalmPt[1] > 220:
                    self.frame = self.drawSkeletonLines(pinkyIndex, thumbIndex, pointerIndex, ringIndex, middleIndex, midPalmPt)
                return self.frame
        
            else:
                # draws circles on fingertips AND contours
                self.frame = self.drawCircleFingers(pinkyIndex, thumbIndex, pointerIndex, ringIndex, middleIndex)
                return cv2.drawContours(self.frame, self.specConts, -1, (0, 255, 0), 3)
        
        # user switches to pointing their finger 
        elif self.thresholdDetect or self.playChallenge:
            # first determine the largest contour in ROI (detect hand)
            if len(self.specConts) == 0:
                self.showErrorMsg = True
                return self.frame

            self.showErrorMsg = False
            largestContours = self.findLargestCont(self.specConts)

            # next determine which point on the largest contour represents the pointer finger
            minDistBetweenCorners = 10000
            pointerFingerInd = 0
            for _ in range(len(largestContours)):
                [kx, ky] = largestContours[_]
                actualDist = (self.distance(kx, ky, 950, 220) ** 2) + (self.distance(kx, ky, 1180, 220) ** 2)
                if (actualDist < minDistBetweenCorners):
                    minDistBetweenCorners = actualDist
                    pointerFingerInd = _

            # draw a circle to represent the virtual mouse
            (x,y) = largestContours[pointerFingerInd]           # (x, y) is current pos
            FingerDetect.FingerMouse.updateCoords(x, y)

            # look at prev pos, approxiate new pos of Player, append currentPos to pos and remove old pos
            if (len(FingerDetect.FingerMouse.positions) > 0):
                (oldX, oldY) = FingerDetect.FingerMouse.positions[0]
                deltaX = x - oldX
                deltaY = y - oldY

                (px, py) = FingerDetect.Player.getCoords()
                units = 10

                # find new pos for Player based upon deltaX, deltaY
                if (deltaX < 0):
                    if (deltaY < 0):
                        newCoords = (px - units, py - units)
                    elif (deltaY > 0):
                        newCoords = (px - units, py + units)
                    else:
                        newCoords = (px - units, py)
                elif (deltaX > 0):
                    if (deltaY < 0):
                        newCoords = (px + units, py - units)
                    elif (deltaY > 0):
                        newCoords = (px + units, py + units)
                    else:
                        newCoords = (px - units, py)
                else:
                    if (deltaY < 0):
                        newCoords = (px, py - units)
                    elif (deltaY > 0):
                        newCoords = (px, py + units)
                    else:
                        newCoords = (px, py)
                
                FingerDetect.Player.updateCoords(newCoords[0], newCoords[1])
                FingerDetect.FingerMouse.positions.append((x, y))
                FingerDetect.FingerMouse.positions.pop(0)

            # show / hide contours + virtual mouse depending on button pressed
            if self.showEye:
                self.frame = cv2.drawContours(self.frame, [largestContours], -1, (0, 255, 255), 3) 
                return cv2.circle(self.frame, FingerDetect.FingerMouse.getCoords(), 8, (255,0,0), -1)
            return self.frame
        
        else:
            return self.frame

    def findLargestCont(self, contours):
        ind = maxP = 0
        minimumPer = 100
        for layer in range(len(contours)):
            cont = contours[layer]
            currentP = cv2.arcLength(cont, closed=False)
            if ((currentP > minimumPer) and (currentP > maxP)):
                maxP = currentP
                ind = layer
        return contours[ind]

    def drawCircleFingers(self, pinkyIndex, thumbIndex, pointerIndex, ringIndex, middleIndex):
        self.frame = cv2.circle(self.frame, pinkyIndex, 6, (0,0,255), -1)
        self.frame = cv2.circle(self.frame, thumbIndex, 6, (0,0,255), -1)
        self.frame = cv2.circle(self.frame, pointerIndex, 6, (0,0,255), -1)
        self.frame = cv2.circle(self.frame, ringIndex, 6, (0,0,255), -1)
        return cv2.circle(self.frame, middleIndex, 6, (0,0,255), -1)

    def drawSkeletonLines(self, pinkyIndex, thumbIndex, pointerIndex, ringIndex, middleIndex, midPalmPt):
        self.frame = cv2.line(self.frame, pinkyIndex, midPalmPt, (255,0,0), 2)
        self.frame = cv2.line(self.frame, thumbIndex, midPalmPt, (255,0,0), 2)
        self.frame = cv2.line(self.frame, pointerIndex, midPalmPt, (255,0,0), 2)
        self.frame = cv2.line(self.frame, ringIndex, midPalmPt, (255,0,0), 2)
        return cv2.line(self.frame, middleIndex, midPalmPt, (255,0,0), 2)

    def distance(self, x0, y0, x1, y1):
        return ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5

    def midPoint(self, x0, y0, x1, y1):
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    def grayscaleImg(self):
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def keyPressed(self, event):
        # terminate webcam frame read if 't' pressed at any time
        if event.key == 't':
            FingerDetect.vs.stop()

        if self.introductionPage and event.key == 's':
            self.isStartPressed = True
            self.introductionPage = False

        if self.isStartPressed and event.key == 'd':
            self.onlyPoints = True

    def timerFired(self):
        if self.isStartPressed or self.thresholdDetect or self.playChallenge:
            # draw Box for hand:
            (x0, y0) = (950, 220)
            (x1, y1) = (1180, 450)

            self.finalImg = cv2.rectangle(self.createFrames(), (x0,  y0), (x1, y1), (128,0,128), 3)

            if self.playChallenge:
                # add up to 5 old positions to construct "tail" of Player
                if (len(FingerDetect.Player.positions) < 5):
                    currentPos = FingerDetect.Player.getCoords()
                    FingerDetect.Player.positions.append(currentPos)
                else:
                    FingerDetect.Player.positions.pop(0)

    def mousePressed(self, event):
        (x, y) = (event.x, event.y)

        # Mouse clicks START button
        if (self.isFirstPage == True) and (self.width / 2 - 150 <= x <= self.width / 2 + 150) and (self.height / 2 + 40 <= y <= self.height / 2 + 80):
            self.introductionPage = True
            self.isFirstPage = False

        # Mouse clicks ARROW button on webcam screen
        if ((self.width - 50 <= x <= self.width - 15) and (self.height - 100 <= y <= self.height - 65)):
            # FIRST click: transition to thresholding detection algorithm
            if (self.isStartPressed):
                self.isStartPressed = False
                self.thresholdDetect = True
                self.hideEye = True

            # SECOND click: transition to solving drawing/maze challenge
            elif (self.thresholdDetect):
                self.thresholdDetect = False
                self.playChallenge = True
                FingerDetect.Player.updateCoords(150, 490)
                FingerDetect.FingerMouse.positions.append((150, 490))

        # Mouse clicks HIDE/SHOW button
        if self.thresholdDetect or self.playChallenge:
            if (self.showEye) and (self.width - 50 <= x <= self.width - 15) and (self.height - 150 <= y <= self.height - 115):
                self.showEye = False
                self.hideEye = True
            elif (self.hideEye) and (self.width - 50 <= x <= self.width - 15) and (self.height - 150 <= y <= self.height - 115):
                self.hideEye = False
                self.showEye = True
        
        # Mouse clicks LEARN button
        if ((self.isStartPressed) or (self.thresholdDetect) or (self.playChallenge)) and (self.width - 50 <= x <= self.width - 15) and (self.height - 50 <= y <= self.height - 15):
            FingerDetect.vs.stop()
            self.learnPage1 = True
            if self.isStartPressed:
                self.isStartPressed = False
            elif self.thresholdDetect:
                self.thresholdDetect = False
            elif self.playChallenge:
                self.playChallenge = False

    def mouseMoved(self, event):
        (x, y) = (event.x, event.y)

        # Mouse hovers over START button
        if (self.isFirstPage == True) and (self.width / 2 - 150 <= x <= self.width / 2 + 150) and (self.height / 2 + 40 <= y <= self.height / 2 + 80):
            self.mouseHoverStart = True
        else:
            self.mouseHoverStart = False

        # Mouse hovers over LEARN button
        if ((self.isStartPressed == True) or (self.thresholdDetect == True)) and (self.width - 50 <= x <= self.width - 15) and (self.height - 50 <= y <= self.height - 15):
            self.mouseHoverLearn = True
        else:
            self.mouseHoverLearn = False

        ''' if self.playChallenge:
            print(x, y) '''

    def redrawAll(self, canvas):
        if self.isStartPressed or self.thresholdDetect or self.playChallenge:
            canvas.create_image((350, 300), image=ImageTk.PhotoImage(self.fromOpenCVtoPIL(self.finalImg)))
            self.drawLearnButton(canvas)
            self.drawRightArrow(canvas)
            
            if self.isStartPressed:
                canvas.create_text(self.width - 150, 15, text="Press 'd' to only show dots", font="Courier 16 italic", fill="white")
           
            elif self.thresholdDetect or self.playChallenge:
                self.drawHideShow(canvas)
                if self.showErrorMsg:
                    self.drawErrorMessage(canvas)
                
                if self.playChallenge:
                    # draw Player
                    (px, py) = FingerDetect.Player.getCoords()
                    r = FingerDetect.Player.radius
                    canvas.create_oval(px - r, py - r, px + r, py + r, fill=FingerDetect.Player.color, outline="white", width=2)

                    # draw Player tail
                    self.drawPlayerTail(canvas)

            if self.mouseHoverLearn:
                # shows border when mouse hovers
                canvas.create_rectangle(self.width - 50, self.height - 50, self.width - 15, self.height - 15, fill = "lightGreen", outline="white", width=3)
                canvas.create_image(self.width - 32.75, self.height - 32.75, image=ImageTk.PhotoImage(self.learningBlocks))
        
        elif self.isFirstPage:
            self.drawStartPage(canvas)
            if self.mouseHoverStart:
                canvas.create_rectangle(self.width / 2 - 150, self.height / 2 + 40, self.width / 2 + 150, self.height / 2 + 80, outline="black", width = 7)
        
        elif self.introductionPage:
            self.drawIntroPage(canvas)
        
        elif self.learnPage1:
            self.drawBackground(canvas)
            canvas.create_text(self.width / 2, self.height - 20, text="Press 'c' to continue.", font="Courier 16 bold", fill="black")
            self.drawLearnPage1(canvas)

    def drawPlayerTail(self, canvas):
        L = FingerDetect.Player.positions
        r = 3
        for (x, y) in L:
            canvas.create_oval(x - r, y - r, x + r, y + r, fill="lightGreen", width=0)

    def drawBackground(self, canvas):
        canvas.create_rectangle(0, 0, 70, self.height, fill="lightPink", width = 0)
        canvas.create_rectangle(self.width - 70, 0, self.width, self.height, fill="lightPink", width = 0)
        canvas.create_rectangle(70, 0, self.width - 70, self.height, fill="lightBlue", width = 0)

    def drawStartPage(self, canvas):
        self.drawBackground(canvas)
        canvas.create_text(self.width / 2, self.height / 2 - 73, text="THE VIRTUAL MOUSE SIMULATOR:", font = "Courier 16 bold")
        canvas.create_text(self.width / 2, self.height / 2 - 40, text="A Touchless Turn of Events!", font = "Courier 32 bold")
        canvas.create_text(self.width / 2, self.height / 2 + 60, text="CLICK TO START", font = "Courier 28 bold")

    def drawIntroPage(self, canvas):
        self.drawBackground(canvas)
        canvas.create_text(self.width // 2 - 4, 40, text="Welcome to the Virtual Mouse Simulator!", font = "Courier 33 bold", fill="white")
        canvas.create_text(self.width // 2, 40, text="Welcome to the Virtual Mouse Simulator!", font = "Courier 33 bold")

        canvas.create_text(self.width // 2, 105, text="When the simulation begins, place your hand in the purple\nbox such that ALL fingers fit within. (See below):", font = "Courier 20")
        canvas.create_image(self.width // 2, 240, image=ImageTk.PhotoImage(self.tutorImg))
        canvas.create_rectangle(342, 148, 650, 333, outline="white", width = 2)

        canvas.create_text(self.width // 2, 355, text="You can interact with the screen buttons on the side panel.", font = "Courier 20")
        canvas.create_image(200, 450, image=ImageTk.PhotoImage(self.buttonsEx))
        canvas.create_rectangle(177, 378, 225, 526, outline="white", width = 2)

        canvas.create_text(590, 450, text="1. Press the Eye button to show/hide contours & virtual mouse.\n2. Press the arrow to advance steps in the simulation.\n3. Press the ABC's to learn more about the algorithms. ", font = "Courier 16")

        canvas.create_text(self.width // 2, self.height - 50, text="When you're ready to begin, please press 'S'. Enjoy!", font = "Courier 20 bold")

    def drawErrorMessage(self, canvas):
        canvas.create_text(775, 140, text="Please move hand & fingers back in box.", font="Courier 10 italic", fill="white")

    # Drawing buttons on webcam screen
    def drawLearnButton(self, canvas):
        canvas.create_rectangle(self.width - 50, self.height - 50, self.width - 15, self.height - 15, fill = "lightGreen", width=0)
        canvas.create_image(self.width - 32.75, self.height - 32.75, image=ImageTk.PhotoImage(self.learningBlocks))
    
    def drawRightArrow(self, canvas):
        canvas.create_image(self.width - 32.75, self.height - 82.75, image=ImageTk.PhotoImage(self.rightArr))
        canvas.create_rectangle(self.width - 50, self.height - 100, self.width - 15, self.height - 65, outline="lightGreen", width=3)

    def drawHideShow(self, canvas):
        canvas.create_rectangle(self.width - 50, self.height - 150, self.width - 15, self.height - 115, fill="lightGreen", outline="white", width=3)
        if self.hideEye:
            canvas.create_image(self.width - 32.75, self.height - 132.75, image=ImageTk.PhotoImage(self.openEye))
        elif self.showEye:
            canvas.create_image(self.width - 32.75, self.height - 132.75, image=ImageTk.PhotoImage(self.closedEye))

    
    # Part 2 Drawings
    def drawLearnPage1(self, canvas):
        canvas.create_text(self.width / 2, 40, text="Wonder how this Finger Detection works?", font="Courier 30 bold")
        canvas.create_text(self.width / 2 - 250, 110, text="Detecting the edges or", font="Courier 18")
        canvas.create_text(self.width / 2 - 75, 110, text="contours", font="Courier 18 italic")
        canvas.create_text(self.width / 2 + 140, 110, text="of your hand is achieved with", font="Courier 18")
        canvas.create_text(self.width / 2 - 181, 130, text="the Canny Edge Detection algorithm.", font="Courier 18")

        canvas.create_text(self.width / 2 - 15, 170, text="Although openCV has a built-in Canny function optimized for speed", font="Courier 18")
        canvas.create_text(self.width / 2 - 20, 190, text="and efficiency, (which was implemented for finger detection), we", font="Courier 18")
        canvas.create_text(self.width / 2 - 141, 210, text="will explore a hand-written version of it.", font="Courier 18")

    # Learning how to convert from PIL image to OpenCV: https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format
    def fromPILtoOpenCV(self, PILimg):
        np_img = np.array(PILimg)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
    def fromOpenCVtoPIL(self, openCVimg):
        convert = cv2.cvtColor(openCVimg, cv2.COLOR_BGR2RGB)
        return Image.fromarray(convert)

    # Part 2 of Project

    def reduceNoiseAndBlur(self, img):
        averagedFilter = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
        blurred = self.convolveWith(img, averagedFilter)
        blurred = cv2.GaussianBlur(blurred, (11, 11), 0) 
        return blurred 

    def sobelKernelConvolutionX(self):
        sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])        
        hwOutputSobelX = self.convolveWithout(self.image2, sobelX)
        return hwOutputSobelX

    def sobelKernelConvolutionY(self):
        sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        hwOutputSobelY = self.convolveWithout(self.image2, sobelY)
        return hwOutputSobelY
    
    def convolveWithout(self, theImg, kernel):
        (imH, imW) = theImg.shape
        (kerH, kerW) = kernel.shape     # 3 x 3 matrix
        kernelOutput = 0

        padding = 1
        newImg = cv2.copyMakeBorder(theImg, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        newImgArr = np.zeros((imH, imW), dtype=float) 

        for cRow in np.arange(padding, imH - padding):
            for cCol in np.arange(padding, imW - padding):

                # from Part 1: image[startY:endY, startX:endX]
                roi = newImg[cRow - 1:cRow + 2, cCol - 1:cCol + 2]

                for row in range(len(roi)):
                    for col in range(len(roi[0])):
                        kernelOutput += roi[row][col] * kernel[row][col]
                
                newImgArr[cRow][cCol] = kernelOutput
                kernelOutput = 0

        return newImgArr

    def convolveWith(self, theImg, kernel):
        (imH, imW) = theImg.shape
        (kerH, kerW) = kernel.shape     # 3 x 3 matrix
        kernelOutput = 0

        padding = 1
        newImg = cv2.copyMakeBorder(theImg, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        newImgArr = np.zeros((imH, imW), dtype=float) 

        for cRow in np.arange(padding, imH - padding):
            for cCol in np.arange(padding, imW - padding):
                roi = newImg[cRow - 1:cRow + 2, cCol - 1:cCol + 2]

                for row in range(len(roi)):
                    for col in range(len(roi[0])):
                        kernelOutput += roi[row][col] * kernel[row][col]
                
                newImgArr[cRow][cCol] = kernelOutput
                kernelOutput = 0

        # Citation: https://stackoverflow.com/questions/49922460/scale-a-numpy-array-with-from-0-1-0-2-to-0-255
        newImgArr = ((newImgArr - newImgArr.min()) * (1/(newImgArr.max() - newImgArr.min()) * 255)).astype('uint8')

        return newImgArr

    def sobelIntensityGradient(self):
        theImg = self.reduceNoiseAndBlur(self.image2)

        sobelXImg = self.sobelKernelConvolutionX()
        sobelYImg = self.sobelKernelConvolutionY()

        (sizeH, sizeW) = sobelXImg.shape

        finalImg = np.zeros((sizeH, sizeW), dtype=float)
        grad_dir = np.empty((sizeH, sizeW), dtype=object)

        for row in np.arange(sobelXImg.shape[0]):
            for col in np.arange(sobelYImg.shape[1]):
                
                # Find Edge Gradient
                G_x = sobelXImg[row][col]
                G_y = sobelYImg[row][col]
                edge_Grad = math.sqrt((G_x ** 2) + (G_y ** 2))
                finalImg[row][col] = edge_Grad

                # Find Gradient Direction
                if G_x != 0:
                    angle = abs(math.atan(G_y / G_x))
                else:
                    angle = 0

                if (angle < math.pi / 2):
                    if abs(math.pi / 2 - angle) < angle:
                        angle = math.pi / 2
                    else: angle = math.pi
                else:
                    if abs(math.pi - angle) < abs(angle - math.pi / 2):
                        angle = math.pi
                    else: angle = math.pi / 2

                grad_dir[row][col] = (edge_Grad, angle)
        
        # See citation from above
        finalImg = ((finalImg - finalImg.min()) * (1/(finalImg.max() - finalImg.min()) * 255)).astype('uint8')

        return finalImg

    def nonMaxSup(self):
        arr = self.sobelIntensityGradient()
        suppressed = np.zeros((len(arr), len(arr[0])), dtype=float)

        for cRow in np.arange(1, len(suppressed) - 1):
            for cCol in np.arange(1, len(suppressed[0]) - 1):

                # from Part 1: image[startY:endY, startX:endX]
                roi = arr[cRow - 1:cRow + 2, cCol - 1:cCol + 2]

                (edge_Grad, angle) = arr[cRow][cCol]

                if (angle == math.pi):
                    first = arr[cRow][cCol - 1][1]
                    current = edge_Grad
                    last = arr[cRow][cCol + 1][1]
                    
                    if max(first, current, last) == first:
                        suppressed[cRow][cCol - 1] = first
                        suppressed[cRow][cCol] = suppressed[cRow][cCol + 1] = 0
                    elif max(first, current, last) == current:
                        suppressed[cRow][cCol] = current
                        suppressed[cRow][cCol - 1] = suppressed[cRow][cCol + 1] = 0
                    else:
                        suppressed[cRow][cCol + 1] = last
                        suppressed[cRow][cCol - 1] = suppressed[cRow][cCol] = 0

                else:
                    first = arr[cRow - 1][cCol][0]
                    current = edge_Grad
                    last = arr[cRow + 1][cCol][0]
                    
                    if max(first, current, last) == first:
                        suppressed[cRow - 1][cCol] = first
                        suppressed[cRow][cCol] = suppressed[cRow + 1][cCol] = 0
                    elif max(first, current, last) == current:
                        suppressed[cRow][cCol] = current
                        suppressed[cRow - 1][cCol] = suppressed[cRow + 1][cCol] = 0
                    else:
                        suppressed[cRow + 1][cCol] = last
                        suppressed[cRow - 1][cCol] = suppressed[cRow][cCol] = 0
        
        return suppressed


    # Miscellaneous Functions
    def debuggingScript(self, place):
        # prints line to test if program runs up to that line
        print(f"Working up to {place}!!")

FingerDetect(width=990, height=600)