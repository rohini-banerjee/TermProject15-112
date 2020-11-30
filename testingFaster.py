from cmu_112_graphics import *
from threading import Thread
import numpy as np
import math, random, time
import cv2, dlib

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

class MySecondApp(App):
    vs = WebcamVideoStream(src=0).start()

    def appStarted(self):  
        self.specConts = []
        self.finalImg = self.createFrames()

        # Imported images
        self.image1 = self.loadImage('learningBlocks.png')
        self.learningBlocks = self.scaleImage(self.image1, 1/19)

        # BOOLEANS
        self.isFirstPage = True
        self.isStartPressed = False
        self.mouseHoverStart = False
        self.introductionPage = False
        self.mouseHoverLearn = False
        self.learnPage1 = False

    def createFrames(self):
        self.frame = MySecondApp.vs.read()

        self.frame2 = cv2.Canny(self.frame, 100, 200)

        self.spectConts = []

        self.contours, self.hierarchy = cv2.findContours(self.frame2,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # self.contours is a non-rectangular array, each dimension being a new contour level
        temp = []
        fullArrForTest = []
        level1Len = len(self.contours)
        for level1 in range(level1Len):
            level2Len = len(self.contours[level1])
            for level2 in range(level2Len):
                [[x, y]] = self.contours[level1][level2]
                if ((320 <= x <= 550) and (220 <= y <= 450)):
                    temp.append([x, y])
                fullArrForTest.append([x, y])
            if temp != []:
                newArr = np.array(temp, dtype=np.int32)
                self.specConts.append(newArr)
            temp = []
        
        ''' for cont in self.specConts:
            finalImg = cv2.drawContours(self.frame, [cont], -1, (0, 255, 0), 3)
         '''
        
        return cv2.drawContours(self.frame, self.specConts, -1, (0, 255, 0), 3)
        
    def resizeByWidth(self, imgShape, newHeight):
        (oldHeight, oldWidth, depth) = imgShape
        factor = (newHeight / oldHeight)
        newWidth = int(oldWidth * factor)
        return (newWidth, newHeight)

    def grayscaleImg(self):
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def keyPressed(self, event):
        # terminate webcam frame read if 't' pressed
        if event.key == 't':
            MySecondApp.vs.stop()

        if self.introductionPage and event.key == 's':
            self.isStartPressed = True
            self.introductionPage = False

    def timerFired(self):
        if self.isStartPressed:
            # draw Box for hand:
            (x0, y0) = (320, 220)
            (x1, y1) = (550, 450)

            self.finalImg = cv2.rectangle(self.createFrames(), (x0,  y0), (x1, y1), (128,0,128), 3)

    def mousePressed(self, event):
        (x, y) = (event.x, event.y)

        # Mouse clicks START button
        if (self.isFirstPage == True) and (self.width / 2 - 150 <= x <= self.width / 2 + 150) and (self.height / 2 + 40 <= y <= self.height / 2 + 80):
            self.introductionPage = True
            self.isFirstPage = False
        
        # Mouse clicks LEARN button
        if (self.isStartPressed == True) and (self.width - 50 <= x <= self.width - 15) and (self.height - 50 <= y <= self.height - 15):
            MySecondApp.vs.stop()
            self.learnPage1 = True
            self.isStartPressed = False

    def mouseMoved(self, event):
        (x, y) = (event.x, event.y)

        # Mouse hovers over START button
        if (self.isFirstPage == True) and (self.width / 2 - 150 <= x <= self.width / 2 + 150) and (self.height / 2 + 40 <= y <= self.height / 2 + 80):
            self.mouseHoverStart = True
        else:
            self.mouseHoverStart = False

        # Mouse hovers over LEARN button
        if (self.isStartPressed == True) and (self.width - 50 <= x <= self.width - 15) and (self.height - 50 <= y <= self.height - 15):
            self.mouseHoverLearn = True
        else:
            self.mouseHoverLearn = False

    def redrawAll(self, canvas):
        if self.isStartPressed:
            canvas.create_image((350, 300), image=ImageTk.PhotoImage(self.fromOpenCVtoPIL(self.finalImg)))
            self.drawLearnButton(canvas)
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
            canvas.create_text(self.width / 2, self.height - 20, text="Press 'c' to continue.", font="Courier 11 bold")
            self.drawLearnPage1(canvas)

    def drawBackground(self, canvas):
        canvas.create_rectangle(0, 0, 70, self.height, fill="lightPink", width = 0)
        canvas.create_rectangle(self.width - 70, 0, self.width, self.height, fill="lightPink", width = 0)
        canvas.create_rectangle(70, 0, self.width - 70, self.height, fill="lightBlue", width = 0)

    def drawStartPage(self, canvas):
        self.drawBackground(canvas)

        canvas.create_text(self.width / 2, self.height / 2 - 40, text="A Touchless Turn of Events!", font = "Courier 32 bold")

        canvas.create_text(self.width / 2, self.height / 2 + 60, text="CLICK TO START", font = "Courier 28 bold")

    def drawIntroPage(self, canvas):
        canvas.create_text(self.width // 2, 20, text="Welcome to the Camera!", font = "Arial 36 bold")
        canvas.create_text(self.width // 2, 70, text="Move close to the camera so your Webcam can track both\nyour eyes and fingers", font = "Arial 24 bold")
        canvas.create_text(self.width // 2, 130, text="When you're ready, press 'S' and close your eyes to begin!", font = "Arial 24 bold")

    def drawLearnButton(self, canvas):
        canvas.create_rectangle(self.width - 50, self.height - 50, self.width - 15, self.height - 15, fill = "lightGreen", width=0)
        canvas.create_image(self.width - 32.75, self.height - 32.75, image=ImageTk.PhotoImage(self.learningBlocks))
    
    def drawLearnPage1(self, canvas):
        canvas.create_text(self.width / 2, 40, text="Wonder how this Finger Detection works?", font="Courier 30 bold")

    def fromPILtoOpenCV(self, PILimg):
        np_img = np.array(PILimg)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
    def fromOpenCVtoPIL(self, openCVimg):
        convert = cv2.cvtColor(openCVimg, cv2.COLOR_BGR2RGB)
        return Image.fromarray(convert)

    def debuggingScript(self, place):
        # prints line to test if program runs up to that line
        print(f"Working up to {place}!!")

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

MySecondApp(width=990, height=600)