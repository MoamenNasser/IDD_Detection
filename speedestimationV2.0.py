import cv2
import time
import math
import dlib
from google.colab.patches import cv2_imshow
from cv2 import rectangle


carCascade = cv2.CascadeClassifier('/content/file.xml')
video = cv2.VideoCapture('/content/traffic.mp4')
Width = 1280
Height = 720



# mounting drive
from google.colab import drive
drive.mount('/content/drive')

# function to calculate the speed
def EstimateSpeed(Location1,Location2):
    d_pixels = math.sqrt(math.pow(Location2[0]-Location1[0],2) + math.pow(Location2[1]-Location1[1],2))
    ppm = 9.25  # number of pixels per frame >> the ratio between distance of road on picture to the real one
    d_meters = d_pixels / ppm
    fps = 30  # depends of the camera that took the video
    speed = d_meters * fps * 8  # 8 is the ratio between the actual [car speed / (d_meters * fps)]
    return speed
# function to track the objects " cars "
def trackMultipleObjectives():
    rectangleColor1 = (0,0,255)
    rectangleColor2 = (0,255,0)
    frameCounter = 0
    currentCarID = 0
    fps = 0
    
    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    
    speed = [None ]* 1000
    
    while True:
        start_time = time.time()
        rc,image = video.read()  # reading the input video
        if type(image) == type(None):
            break
            
        image = cv2.resize(image,(Width,Height))
        resultImage = image.copy()
        frameCounter = frameCounter + 1
        
        carIDtobeDeleted = []
    
        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            
            if trackingQuality < 7:
                carIDtobeDeleted.append(carID)
                
        for carID in carIDtobeDeleted:
            print('Removing Car ID' + str(carID) + 'from list of trackers')
            print('Removing Car ID' + str(carID) + 'previous location')
            print('Removing Car ID' + str(carID) + 'current trackers')
            carTracker.pop(carID,None)
            carLocation1.pop(carID,None)
            carLocation2.pop(carID,None)
            
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray,1.1,13,18,(24,24))
            for (_x,_y,_w,_h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                cv2.rectangle(resultImage, (x,y), (x+w,y+h), rectangleColor1,4)
                
                matchcarID = None
                
                for carID in carTracker.keys():
                    trackedPositions = carTracker[carID].get_position()
                    t_x = int(trackedPositions.left())
                    t_y = int(trackedPositions.top())
                    t_w = int(trackedPositions.width())
                    t_h = int(trackedPositions.height())
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if((t_x <= x_bar <= (t_x * t_w )) and (t_y <= y_bar <= (t_y + t_h)) and 
                        (x <= t_x_bar <= (x+w)) and (y<= t_y_bar <= (y+h))):
                        matchcarID = carID
                if matchcarID is None:
                    print('Creating new tracker' + str(currentCarID))
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y ,x+w,y+h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x,y,w,h]
                    currentCarID = currentCarID + 1

            for carID in carTracker.keys():
                trackedPositions = carTracker[carID].get_position()
                t_x = int(trackedPositions.left())
                t_y = int(trackedPositions.top())
                t_w = int(trackedPositions.width())
                t_h = int(trackedPositions.height())
                cv2.rectangle(resultImage,(t_x, t_y),(t_x + t_w , t_y + t_h),rectangleColor2,4)
                # here we get the speed estimation
                carLocation2[carID] = [t_x,t_y,t_w,t_h]
            end_time = time.time()

            # to avoid dividing by 0
            if not (end_time == start_time):
                fps = 1.0 / (end_time - start_time)

            for i in carLocation1.keys():
                if frameCounter % 1 == 0:
                    [x1,y1,w1,h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]
                    carLocation1[i] = [x2, y2, w2, h2]
                    
                    # print 'new previous location: + str(carLocation1[i])
                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        if(speed[i] == None or speed[i] == 0) and y1 >= 100 and y1 <= 700:
                            speed[i] = EstimateSpeed(
                                [x1, y1, w1, h1], [x2, y2, w2, h2])
                        if speed[i] != None and y1 >= 180:
                            cv2.putText(resultImage,str(int(speed[i]))+ 'km/hr' , (int(x1+w1/2),int(y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
                        cv2_imshow(resultImage)
                        if cv2.waitKey(33) == 27:
                            break

cv2.destroyAllWindows()

if __name__ == "__main__":
    trackMultipleObjectives()

