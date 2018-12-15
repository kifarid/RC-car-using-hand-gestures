from keras.models import load_model
import sys
import os
import pickle as pkl
import matplotlib.pyplot as plt
import keras
import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
from keras.preprocessing.image import load_img, img_to_array
import Low_level as ll


def main():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    model = load_model('model/' + 'hand_model_gray_second.hdf5')
    ok, frame = video.read()
    if not ok:
        print("Cannot read video")
        sys.exit()
    # Use the first frame as an initial background frame
    bg = frame.copy()

    # Kernel for erosion and dilation of masks
    kernel = np.ones((3, 3), np.uint8)

    # Tracking
    # Bounding box -> (TopRightX, TopRightY, Width, Height)
    bbox_initial = (60, 60, 170, 170)
    bbox = bbox_initial
    
    counter_left=0
    counter_right=0
    counter_forward=0
    counter_stop=0
while True:
        # Read a new frame
        ok, frame = video.read()
        display = frame.copy()
        if not ok:
            break

        # Processing
        # First find the absolute difference between the two images
        diff = cv2.absdiff(bg, frame)
        mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        # Threshold the mask
        th, thresh = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY_INV)
        thresh_2 = cv2.bitwise_not(thresh)
        # Opening, closing and dilation
        opening = cv2.morphologyEx(thresh_2, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img_dilation = cv2.dilate(closing, kernel, iterations=2)

        # Use numpy array indexing to crop the foreground frame
        hand_crop = img_dilation[ int(bbox[ 1 ]):int(bbox[ 1 ] + bbox[ 3 ]), int(bbox[ 0 ]):int(bbox[ 0 ] + bbox[ 2 ]) ]

        # Draw bounding box
        p1 = (int(bbox[ 0 ]), int(bbox[ 1 ]))
        p2 = (int(bbox[ 0 ] + bbox[ 2 ]), int(bbox[ 1 ] + bbox[ 3 ]))
        cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

        # Display result
        cv2.imshow("display", display)
        # Display diff
        cv2.imshow("diff", diff)
        # Display thresh
        cv2.imshow("thresh", thresh)
        # Display mask
        cv2.imshow("img_dilation", img_dilation)
        try:
            # Display hand_crop
            cv2.imshow("hand_crop", hand_crop)
        except:
            pass
        img = cv2.resize(hand_crop, (64, 64))
        img = np.reshape(img, (1, img.shape[ 0 ], img.shape[ 1 ], 1))
        y = model.predict(img)
        img = cv2.resize(thresh_2, (64, 64))
                
        index = np.argmax(y)
        #print("index is %d of value %f" % (index,y[index]))
        print(y)
        print (index)
        print (y[0,index])
        
        if y[0,index] > 0.8:
            
            if (index == 0): #Pose 5
                Label = "right"
                counter_right=counter_right+1
                if counter_right >= 20:
                    counter_left=0
                    counter_right=0
                    counter_forward=0
                    counter_stop=0
                    ll.turn_right()
                    
            if (index == 1): #Pose 1
                Label = "left"
                counter_left=counter_left+1
                if counter_left >= 20:
                    counter_left=0
                    counter_right=0
                    counter_forward=0
                    counter_stop=0
                    ll.turn_left()
                    
            if (index == 2): #Pose 2
                Label = "forward"
                counter_forward=counter_forward+1
                if counter_forward >= 20:
                    counter_left=0
                    counter_right=0
                    counter_forward=0
                    counter_stop=0                
                    ll.forward()
                    
            if (index == 3): #lma n3ml 0
                Label = "stop"
                counter_stop=counter_stop+1
                if counter_stop >= 20:
                    counter_left=0
                    counter_right=0
                    counter_forward=0
                    counter_stop=0
                    ll.stop()
                
        else:
            Label = "stop"
            #ll.stop()
            
        print(Label)
        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break  # ESC pressed
        elif k == 114 or k == 112:
            # r pressed
            counter_left=0
            counter_right=0
            counter_forward=0
            counter_stop=0
            bg = frame.copy()
            bbox = bbox_initial

        elif k != 255:
            print(k)
    cv2.destroyAllWindows()
    video.release()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
