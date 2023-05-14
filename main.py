import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

beforeImg = cv2.imread('images/White_3.jpg',cv2.IMREAD_GRAYSCALE)
testImg =  cv2.imread('images/White_4.jpg',cv2.IMREAD_GRAYSCALE) 

beforeImg2 = cv2.imread('images/White_3.jpg')


#threshhold before and after images
# ret, thresh1 = cv2.threshold(beforeImg, 120, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(testImg, 120, 255, cv2.THRESH_BINARY)
#diff=cv2.absdiff(thresh1, thresh2)

#subtract theshold images

#use subtracted images to determine number of holes, stickered holes

#present results
thresh1 = cv2.adaptiveThreshold(beforeImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
thresh2 = cv2.adaptiveThreshold(testImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
thresh11 = cv2.adaptiveThreshold(beforeImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 259, 5)
thresh22 = cv2.adaptiveThreshold(testImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 259, 5)

# blur
blur = cv2.GaussianBlur(beforeImg, (0,0), sigmaX=33, sigmaY=33)
blur2 = cv2.GaussianBlur(testImg, (0,0), sigmaX=33, sigmaY=33)

# divide
divide = cv2.divide(beforeImg, blur, scale=255)
divide2 = cv2.divide(testImg, blur2, scale=255)


thresh199=cv2.absdiff(divide, divide2)
thresh259=cv2.absdiff(thresh22, thresh11)


# cv2.imshow('image',thresh1)
# cv2.imshow('image2',thresh2)
# # cv2.imshow('image3', diff)
# cv2.imshow('image4',thresh11)
# cv2.imshow('image5',thresh22)
#cv2.imshow('199',thresh199)

# cv2.imshow('259',thresh259)


#cv2.imshow('blur', cv2.blur(diff,(5,5)))


#detect shape

def houghCircleDetector(path_to_img):
    img = path_to_img #cv2.imread(path_to_img)
    img = cv2.medianBlur(img,3)
    img_edge = cv2.Canny(img,100,200)

    circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,minDist=20,param1=200,param2=70)
    print(circles)
    circles = np.uint16(np.round(circles))
    for val in circles[0,:]:
        cv2.circle(img,(val[0],val[1]),val[2],(255,0,0),2)

    plt.figure(figsize=(20,10))
    plt.subplot(121),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
    plt.subplot(122),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('Result',color='c')
    plt.show()
    return


 
    #function for inverting color

    #def inverte(imagem, name):
    #    imagem = (255-imagem)
    #    cv2.imwrite(name, imagem)

    #builtin function of open cv for inverting

    #thresh199 = cv2.bitwise_not(thresh199)

#count circles
def count_circles(image_filename):
    # Load the image
    img = image_filename

    # Blur the image to reduce noise
    #blur = cv2.medianBlur(img, 5)

    # Detect circles using HoughCircles function

    circles = cv2.HoughCircles(image_filename, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=211, maxRadius=300)

    img = cv2.medianBlur(beforeImg2,5)
    
    circles2 = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    circles2 = np.uint16(np.around(circles))
    for i in circles2[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',cimg)


    # If circles are detected, count them
    if circles is not None:
        print(circles)
        count = len(circles[0])
        print("Number of Circles: " + str(count))
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         center = (i[0], i[1])
        #         # circle center
        #         cv2.circle(image_filename, center, 1, (0, 100, 100), 3)
        #         # circle outline
        #         radius = i[2]
        #         cv2.circle(image_filename, center, radius, (255, 0, 255), 3)
        # Plotting circles
        
        figure, axes = plt.subplots()
            
        h,w = image_filename.shape
            
        for i in circles[0,:]:
            Drawing_colored_circle = plt.Circle((i[0], i[1]), i[2])

            # # draw the outer circle
            # circles = np.uint16(np.around(circles))
            # cv2.circle(image_filename,(100,100),50,(0,255,0),2)
            # # draw the center of the circle
            # cv2.circle(image_filename,(0,1),10,(0,0,255),3)
            
            
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            img = cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            #img=cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
            #ext for eliminate circle function, make the set
        #     s = {}~
        #     s[i] = cv2.circle(center)
        # cv2.imshow("circle",image_filename)

        # axes.set_aspect( 1 )
        # axes.add_artist( Drawing_colored_circle )
        # plt.title( 'Colored Circle' )
        # plt.xlim(0, w)
        # plt.ylim(0, h)
        # plt.show()  
        img = cv2.circle (img, (999, 454), 270, (0, 255, 0), 2)
        cv2.imshow("with circles", beforeImg2)
        # return {count,s}
    else:
        print("failed")
        return 0


# def eliRepCir(s):
    #s1 = {}
    #s2 = {}
    #s2 = {}
    #dis = {}
    #for i in s[0::]:
        #for j in s[0::]:
            #n = sqrt((s[i,0]-s[j,0])**2+(s[i,1]-s[j,1])**2)
            #if n >= 50:
                #dis[i,j] = n
            #else:
                #dis[i,j] =
    

                    
                    

def detect_arcs(image_filename):
    # Load the image
    img = image_filename

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        # Approximate the contour as a polygon
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)

        # Check if the polygon has exactly 2 vertices, indicating an arc
        if len(approx) == 2:
            # Draw the arc contour on the original image
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
            
    print(contours)
    cv2.imshow("image arcs", img) 

# ?function to eliminate repetitive circle 
#def elicir(imgset;num):


# detect_arcs(beforeImg2)
count_circles(thresh199)
cv2.waitKey(0)
cv2.destroyAllWindows()


