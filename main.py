import cv2
import numpy as np
import matplotlib.pyplot as plt

beforeImg = cv2.imread('images/White_3.jpg',cv2.IMREAD_GRAYSCALE)
testImg =  cv2.imread('images/White_4.jpg',cv2.IMREAD_GRAYSCALE)

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
cv2.imshow('199',thresh199)

# cv2.imshow('259',thresh259)


#cv2.imshow('blur', cv2.blur(diff,(5,5)))


#detect shape


# def detectShapes(img_path):
#     img = img_path
#     _,img_Otsubin = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#     contours,_ = cv2.findContours(img_Otsubin.copy(),1,2)
#     for num,cnt in enumerate(contours):
#         x,y,w,h = cv2.boundingRect(cnt)
#         approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#         # print(num, approx)
#         if len(approx) == 3:
#             cv2.putText(img,"Triangle",(int(x+w/2),int(y+h/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.drawContours(img,[cnt],-1,(0,255,0),10)
#         if len(approx) == 4:
#             cv2.putText(img,"Rect",(int(x+w/2),int(y+h/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.drawContours(img,[cnt],-1,(0,255,0),10)
#         if len(approx) > 10:
#             cv2.putText(img,"Circle",(int(x+w/2),int(y+h/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#             cv2.drawContours(img,[cnt],-1,(0,255,0),10)
            
#     plt.figure(figsize=(20,30))
#     #plt.subplot(131),plt.imshow(cv2.cvtColor(img_Otsubin,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
#     plt.subplot(132),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('Result')
#     #plt.subplot(133),plt.imshow(cv2.cvtColor(img_path,cv2.COLOR_BGR2RGB)),plt.title('Original')
#     plt.show()
#     return

# detectShapes(thresh199)

def houghCircleDetector(path_to_img):
    img = path_to_img #cv2.imread(path_to_img)
    img = cv2.medianBlur(img,3)
    img_edge = cv2.Canny(img,100,200)

    circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,minDist=20,param1=200,param2=70)
    print(circles)
    # circles = np.uint16(np.round(circles))
    # for val in circles[0,:]:
    #     cv2.circle(img,(val[0],val[1]),val[2],(255,0,0),2)

    # plt.figure(figsize=(20,10))
    # plt.subplot(121),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
    # plt.subplot(122),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('Result',color='c')
    # plt.show()
    return


                
houghCircleDetector(thresh199)
cv2.waitKey(0)
cv2.destroyAllWindows()


