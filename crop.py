#importing all the required modules
import cv2 as cv
#reading the image on which bounding box is to be drawn using imread() function
imageread = cv.imread("data/data_tagged/1377.png")
#using selectROI() function to draw the bounding box around the required objects
imagedraw = cv.selectROI(imageread)
#cropping the area of the image within the bounding box using imCrop() function
croppedimage = imageread[int(imagedraw[1]):int(imagedraw[1]+imagedraw[3]), int(imagedraw[0]):int(imagedraw[0]+imagedraw[2])] #displaying the cropped image as the output on the screen
cv.imshow('Cropped_image',croppedimage)
cv.waitKey(0)
cv.destroyAllWindows()