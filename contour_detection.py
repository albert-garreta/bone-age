import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/example.png', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_gray = cv2.equalizeHist(img_gray)
cv2.imshow('Equalized', img_gray)
cv2.waitKey(0)
tr= 100
ret, threshold = cv2.threshold(img_gray, tr, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary image', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()