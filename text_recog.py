# USAGE
# python correct_skew.py --image images/neg_28.png

# import the necessary packages
import numpy as np
import argparse
import pytesseract
import cv2
import imutils


def get_vin_text(image):
	text = pytesseract.image_to_string(image)
	return text

image = cv2.imread("pic1.png")
image = imutils.resize(image, width=1000)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
filtered_image = cv2.inRange(image, 140, 255)
contours = cv2.findContours(filtered_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

print("the length of contours are ", len(contours))

biggest_contour = contours[0]

idx = 0 # The index of the contour that surrounds your object
mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
out = np.zeros_like(image) # Extract out the object and place into output image
out[mask == 255] = image[mask == 255]

cv2.imshow("output image", out)

cv2.waitKey(0)

image = out.copy()

gray = out.copy()
#cv2.imshow("our orignal image", image)


'''

cv2.drawContours(image, [biggest_contour], 0, (0,255,0), 3)

cv2.imshow("contour image", image)

cv2.waitKey(0)

#cv2.contourArea(contours)

cv2.imshow("barcode image", filtered_image)
cv2.waitKey(0)

'''

# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.bitwise_not(image)

#cv2.imshow("gray is ", gray)
#cv2.waitKey(0)
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#cv2.imshow("thresh", thresh)
#cv2.waitKey(0)
# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

#print(angle)
#cv2.waitKey(0)
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)

# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)
text = get_vin_text(rotated)
print(text)
cv2.waitKey(0)
