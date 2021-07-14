import sys, os
import glob
import cv2
import numpy as np


def contours(path_to_image, subfolder):
    try:
        os.mkdir('results/'+subfolder)
    except Exception as e:
        print(e)
        pass

    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to hsv
    # min and max are the boundaries which the cv2.inRange function will apply to hsv format to get an threshhold image
    # min and max are approximate and will need to be defined more precisely if accuracy of other samples is poor
    min = np.array((50, 20, 20), np.uint8)
    max = np.array((155, 140, 170), np.uint8)

    blurred = cv2.GaussianBlur(img, (21, 21), 0)

    # locate the box using Canny edge detection operator
    canny = cv2.Canny(blurred, 100, 150, apertureSize=3)
    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)
    # crop the box
    cropped_orig = img[y1:y2, x1:x2]

    # threshold by color
    thresh = cv2.inRange(cropped_orig, min, max)
    # find the contours. Maybe useful to check other methods of approximation
    contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    # check all contours
    for cnt in contours0:
        if len(cnt) > 400:

            rect = cv2.minAreaRect(cnt)  # try to set an rechtangle

            area = int(rect[1][0] * rect[1][1])  # area of rechtangle

            if 10000 < area < 500000:  # if the area of rechtangle is big enough

                [X, Y, W, H] = cv2.boundingRect(cnt)

                crop_img = cropped_orig[Y:Y+H, X:X+W]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_HSV2BGR)
                cv2.imwrite('results/'+subfolder+'/'+str(i)+'.jpg', crop_img)
                i=i+1


try:
    folderName = sys.argv[1]
except:
    print("Usage: python core_samples.py someFolder")
    sys.exit(1)

filePaths = glob.glob(folderName + "/*.JPG") #search all JPG in the folder

subfolder = 0

for filePath in filePaths:
    contours(filePath, str(subfolder))
    subfolder = subfolder+1

print('Cropped images are placed in /results/ folder')