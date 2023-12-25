import matplotlib.pyplot as plt
import cv2 as cv

if __name__ == '__main__':
    imgIn = cv.imread("coins.png")
    imgHsv = cv.cvtColor(imgIn, cv.COLOR_BGR2HSV)
    plt.imshow(imgHsv)
    plt.show()
    saturation_channel = imgHsv[:, :, 1]
    plt.imshow(saturation_channel, cmap="gray")
    plt.show()
    retval, img = cv.threshold(saturation_channel, 45, 255, cv.THRESH_BINARY_INV)
    cv.imshow("Input", img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(30, 30))

    imgOpen2 = cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel)

    imgOpenClose = cv.morphologyEx(imgOpen2,op=cv.MORPH_CLOSE, kernel=kernel)
    cv.imshow("OpenClose",imgOpenClose)

    mask = 255 - imgOpenClose

    img_res=cv.bitwise_and(imgIn,imgIn,mask=mask)
    cv.imshow("res1",img_res)

    cv.waitKey(0)
    cv.destroyAllWindows()