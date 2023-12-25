# from cv2 import Mat
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils


def crop_image(img, new_width, new_height):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)[1]

    contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]

    coords = [0, 0]
    diff_x = new_height
    diff_y = new_width

    for c in contours:
        if cv.contourArea(c) >= new_width * new_height:
            arc_len = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.1 * arc_len, True)
            if len(approx) == 4:
                l_x, l_y = [], []

                for a in approx:
                    coord_y = a[0][0]
                    coord_x = a[0][1]

                    l_y.append(coord_y)
                    l_x.append(coord_x)

                start_x = np.min(l_x)
                end_x = np.max(l_x)
                start_y = np.min(l_y)
                end_y = np.max(l_y)

                tmp_diff_x = abs(new_height - (end_x - start_x))
                tmp_diff_y = abs(new_width - (end_y - start_y))

                if tmp_diff_x < diff_x and tmp_diff_y < diff_y:
                    diff_x = tmp_diff_x
                    diff_y = tmp_diff_y

                    coords[0] = start_x
                    coords[1] = start_y

                    break

    return img[coords[0]:coords[0] + new_height, coords[1]:coords[1] + new_width]


def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':
    input = cv.imread('./ulazna-slika.png')

    rows = open('./googlenet/synset_words.txt').read().strip().split('\n')
    classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]
    net = cv.dnn.readNetFromCaffe('./googlenet/bvlc_googlenet.prototxt', './googlenet/bvlc_googlenet.caffemodel')

    cropped_img = crop_image(input, 1440, 720)

    confidence_threshold = 0.5

    for resized in pyramid(cropped_img, scale=2, minSize=(180, 180)):
        for (x, y, img) in sliding_window(resized, 180, (180, 180)):
            if img.shape[0] != 180 or img.shape[1] != 180:
                continue

            blob = cv.dnn.blobFromImage(img, 2, (224, 224), (105, 117, 123))
            scale = cropped_img.shape[1] / resized.shape[1]
            net.setInput(blob)
            preds = net.forward()

            idx = np.argsort(preds[0])[::-1][0]
            if preds[0][idx] > confidence_threshold:
                if 'dog' in classes[idx]:
                    color = (0, 255, 255)
                    text = 'Dog'
                elif 'cat' in classes[idx]:
                    color = (0, 0, 255)
                    text = 'Cat'
                else:
                    continue

                nx = int(x * scale)
                ny = int(y * scale)
                size = int(180 * scale)
                rect = (size, nx, ny)

                cv.rectangle(cropped_img, (nx, ny), (nx + size, ny + size), color, 2)
                cv.putText(cropped_img, text, (nx + 10, ny + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv.imshow('Output', cropped_img)
    cv.imwrite('izlazna-slika.jpg', cropped_img)

    cv.waitKey()
