import cv2
import numpy as np

def match_images(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)

    result12 = ransac(img1, img2, matches12, keypoints1, keypoints2)
    return result12


def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() - 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img2

def ransac(img1, img2, matches, keypoints1, keypoints2):
    good_matches = []
    for m, n in matches:
        if m.distance<0.75*n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    warped_img = warp_images(img1, img2, homography)
    return warped_img


if __name__ == '__main__':
    image1 = cv2.imread('1.JPG')
    image2 = cv2.imread('2.JPG')
    image3 = cv2.imread('3.JPG')

    img12 = match_images(image1, image2)
    img23 = match_images(image2, image3)
    result = match_images(img12, img23)

    cv2.imshow('Panoramska slika', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()