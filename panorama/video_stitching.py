#! /usr/bin/python
import numpy as np
import cv2 as cv
import time


FLANN_INDEX_LSH = 6


def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt(anorm2(a))

def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2):

    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12 number of hash tables
                       key_size = 12,     # 20 size of key bits
                       multi_probe_level = 1) #2 nearest neighbor search

	# FLANN = Fas Library for Approximate Nearest Neighbors
    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k = 2) #2 k : Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.

    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
            matches.append((m[0].trainIdx, m[0].queryIdx))


    if len(matches) >= 4:

        keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
        keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])


		# homography = a constant trasformation relationship that occurs between the projected correspondence points when one plane is projected onto another plane
        H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC,4.0)

        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))


    return matches, H, status


   
def drawMatches(image1, image2, keyPoints1, keyPoints2, matches, status):


    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]


    img_matching_result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")


    img_matching_result[0:h2, 0:w2] = image2
    img_matching_result[0:h1, w2:] = image1


    for ((trainIdx, queryIdx), s) in zip(matches, status):

        if s == 1:
            keyPoint2 = (int(keyPoints2[trainIdx][0]), int(keyPoints2[trainIdx][1]))
            keyPoint1 = (int(keyPoints1[queryIdx][0]) + w2, int(keyPoints1[queryIdx][1]))
            cv.line(img_matching_result, keyPoint1, keyPoint2, (0, 255, 0), 1)


    return img_matching_result


def main():

    cap_left = cv.VideoCapture("road_video_left.avi")
    cap_right = cv.VideoCapture("road_video_right.avi")

    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    cv.imshow('left', frame_left)
    cv.moveWindow('left', 1000, 100)
    cv.imshow('right', frame_right)
    cv.moveWindow('right', 1400, 100)

    gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)


    detector = cv.BRISK_create()
    keyPoints_left, descriptors_left = detector.detectAndCompute(gray_left, None)
    keyPoints_right, descriptors_right = detector.detectAndCompute(gray_right, None)
    print('img_left - %d features, img_right - %d features' % (len(keyPoints_left), len(keyPoints_right)))

    
    keyPoints_left = np.float32([keypoint.pt for keypoint in keyPoints_left])
    keyPoints_right = np.float32([keypoint.pt for keypoint in keyPoints_right])
    

    matches, H, status = matchKeypoints(keyPoints_right, keyPoints_left, descriptors_right, descriptors_left)

    #img_matching_result = drawMatches(frame_right, frame_left, keyPoints_right, keyPoints_left, matches, status)

    result = cv.warpPerspective(frame_right, H,
        (frame_right.shape[1] + frame_left.shape[1], frame_right.shape[0]))
    result[0:frame_left.shape[0], 0:frame_left.shape[1]] = frame_left

    prevTime = time.time()

    cv.imshow('result', result)
    #cv.imshow('matching result', img_matching_result)


    while(cap_left.isOpened() and cap_right.isOpened()):


        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        cv.imshow('left', frame_left)
        cv.imshow('right', frame_right)

        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        fps = 1/sec

        str = "FPS : %0.2f" %fps

        result = cv.warpPerspective(frame_right, H,
            (frame_right.shape[1] + frame_left.shape[1], frame_right.shape[0]))
        result[0:frame_left.shape[0], 0:frame_left.shape[1]] = frame_left

        cv.putText(result, str, (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv.imshow('result', result)

        cv.waitKey(1)

    cap_left.release()
    cap_right.release()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
