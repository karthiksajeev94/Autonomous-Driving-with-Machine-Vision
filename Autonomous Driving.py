### IE-590 FINAL PROJECT ###

import numpy as np
import cv2

# Input video, set starting frame and initialize variables
cap = cv2.VideoCapture('cars2.mp4')
cap.set(1,300)
ret, img = cap.read()
m, n = img.shape[:2]
blank1 = np.zeros((m, n, 3), dtype=np.uint8) * 255
frame_No = 1
frame = []
speed=[]
kp = []
des = []
speedHt=[]

# While video is not over
while(True):
    # Initialize varibales, read video frames and obtain frame dimensions
    dst0 = []
    lab = []
    ret, img = cap.read()
    m, n = img.shape[:2]
    blank = np.zeros((m, n, 3), dtype=np.uint8) * 255
    ret, img1 = cap.read()

    # Isolating region of interest
    a, b, c = img1.shape
    rect1 = np.array([[0, m-150], [n, m-150], [b, a], [0, a]])
    color = [255, 255, 255]
    cv2.fillConvexPoly(img1, rect1, color)

    ## LANE DETECTION
    # Corners for Inverse perspective transform
    tlX = 200
    tlY = 40
    trX = n-150
    trY = 40
    blX = 0
    blY = m-220
    brX = n
    brY = m-220

    # Inverse Perspective transform
    pts1 = np.float32([[tlX, tlY], [trX, trY], [blX, blY], [brX, brY]])
    pts2 = np.float32([[0, 0], [600, 0], [0, 402], [600, 402]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(img, matrix, (600, 402))

    # Image segementation to isolate white and yellow regions
    hsv = res
    boundaries = [([0, 59, 119], [100, 230, 255]), ([200, 200, 200], [255, 255, 255])]

    List = []

    for (l, h) in boundaries:
        l = np.array(l, dtype='uint8')
        h = np.array(h, dtype='uint8')
        col = cv2.inRange(hsv, l, h)
        out = cv2.bitwise_and(hsv, hsv, mask=col)
        List.append(out)

    List = np.asarray(List)
    gwhite = cv2.cvtColor(List[1], cv2.COLOR_BGR2GRAY)
    gyellow = cv2.cvtColor(List[0], cv2.COLOR_BGR2GRAY)
    comb = gwhite + gyellow

    # Canny edge detection
    smooth = cv2.GaussianBlur(comb, (5, 5), 0)
    can = cv2.Canny(smooth, 75, 150)

    # Hough Transform
    houghT = cv2.HoughLinesP(can, 1, np.pi / 180, 30, maxLineGap=200)  # first param-min line length
    temp = np.zeros((402, 600, 3), dtype=np.uint8) * 255

    if houghT is not None:
        for i in houghT:
            x1, y1, x2, y2 = i[0]
            # To avoid slope=infinity
            if(x1!=x2):
                slope=int((y2-y1)/(x2-x1))
                # Desired ranges of slope to be identified correctly as a lane
                if(slope<-1 and slope>-50) or (slope>1 and slope<20):
                    cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 10)

    # Morphological operations to remove noise
    kernel=np.ones((7,7),np.uint8)
    temp = cv2.dilate(temp, kernel, iterations=2)

    # Reverse inverse perspective mapping
    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    n, m, p = img.shape
    temp = cv2.warpPerspective(temp, matrix, (m, n))

    # To find 4 corners of Hough lines to draw smooth lanes
    tempm, tempn, o = temp.shape
    lanecord = []

    # Top-left
    for i in range(0,tempm):
        for j in range(0, int(tempn/2)):
            if (temp[i][j][2] > 0):
                lanecord.append(j)
                lanecord.append(i)
                break
        if(len(lanecord)==2):
            break

    # Bottom-left
    for j in range(0, tempn):
        for i in range(tempm - 1, -1, -1):
            if (temp[i][j][2] > 0):
                lanecord.append(j)
                lanecord.append(i)
                break
        if (len(lanecord) == 4):
            break

    # Bottom-right
    for j in range(tempn - 1, -1, -1):
        for i in range(tempm - 1, -1, -1):
            if (temp[i][j][2] > 0):
                lanecord.append(j)
                lanecord.append(i)
                break
        if (len(lanecord) == 6):
            break

    # Top-right
    if (len(lanecord) != 0):
        for i in range(0, tempm):
            for j in range(tempn - 1, lanecord[0] + 30, -1):  # int(tempn/2)
                if (temp[i][j][2] > 0):
                    lanecord.append(j)
                    lanecord.append(i)
                    break
            if (len(lanecord) == 8):
                break

    # Draw lane lines superimposed on original frame
    if (len(lanecord) >= 4):
        cv2.line(img,(lanecord[0],lanecord[1]),(lanecord[2],lanecord[3]),(0,0,255),10)
    if (len(lanecord) == 8):
        cv2.line(img, (lanecord[4], lanecord[5]), (lanecord[6], lanecord[7]), (0, 0, 255), 10)
    
    ## OBJECT DETECTION
    # Import cars and speed limit training sets
    car_cascade = cv2.CascadeClassifier('cas1.xml')
    sign_cascade1 = cv2.CascadeClassifier('signs1.xml')
    sign_cascade2 = cv2.CascadeClassifier('signs2.xml')
    sign_cascade3 = cv2.CascadeClassifier('signs3.xml')
    sign_cascade4 = cv2.CascadeClassifier('signs4.xml')

    # Haar cascade
    cars = car_cascade.detectMultiScale(img1, 1.1,10)
    signs1=sign_cascade1.detectMultiScale(img1, 1.1,10)
    signs2 = sign_cascade2.detectMultiScale(img1, 1.1, 10)
    signs3 = sign_cascade3.detectMultiScale(img1, 1.1, 10)
    signs4 = sign_cascade4.detectMultiScale(img1, 1.1, 10)

    # Keypoint matching for consistent car detection
    dst = [[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]]
    if (len(kp)>0 and len(kp[len(kp)-1])!=0):
        for f in range(0, len(kp)):
            # Extract features using SIFT
            kp2, des2 = sift.detectAndCompute(img1, None)
            des1 = des[f]
            des1 = np.asarray(des1)
            kp1 = np.asarray(kp[f])

            # Perform keypoint matching using FLANN matcher and k-nearest neighbors (k=2)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            # Consider match to be good only if it is above a threshold
            if len(good) > 7:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M = 0
                mask = 0
                # RANSAC to estimate homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                h, w, _ = roi.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                dst0.append(dst[0][0][0])
            else:
                # If number of matches is not above the threshold
                print("Not enough matches are found")
                kp = []
                des = []
                dst = [[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]]
                dst0 = []
                matchesMask = None
                break

            # Find coordinates of rectangle to draw around detected object
            u1 = (dst[0][0][0] + dst[3][0][0]) / 2
            u2 = (dst[1][0][0] + dst[2][0][0]) / 2
            v1 = (dst[0][0][1] + dst[1][0][1]) / 2
            v2 = (dst[3][0][1] + dst[2][0][1]) / 2
            u = np.int32((u1 + u2) / 2)
            v = np.int32((v1 + v2) / 2)
            lab.append(u)
            lab.append(v)
            # Draw rectangle around object
            img = cv2.rectangle(img, (u - 40, v - 20), (u + 40, v + 20), (0, 255, 0), 2)
            # Draw line on blank- will help for distance calculation later
            cv2.line(blank, (u - 40, v + 20), (u + 40, v + 20), (0, 255, 0), 5)

    # Examine rectangle returned by Haar cascade
    for (x, y, w, h) in cars:
        que = np.asarray(dst0)
        que = abs(que - x)
        if(len(que)>0):
            que = min(que)

        # If rectangles are not too large and not very close to each other, use SIFT to extract features
        if (len(kp) == 0) or ((w < 100) and (que > 45)):
            roi = img1[y:y + h, x:x + w]
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(roi, None)
            if (len(kp1) != 0):
                kp.append(kp1)
                des.append(des1)

    # Draw rectangles based on Haar cascade detection for traffic signs
    for (x, y, w, h) in signs1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in signs2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in signs3:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in signs4:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    ## DISTANCE DETECTION
    m, n, o = img.shape
    # Corners for Inverse perspective transform
    tlX = 150
    tlY = 10
    trX = n - 100
    trY = 10
    blX = 0
    blY = m - 150
    brX = n
    brY = m - 150

    # Inverse perspective transform
    pts1=np.float32([[tlX,tlY],[trX, trY],[blX, blY],[brX, brY]])
    pts2=np.float32([[0,0],[n/2,0],[0,m/2],[n/2,m/2]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    res=cv2.warpPerspective(blank,matrix, (int(n/2),int(m/2)))
    resm,resn,o = res.shape

    # Store bottom y-coordinates of detected rectangles
    cord=[]
    width = []
    for i in range(resm-2,-1,-1):
        for j in range(resn-2,-1,-1):
            # Find bottom right pixel of line (line may be multiple pixels thick)
            if(res[i][j][1]>0)and(res[i+1][j][1]==0)and(res[i][j+1][1]==0):
                cord.append(i)
                width.append(j)

    # Sorting width and height of the detected pixels
    widthOrder=np.argsort(width)
    widthOrder=np.array(widthOrder).tolist()
    width=sorted(width)
    cordNew=[]
    for i in widthOrder:
        cordNew.append(cord[i])

    # Remove duplicate y-coordinates corresponding to the same rectangle
    height=[];length = []
    for i in range(0,len(cordNew)):
        height.append(cordNew[i])
        if(i!=0) and (abs(cordNew[i] - cordNew[i-1]) <15) and (abs(width[i] - width[i-1])<40):
            a = min(height[len(height)-1], height[len(height)-2])
            height.pop();
            height.pop();
            height.append(a)

    # To scale up height from bird's-eye to perspective view
    height = np.asarray(height)
    height = height*2

    # Distance from the bottom of screen, instead of top
    for i in range(0,len(height)):
        height[i] = m-height[i]
    height = np.array(height).tolist()
    speedHt.append(height)

    # To convert height to real world distance
    # Focal length calculated using y/f = Y/d
    focal = 0.3525
    # Scale determined experimentally
    scale = 8.5
    dist = []
    for i in range(len(height)):
        # Convert from pixels to meters
        yTemp = ((2.836 * height[i] * 0.00047 * scale)- 0.24)**1.6
        d=yTemp*focal/(height[i]*0.00047)
        dist.append(d-1.25)

    # Calculated number of height values should not exceed number of cars
    if(len(height)==len(lab)/2):
        for i in range(0,len(height)):
            # Print distance values
            cv2.putText(img, str(round(dist[i],2)) + ' m', (lab[2*i]-40,lab[(2*i)+1]+40), 4, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)

    ## CALCULATE SPPED
    height=dist
    if(len(height)>0):
        # Update speed every frame
        if (frame_No % 1 == 0):
            xt=[]
            yt = []
            tt = []
            # Store coordinates of lower-left corner of detected rectangle
            for i in range(0,len(height)):
                tt.append(lab[2*i]- 40)
                tt.append(lab[(2 * i) + 1] + 20)
            frame.append(frame_No)
            speed.append(tt)

            # To calculate speed separately for multiple detected objects in consecutive frames
            if (len(speed) > 1)and(len(speedHt[-2])!=0):
                for i in range(int(len(speed[-1])/2)):
                    for j in range(int(len(speed[-2])/2)):
                        if (len(speed[-1])>1)and(len(speed[-2])>1)and(abs(speed[-1][2*i] - speed[-2][2*j]) < 30) and (abs(speed[-1][(2*i)+1] - speed[-2][(2*j)+1]) < 30):
                            # Calculate speed as change in y-coordinate of lower-left corner per second
                            ht = speedHt[-1][i] - speedHt[-2][j]
                            spd = round(ht / ((frame[-1] - frame[-2]) * 0.066), 2)
                            cv2.putText(img, str(round(spd,2)) + " m/s", (lab[2*i] - 40, lab[(2*i)+1] + 60), 4, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Update frame number
    frame_No = frame_No+1

    # Display output
    cv2.imshow('Final Lane & Cars', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break