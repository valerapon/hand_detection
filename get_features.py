import sys
import numpy as np
import cv2

def get_image():
    return cv2.imread(sys.argv[1])

def get_contour(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    threshold = gray.copy()
    threshold[threshold < 50] = 0
    threshold[threshold > 0] = 1

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [filter_contours[0]]

    return contours

def get_key_points(cnt):
    hull = cv2.convexHull(contours[0], returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    P = []
    for i in range(defects.shape[0]):
        P.append(defects[i, 0][0])
        P.append(defects[i, 0][2])
        P.append(defects[i, 0][1])

    K = [P[0]]
    K_i = [0]
    for i, p in enumerate(P):
        s = tuple(cnt[K[-1]][0])
        e = tuple(cnt[p][0])
        if (s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2 > 4000:
            K.append(p)
            K_i.append(i)  
    s = tuple(cnt[K[0]][0])
    e = tuple(cnt[K[-1]][0])
    if (s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2 < 4000:
        K = K[:len(K) - 1]
        K_i = K_i[:len(K_i) - 1]

    return P, K, K_i

def check_points(ind, cnt, K):
    pre_is_longer = True
    m = cv2.moments(cnt)
    cen = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
    p = tuple(cnt[K[ind]][0])
    
    pre_dist = (cen[0] - p[0]) ** 2 + (cen[1] - p[1]) ** 2
    for i in range(ind + 1, ind + 9):
        p = tuple(cnt[K[i]][0])
        dist = (cen[0] - p[0]) ** 2 + (cen[1] - p[1]) ** 2
        if dist > pre_dist and pre_is_longer:
            return False
        if dist < pre_dist and not pre_is_longer:
            return False
        pre_is_longer = not pre_is_longer
        pre_dist = dist

    for i in range(ind + 1, ind + 7):
        p1 = tuple(cnt[K[i]][0])
        p2 = tuple(cnt[K[i + 1]][0])
        p3 = tuple(cnt[K[i + 2]][0])
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        if v1[0] * v2[0] + v1[1] * v2[1]  > 0:
            return False

    p1 = tuple(cnt[K[ind]][0])
    p2 = tuple(cnt[K[ind + 8]][0])
    v1 = (p1[0] - cen[0], p1[1] - cen[1])
    v2 = (p2[0] - cen[0], p2[1] - cen[1])
    if v1[0] * v2[1] - v1[1] * v2[0] > 0:
        return False

    return True

def find_hand(K, cnt):
    hand = []
    end = -1000
    for M in range(-9, len(K) - 8):
            if check_points(M, cnt, K):
                    if M + 8 <= 0:
                            end = max(end, len(K) - M - 8)
                    else:
                            end = max(end, M + 8)
    end = end % len(K)
    
    for i in range(8, -1, -1):
            hand.append(end - i)
    return hand

def adjust_hand(cnt, P, K, K_i, hand):
    dist, ind = 0, 0
    r = tuple(cnt[K[hand[3]]][0])
    l = tuple(cnt[K[hand[5]]][0])
    for i in (range(K_i[hand[3]] + 1, K_i[hand[5]])):
        p = tuple(cnt[P[i]][0])
        if  (l[0] - p[0]) ** 2 + (l[1] - p[1]) ** 2 + (r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2 > dist:
            dist = (l[0] - p[0]) ** 2 + (l[1] - p[1]) ** 2 + (r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2
            ind = i
    K_i[hand[4]] = ind
    K[hand[4]] = P[ind]

    dist, ind = 0, 0
    r = tuple(cnt[K[hand[1]]][0])
    l = tuple(cnt[K[hand[3]]][0])
    for i in (range(K_i[hand[1]] + 1, K_i[hand[3]])):
        p = tuple(cnt[P[i]][0])
        if  (l[0] - p[0]) ** 2 + (l[1] - p[1]) ** 2 + (r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2 > dist:
            dist = (l[0] - p[0]) ** 2 + (l[1] - p[1]) ** 2 + (r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2
            ind = i
    K_i[hand[2]] = ind
    K[hand[2]] = P[ind]

    p1 = tuple(cnt[K[hand[1]]][0])
    p2 = tuple(cnt[K[hand[2]]][0])
    v = (p2[0] - p1[0], p2[1] - p1[1])

    dist, ind = 0, 0
    i = K_i[hand[1]]

    while True:
        i -= 1
        p = tuple(cnt[P[i]][0])
        v1 = (p[0] - p1[0], p[1] - p1[1])
        if v[0] * v1[1] - v[1] * v1[0] <= 0:
            break
        if (v[0] * v1[0] + v[1] * v1[1]) / (v[0] ** 2 + v[1] ** 2) ** 0.5 / (v1[0] ** 2 + v1[1] ** 2) ** 0.5 < -1 / 2 ** 0.5:
            break
        if v1[0] ** 2 + v1[1] ** 2 > dist:
            dist = v1[0] ** 2 + v1[1] ** 2
            ind = i

    K_i[hand[0]] = ind
    K[hand[0]] = P[ind]
    return hand

def draw_marks(image, hand, contours):
    cnt = contours[0]

    for i in hand:
            cv2.circle(image, tuple(cnt[K[i]][0]), 5, [0,0,255], -1)
    for i in range(len(hand) - 1):
            cv2.line(image, tuple(cnt[K[hand[i]]][0]), tuple(cnt[K[hand[i + 1]]][0]), [0,255,0], 1)

    hand_img = cv2.drawContours(image * 0, contours, -1, (255,255,255), -1)[:,:,0]
    trans = cv2.distanceTransform(hand_img, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(trans)

    center = max_loc
    radius = max_val

    cv2.circle(image, center, int(radius), [255,0,0], 1)
    cv2.circle(image, center, 5, [255,255,255], -1)
    return image, center, radius

def print_features(image, K, hand, center, radius):
    r = [0, 0, 0, 0]
    r[0] = ((center[0] - cnt[K[hand[1]]][0][0]) ** 2 + (center[1] - cnt[K[hand[1]]][0][1]) ** 2) ** 0.5
    r[1] = ((center[0] - cnt[K[hand[3]]][0][0]) ** 2 + (center[1] - cnt[K[hand[3]]][0][1]) ** 2) ** 0.5
    r[2] = ((center[0] - cnt[K[hand[5]]][0][0]) ** 2 + (center[1] - cnt[K[hand[5]]][0][1]) ** 2) ** 0.5
    r[3] = ((center[0] - cnt[K[hand[7]]][0][0]) ** 2 + (center[1] - cnt[K[hand[7]]][0][1]) ** 2) ** 0.5

    p = [0, 0, 0]
    tmp = (cnt[K[hand[1]]][0] + cnt[K[hand[3]]][0]) / 2
    p[0] = ((cnt[K[hand[2]]][0][0] - tmp[0]) ** 2 + (cnt[K[hand[2]]][0][1] - tmp[1]) ** 2) ** 0.5
    tmp = (cnt[K[hand[3]]][0] + cnt[K[hand[5]]][0]) / 2
    p[1] = ((cnt[K[hand[4]]][0][0] - tmp[0]) ** 2 + (cnt[K[hand[4]]][0][1] - tmp[1]) ** 2) ** 0.5
    tmp = (cnt[K[hand[5]]][0] + cnt[K[hand[7]]][0]) / 2
    p[2] = ((cnt[K[hand[6]]][0][0] - tmp[0]) ** 2 + (cnt[K[hand[6]]][0][1] - tmp[1]) ** 2) ** 0.5

    circle = image * cv2.circle(image * 0, center, int(radius), (1, 1, 1), -1)
    R = circle[:,:,2]
    G = circle[:,:,1]
    B = circle[:,:,0]
    R = R[R != 0].mean()
    G = G[G != 0].mean()
    B = B[B != 0].mean()

    print('@', radius, *r, *p, R, G, B, '@')

def save_result(image, name):
    cv2.imwrite('output/' + name, image)

if __name__ == '__main__':
    image = get_image()
    if image is None:
        print('Image not found')
        exit()
    img = image.copy()

    contours = get_contour(img)
    P, K, K_i = get_key_points(contours[0])

    cnt = contours[0]
    hand = find_hand(K, contours[0])
    hand = adjust_hand(contours[0], P, K, K_i, hand)
            
    img, center, radius = draw_marks(img, hand, contours)
    print_features(image, K, hand, center, radius)

    if len(sys.argv) >= 3 and sys.argv[2] == 'True':
        save_result(img, sys.argv[1][-7:])
        