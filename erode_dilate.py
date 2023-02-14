import cv2 as cv
import numpy as np


def repair(n):
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))

    img = cv.imread('colormap.jpg')
    rows, cols, channels = img.shape
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    res = np.copy(img)

    cv.imshow("img", img)
    cv.waitKey(10)

    lower_list = []
    upper_list = []

    # HSV 的上下界限
    lower_list.append(np.array([0, 43, 100]))
    upper_list.append(np.array([10, 255, 255]))

    lower_list.append(np.array([10, 43, 100]))
    upper_list.append(np.array([25, 255, 255]))

    lower_list.append(np.array([25, 43, 100]))
    upper_list.append(np.array([45, 255, 255]))

    lower_list.append(np.array([45, 43, 100]))
    upper_list.append(np.array([55, 255, 255]))

    lower_list.append(np.array([55, 43, 100]))
    upper_list.append(np.array([75, 255, 255]))

    lower_list.append(np.array([75, 43, 100]))
    upper_list.append(np.array([85, 255, 255]))

    lower_list.append(np.array([85, 43, 100]))
    upper_list.append(np.array([125, 255, 255]))

    lower_list.append(np.array([125, 43, 100]))
    upper_list.append(np.array([255, 255, 255]))

    mask_list = []
    layer_list = []
    eroded_list = []
    dilated_list = []

    # 通过上下限提取范围内的掩模mask
    for i in range(n):
        mask_list.append(cv.inRange(hsv, lower_list[i], upper_list[i]))
        eroded_list.append(cv.erode(mask_list[i], kernel1))
        layer_list.append(cv.bitwise_and(img, img, mask=eroded_list[i]))
        dilated_list.append(cv.dilate(layer_list[i], kernel2))

    for r in range(rows):
        for c in range(cols):
            for x in range(n):
                if dilated_list[x][r, c].all() != 0:
                    res[r, c] = dilated_list[x][r, c]
                    break

    cv.imshow("result", res)
    cv.waitKey(10)
    cv.imwrite("result.jpg", res)



def rainbowcolor(grayMat):
    res = cv.applyColorMap(255 - grayMat, cv.COLORMAP_RAINBOW)
    return res


def hotcolor(grayMat):
    res = cv.applyColorMap(255 - grayMat, cv.COLORMAP_HOT)
    return res