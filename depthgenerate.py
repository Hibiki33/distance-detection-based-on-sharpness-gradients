import cv2
import numpy as np
import os
import time
from scipy import stats

time1 = time.time()
threshold_img = 25
threshold_gray = 20
scale = [(104, 156), (52, 78), (26, 26), (13, 13), (5, 5), (3, 3)]
# group = "new_warped/10"


def getdistance(a, b):
    return np.square(int(a[0]) - int(b[0])) + np.square(int(a[1]) - int(b[1])) + np.square(int(a[2]) - int(b[2]))


def getdepthmap(group):
    # np.set_printoptions(threshold=np.inf)
    imgs_path = "data/" + group
    imgs_list = os.listdir(imgs_path)

    # 获取图片大小
    tmp = cv2.imread(imgs_path + '/' + "1.jpg")
    tmp = cv2.resize(tmp, None, fx=0.05, fy=0.05)
    height, width, _ = tmp.shape
    print(height, width)

    # 申请输出矩阵
    max_clarity = np.zeros((height, width))
    dist = []

    def get_mean(x, y):
        d = 8
        M = np.sum(res[max(0, x - d):min(height - 1, x + d), max(0, y - d):min(width - 1, y + d)]) / np.count_nonzero(
            res[max(0, x - d):min(height - 1, x + d), max(0, y - d):min(width - 1, y + d)])
        # M = np.max(res[max(0, x - d):min(height - 1, x + d), max(0, y - d):min(width - 1, y + d)])
        return M

    def get_near(x, y):
        d = 5
        M = np.max(res[max(0, x - d):min(height - 1, x + d), max(0, y - d):min(width - 1, y + d)])
        return M

    for i in range(len(imgs_list)):
        dist1 = np.zeros((height, width))
        print('processing...', i + 1, len(imgs_list))
        # 读取图片
        path = imgs_path + '/' + str(i + 1) + '.jpg'
        print(path)
        img = cv2.imread(path)
        # 读取图像
        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        # img = img // 2
        # cv2.imwrite(str(i + 1) + ".jpg", img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = img_gray.astype(np.float64)
        img = img.astype(np.float64)
        # 梯度检测
        # clar_mat_x = cv2.Scharr(img, -1, 1, 0)
        # clar_mat_y = cv2.Scharr(img, -1, 0, 1)
        # clar_mat_hsv = cv2.addWeighted(clar_mat_y, 0.5, clar_mat_x, 0.5, 0)
        # clar_mat_hsv = np.sum(clar_mat_hsv, axis=2)
        # clar_mat_hsv = np.abs(clar_mat_hsv)
        clar_mat_gray_x = np.abs(cv2.Sobel(img_gray, -1, 1, 0))
        clar_mat_gray_y = np.abs(cv2.Sobel(img_gray, -1, 0, 1))
        clar_mat_gray = 0.5 * clar_mat_gray_x + 0.5 * clar_mat_gray_y
        print(np.max(clar_mat_gray))
        clar_mat = clar_mat_gray
        clar_mat[clar_mat < 20] = 0
        clar_mat = cv2.resize(clar_mat, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        # 清晰度结果
        # out_clar = (clar_mat - np.min(clar_mat)) / (np.max(clar_mat) - np.min(clar_mat)) * 255
        # cv2.imwrite("image/gardenFusion/" + group + "/fusionEach/" + "Scharr_" + str(i + 1) + ".jpg", out_clar)
        # cv2.imwrite(str(i + 1) + ".jpg", img_gray)
        dist.append(clar_mat)

    tmp = np.zeros_like(dist)
    out_clar = (dist - np.min(dist)) / (np.max(dist) - np.min(dist)) * 255
    # for k in range(len(imgs_list)):
    # cv2.imwrite(str(k + 1) + ".jpg", out_clar[k])
    for (w, h) in scale:
        if w == 3:
            size = np.power(w * h, 4)
        else:
            size = np.power(w * h, 5)
        for i in range(0, height, w // 2):
            for j in range(0, width, h // 2):
                for k in range(len(imgs_list)):
                    if np.count_nonzero(dist[k][i:min(i + w, height), j:min(j + h, width)]) > 0:
                        small_sum = np.sum(dist[k][i:min(i + w, height), j:min(j + h, width)])
                        tmp[k][i:min(i + w, height), j:min(j + h, width)] = \
                            max(small_sum / size * 10000, tmp[k][min(i + w, height) - 1][min(j + h, width) - 1])
                        # tmp[k][i:min(i + w, height), j:min(j + h, width)] = small_sum

    # for k in range(len(imgs_list)):
    #     aa = np.array([tmp[k], dist[k] * 10000])
    #     tmp[k] = np.max(aa, axis=0)

    for k in range(len(imgs_list)):
        out = (tmp[k] - np.min(tmp[k])) / (np.max(tmp[k] - np.min(tmp[k]))) * 255
        # cv2.imwrite(str(k + 1) + "_aa.jpg", out)

    dist = tmp.copy()
    maxx = np.max(dist, axis=0)
    for i in range(len(imgs_list)):
        dist[i][dist[i] < maxx] = 0
        dist[i][dist[i] != 0] = 1
        dist[i] = dist[i] * (255 - i / len(imgs_list) * 255)

    res = np.sum(dist, axis=0)
    exist = np.count_nonzero(dist, axis=0)
    exist[exist == 0] = 1
    res = res / exist
    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    res = res.astype(np.uint8)
    bottom = res.copy()
    # cv2.imwrite("ori1.jpg", res)
    cv2.imwrite("color_ori1.jpg", cv2.applyColorMap(255 - res, cv2.COLORMAP_RAINBOW))
    cv2.imshow("colormap", cv2.applyColorMap(255 - res, cv2.COLORMAP_RAINBOW))


    for i in range(height):
        for j in range(width):
            if res[i][j] < 60:
                aa = bottom[max(0, i - 3):min(height, i + 4), max(0, j - 3):min(width, j + 4)]
                if aa[aa > 60].any():
                    res[i][j] = stats.mode(aa, axis=None)[0]

    res = cv2.GaussianBlur(res, (3, 3), None)
    cv2.imwrite("colormap.jpg", cv2.applyColorMap(255 - res, cv2.COLORMAP_RAINBOW))
