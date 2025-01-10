from imutils import contours
import numpy as np
import cv2
import argparse
import myutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())
# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取一个模板图像
img = cv2.imread(args["template"])
# cv_show('img',img)
# 灰度图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2_show('ref',ref)
# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，就黑白的（不是灰度图像），cv2.RETR_EXTERNAL只检测外轮廓
#cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中，每个元素都是图像中的一个轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#外面轮廓





