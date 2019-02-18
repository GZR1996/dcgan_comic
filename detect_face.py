import cv2
import sys
import os.path
from glob import glob

def detect(filename, cascade_file='lbpcascade_animeface.xml'):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    # CascadeClassifier分类器
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度直方图均衡化
    gray = cv2.equalizeHist(gray)

    # image=gray: 输入图片
    # scaleFactor：表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1，即每次搜索窗口依次扩大10%
    # minNeighbors：
    # minSize：目标区域的最小范围
    # maxSize：目标区域的最大范围
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.2,
                                     minNeighbors=5,
                                     minSize=(48, 48))

    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        save_filename = '%s.jpg' % (os.path.basename(filename).split('.')[0])
        cv2.imwrite("faces/" + save_filename, face)

if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob('imgs/*.jpg')
    for filename in file_list:
        detect(filename)

