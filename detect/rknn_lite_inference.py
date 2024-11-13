import glob
import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from math import exp

RKNN_MODEL = './model/detect_FQ.rknn'

dataset_file = './dataset.txt'
img_folder = "./dataset"
video_path = "00001.mp4"
video_inference = True

result_path = './detect_result'
CLASSES = ['broke', 'good', 'lose']

meshgrid = []

class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
nmsThresh = 0.5
objectThresh = 0.5

input_imgH = 640
input_imgW = 640


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def GenerateMeshgrid():
    for index in range(headNum):
        for i in range(mapSize[index][0]):
            for j in range(mapSize[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []
    output = []
    for i in range(len(out)):
        print(out[i].shape)
        output.append(out[i].reshape((-1)))

    scale_h = img_h / input_imgH
    scale_w = img_w / input_imgW

    gridIndex = -2
    cls_index = 0
    cls_max = 0

    for index in range(headNum):
        reg = output[index * 2 + 0]
        cls = output[index * 2 + 1]

        for h in range(mapSize[index][0]):
            for w in range(mapSize[index][1]):
                gridIndex += 2

                if 1 == class_num:
                    cls_max = sigmoid(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])
                    cls_index = 0
                else:
                    for cl in range(class_num):
                        cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]
                        if 0 == cl:
                            cls_max = cls_val
                            cls_index = cl
                        else:
                            if cls_val > cls_max:
                                cls_max = cls_val
                                cls_index = cl
                    cls_max = sigmoid(cls_max)

                if cls_max > objectThresh:
                    regdfl = []
                    for lc in range(4):
                        sfsum = 0
                        locval = 0
                        for df in range(16):
                            temp = exp(reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h *
                                           mapSize[index][1] + w])
                            reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][
                                1] + w] = temp
                            sfsum += temp

                        for df in range(16):
                            sfval = reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][
                                1] + w] / sfsum
                            locval += sfval * df
                        regdfl.append(locval)

                    x1 = (meshgrid[gridIndex + 0] - regdfl[0]) * strides[index]
                    y1 = (meshgrid[gridIndex + 1] - regdfl[1]) * strides[index]
                    x2 = (meshgrid[gridIndex + 0] + regdfl[2]) * strides[index]
                    y2 = (meshgrid[gridIndex + 1] + regdfl[3]) * strides[index]

                    xmin = x1 * scale_w
                    ymin = y1 * scale_h
                    xmax = x2 * scale_w
                    ymax = y2 * scale_h

                    xmin = xmin if xmin > 0 else 0
                    ymin = ymin if ymin > 0 else 0
                    xmax = xmax if xmax < img_w else img_w
                    ymax = ymax if ymax < img_h else img_h

                    box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax)
                    detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)

    return predBox


def export_rknnlite_inference(img):
    # Create RKNN object
    rknnlite = RKNNLite(verbose=False)

    # Load ONNX model
    print('--> Loading model')
    ret = rknnlite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    # ret = rknnlite.init_runtime()
    ret = rknnlite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknnlite.inference(inputs=[img])
    rknnlite.release()
    print('done')

    return outputs


def get_dataset_txt(dataset_path, dataset_savefile):
    file_data = glob.glob(os.path.join(dataset_path, "*.png"))
    with open(dataset_savefile, "r") as f:
        for file in file_data:
            f.readlines(f"{file}\n")


if __name__ == '__main__':
    print('This is main ...')
    GenerateMeshgrid()
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

    if video_inference == False:
        print('--> image -----------------------------------------')
        img_names = os.listdir(img_folder)
        initime = time.time()
        num = 0
        allfps = 0
        for name in img_names:
            img_path = os.path.join(img_folder, name)
            num += 1

            orig_img = cv2.imread(img_path)
            img_h, img_w = orig_img.shape[:2]

            origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
            origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)

            img = np.expand_dims(origimg, 0)
            start = time.time()
            outputs = rknn_lite.inference(inputs=[img])  # outputs = export_rknnlite_inference(img)

            out = []
            for i in range(len(outputs)):
                out.append(outputs[i])

            predbox = postprocess(out, img_h, img_w)

            print('detect:', len(predbox))
            fps = 1 / (time.time() - start)
            allfps += fps

            print('fps: ', fps)
            for i in range(len(predbox)):
                xmin = int(predbox[i].xmin)
                ymin = int(predbox[i].ymin)
                xmax = int(predbox[i].xmax)
                ymax = int(predbox[i].ymax)
                classId = predbox[i].classId
                score = predbox[i].score

                cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                ptext = (xmin, ymin)
                title = CLASSES[classId] + ":%.2f" % (score)
                cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imwrite(f'./{result_path}/{name}', orig_img)
            cv2.imshow("test", orig_img)
            cv2.waitKey(1)

        end = time.time()
        print('avgFPS, avgTime:', allfps / num, (end - initime) / num)
    else:
        print('--> video -----------------------------------------')
        cap = cv2.VideoCapture(video_path)
        initime = time.time()
        num = 0
        v = cv2.VideoWriter(f'./{result_path}/detect.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))
        allfps = 0
        while (cap.isOpened()):
            num += 1
            ret, frame = cap.read()
            print('ret:', ret)
            if not ret:
                break
            img_h, img_w = frame.shape[:2]

            orig_img = cv2.resize(frame, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            img = np.expand_dims(orig_img, 0)
            start = time.time()
            outputs = rknn_lite.inference(inputs=[img])  # outputs = export_rknnlite_inference(img)

            out = []
            for i in range(len(outputs)):
                out.append(outputs[i])

            predbox = postprocess(out, img_h, img_w)

            print('detect:', len(predbox))
            fps = 1 / (time.time() - start)
            allfps += fps

            print('fps: ', fps)
            for i in range(len(predbox)):
                xmin = int(predbox[i].xmin)
                ymin = int(predbox[i].ymin)
                xmax = int(predbox[i].xmax)
                ymax = int(predbox[i].ymax)
                classId = predbox[i].classId
                score = predbox[i].score
                print(f'point  score :', CLASSES[classId], score)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                ptext = (xmin, ymin)
                title = CLASSES[classId] + ":%.2f" % (score)
                cv2.putText(frame, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("output", frame)
            # i = cv2.resize(frame, (640, 640))
            v.write(frame)
            cv2.imwrite(f'./{result_path}/test_rknn_result.jpg', frame)
            cv2.waitKey(1)
        end = time.time()
        print('avgFPS, avgTime:', allfps / num, (end - initime) / num)
    rknn_lite.release()


