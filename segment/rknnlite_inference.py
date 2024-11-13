import os, cv2, time, numpy as np
from utils import *
from rknnlite.api import RKNNLite

conf_thres = 0.8
iou_thres = 0.9

input_width = 640
input_height = 640
model_name = 'ALL'
model_path = "./model"
config_path = "./config"
result_path = "./result"
image_path = "./dataset/032.png"
video_path = "1.mp4"
video_inference = True
RKNN_MODEL = f'./model/green-640-640.rknn'
CLASSES = ['road', 'lane_line']

if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

    if video_inference == True:
        cap = cv2.VideoCapture(video_path)
        imgs = []
        v = cv2.VideoWriter(f'./result/vis_img2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (720, 320))
        initime = time.time()
        num = 0
        while (cap.isOpened()):
            num += 1
            start = time.time()
            ret, image_3c = cap.read()
            print('ret:', ret)
            if not ret:
                break
            print('--> Running model for video inference')

            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            # ret = rknn_lite.init_runtime()

            image_3C = image_3c[np.newaxis, :]
            # print('111111',image_3C.shape)
            outputs = rknn_lite.inference(inputs=[image_3C])
            stop = time.time()
            fps = round(1 / (stop - start), 2)

            outputs[0] = np.squeeze(outputs[0])

            outputs[0] = np.expand_dims(outputs[0], axis=0)

            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres,
                                  classes=len(CLASSES))  ##[box,mask,shape]

            results = results[0]  ## batch=1
            boxes, masks, shape = results

            if type(masks) != list and masks.ndim == 3:
                mask_img, vis_img = vis_result(image_3c, results, colorlist, CLASSES, result_path)
                # cv2.imshow("mask_img", mask_img)
                cv2.putText(vis_img, str(fps), (1, 571), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.imshow("vis_img", vis_img)
                imgs.append(vis_img)

                # for i in imgs:
                i = cv2.resize(vis_img, (720, 320))
                v.write(i)

            else:
                print("-------------No segmentation result-------------")

                # img5 = image_3c[185:455,:]
                # img2 = np.zeros_like(img5)
                # cv2.imshow("1", image_3c)
            cv2.waitKey(1)
        end = time.time()
        print('avgTimes:', num / (end - initime), num, end, initime)

    else:
        image_3c = cv2.imread(image_path)  # (640,640,3)
        image_4c, image_3c = preprocess(image_3c, input_height, input_width)
        print('--> Running model for image inference')
        # ret = rknn_lite.init_runtime()
        start = time.time()
        image_3C2 = image_3c[np.newaxis, :]  # (1, 640, 640, 3)
        outputs = rknn_lite.inference(inputs=[image_3C2])  # len(outputs)->2
        stop = time.time()
        fps = round(1 / (stop - start), 2)

        outputs[0] = np.squeeze(outputs[0])

        outputs[0] = np.expand_dims(outputs[0], axis=0)

        colorlist = [(255, 255, 255), (0, 0, 0)]  # colorlist = gen_color(len(CLASSES))

        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres,
                              classes=len(CLASSES))  ##[box,mask,shape]

        results = results[0]  ## batch=1

        boxes, masks, shape = results
        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=0).astype(np.float32)

        if type(masks) != list and masks.ndim == 3:
            mask_img, vis_img = vis_result(image_3c, results, colorlist, CLASSES, result_path)
            print('--> Save inference result')
        else:
            print("-------------No segmentation result-------------")
    print("rknn_liteLite inference finish")
    rknn_lite.release()
    cv2.destroyAllWindows()
