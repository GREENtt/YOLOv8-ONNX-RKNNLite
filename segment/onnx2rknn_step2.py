import os, shutil, numpy as np, cv2
from utils import *
from rknn.api import RKNN

conf_thres = 0.65
iou_thres = 0.65
input_width = 640
input_height = 640
model_name = 'best'
model_path = "./model"
config_path = "./config"
result_path = "./result"
image_path = "./dataset/1.png"
video_path = "test.mp4"
video_inference = False
RKNN_MODEL = f'best-{input_height}-{input_width}.rknn'
CLASSES = ['lane', 'lane_line']

if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Build model
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input=f'{config_path}/{model_name}-{input_height}-{input_width}.model',
                                         data_input=f'{config_path}/{model_name}-{input_height}-{input_width}.data',
                                         model_quantization_cfg=f'{config_path}/{model_name}-{input_height}-{input_width}.quantization.cfg')
    
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    print('--> Move RKNN file into model folder')
    shutil.move(RKNN_MODEL, f"{model_path}/{RKNN_MODEL}")

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    if video_inference == True:
        cap = cv2.VideoCapture(video_path)
        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            print('--> Running model for video inference')
            outputs = rknn.inference(inputs=[image_3c])
            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, masks, shape = results
            if isinstance(masks, np.ndarray):
                mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
                cv2.imshow("mask_img", mask_img)
                cv2.imshow("vis_img", vis_img)
            else:
                print("No segmentation result")
            cv2.waitKey(10)
    else:
        # Preprocess input image
        image_3c = cv2.imread(image_path)
        image_4c, image_3c =  preprocess(image_3c, input_height, input_width)
        print('--> Running model for image inference')
        print( image_3c.shape)
        outputs = rknn.inference(inputs=[image_3c])

        colorlist = gen_color(len(CLASSES))
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, masks, shape = results
        if isinstance(masks, np.ndarray):
            mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            print('--> Save inference result')
        else:
            print("No segmentation result")
    print("RKNN inference finish")
    rknn.release()
    cv2.destroyAllWindows()
