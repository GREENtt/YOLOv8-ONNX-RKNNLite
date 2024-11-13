import os, shutil
from ultralytics import YOLO

model_name = 'best11'
input_width = 640
input_height = 640
model_path = "model"

isExist = os.path.exists(model_path)
if not isExist:
   os.makedirs(model_path)

model = YOLO(f"./{model_path}/{model_name}.pt")
# model.export(format="onnx", opset=12, dynamic=False)
# os.rename(f"./{model_path}/{model_name}.onnx", f"{model_name}-{input_height}-{input_width}.onnx")
# shutil.move(f"{model_name}-{input_height}-{input_width}.onnx", f"./{model_path}/{model_name}-{input_height}-{input_width}.onnx")
# shutil.move(f"./{model_path}/{model_name}.pt", f"{model_path}/{model_name}.pt")

#
# print("===========  onnx =========== ")
# from ultralytics import YOLO

# model = YOLO('./model/best11.pt')
results = model(task='detect', mode='predict', source='C:/Users\lzy06\Desktop\zxq/relaticdata',
                line_width=3, show=False, save=True, device='cpu')

