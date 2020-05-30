import platform
import os

is_win = 'windows' in platform.platform().lower()
if is_win:
    message = 'Please run "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat" before running this.'
else:
    message = "Add the following line to ~/.bashrc and re-run.\nsource /opt/intel/openvino/bin/setupvars.sh"
print (message)


from PIL import Image
import numpy as np
import cv2

os.chdir("C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\python\python3.7")
import openvino
from openvino import inference_engine
from openvino.inference_engine import IEPlugin, IENetwork

acc = 0
with open("C:/Users/Dell/Desktop/DL/label.txt", 'r') as f:
        labels_map = [x.split("\n") for x in f]
        
with open("C:/Users/Dell/Desktop/DL/inference_labels.txt", 'r') as f1:
        inf_labels_map = [x.split("\n") for x in f1]

def pre_process_image(imagePath, img_height=100):
    n, h, w,c = [1, img_height, img_height, 1]
    image = Image.open(imagePath)
    image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    processedImg = cv2.resize(image,(h, w))  
    processedImg = (np.array(processedImg)) 

    return image, processedImg, imagePath

plugin_dir = None
os.chdir("C:/Users/Dell/Desktop/DL")
model_xml = './model/frozen_model.xml'
model_bin = './model/frozen_model.bin'
plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
net = IENetwork(model=model_xml, weights=model_bin)
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = plugin.load(network=net)
del net

for i in range(36):
    image, processedImg, imagePath = pre_process_image("C:/Users/Dell/Desktop/DL/Inference_Data/Case "+str(i+1)+"/DWI.jpg")
    res = exec_net.infer(inputs={input_blob: processedImg})
    transformed_label = [39, 36, 28, 0, 31, 35, 13, 20, 14, 38, 3, 24, 8, 16, 33, 17,  2, 19, 12, 27,  9, 36,  5,  4, 0, 28, 18, 25,  6,  7, 10,  2,  1, 23, 34, 36, 11, 30, 21, 15, 26, 22, 29, 37, 32, 20]
    a = list(res.values())[0][0]
    res_list = list(a)
    class_id  = res_list.index(max(res_list))
    print('Predicted Disease label : ',transformed_label.index(int(class_id)))
    predicted_class = labels_map[transformed_label.index(int(class_id))+1][0]
    print('Predicted Disease name : ',predicted_class)
    if(predicted_class == inf_labels_map[i][0]):
        acc += 1
        print(i)
accuracy = (acc/36)*100
print("Accuracy is ",accuracy)        

