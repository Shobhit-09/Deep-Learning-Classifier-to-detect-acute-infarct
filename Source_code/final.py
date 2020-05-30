from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
import openvino
from openvino import inference_engine
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import platform


is_win = 'windows' in platform.platform().lower()

if is_win:
    message = 'Please run "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat" before running this.\n'
else:
    message = "Add the following line to ~/.bashrc and re-run.\nsource /opt/intel/openvino/bin/setupvars.sh\n"
print (message)

def build_agrparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Useful commands')
    args.add_argument("-x", "--xml", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-b", "--bins", help="Required. Path to an .bin file with the trained model weights.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to image.",
                      required=True, type=str)
    args.add_argument("--labels", help=" Path to labels mapping file", required =True, type=str)
    

    return parser

    
def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_agrparser().parse_args()
    model_xml = args.xml
    model_bin = args.bins 

    log.info("Inference Engine Inialised")
    ie = IECore()
    plugin_dir = None

    #Read IR
    log.info("Loading XML and BIN files:\n\t{}\n\t{}".format(model_xml, model_bin))
    plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
    net = IENetwork(model=model_xml, weights=model_bin)
    dict_for_input = {}
    input_blob = next(iter(net.inputs))
 

    assert len(net.outputs) == 1, "Infer one image at a time"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net)
    
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    
    input_image = args.input
    assert os.path.isfile(args.input), "Specified input file doesn't exist"
    with open(args.labels, 'r') as f:
        labels_map = [x.split("\n") for x in f]
        
            
    
       
    
    
            
 
    img=cv2.imread(input_image,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(w,h))
    img = img.reshape((n, c, h, w))
    dict_for_input[input_blob] = img
    res = exec_net.infer(inputs = dict_for_input)
    a = list(res.values())[0][0]
    res_list = list(a)
    print("\nThe output array is: \n",res_list)
    transformed_label = [39, 36, 28, 0, 31, 35, 13, 20, 14, 38, 3, 24, 8, 16, 33, 17,  2, 19, 12, 27,  9, 36,  5,  4, 0, 28, 18, 25,  6,  7, 10,  2,  1, 23, 34, 36, 11, 30, 21, 15, 26, 22, 29, 37, 32, 20]
    class_id  = res_list.index(max(res_list))
    print('\nPredicted Disease label : ',transformed_label.index(int(class_id)))
    predicted_class = labels_map[transformed_label.index(int(class_id))+1][0]
    print('\nPredicted Disease name : ',predicted_class)
  
main()
