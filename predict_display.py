#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes,get_boxes_dimensions
from frontend import YOLO
import json
import pytesseract
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = "tesseract.exe"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def equalize_img(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def equalize_hist(img):
    if (len(img.shape) >2):
        for c in range(0, img.shape[2]):
            img[:,:,c] = cv2.equalizeHist(img[:,:,c])
        return img
    return cv2.equalizeHist(img)

def saturate(img, satadj, valadj):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h,s,v) = cv2.split(imghsv)
    s = s * satadj
    v = v * valadj
    s = np.clip(s,0,255)
    v = np.clip(v,0,255)
    imghsv = cv2.merge([h,s,v])
    imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return imgrgb


def eq_sat(img,satadj):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#.astype("float32")
    (h,s,v) = cv2.split(imghsv)
    #s = s.astype("float") * satadj
    #print ("v",v)
    #cv2.imshow("h", h)
    #cv2.imshow("s", s)
    #cv2.imshow("v",v)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    s = np.clip(s,0,255)
    v = np.clip(v,0,255)
    imghsv = cv2.merge([h.astype("uint8"),s.astype("uint8"),v.astype("uint8")])
    imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return imgrgb

def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print (weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    license_counter = 0;

    if (image_path[-4:]).lower() == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        print ("video params", nb_frames, frame_h, frame_w)

        # video_writer = cv2.VideoWriter(video_out,
        #                        cv2.VideoWriter_fourcc(*'MPEG'), 
        #                        50.0, 
        #                        (frame_w, frame_h))

        grow_rate = 10

        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (i % 100 != 0):
                continue

            
            
            boxes = yolo.predict(image)

            #marked_image = draw_boxes(image.copy(), boxes, config['model']['labels'])

            refitted = get_boxes_dimensions(image,boxes)

            j = 0

            for box in refitted:
                box[0] = box[0] - grow_rate
                box[1] = box[1] - grow_rate
                box[2] = box[2] + grow_rate
                box[3] = box[3] + grow_rate
                if (box[0] < 0):
                    box[0] = 0
                if (box[1] < 0):
                    box[1] = 0
                if (box[2] > image.shape[1]):
                    box[2] = image.shape[1]
                if (box[3] > image.shape[0]):
                    box[3] = image.shape[0]

                
                print ("box", box)

                crop_img = image[box[1]:box[3],box[0]:box[2]]

                print ("crop shape", crop_img.shape)

                #saturated = saturate(crop_img,2,2.00)
                saturated = eq_sat(crop_img,1.5)
                #saturated = equalize_img(saturated)
                #saturated = equalize_hist(saturated)
                cv2.imshow("sat", saturated)


                gray_image = cv2.cvtColor(saturated, cv2.COLOR_BGR2GRAY)

                height, width = gray_image.shape[:2]

                gray_image = cv2.resize(gray_image, (width*4, height*2), interpolation = cv2.INTER_NEAREST)

                cv2.imshow("big", gray_image)

                cv2.imwrite("sample_licenses/" + str(license_counter) + ".png", gray_image)
                license_counter = license_counter + 1

                #--psm 10 --eom 3
                target = pytesseract.image_to_string(Image.fromarray(gray_image), lang='eng', boxes=False, config='-psm 12 -oem 2 -c tessedit_char_whitelist="0123456789"')
                #target = target.translate("0123456789")
                #target = re.sub('[0-9]','',target)
                set = '01234567890'
                target = ''.join([c for c in target if c in set])
                if (len(target) < 5 or len(target) > 8):
                    target = ""
                
                print ("target------------------------>", target)

                #cv2.imshow("cropped", np.uint8(bw_img))
                cv2.waitKey(1)    
                j = j + 1
                cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,255,0), 3)
                cv2.putText(image, 
                    target,
                    (box[0], box[1] - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,255,0), 2)

            siheight, siwidth = image.shape[:2]
            print ("si", siheight,siwidth)
            small_image = cv2.resize(image, (int(siwidth/3), int(siheight/3)), interpolation = cv2.INTER_NEAREST)

            cv2.imshow("display", np.uint8(small_image))
            #cv2.imshow("display", np.uint8(marked_image))
            cv2.waitKey(1)
            #video_writer.write(np.uint8(image))

        video_reader.release()
        #video_writer.release()  
    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
