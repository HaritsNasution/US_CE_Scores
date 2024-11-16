import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

def resize(img):
    h,w = img.shape[:2]
    new_h = int((h/1080)*720)
    new_w = int((w/1080)*720)
    return cv2.resize(img,(new_w,new_h))

def crop(img):
    h,w = img.shape[:2]
    return img[350:h-275,150:w-75,:]

def extract(result):
    data = []
    players = {}
    for r in result:
        text = r[1]
        if text.isdigit():
            score = text
            players = {'IGN':name,'CE':score}
            data.append(players)
            name = score = ''
        else:
            name = text
    return data

def process_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    resized = resize(img)
    cropped = crop(resized)
    result, elapse = engine(cropped)
    data = extract(result)
    return data