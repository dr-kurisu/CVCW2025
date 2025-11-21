import numpy as np
import cv2
import logging
from typing import List

model = "cascade.xml"
cascade_name = f"./Dartboardcascade/{model}"
result = []
IOU_thr = 0.5

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename=f'withoutAnyOptimise_{model}.txt',
    filemode='w'
)

class BoundingBox:
    def __init__(self, *args):
        if len(args) == 4:
            # x, y, width, height
            self.x, self.y, self.width, self.height = args

        elif len(args) == 2:
            # start, end
            start, end = args
            self.x = start[0]
            self.y = start[1]
            self.width = end[0] - start[0]
            self.height = end[1] - start[1]
        else:
            raise RuntimeError("Wrong BBox init")
    
    @property
    def twoPoints(self):
        return ((self.x, self.y), (self.x + self.width, self.y + self.height))
    
    def __iter__(self):
        #(x, y, width, height)
        yield self.x
        yield self.y
        yield self.width
        yield self.height

    def __str__(self):
        return f"x: {int(self.x)}\ty: {int(self.y)}\tw: {int(self.width)}\th: {int(self.height)}"
    
def calculateIOU(box1: BoundingBox, box2: BoundingBox) -> float:
    # May need to check
    x1_min, y1_min = box1.x, box1.y
    x1_max, y1_max = box1.x + box1.width, box1.y + box1.height
    x2_min, y2_min = box2.x, box2.y
    x2_max, y2_max = box2.x + box2.width, box2.y + box2.height

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def calculateF1Score(recall: float, precision: float):
    # May have chance XD
    if recall + precision == 0:
        return 0.0
    return 2 * (recall * precision) / (recall + precision)

def calculateTP(detectBBoxes: List[BoundingBox], realBBoxes: List[BoundingBox]):
    matched_real = set()
    TP = 0
    for detect_i, detect_box in enumerate(detectBBoxes):
        best_IOU = 0
        best_real_i = -1
        for real_i, real_box in enumerate(realBBoxes):
            # prevent repeated matching same real box
            if real_i in matched_real:
                continue
            IOU = calculateIOU(real_box, detect_box)
            if IOU > best_IOU:
                best_IOU = IOU
                best_real_i = real_i
        if best_IOU >= IOU_thr:
            logging.info(f"Matched GT       : {realBBoxes[best_real_i]}")
            logging.info(f"Matched DetectBox: {detect_box}")
            logging.info(f"Best IOU         : {round(best_IOU, 3)}")
            TP += 1
            matched_real.add(best_real_i)
    return TP

def detectAndDisplay(frame):
	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
    # 3. Print number of Faces found
    bboxes = []
    # 4. Draw box around faces found
    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
        bboxes.append(BoundingBox(start_point, end_point))
    return bboxes

def readGroundtruth(filename='groundtruth.txt'):
    result = {}
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            x = float(content_list[1])
            y = float(content_list[2])
            width = float(content_list[3])
            height = float(content_list[4])
            if img_name not in result:
                result[img_name] = []
                imageNames.append(img_name)
            result[img_name].append(BoundingBox(x, y, width, height))
    return result

def drawGTBox(frame, ground_truth):
    for (x, y, w, h) in ground_truth:
        start = (int(x), int(y))
        end   = (int(x + w), int(y + h))
        cv2.rectangle(frame, start, end, (0, 0, 255), 2)   # red box
    return frame

model = cv2.CascadeClassifier(cascade_name)
imageNames = []
ground_truth = readGroundtruth("./ground_truth.txt")


for name in imageNames:
    logging.info(name + ".jpg")
    frame = cv2.imread("./Dartboard/" + name + ".jpg", 1)
    detectBBoxes = detectAndDisplay(frame)
    realBBoxes = ground_truth[name]
    TP = calculateTP(detectBBoxes=detectBBoxes, realBBoxes=realBBoxes)
    FP = len(detectBBoxes) - TP
    FN = len(realBBoxes) - TP
    TPR = TP / len(realBBoxes)
    FPR = FP / len(detectBBoxes)
    precision = TP / (TP + FP)
    f1 = calculateF1Score(recall=TPR, precision=precision)
    logging.info(f"TPR: {round(TPR, 3)}")
    logging.info(f"FPR: {round(FPR, 3)}")
    logging.info(f"F1 : {round(f1, 3)}")
    drawGTBox(frame, ground_truth[name])
    cv2.imwrite(f"./DartboardOut/{name}.jpg", frame)
    logging.info("")