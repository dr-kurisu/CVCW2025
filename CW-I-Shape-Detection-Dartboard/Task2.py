import numpy as np
import cv2
import logging
from typing import List
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename='WithShapeDetectors.txt',
    filemode='w'
)


cascade_name = "./Dartboardcascade/cascade.xml"
IOU_thr = 0.4

# Circle Detector
MIN_R = 30
MAX_R = 100
SAMPLE_R = 10
MAG_THR = 0.4
CENTER_BIAS_X = 0.07
CENTER_BIAS_Y = 0.07
RADIUS_THR = 0.8


class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), int(self.r), (0, 255, 0), 2)

    def __str__(self):
        return f"center: {self.x, self.y} \tradius: {round(self.r, 2)}"

class BoundingBox:
    def __init__(self, *args):
        if len(args) == 4:
            # x, y, width, height
            self.x, self.y, self.width, self.height = args
            self.best_circle = None

        elif len(args) == 2:
            # start, end
            start, end = args
            self.x = start[0]
            self.y = start[1]
            self.width = end[0] - start[0]
            self.height = end[1] - start[1]
            self.best_circle = None
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
    
    def draw(self, frame):
        start_point, end_point = self.twoPoints
        frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

def correlation(image, kernel):
    height, width = image.shape
    newImage = np.zeros((height, width))
    kernelSize = kernel.shape[0]
    # padding
    padImage = np.pad(image, 
                  ((kernelSize//2, kernelSize//2), 
                   (kernelSize//2, kernelSize//2)), 
                  mode="edge")
    # convolution
    for y in range(height):
        for x in range(width):
            region = padImage[y:y+kernelSize, x:x+kernelSize]
            newImage[y, x] = np.sum(region * kernel)

    newImage = np.clip(newImage, 0, 255)
    return newImage

def sobel(frame_gray):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = correlation(frame_gray, kernel=kernel_x)
    grad_y = correlation(frame_gray, kernel=kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.degrees(np.arctan2(grad_y, grad_x))
    direction = (direction + 360) % 360
    return magnitude, direction

def selectRegion(frame, bbox: BoundingBox):
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)
    return frame[y1:y2, x1:x2]

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

def drawBBox(frame, BBoxes: List[BoundingBox]):
    for box in BBoxes:
        if box.best_circle is not None:
            box.draw(frame)

def hough_fix_r(mag, dir, r, mask):
    h, w = mag.shape
    acc = np.zeros((h, w))
    edge_points = np.argwhere(mask)
    for (y, x) in edge_points:
        theta0 = dir[y, x]
        if np.isnan(theta0):
            continue
        for theta in range(int(theta0 - 20), int(theta0 + 20), 2):
            a = int(x + r*math.cos(math.radians(theta)))
            b = int(y + r*math.sin(math.radians(theta)))
            if 0 <= a < w and 0 <= b < h:
                acc[b, a] += 1
    return acc

def circlesDectection(bboxes: List[BoundingBox], min_r: int, max_r: int, sample_r: int):
    circles = []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    for box in bboxes:
        region = selectRegion(frame_gray, box)
        magnitude, direction = sobel(region)
        mask = magnitude > (MAG_THR * magnitude.max())  # dynamic
        
        # check thr
        # mask_vis = (mask.astype(np.uint8)) * 255
        # cv2.imshow("Edge Mask", mask_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        radii = np.linspace(min_r, max_r, sample_r)
        acc = np.zeros((len(radii), box.height, box.width), dtype=np.uint64)
        for i, r in enumerate(radii):
            acc[i] = hough_fix_r(magnitude, direction, r, mask)

        best_r = 0
        best_center = (0, 0)
        for i, r in enumerate(radii):
            acc_i = acc[i]
            max_vote = acc_i.max()
            threshold = max(max_vote * 0.95, 20)  #dynamic thr
            centers = np.argwhere(acc_i > threshold)
            if centers.size == 0:
                continue
            # add every center
            # center (y, x)
            for center in centers:
                # center filter
                global_x = box.x + center[1]
                global_y = box.y + center[0]
                x_flag = box.x + (0.5 - CENTER_BIAS_X) * box.width  < global_x < box.x + (0.5 + CENTER_BIAS_X) * box.width
                y_flag = box.y + (0.5 - CENTER_BIAS_Y) * box.height < global_y < box.y + (0.5 + CENTER_BIAS_Y) * box.height
                if x_flag and y_flag:
                    if r > best_r:
                        best_r = r
                        best_center = global_x, global_y
        # Best r is greater than (average width + height) * RADIUS_THR
        if best_r > RADIUS_THR * (box.height + box.width) / 4:
            circle = Circle(best_center[0], best_center[1], best_r)
            box.best_circle = circle
            circles.append(circle)
            print(circle)
    return circles
        
def benchmark(detectBBoxes: List[BoundingBox], realBBoxes: List[BoundingBox]):
    TP = calculateTP(detectBBoxes=detectBBoxes, realBBoxes=realBBoxes)
    FP = len(detectBBoxes) - TP
    FN = len(realBBoxes) - TP
    TPR = TP / len(realBBoxes)
    if len(detectBBoxes) == 0:
        FPR = 0.0
    else:
        FPR = FP / len(detectBBoxes)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    f1 = calculateF1Score(recall=TPR, precision=precision)
    logging.info(f"TPR: {round(TPR, 3)}")
    logging.info(f"FPR: {round(FPR, 3)}")
    logging.info(f"F1 : {round(f1, 3)}")

model = cv2.CascadeClassifier(cascade_name)
imageNames = []
ground_truth = readGroundtruth("./ground_truth.txt")


for name in imageNames:
    logging.info(name + ".jpg")
    print(name)

    frame = cv2.imread("./Dartboard/" + name + ".jpg", 1)
    detectBBoxes = detectAndDisplay(frame)
    realBBoxes = ground_truth[name]

    circles = circlesDectection(bboxes=detectBBoxes, min_r=MIN_R, max_r=MAX_R, sample_r=SAMPLE_R)
    detectBBoxes = [box for box in detectBBoxes if box.best_circle is not None]

    # Drawing
    drawBBox(frame, BBoxes=detectBBoxes)
    for circle in circles:
        circle.draw(frame)
    drawGTBox(frame, ground_truth[name])
    benchmark(detectBBoxes, realBBoxes)
    cv2.imwrite(f"./DartboardOut/{name}.jpg", frame)
    logging.info("")