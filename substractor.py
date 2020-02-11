import cv2
import numpy
from tracker.multitracker import STrack
from utils import utils
from torchvision.ops import nms
import torch



class Rectangle:
    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        self.min_x = min(x1, x2)
        self.max_x = max(x1, x2)
        self.min_y = min(y1, y2)
        self.max_y = max(y1, y2)

    def __str__(self):
        return 'Rectangle({0}, {1})'.format(self.points()[0], self.points()[1])

    def __repr__(self):
        return 'Rectangle({0}, {1})'.format(self.points()[0], self.points()[1])

    def points(self):
        return (self.min_x, self.min_y), (self.max_x, self.max_y)

    def is_intersect(self, other):
        if self.min_x > other.max_x or self.max_x < other.min_x:
            return False
        if self.min_y > other.max_y or self.max_y < other.min_y:
            return False
        return True

    def __and__(self, other):
        if not self.is_intersect(other):
            return Rectangle()
        min_x = max(self.min_x, other.min_x)
        max_x = min(self.max_x, other.max_x)
        min_y = max(self.min_y, other.min_y)
        max_y = min(self.max_y, other.max_y)
        return Rectangle(min_x, min_y, max_x, max_y)

    intersect = __and__

    def __or__(self, other):
        min_x = min(self.min_x, other.min_x)
        max_x = max(self.max_x, other.max_x)
        min_y = min(self.min_y, other.min_y)
        max_y = max(self.max_y, other.max_y)
        return Rectangle(min_x, min_y, max_x, max_y)

    union = __or__

    def __str__(self):
        return 'Rectangle({self.min_x},{self.max_x},{self.min_y},{self.max_y})'.format(self=self)

    def area(self):
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)


def remove_intersected(bboxes):
    result = []
    for i, a in enumerate(bboxes):
        ok = True
        for j, b in enumerate(bboxes):
            if (i != j):
                if ( (a.area() / 2.0)  < (a & b).area() ):
                   ok = False
                   break

        if (ok):
            result.append(a)

    return result


class SubstracktorDetector:
    def __init__(self):
        self.substractor = cv2.createBackgroundSubtractorMOG2(100, 90)


    def detect(self, img):
        """
        Detects moving objects
        :param img:
        :return: numpy.array
        bounding boxes with format top-left x, y, width, height
        """
        frame = cv2.blur(img, (4, 4))
        #apply background substraction
        fgmask = self.substractor.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 3)
        elemDilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        fgmask = cv2.dilate(fgmask, elemDilate)
        contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = []
        #looping for contours
        for c in contours:
            if cv2.contourArea(c) < 100:
                continue

            #get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)
            bbox.append((x, y, x + w, y + h))

            #draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        result = []
        if bbox:
            bbox_two_points = [Rectangle(*x) for x in bbox]
            filtered = remove_intersected(bbox_two_points)

            for rect in filtered:
                (x1, y1), (x2, y2) = rect.points()
                result.append(STrack((rect.min_x,
                                             rect.min_y,
                                             rect.max_x - rect.min_x,
                                             rect.max_y - rect.min_y), 0.5, None))
                #draw bounding box
                cv2.rectangle(fgmask, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.imshow('foreground and background',fgmask)

        return result


def test():
    video_path = '/home/noskill/1/a.avi'
    #read video file
    cap = cv2.VideoCapture(video_path)
    det = SubstracktorDetector()
    while (cap.isOpened):
        #if ret is true than no error with cap.isOpened
        ret, frame = cap.read()
        if ret:
            det.detect(frame)



if __name__ == '__main__':
    test()




