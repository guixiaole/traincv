import cv2 as cv
import time


def yolov4_detect(frame):
    net = cv.dnn_DetectionModel('yolo_data/yolo-obj.cfg', 'yolo_data/yolo-obj_best.weights')
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    # frame = cv.imread('images/test4.jpg')

    with open('yolo_data/obj.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    print(names)
    startTime = time.time()
    classes, confidences, boxes = net.detect(frame, confThreshold=0.8, nmsThreshold=0.5)
    endTime = time.time()
    print("Time: {}s".format(endTime - startTime))
    # for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
    #     label = '%.2f' % confidence
    #     label = '%s: %s' % (names[classId], label)
    #     labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #     left, top, width, height = box
    #     top = max(top, labelSize[1])
    #     cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
    #     cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
    #                  cv.FILLED)
    #     cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    #
    # cv.imshow('out', frame)
    # cv.waitKey(0)
    return boxes, confidences


if __name__ == '__main__':
    image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
    for image_id in image_ids:
        image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
        image = cv.imread(image_path)
        box, claasses = yolov4_detect(image)
        print(box)
        print(claasses)