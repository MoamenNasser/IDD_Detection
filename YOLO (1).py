import cv2
import time
import random
class YOLOV4:
    def __init__(self,
                 cfg = 'yolov4.cfg',
                 weights = 'yolov4.weights',
                 use_gpu = False,
                 input_size = 512,
                 classes = 'classes.names'):
        self.cfg = cfg
        self.weights = weights
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.classes = classes
        self.net = None
        self.build_model()
    def build_model(self):
        self.net =cv2.dnn_DetectionModel(self.cfg, self.weights)
        if self.use_gpu is True:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        if self.input_size % 32 != 0:
            raise Exception('Invalid Input Size')

        self.net.setInputSize(self.input_size, self.input_size)
        self.net.setInputScale(1.0/ 255) #normalize input
        self.net.setInputSwapRB(True) #open cv uses BGR instead of RGB
        with open(self.classes, 'r') as f:
            self.names = f.read().splitlines()
    def image_inf(self, image, is_it_path = False, show_image = True, save_image = True):
        if is_it_path:
            frame = cv2.imread(image)
        else:
            frame = image
        timer = time.time()
        #nmsThreshold: non maximum supression threshold
        classes, confidences, boxes = self.net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
        print(f'[Info] Time Taken: {time.time() - timer}')
        if not len(classes) == 0:
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%s: %.2f' % (self.names[classId], confidence)
                left, top, width, height = box
                print(box)
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)
                cv2.rectangle(frame, box, color=(b, g, r), thickness=2)
                cv2.rectangle(frame, (left, top), (left + len(label) * 20, top - 30), (b, g, r), cv2.FILLED)
                cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255 - b, 255 - g, 255 - r), 1,
                            cv2.LINE_AA)

        if save_image: cv2.imwrite('result.jpg', frame)
        if show_image:
            cv2.imshow('Inference', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                return
    def return_boxes(self, image):
        classes, confidences, boxes = self.net.detect(image, confThreshold=0.1, nmsThreshold=0.4)
        return classes, confidences, boxes
