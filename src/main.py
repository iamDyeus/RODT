import warnings
import os
import random
from pathlib import Path
import cv2
from ultralytics import YOLO
from utils.tracker import Tracker

class Main:
    def __init__(self, input_source, output_path='demo/people_out.mp4', detection_threshold=0.5):
        self.input_source = input_source
        self.output_path = self.relative_to_absolute(output_path)
        self.detection_threshold = detection_threshold
        self.webcamID = 0

        # Initialize video capture
        self.cap = self.init_video_capture()
        ret, self.frame = self.cap.read()

        if not ret:
            print(f"Error: Unable to read video input {self.input_source}")
            exit()

        # Initialize video writer if not in webcam mode
        if self.input_source.lower() != 'webcam':
            self.cap_out = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*'MP4V'),
                self.cap.get(cv2.CAP_PROP_FPS),
                (self.frame.shape[1], self.frame.shape[0])
            )

        # Initialize YOLO model
        self.model = YOLO("models/yolov8n.pt")

        # Initialize tracker
        self.tracker = Tracker()

        # Initialize colors for bounding boxes
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    def change_webcam(self, webcamID):
        self.webcamID = webcamID

    def relative_to_absolute(self, path) -> str:
        return str(Path(__file__).parent / path)

    def init_video_capture(self):
        if self.input_source.lower() == 'webcam':
            return cv2.VideoCapture(0)
        else:
            video_path = self.relative_to_absolute(self.input_source)
            return cv2.VideoCapture(video_path)

    def process_video(self):
        ret, frame = self.cap.read()
        while ret:
            results = self.model(frame)

            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    class_id = int(class_id)
                    if score > self.detection_threshold:
                        detections.append([x1, y1, x2, y2, score])

                self.tracker.update(frame, detections)

                for track in self.tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (self.colors[track_id % len(self.colors)]), 3)

            if self.input_source.lower() != 'webcam':
                self.cap_out.write(frame)
            else:
                cv2.imshow('Live Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            ret, frame = self.cap.read()

        self.cap.release()
        if self.input_source.lower() != 'webcam':
            self.cap_out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    # Removing DeprecationWarning for linear_sum_assignment from scipy  
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    
    # Example usage:
    print("\n\tEnter the Mode you'll like to run the program in:\n")
    print("\t1. For video file input")
    print("\t2. For webcam input\n")
    mode = int(input(">> "))

    try :
        if mode == 1:
            """
            For video file input ->
                if you want to test on a different video file, change the input_source to the 
                path of the video file relative to src directory
            """
            main = Main(input_source='demo/people.mp4')
            main.process_video()
        else: 
            """
            For webcam input ->
                default webcamID is 0, if you have multiple webcams, you can change the webcamID
                Use the change_webcam method to change the webcamID (remember to change the webcamID before calling process_video)
            """
            main = Main(input_source='webcam')
            # main.change_webcam(YOUR-WEBCAM-ID-INTEGER) # default ID is 0
            main.process_video()
    except Exception as e:
        print(f"Some Error Occurred : {e}")
        exit()

