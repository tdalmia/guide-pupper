#!/usr/bin/env python3
"""
ROS2 node for Hailo object detection and tracking
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image, CompressedImage
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import supervision as sv
import numpy as np
import cv2
import queue
import sys
import os
from typing import Dict, List, Tuple
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference


class HailoDetectionNode(Node):
    def __init__(self):
        super().__init__('hailo_detection_node')

        # Declare and get parameters
        self.declare_parameter('model_path', 'yolov5m_wo_spp_60p.hef')
        self.declare_parameter('labels_path', 'coco.txt')
        self.declare_parameter('score_threshold', 0.5)

        self.model_path = self.get_parameter('model_path').value
        self.labels_path = self.get_parameter('labels_path').value
        self.score_threshold = self.get_parameter('score_threshold').value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Set up publishers and subscribers
        self.detection_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.annotated_pub = self.create_publisher(CompressedImage, 'annotated_image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'detection_markers', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # Initialize Hailo inference
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.hailo_inference = HailoAsyncInference(
            hef_path=self.model_path,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
        )
        self.model_h, self.model_w, _ = self.hailo_inference.get_input_shape()

        # Initialize tracking and annotation
        self.box_annotator = sv.RoundBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.tracker = sv.ByteTrack()

        # Load class names
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.class_names = f.read().splitlines()

        # Start inference thread
        self.inference_thread = threading.Thread(target=self.hailo_inference.run)
        self.inference_thread.start()

    def image_callback(self, msg):
        # Convert ROS Image to CV2
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        video_h, video_w = frame.shape[:2]

        # Rotate 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Swap r and b channels, then multiply r by 0.5 to fix the colors
        frame = frame[:, :, ::-1]
        frame[:, :, 0] = frame[:, :, 0] * 0.5

        # Preprocess frame
        preprocessed_frame = self.preprocess_frame(frame, self.model_h, self.model_w, video_h, video_w)

        # Run inference
        self.input_queue.put([preprocessed_frame])
        _, results = self.output_queue.get()

        if len(results) == 1:
            results = results[0]

        # Process detections
        detections = self.extract_detections(results, video_h, video_w, self.score_threshold)

        # Create Detection2DArray message
        detection_msg = Detection2DArray()
        detection_msg.header = msg.header

        # Create MarkerArray message
        marker_array = MarkerArray()

        # Convert detections to ROS messages
        for i in range(detections["num_detections"]):
            print("Class ID: ", detections["class_id"][i])
            if str(detections["class_id"][i]) != "0":
                continue
            det = Detection2D()
            det.bbox.center.position.x = float((detections["xyxy"][i][0] + detections["xyxy"][i][2]) / 2)
            det.bbox.center.position.y = float((detections["xyxy"][i][1] + detections["xyxy"][i][3]) / 2)
            det.bbox.size_x = float(detections["xyxy"][i][2] - detections["xyxy"][i][0])
            det.bbox.size_y = float(detections["xyxy"][i][3] - detections["xyxy"][i][1])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(detections["class_id"][i])
            hyp.hypothesis.score = float(detections["confidence"][i])
            det.results.append(hyp)

            detection_msg.detections.append(det)

            # Create marker for bounding box
            marker = Marker()
            marker.header = msg.header
            marker.ns = "detection_boxes"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.01  # Line width
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Add points to form rectangle
            x1, y1 = float(detections["xyxy"][i][0]), float(detections["xyxy"][i][1])
            x2, y2 = float(detections["xyxy"][i][2]), float(detections["xyxy"][i][3])
            points = [
                (x1, y1, 0.0),
                (x2, y1, 0.0),
                (x2, y2, 0.0),
                (x1, y2, 0.0),
                (x1, y1, 0.0)  # Close the rectangle
            ]
            for x, y, z in points:
                p = Point()
                p.x = x
                p.y = y
                p.z = z
                marker.points.append(p)

            marker_array.markers.append(marker)

        # Publish detections
        self.detection_pub.publish(detection_msg)
        self.marker_pub.publish(marker_array)

        # Create and publish annotated image
        if detections["num_detections"]:
            annotated_frame = self.postprocess_detections(
                frame, detections, self.class_names, self.tracker,
                self.box_annotator, self.label_annotator
            )
            _, jpg_buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_msg = CompressedImage()
            annotated_msg.format = "jpeg"
            annotated_msg.data = jpg_buffer.tobytes()
        else:
            return
            # _, jpg_buffer = cv2.imencode('.jpg', frame)
            # annotated_msg = CompressedImage()
            # annotated_msg.format = "jpeg"
            # annotated_msg.data = jpg_buffer.tobytes()
        annotated_msg.header = msg.header
        self.annotated_pub.publish(annotated_msg)

    def preprocess_frame(
        self, frame: np.ndarray, model_h: int, model_w: int, video_h: int, video_w: int
    ) -> np.ndarray:
        if model_h != video_h or model_w != video_w:
            frame = cv2.resize(frame, (model_w, model_h))
        return frame

    def extract_detections(
        self, hailo_output: List[np.ndarray], h: int, w: int, threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        xyxy: List[np.ndarray] = []
        confidence: List[float] = []
        class_id: List[int] = []
        num_detections: int = 0

        for i, detections in enumerate(hailo_output):
            if len(detections) == 0:
                continue
            for detection in detections:
                bbox, score = detection[:4], detection[4]

                if score < threshold:
                    continue

                bbox[0], bbox[1], bbox[2], bbox[3] = (
                    bbox[1] * w,
                    bbox[0] * h,
                    bbox[3] * w,
                    bbox[2] * h,
                )

                xyxy.append(bbox)
                confidence.append(score)
                class_id.append(i)
                num_detections += 1

        return {
            "xyxy": np.array(xyxy),
            "confidence": np.array(confidence),
            "class_id": np.array(class_id),
            "num_detections": num_detections,
        }

    def postprocess_detections(
        self, frame: np.ndarray,
        detections: Dict[str, np.ndarray],
        class_names: List[str],
        tracker: sv.ByteTrack,
        box_annotator: sv.RoundBoxAnnotator,
        label_annotator: sv.LabelAnnotator,
    ) -> np.ndarray:
        sv_detections = sv.Detections(
            xyxy=detections["xyxy"],
            confidence=detections["confidence"],
            class_id=detections["class_id"],
        )

        sv_detections = tracker.update_with_detections(sv_detections)

        labels: List[str] = [
            f"#{tracker_id} {class_names[class_id]}"
            for class_id, tracker_id in zip(sv_detections.class_id, sv_detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), detections=sv_detections
        )
        annotated_labeled_frame = label_annotator.annotate(
            scene=annotated_frame, detections=sv_detections, labels=labels
        )
        return annotated_labeled_frame


def main(args=None):
    rclpy.init(args=args)
    node = HailoDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
