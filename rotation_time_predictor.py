# --------------------------- Imports ---------------------------
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from datetime import datetime, timezone
import json
import time
import os
from shapely.geometry import Point, Polygon
from google.colab import files

LOGGER.setLevel("WARNING")

# --------------------------- Logger Setup ---------------------------
log_dir = "logs_speed"
os.makedirs(log_dir, exist_ok=True)

class DualLogger:
    current_date = datetime.now(timezone.utc).strftime("%d-%m-%Y")
    log_file_name = os.path.join(log_dir, f"logs_speed_{current_date}.log")
    log = open(log_file_name, "a", encoding="utf-8")

    @staticmethod
    def log_message(message, to_console=False):
        today = datetime.now(timezone.utc).strftime("%d-%m-%Y")

        if today != DualLogger.current_date:
            DualLogger.log.close()
            DualLogger.current_date = today
            DualLogger.log_file_name = os.path.join(log_dir, f"logs_speed_{today}.log")
            DualLogger.log = open(DualLogger.log_file_name, "a", encoding="utf-8")

        if not message.endswith("\n"):
            message += "\n"

        msg = f"{datetime.now(timezone.utc).isoformat()} - {message}"
        DualLogger.log.write(msg)
        DualLogger.log.flush()

        if to_console:
            print(message, end="")

    @staticmethod
    def close():
        DualLogger.log.close()


# ============================ Main Class ============================
class LineMonitor:

    def __init__(self):
        self.marker_model_path = "filler_marker_bright_26_nov_object_detection.pt"
        self.video_path = "15_fps_rtmp_stream_part_1.mp4"
        self.output_path = "15_fps_rtmp_stream_part_1_visual.mp4"

        self.rotation_time = 0
        self.has_completed_rotation = False
        self.number_of_classes = 2

        self.last_marker_position = None
        self.updated_color_flag = True
        self.show_updated = False

        self._init_speed_config()
        self._initialize_models()

    # -------------------- Speed & Color Configuration --------------------
    def _init_speed_config(self):

        # ROI Polygon for RTMP stream
        self.polygon_x = [0, 0, 1380, 1380]
        self.polygon_y = [0, 300, 0, 0]
        self.polygon_pts = np.array(list(zip(self.polygon_x, self.polygon_y)), dtype=np.int32)

        # HSV Red Range
        self.lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
        self.upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        self.lower_red2 = np.array([160, 70, 50], dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        self.kernel = np.ones((3, 3), np.uint8)

    # -------------------- ROI Mask Initialization --------------------
    def _init_speed_frame_logic(self):
        mask_full = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.fillPoly(mask_full, [self.polygon_pts], 255)

        self.x, self.y, self.w, self.h = cv2.boundingRect(self.polygon_pts)
        mask_roi = mask_full[self.y:self.y+self.h, self.x:self.x+self.w]
        self.mask_roi_bool = mask_roi.astype(bool)

    # -------------------- Model & Video Initialization --------------------
    def _initialize_models(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.marker_detection_model = YOLO(self.marker_model_path).to(self.device).eval()

        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")

        self.frame_height, self.frame_width = first_frame.shape[:2]
        self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        self._init_speed_frame_logic()

    # -------------------- Core Processing --------------------
    def find_line_rotation(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open input video")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.video_fps,
                              (self.frame_width, self.frame_height))

        self.current_frame = 1
        self.frame_crossed = 1
        start_x = 130

        poly = Polygon(self.polygon_pts.reshape(-1, 2))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            print("self.video fps : ", self.video_fps)

            cv2.polylines(frame, [self.polygon_pts], True, (255, 255, 0), 2)
            cv2.line(frame, (start_x, 0), (start_x, self.frame_height), (0, 255, 255), 2)

            # -------------------- Red Highlighting --------------------
            roi_bgr = frame[self.y:self.y+self.h, self.x:self.x+self.w]
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(roi_hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(roi_hsv, self.lower_red2, self.upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            red_in_poly = np.zeros_like(red_mask)
            red_in_poly[self.mask_roi_bool] = red_mask[self.mask_roi_bool]
            roi_bgr[red_in_poly > 0] = (0, 0, 255)

            # -------------------- YOLO Detection --------------------
            results = self.marker_detection_model.predict(frame, verbose=False)[0]
            self.last_marker_position = None

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if not poly.contains(Point(cx, cy)):
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                potential_rotation_time = (self.current_frame - self.frame_crossed) / self.video_fps

                if cx >= start_x and potential_rotation_time >= 4.85:
                    self.rotation_time = potential_rotation_time
                    self.frame_crossed = self.current_frame
                    self.has_completed_rotation = True
                    self.show_updated = True
                    self.updated_color_flag = not self.updated_color_flag
                    self.last_marker_position = [cx, cy]

                    DualLogger.log_message(f"Rotation time: {self.rotation_time:.2f}s")

            # -------------------- Marker Position Text --------------------
            if self.last_marker_position:
                cx, cy = self.last_marker_position
                pos_text = f"Marker: (x={cx}, y={cy})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.8
                thickness = 4

                text_size, _ = cv2.getTextSize(pos_text, font, font_scale, thickness)
                text_x = (self.frame_width - text_size[0]) // 2
                text_y = (self.frame_height // 2) + 80

                cv2.putText(frame, pos_text, (text_x, text_y), font, font_scale, (0, 0, 0), 6)
                cv2.putText(frame, pos_text, (text_x, text_y), font, font_scale, (255, 255, 0), 4)

            # -------------------- Updated Overlay --------------------
            if self.show_updated:
                updated_text = "Updated"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.5
                thickness = 5

                text_size, _ = cv2.getTextSize(updated_text, font, font_scale, thickness)
                text_x = (self.frame_width - text_size[0]) // 2
                text_y = (self.frame_height + text_size[1]) // 2

                color = (128, 0, 128) if self.updated_color_flag else (0, 100, 0)
                cv2.putText(frame, updated_text, (text_x, text_y), font, font_scale, color, thickness)
                self.show_updated = False

            # -------------------- Rotation Time Display --------------------
            rot_text = f"Rotation Time: {self.rotation_time:.2f} s"
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(frame, rot_text, (50, 80), font, 2.0, (0, 0, 0), 6)
            cv2.putText(frame, rot_text, (50, 80), font, 2.0, (0, 255, 255), 4)

            out.write(frame)
            self.current_frame += 1

        cap.release()
        out.release()
        files.download(self.output_path)
        DualLogger.log_message("Processing complete. Output saved to: " + self.output_path)


# ============================ Execution ============================
monitor = LineMonitor()
monitor.find_line_rotation()
