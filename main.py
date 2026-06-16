import sys
import os

# ===== SUPPRESS TENSORFLOW VERBOSITY BEFORE IMPORT =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO/WARNING messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to fail fast on CUDA errors

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import tensorflow as tf
import time as time_module
from dotenv import load_dotenv
import threading, queue, time
from pathlib import Path
from collections import deque

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QDialog,
)
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt, QTime

from mongodb_handler import MongoDBHandler
from encryption_handler import EncryptionHandler
from violation_logger import ViolationLogger
from src.inference import load_model, infer, process_with_logits, crop
from src.detection import load_detector, detect

# ML optimizations
try:
    from model_monitor import ModelMonitor, print_system_status

    monitor = ModelMonitor()
except ImportError:
    monitor = None
    print_system_status = None

# Load environment variables
load_dotenv()

# ================= CONFIG =================

MODEL_DIR = "models"
DATASET_ROOT = "dataset"

ARC_ONNX = f"{MODEL_DIR}/glintr100.onnx"
CLASSIFIER_PATH_H5 = f"{MODEL_DIR}/best_model.h5"  # Backup/fallback
CLASSIFIER_PATH_ONNX = f"{MODEL_DIR}/best_model.onnx"  # Primary (50-70% faster)
SCALER_MEAN = f"{MODEL_DIR}/scaler_mean.npy"
SCALER_SCALE = f"{MODEL_DIR}/scaler_scale.npy"

ARC_SIZE = (112, 112)
THRESHOLD = 0.80
TOP_K = 3
POSES = ["Forward", "Left", "Right", "Up", "Down"]

os.makedirs(DATASET_ROOT, exist_ok=True)

# ======= Phone detector config =======
PHONE_MODEL = os.getenv("PHONE_MODEL_PATH", os.path.join(MODEL_DIR, "best v5.pt"))
PHONE_CONF = float(
    os.getenv("PHONE_CONF", 0.55)
)  # High confidence threshold to avoid false positives (cupboards, doors, etc)
PHONE_IOU = float(os.getenv("PHONE_IOU", 0.45))
PHONE_IMG_SIZE = int(os.getenv("PHONE_IMG_SIZE", 640))
PHONE_WARNING_FRAMES = 90  # 3 seconds at 30fps
PHONE_VIOLATION_FRAMES = 300  # 10 seconds at 30fps

# Try to import ultralytics YOLO
PHONE_DETECTOR_AVAILABLE = False
try:
    from ultralytics import YOLO
    import torch

    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    PHONE_DETECTOR_AVAILABLE = True
except Exception:
    PHONE_DETECTOR_AVAILABLE = False

# ======= Anti-spoofing detector config (ONNX-based) =======
ANTI_SPOOF_MODEL = os.path.join(MODEL_DIR, "best_model_quantized.onnx")
FACE_DETECTOR_MODEL = os.path.join(MODEL_DIR, "detector_quantized.onnx")
ANTI_SPOOF_CONF_THRESHOLD = float(os.getenv("ANTI_SPOOF_CONF_THRESHOLD", 0.0))
ANTI_SPOOF_LOGIT_THRESHOLD = float(os.getenv("ANTI_SPOOF_LOGIT_THRESHOLD", 0.0))
ANTI_SPOOF_TEMPORAL_FRAMES = int(os.getenv("ANTI_SPOOF_TEMPORAL_FRAMES", 2))
ANTI_SPOOF_IMG_SIZE = 128  # Model expects 128x128 input (was incorrectly set to 224)

ANTI_SPOOF_AVAILABLE = False
anti_spoof_session = None
anti_spoof_input_name = None
face_detector = None

# ================= MONGODB & SECURITY =================

db_handler = MongoDBHandler()
encryption_handler = EncryptionHandler()
violation_logger = ViolationLogger()

# ================= LOAD MODELS =================

print("\n" + "=" * 70)
print("🚀 ContinuAuth Starting - OPTIMIZED Model Loading")
print("=" * 70)

# Load only ESSENTIAL models at startup
print("⚡ Fast startup (CRITICAL models only):")
fast_start = time_module.time()

# These are truly essential and fast to load
print("  • Loading embeddings model (ONNX-ARC)...")

arc_sess = ort.InferenceSession(ARC_ONNX, providers=["CPUExecutionProvider"])
arc_in = arc_sess.get_inputs()[0].name
arc_out = arc_sess.get_outputs()[0].name
print("  ✓ ONNX embeddings loaded")

print("  • Loading Haar Cascade (face detection)...")
haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("  ✓ Haar Cascade loaded")

print("  • Loading MediaPipe (face mesh)...")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
print("  ✓ MediaPipe loaded")

fast_elapsed = time_module.time() - fast_start
print(f"\n✅ Fast startup completed in {fast_elapsed:.2f}s")

print("\n⏳ Lazy loading (on first verify):")
print("  • Face classifier (TensorFlow) - ~15s")
print("  • Feature scalers (NumPy)")
print("  • Anti-spoofing model (ONNX) - ~5s")
print("  • Face detector (OpenCV)")
print("  • Phone detector (YOLO) - ~10-20s")
print("=" * 70 + "\n")

# ===== LAZY LOADED MODELS (will load on first verify) =====
clf = None
sc_mean = None
sc_scale = None
CLASSIFIER_LOADED = False

# ===== ONNX Classifier (H5 Alternative) =====
classifier_session = None
classifier_input_name = None

# ======= Lazy Load All Heavy TensorFlow/ONNX Models =======
ANTI_SPOOF_AVAILABLE = False
anti_spoof_session = None
anti_spoof_input_name = None
face_detector = None


def load_verification_models():
    """Lazy load all verification models when first needed (face classifier + anti-spoofing)."""
    global clf, sc_mean, sc_scale, CLASSIFIER_LOADED
    global anti_spoof_session, anti_spoof_input_name, face_detector, ANTI_SPOOF_AVAILABLE

    # Load face classifier and scalers if not already loaded
    if not CLASSIFIER_LOADED:
        try:
            classifier_start = time_module.time()

            # TRY ONNX FIRST (50-70% faster)
            if ort is not None and os.path.exists(CLASSIFIER_PATH_ONNX):
                try:
                    print("  🔄 Loading face classifier (ONNX)...")
                    if monitor:
                        monitor.start_tracking("classifier_load_onnx")

                    classifier_session = ort.InferenceSession(
                        CLASSIFIER_PATH_ONNX, providers=["CPUExecutionProvider"]
                    )
                    classifier_input_name = classifier_session.get_inputs()[0].name

                    # Wrap ONNX to match TensorFlow API
                    clf = ONNXClassifierWrapper(
                        classifier_session, classifier_input_name
                    )

                    if monitor:
                        metrics = monitor.end_tracking("classifier_load_onnx")

                    classifier_elapsed = time_module.time() - classifier_start
                    print(
                        f"  ✓ ONNX classifier loaded in {classifier_elapsed:.2f}s ⚡ (50-70% faster)"
                    )

                except Exception as e:
                    print(
                        f"  ⚠️ ONNX loading failed ({str(e)[:50]}), using H5 fallback..."
                    )
                    raise
            else:
                raise FileNotFoundError("ONNX model not found, falling back to H5")

        except:
            # FALLBACK TO H5 IF ONNX FAILS
            try:
                print("  🔄 Loading face classifier (TensorFlow H5)...")
                clf = tf.keras.models.load_model(CLASSIFIER_PATH_H5)
                classifier_elapsed = time_module.time() - classifier_start
                print(
                    f"  ✓ H5 classifier loaded in {classifier_elapsed:.2f}s (fallback mode)"
                )
            except Exception as e:
                print(f"  ⚠️ Failed to load face classifier: {e}")
                CLASSIFIER_LOADED = False
                return

        try:
            sc_mean = np.load(SCALER_MEAN)
            sc_scale = np.load(SCALER_SCALE)
            CLASSIFIER_LOADED = True

            # Show system status after loading
            if print_system_status:
                print_system_status("After model load")

        except Exception as e:
            print(f"  ⚠️ Failed to load scalers: {e}")
            CLASSIFIER_LOADED = False

    # Load anti-spoofing models if not already loaded
    if not ANTI_SPOOF_AVAILABLE:
        try:
            print("  🔄 Loading anti-spoofing models (ONNX)...")
            antispoof_start = time_module.time()
            anti_spoof_session, anti_spoof_input_name = load_model(ANTI_SPOOF_MODEL)
            face_detector = load_detector(
                FACE_DETECTOR_MODEL, (320, 320), confidence_threshold=0.5
            )
            antispoof_elapsed = time_module.time() - antispoof_start
            if anti_spoof_session and face_detector:
                ANTI_SPOOF_AVAILABLE = True
                print(f"  ✓ Anti-spoof models loaded in {antispoof_elapsed:.2f}s")
            else:
                print("  ⚠️ Anti-spoof models failed to load (session/detector None)")
        except Exception as e:
            print(f"  ⚠️ Anti-spoof models failed to load: {e}")
            ANTI_SPOOF_AVAILABLE = False


# ======= Temporal Filter Class for Anti-Spoofing =======
class TemporalFilter:
    """Track face liveness predictions across frames with temporal smoothing."""

    def __init__(self, required_frames=2, confidence_threshold=0.0):
        self.required_frames = required_frames
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=required_frames)
        self.current_prediction = None

    def update(self, logit_diff, is_real):
        """Update with new frame prediction."""
        if abs(logit_diff) >= self.confidence_threshold:
            self.prediction_history.append((is_real, logit_diff))
        else:
            self.prediction_history.append(None)

        confident_preds = [p for p in self.prediction_history if p is not None]

        if len(confident_preds) >= self.required_frames:
            real_count = sum(1 for p, _ in confident_preds if p)
            spoof_count = len(confident_preds) - real_count

            if real_count >= self.required_frames - 1:
                self.current_prediction = ("REAL", True)
            elif spoof_count >= self.required_frames - 1:
                self.current_prediction = ("SPOOF", False)
            else:
                self.current_prediction = None
        else:
            self.current_prediction = None

        return self.current_prediction

    def get_progress(self):
        """Get progress towards confirmation (0.0 to 1.0)."""
        if self.required_frames == 0:
            return 1.0
        confident = len([p for p in self.prediction_history if p is not None])
        return confident / self.required_frames


# ======= Phone detection helper classes (threaded) =======
if PHONE_DETECTOR_AVAILABLE:

    class InferenceThread:
        def __init__(self, model, conf, iou, img_size, device):
            self.model = model
            self.conf = conf
            self.iou = iou
            self.img_size = img_size
            self.device = device
            self._ms = 0.0
            self._stop = False
            self._in_q = queue.Queue(maxsize=1)
            self._out_q = queue.Queue(maxsize=1)
            threading.Thread(target=self._worker, daemon=True).start()

        def _worker(self):
            while not self._stop:
                try:
                    frame = self._in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                t0 = time.time()
                try:
                    results = self.model.predict(
                        source=frame,
                        conf=self.conf,
                        iou=self.iou,
                        imgsz=self.img_size,
                        verbose=False,
                        device=self.device,
                    )
                except Exception as e:
                    # Log the first time this error occurs, then suppress
                    if not hasattr(self, "_error_logged"):
                        print(
                            f"  ⚠️ Phone detector error: {type(e).__name__}: {str(e)[:100]}",
                            file=sys.stderr,
                        )
                        print(
                            f"     (This is likely a torchvision::nms compatibility issue. Run: pip install -r requirements.txt)",
                            file=sys.stderr,
                        )
                        self._error_logged = True
                    continue
                self._ms = (time.time() - t0) * 1000

                dets = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        dets.append(
                            (
                                *map(float, box.xyxy[0]),
                                float(box.conf[0]),
                                int(box.cls[0]),
                            )
                        )

                if self._out_q.full():
                    try:
                        self._out_q.get_nowait()
                    except:
                        pass
                self._out_q.put(dets)

        def submit(self, frame):
            if self._in_q.full():
                try:
                    self._in_q.get_nowait()
                except:
                    pass
            try:
                self._in_q.put_nowait(frame)
            except:
                pass

        def result(self):
            try:
                return self._out_q.get_nowait()
            except:
                return None

        @property
        def ms(self):
            return self._ms

        def set_conf(self, c):
            self.conf = c

        def stop(self):
            self._stop = True

    def draw_phone_box(frame, x1, y1, x2, y2, conf):
        col = (57, 255, 20)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)


# ================= FUNCTIONS =================


def crop_face(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = haar.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
    if len(faces) != 1:
        return None, None

    x, y, w, h = faces[0]
    pad = int(0.2 * w)

    crop = rgb[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]

    return cv2.resize(crop, ARC_SIZE), (x, y, w, h)


def run_arc(img112):
    x = (img112.astype(np.float32) - 127.5) / 128.0
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
    out = arc_sess.run([arc_out], {arc_in: x})[0]
    emb = np.squeeze(out)
    return emb / (np.linalg.norm(emb) + 1e-12)


def build_features(e1, e2):
    diff = e1 - e2
    return np.hstack([np.abs(diff), [np.linalg.norm(diff), np.dot(e1, e2)]]).astype(
        np.float32
    )


def detect_head_pose(rgb):
    """
    Detect head pose from face landmarks
    Returns: (pose_name, x_angle, y_angle, face_landmarks) or
             (None, None, None, None) when no face is found.
    """
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None, None, None

    img_h, img_w, _ = rgb.shape
    face_landmarks = results.multi_face_landmarks[0]

    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            px, py = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([px, py])
            face_3d.append([px, py, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = img_w
    cam_matrix = np.array(
        [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
    )

    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    if not success:
        return None, None, None, None

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    x_angle = angles[0] * 360
    y_angle = angles[1] * 360

    if y_angle < -10:
        pose_name = "Left"
    elif y_angle > 10:
        pose_name = "Right"
    elif x_angle < -10:
        pose_name = "Down"
    elif x_angle > 10:
        pose_name = "Up"
    else:
        pose_name = "Forward"

    return pose_name, x_angle, y_angle, face_landmarks


# ================= ML OPTIMIZATION: ONNX WRAPPER =================


class ONNXClassifierWrapper:
    """
    Wraps ONNX model to match TensorFlow .predict() API.
    Allows seamless switching from H5 to ONNX without code changes.
    """

    def __init__(self, session, input_name):
        self.session = session
        self.input_name = input_name

    def predict(self, data, verbose=0):
        """
        Run ONNX inference matching TensorFlow API.
        Input: numpy array (shape: batch_size x features)
        Output: predictions array (shape: batch_size x classes)
        """
        output_name = self.session.get_outputs()[0].name
        result = self.session.run(
            [output_name], {self.input_name: data.astype(np.float32)}
        )
        return result[0]


# ================= MAIN CLASS =================


class ContinuAuth(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ContinuAuth Secure Desktop")
        self.setFixedSize(750, 720)
        self.setStyleSheet(self.style())

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Multi-face detection tracking
        self.last_face_count = 0
        self.face_count_warning_count = 0

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter 3-digit Username")

        self.video_label = QLabel()
        self.video_label.setFixedSize(700, 450)

        self.status_label = QLabel("System Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))

        self.register_btn = QPushButton("Register User")
        self.verify_btn = QPushButton("Verify")
        self.delete_btn = QPushButton("Delete User")
        # Don't create admin_dashboard_btn here since it's only for admin panel

        self.register_btn.clicked.connect(self.start_registration)
        self.verify_btn.clicked.connect(self.verify_user)
        self.delete_btn.clicked.connect(self.delete_user)

        layout = QVBoxLayout()
        # Back navigation button (top-left)
        self.back_btn = QPushButton("\u2190 Back")
        self.back_btn.clicked.connect(self.show_home)
        self.back_btn.setVisible(False)
        layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        # Home buttons (Admin / User)
        self.home_widget = QWidget()
        home_layout = QVBoxLayout()
        self.admin_home_btn = QPushButton("Admin")
        self.user_home_btn = QPushButton("User")
        home_layout.addStretch(1)
        home_layout.addWidget(self.admin_home_btn)
        home_layout.addWidget(self.user_home_btn)
        home_layout.addStretch(1)
        self.home_widget.setLayout(home_layout)

        layout.addWidget(self.home_widget)

        # admin panel separate vertical buttons - WITH PROPER SIZING
        self.admin_reg_btn = QPushButton("Register User")
        self.admin_dashboard_btn = QPushButton("📊 Admin Dashboard")
        self.toggle_spoof_btn = QPushButton("Anti-Spoofing: ON")
        self.admin_del_btn = QPushButton("Delete User")

        # Set minimum heights for visibility
        self.admin_reg_btn.setMinimumHeight(50)
        self.admin_dashboard_btn.setMinimumHeight(50)
        self.toggle_spoof_btn.setMinimumHeight(50)
        self.admin_del_btn.setMinimumHeight(50)

        # Simple, consistent styling for all buttons
        btn_style = """
            QPushButton {
                background-color: #2979FF;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 12px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """
        dashboard_style = """
            QPushButton {
                background-color: #FF6B35;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 12px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #E55100;
            }
            QPushButton:pressed {
                background-color: #BF360C;
            }
        """

        toggle_spoof_style_on = """
            QPushButton {
                background-color: #4CAF50; color: white; font-weight: bold;
                font-size: 13px; padding: 12px; border-radius: 5px; border: none;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3e8e41; }
        """

        self.admin_reg_btn.setStyleSheet(btn_style)
        self.admin_dashboard_btn.setStyleSheet(dashboard_style)
        self.admin_del_btn.setStyleSheet(btn_style)
        self.toggle_spoof_btn.setStyleSheet(toggle_spoof_style_on)

        # connect admin panel buttons
        self.admin_reg_btn.clicked.connect(self.start_registration)
        self.admin_dashboard_btn.clicked.connect(self.show_admin_dashboard)
        self.toggle_spoof_btn.clicked.connect(self.toggle_anti_spoofing)
        self.admin_del_btn.clicked.connect(self.delete_user)

        # Create admin panel layout with all buttons
        self.admin_panel_widget = QWidget()
        admin_layout = QVBoxLayout()
        admin_layout.setSpacing(10)
        admin_layout.addStretch(1)
        admin_layout.addWidget(self.admin_reg_btn)
        admin_layout.addWidget(self.admin_dashboard_btn)
        admin_layout.addWidget(self.toggle_spoof_btn)
        admin_layout.addWidget(self.admin_del_btn)
        admin_layout.addStretch(1)
        self.admin_panel_widget.setLayout(admin_layout)
        self.admin_panel_widget.setVisible(False)
        layout.addWidget(self.admin_panel_widget)

        # Main preview widgets (hidden until a panel selected)
        layout.addWidget(self.username_input)
        layout.addWidget(self.video_label)

        # User panel: single Verify button
        self.user_verify_widget = QWidget()
        user_verify_layout = QVBoxLayout()
        self.user_verify_btn = QPushButton("Verify")
        self.user_verify_btn.clicked.connect(self.verify_user)
        user_verify_layout.addWidget(self.user_verify_btn)
        user_verify_layout.addStretch(1)
        self.user_verify_widget.setLayout(user_verify_layout)
        self.user_verify_widget.setVisible(False)
        layout.addWidget(self.user_verify_widget)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.register_btn)
        btn_layout.addWidget(self.verify_btn)
        btn_layout.addWidget(self.delete_btn)
        # Don't add admin_dashboard_btn here - only for admin panel
        # admin-only commit/cancel buttons (hidden by default)
        self.complete_register_btn = QPushButton("Complete Register")
        self.cancel_register_btn = QPushButton("Cancel Registration")
        self.complete_register_btn.setVisible(False)
        self.cancel_register_btn.setVisible(False)
        btn_layout.addWidget(self.complete_register_btn)
        btn_layout.addWidget(self.cancel_register_btn)

        # wrap horizontal buttons in a widget for easy show/hide
        self.button_row_widget = QWidget()
        self.button_row_widget.setLayout(btn_layout)
        layout.addWidget(self.button_row_widget)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        self.registering = False
        self.pose_index = 0
        self.reg_images = {}
        self.pose_progress = 0
        self.required_hold_frames = 20
        self.last_verification_color = None

        # ================= VERIFICATION MONITORING STATE =================

        self.verifying = False  # in verification/monitoring mode
        self.verify_username = None
        self.bad_condition_frames = 0  # frames with bad pose/faces
        self.violation_popup_showing = False
        self.last_violation_frame = None

        # Thresholds (in frames at 30fps)
        self.WARNING_THRESHOLD_FRAMES = 150  # ~5 seconds
        self.VIOLATION_THRESHOLD_FRAMES = 300  # ~10 seconds

        # Performance optimization
        self.frame_counter = 0
        # Popup timeout (ms)
        self.POPUP_DISMISS_TIMEOUT_MS = int(os.getenv("POPUP_DISMISS_TIMEOUT_MS", 3000))
        # Phone detection state (count-based)
        self.phone_detector = None
        self.phone_model = None  # Will be lazy-loaded on first verify
        self.phone_model_loaded = False
        self.phone_detection_history = (
            []
        )  # Track last 300 frames of phone detections (10 seconds)
        self.phone_warning_triggered = False
        self.phone_violation_triggered = False
        self.phone_no_detection_frames = (
            0  # Debounce counter: frames without phone detected
        )
        # Do NOT load YOLO model at startup - defer to first verification for fast startup

        # Admin/user mode flags
        self.anti_spoof_enabled = True
        self.admin_mode = False
        self.user_mode = False
        self.pending_registration = None  # hold embeddings and images until commit

        # Wire up home/admin navigation
        self.admin_home_btn.clicked.connect(self.show_admin_panel)
        self.user_home_btn.clicked.connect(self.show_user_panel)
        self.complete_register_btn.clicked.connect(self.commit_registration)
        self.cancel_register_btn.clicked.connect(self.cancel_registration)

        # Start showing home
        self.show_home()

    def toggle_anti_spoofing(self):
        self.anti_spoof_enabled = not self.anti_spoof_enabled
        if self.anti_spoof_enabled:
            self.toggle_spoof_btn.setText("Anti-Spoofing: ON")
            self.toggle_spoof_btn.setStyleSheet("""
                QPushButton { background-color: #4CAF50; color: white; font-weight: bold;
                font-size: 13px; padding: 12px; border-radius: 5px; border: none; }
                QPushButton:hover { background-color: #45a049; }
                QPushButton:pressed { background-color: #3e8e41; }
            """)
        else:
            self.toggle_spoof_btn.setText("Anti-Spoofing: OFF")
            self.toggle_spoof_btn.setStyleSheet("""
                QPushButton { background-color: #f44336; color: white; font-weight: bold;
                font-size: 13px; padding: 12px; border-radius: 5px; border: none; }
                QPushButton:hover { background-color: #e53935; }
                QPushButton:pressed { background-color: #c62828; }
            """)
        self.status_label.setText(
            f"Anti-spoofing is now {'ON' if self.anti_spoof_enabled else 'OFF'}"
        )

    def load_phone_detector(self):
        """Lazy load YOLO phone detector model on first verification."""
        if self.phone_model_loaded or not PHONE_DETECTOR_AVAILABLE:
            return  # Already loaded or not available

        try:
            print("  🔄 Loading phone detector (YOLO)...")
            phone_start = time_module.time()
            model_path = Path(PHONE_MODEL)
            if not model_path.exists():
                model_path = Path(MODEL_DIR) / "best v5.pt"
            self.phone_model = YOLO(str(model_path))
            self.phone_detector = InferenceThread(
                self.phone_model, PHONE_CONF, PHONE_IOU, PHONE_IMG_SIZE, DEVICE
            )
            phone_elapsed = time_module.time() - phone_start
            print(f"  ✓ Phone detector loaded in {phone_elapsed:.2f}s")
            self.phone_model_loaded = True
        except Exception as e:
            print(f"  ⚠️ Failed to load phone detector: {e}")
            self.phone_model_loaded = False
            self.phone_detector = None

    # ================= NAVIGATION HELPERS =================
    def show_home(self):
        self.back_btn.setVisible(False)
        self.home_widget.setVisible(True)
        # hide panels/buttons
        self.button_row_widget.setVisible(False)
        self.admin_panel_widget.setVisible(False)
        self.user_verify_widget.setVisible(False)
        # hide preview widgets
        self.username_input.setVisible(False)
        self.video_label.setVisible(False)
        self.register_btn.setVisible(False)
        self.verify_btn.setVisible(False)
        self.delete_btn.setVisible(False)
        self.status_label.setVisible(False)
        self.complete_register_btn.setVisible(False)
        self.cancel_register_btn.setVisible(False)
        self.admin_mode = False
        self.user_mode = False

    def show_admin_panel(self):
        self.back_btn.setVisible(True)
        self.home_widget.setVisible(False)
        self.button_row_widget.setVisible(False)
        # show admin panel vertical buttons
        self.admin_panel_widget.setVisible(True)
        self.user_verify_widget.setVisible(False)
        # hide preview/input
        self.username_input.setVisible(False)
        self.video_label.setVisible(False)
        # hide any pending registration controls
        self.complete_register_btn.setVisible(False)
        self.cancel_register_btn.setVisible(False)
        self.status_label.setVisible(True)
        self.admin_mode = True
        self.user_mode = False
        self.status_label.setText("Admin Panel — choose action")

    def show_user_panel(self):
        self.back_btn.setVisible(True)
        self.home_widget.setVisible(False)
        # hide admin panel
        self.admin_panel_widget.setVisible(False)
        self.button_row_widget.setVisible(False)  # Hide old button row
        self.username_input.setVisible(True)
        self.video_label.setVisible(True)
        # Show user verify widget (single Verify button)
        self.user_verify_widget.setVisible(True)
        # hide pending registration controls
        self.complete_register_btn.setVisible(False)
        self.cancel_register_btn.setVisible(False)
        self.status_label.setVisible(True)
        self.admin_mode = False
        self.user_mode = True
        self.status_label.setText("User Panel — enter username and click Verify")

    def show_admin_dashboard(self):
        # Build XDR-style dashboard
        dlg = QDialog(self)
        dlg.setWindowTitle("Admin Dashboard - Security Log")
        dlg.setFixedSize(1000, 600)
        dlg.setStyleSheet("""
            QDialog { background-color: #1a1a1a; }
            QTableWidget { background-color: #0d0d0d; color: white; }
            QTableWidget::item { padding: 5px; }
            QHeaderView::section { background-color: #2a2a2a; color: white; padding: 5px; }
            QLineEdit { background-color: #2a2a2a; color: white; padding: 5px; border: 1px solid #444; }
            QPushButton { background-color: #2979FF; color: white; padding: 5px 15px; border-radius: 3px; }
            QPushButton:hover { background-color: #1565C0; }
        """)

        main_layout = QVBoxLayout()

        # Header
        title = QLabel("ADMIN LOG - Security Events Dashboard")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #2979FF;")
        main_layout.addWidget(title)

        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: white;")
        search_input = QLineEdit()
        search_input.setPlaceholderText("Filter by User ID, Status, or Type...")
        search_btn = QPushButton("Search")
        search_layout.addWidget(search_label)
        search_layout.addWidget(search_input)
        search_layout.addWidget(search_btn)
        main_layout.addLayout(search_layout)

        # Table
        tbl = QTableWidget()
        tbl.setColumnCount(7)
        tbl.setHorizontalHeaderLabels(
            [
                "Date",
                "Time (UTC)",
                "User ID",
                "Event Type",
                "Status",
                "Description",
                "Evidence",
            ]
        )
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setAlternatingRowColors(True)
        tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tbl.setStyleSheet("""
            QTableWidget { 
                gridline-color: #444; 
                alternate-background-color: #1a1a1a;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
        """)

        # fetch all data: violations + warnings + logs
        violations = db_handler.get_violations(days=30) or []
        warnings = db_handler.get_logs(days=30, warning_type="warning") or []
        all_logs = db_handler.get_logs(days=30) or []

        # Helper function to convert datetime to string
        def ts_to_str(ts):
            if isinstance(ts, str):
                return ts
            try:
                return str(ts)
            except:
                return "Unknown"

        # Helper function to extract date and time from timestamp
        def extract_date_time(ts):
            ts_str = ts_to_str(ts)
            if " " in ts_str:
                parts = ts_str.split(" ")
                return parts[0], parts[1] if len(parts) > 1 else "00:00:00"
            return ts_str, "00:00:00"

        # Combine all events
        rows = []

        # Add violations
        for v in violations:
            ts = v.get("timestamp", "")
            username = v.get("username", "Unknown")
            details = v.get("details", "")
            screenshot_id = v.get("screenshot_id")
            violation_type = v.get("type", "violation")
            date_str, time_str = extract_date_time(ts)
            rows.append(
                {
                    "timestamp": ts,
                    "date": date_str,
                    "time": time_str,
                    "username": username,
                    "event_type": violation_type.upper(),
                    "status": "VIOLATION",
                    "description": details,
                    "screenshot_id": screenshot_id,
                    "color": (200, 40, 40),  # Red for violations
                }
            )

        # Add warnings
        for w in warnings:
            ts = w.get("timestamp", "")
            log_type = w.get("type", "warning")
            details = w.get("details", "")
            date_str, time_str = extract_date_time(ts)
            rows.append(
                {
                    "timestamp": ts,
                    "date": date_str,
                    "time": time_str,
                    "username": "",
                    "event_type": log_type.upper(),
                    "status": "WARNING",
                    "description": details,
                    "screenshot_id": None,
                    "color": (255, 165, 0),  # Orange for warnings
                }
            )

        # Add info logs
        for l in all_logs:
            if l.get("type") not in ["warning"]:
                ts = l.get("timestamp", "")
                log_type = l.get("type", "info")
                details = l.get("details", "")
                date_str, time_str = extract_date_time(ts)
                rows.append(
                    {
                        "timestamp": ts,
                        "date": date_str,
                        "time": time_str,
                        "username": "",
                        "event_type": log_type.upper(),
                        "status": "INFO",
                        "description": details,
                        "screenshot_id": None,
                        "color": (100, 200, 100),  # Green for info
                    }
                )

        # Sort by timestamp (newest first)
        rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Populate table
        tbl.setRowCount(len(rows))
        for i, r in enumerate(rows):
            date_item = QTableWidgetItem(str(r.get("date", "")))
            time_item = QTableWidgetItem(str(r.get("time", "")))
            user_item = QTableWidgetItem(str(r.get("username", "")))
            event_type_item = QTableWidgetItem(str(r.get("event_type", "")))
            status_item = QTableWidgetItem(str(r.get("status", "")))
            desc_item = QTableWidgetItem(str(r.get("description", "")))
            evidence_item = QTableWidgetItem("View" if r.get("screenshot_id") else "")

            # Apply colors
            color = r.get("color", (200, 200, 200))
            status_item.setBackground(QColor(*color))
            status_item.setForeground(QColor(255, 255, 255))

            tbl.setItem(i, 0, date_item)
            tbl.setItem(i, 1, time_item)
            tbl.setItem(i, 2, user_item)
            tbl.setItem(i, 3, event_type_item)
            tbl.setItem(i, 4, status_item)
            tbl.setItem(i, 5, desc_item)
            tbl.setItem(i, 6, evidence_item)

        # Evidence viewer
        def on_cell_clicked(row, col):
            if col == 6:
                screenshot_id = rows[row].get("screenshot_id")
                if screenshot_id:
                    data = db_handler.get_violation_screenshot(screenshot_id)
                    if data:
                        img = QImage.fromData(data)
                        if not img.isNull():
                            px = QPixmap.fromImage(img)
                            img_dlg = QDialog(dlg)
                            img_dlg.setWindowTitle("Evidence Screenshot")
                            img_dlg.setFixedSize(700, 500)
                            lbl = QLabel()
                            lbl.setPixmap(
                                px.scaled(700, 500, Qt.AspectRatioMode.KeepAspectRatio)
                            )
                            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            v = QVBoxLayout()
                            v.addWidget(lbl)
                            img_dlg.setLayout(v)
                            img_dlg.exec()

        # Search filter
        def on_search():
            search_text = search_input.text().lower().strip()
            for row in range(tbl.rowCount()):
                if search_text == "":
                    tbl.setRowHidden(row, False)
                else:
                    user_text = (
                        tbl.item(row, 2).text().lower() if tbl.item(row, 2) else ""
                    )
                    status_text = (
                        tbl.item(row, 4).text().lower() if tbl.item(row, 4) else ""
                    )
                    event_text = (
                        tbl.item(row, 3).text().lower() if tbl.item(row, 3) else ""
                    )
                    desc_text = (
                        tbl.item(row, 5).text().lower() if tbl.item(row, 5) else ""
                    )

                    match = (
                        search_text in user_text
                        or search_text in status_text
                        or search_text in event_text
                        or search_text in desc_text
                    )
                    tbl.setRowHidden(row, not match)

        tbl.cellClicked.connect(on_cell_clicked)
        search_btn.clicked.connect(on_search)
        search_input.returnPressed.connect(on_search)

        main_layout.addWidget(tbl)
        dlg.setLayout(main_layout)
        dlg.exec()

    # ================= FRAME LOOP =================

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("⚠️ Camera not available - check connection")
            return

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))

        # count frames for slower pose evaluation
        self.frame_counter += 1
        do_heavy = self.frame_counter % 3 == 0

        # ================= PRE-VERIFICATION ANTI-SPOOF MODE =================
        if getattr(self, "pending_anti_spoof", False):
            # run anti-spoof on every frame until we decide real or fake
            if ANTI_SPOOF_AVAILABLE:
                # Detect faces
                detected_faces = detect(rgb, face_detector, min_face_size=60, margin=2)

                if not detected_faces:
                    self.status_label.setText("Detecting face... No face detected")
                    display_text = "No face detected"
                else:
                    # Initialize temporal filter if needed
                    if not hasattr(self, "anti_spoof_filter"):
                        self.anti_spoof_filter = TemporalFilter(
                            required_frames=ANTI_SPOOF_TEMPORAL_FRAMES,
                            confidence_threshold=ANTI_SPOOF_CONF_THRESHOLD,
                        )

                    # Process first face only
                    face = detected_faces[0]
                    bbox = face["bbox"]
                    x, y, w, h = (
                        int(bbox["x"]),
                        int(bbox["y"]),
                        int(bbox["width"]),
                        int(bbox["height"]),
                    )

                    try:
                        # Crop and infer
                        face_crop = crop(
                            rgb, (x, y, x + w, y + h), bbox_expansion_factor=1.0
                        )
                        predictions = infer(
                            [face_crop],
                            anti_spoof_session,
                            anti_spoof_input_name,
                            ANTI_SPOOF_IMG_SIZE,
                        )

                        if predictions:
                            result = process_with_logits(
                                predictions[0], ANTI_SPOOF_LOGIT_THRESHOLD
                            )
                            temporal_result = self.anti_spoof_filter.update(
                                result["logit_diff"], result["is_real"]
                            )
                            progress = self.anti_spoof_filter.get_progress()

                            # Draw face box with status
                            if result["is_real"]:
                                box_color = (0, 255, 0)
                                status_text = f"REAL {result['logit_diff']:.2f}"
                            else:
                                box_color = (0, 0, 255)
                                status_text = f"SPOOF {result['logit_diff']:.2f}"

                            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 3)
                            cv2.putText(
                                display,
                                status_text,
                                (x, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                box_color,
                                2,
                            )

                            # Check for final decision
                            if temporal_result:
                                final_status, final_is_real = temporal_result
                                if final_is_real:
                                    # REAL FACE - start verification
                                    self.pending_anti_spoof = False
                                    self.verifying = True
                                    self.status_label.setText(
                                        f"🔍 Monitoring User: {self.verify_username}"
                                    )
                                    self.initial_auth_done = False
                                else:
                                    # SPOOF DETECTED
                                    self.status_label.setText(
                                        "⚠️ SPOOFING DETECTED - Fake face (blocked)"
                                    )
                                    violation_logger.log_violation(
                                        "spoofing_detected", self.verify_username, frame
                                    )
                                    self.pending_anti_spoof = False
                            else:
                                # Still analyzing
                                frames_counted = len(
                                    [
                                        p
                                        for p in self.anti_spoof_filter.prediction_history
                                        if p is not None
                                    ]
                                )
                                display_text = f"Analyzing... {frames_counted}/{ANTI_SPOOF_TEMPORAL_FRAMES}"
                                self.status_label.setText(display_text)
                    except Exception as e:
                        print(f"Anti-spoof error: {e}")
                        self.status_label.setText(f"Anti-spoof error: {str(e)[:50]}")

        # ================= VERIFICATION MONITORING MODE =================
        elif self.verifying and not self.violation_popup_showing:
            if do_heavy:
                current_pose, _, _, landmarks = detect_head_pose(rgb)
                current_face_count = len(faces)
                face_crop, _ = crop_face(rgb)
            else:
                current_pose = getattr(self, "current_pose", None)
                current_face_count = len(faces)

            self.current_pose = current_pose

            # perform initial authentication only once
            if not getattr(self, "initial_auth_done", False):
                score, valid = self.perform_verification_check(
                    rgb, self.verify_username
                )
                if valid and score >= THRESHOLD:
                    self.initial_auth_done = True
                    self.last_verification_color = (0, 255, 0)
                    self.status_label.setText(f"✓ VERIFIED ({score:.4f}) - monitoring")
                else:
                    self.status_label.setText(
                        "✗ Verification failed - face not recognized"
                    )
                    self.show_violation_popup(
                        self.verify_username, "verification_failed", frame
                    )
                    self.verifying = False
                    return

            # after initial auth, only watch conditions
            is_good_condition = current_face_count == 1 and current_pose == "Forward"

            if not is_good_condition:
                self.bad_condition_frames += 1
            else:
                self.bad_condition_frames = 0

            # draw detection box green/red
            if current_face_count == 1:
                x, y, w, h = faces[0]
                box_color = (0, 255, 0) if current_pose == "Forward" else (0, 0, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 3)
            elif current_face_count > 1:
                for x, y, w, h in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # ======= PHONE DETECTION (count-based in windows) =======
            if self.phone_detector and do_heavy:
                try:
                    # submit and read latest detection
                    self.phone_detector.submit(frame.copy())
                    dets = self.phone_detector.result()
                except Exception:
                    dets = None

                phone_found = False
                if dets:
                    for x1, y1, x2, y2, cf, cls in dets:
                        if cf >= PHONE_CONF:
                            phone_found = True
                            draw_phone_box(display, x1, y1, x2, y2, cf)

                # Track phone detection (1=found, 0=not found)
                self.phone_detection_history.append(1 if phone_found else 0)
                # Keep only last 300 frames (10 seconds at 30fps)
                if len(self.phone_detection_history) > 300:
                    self.phone_detection_history.pop(0)

                # Debounce: track consecutive frames of NO detection
                if phone_found:
                    self.phone_no_detection_frames = 0  # Reset when phone detected
                else:
                    self.phone_no_detection_frames += 1  # Increment when no phone

                # Check counts in time windows
                phone_count_5s = (
                    sum(self.phone_detection_history[-150:])
                    if len(self.phone_detection_history) >= 150
                    else sum(self.phone_detection_history)
                )
                phone_count_10s = sum(self.phone_detection_history)

                # Warning: >10 detections in 5 seconds (sustained phone use)
                if phone_count_5s > 10 and not self.phone_warning_triggered:
                    self.phone_warning_triggered = True
                    violation_logger.log_warning(
                        "phone_detected_multiple",
                        f"User {self.verify_username}: phone use detected (detections: {phone_count_5s}/150 frames)",
                    )
                    db_handler.log_warning(
                        "phone_detected_multiple",
                        f"User {self.verify_username}: phone use detected",
                    )
                    self.status_label.setText(
                        f"⚠️ WARNING: PHONE DETECTED ({phone_count_5s}) | User: {self.verify_username}"
                    )

                # Violation: >=18 detections in 10 seconds (serious/sustained phone use)
                if phone_count_10s >= 18 and not self.phone_violation_triggered:
                    self.phone_violation_triggered = True
                    self.show_violation_popup(
                        self.verify_username, "phone_detected_violation", frame
                    )
                    self.phone_detection_history = []  # Reset history
                    self.phone_warning_triggered = False
                    self.phone_no_detection_frames = 0

                # Reset warning only after 90 frames (~3 seconds) of continuous NO detection
                # This prevents false resets and ensures stable state
                if (
                    self.phone_warning_triggered
                    and self.phone_no_detection_frames >= 90
                ):
                    self.phone_warning_triggered = False
                    self.phone_no_detection_frames = 0
                    self.status_label.setText(
                        f"✓ Phone warning cleared | User: {self.verify_username}"
                    )

            # ======= BAD CONDITION CHECKING (5s warning, 10s violation) =======
            if self.bad_condition_frames > 0:
                seconds_bad = self.bad_condition_frames / 30
                cv2.putText(
                    display,
                    f"Bad condition: {seconds_bad:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                # Warning: after 5 seconds of bad conditions
                if self.bad_condition_frames >= 150 and self.bad_condition_frames < 300:
                    warning_msg = (
                        "⚠️ WARNING: Bad conditions detected (5s+) - look forward!"
                    )
                    self.status_label.setText(warning_msg)
                    cv2.putText(
                        display,
                        warning_msg,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 165, 255),
                        2,
                    )

                # Violation: after 10 seconds of bad conditions
                if (
                    self.bad_condition_frames >= 300
                    and not self.violation_popup_showing
                ):
                    self.show_violation_popup(
                        self.verify_username, "bad_conditions_violation", frame
                    )
                    self.bad_condition_frames = 0

            # ======= BAD CONDITION CHECKING (5s warning, 10s violation) =======
            if self.bad_condition_frames > 0:
                seconds_bad = self.bad_condition_frames / 30
                cv2.putText(
                    display,
                    f"Bad condition: {seconds_bad:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                # Warning: after 5 seconds of bad conditions
                if self.bad_condition_frames >= 150 and self.bad_condition_frames < 300:
                    warning_msg = (
                        "⚠️ WARNING: Bad conditions detected (5s+) - look forward!"
                    )
                    self.status_label.setText(warning_msg)
                    cv2.putText(
                        display,
                        warning_msg,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 165, 255),
                        2,
                    )

                # Violation: after 10 seconds of bad conditions
                if (
                    self.bad_condition_frames >= 300
                    and not self.violation_popup_showing
                ):
                    self.show_violation_popup(
                        self.verify_username, "bad_conditions_violation", frame
                    )
                    self.bad_condition_frames = 0

        # ================= REGISTRATION MODE =================
        elif self.registering:
            face_crop, bbox = crop_face(rgb)

            if bbox:
                x, y, w, h = bbox
                detected_pose, _, _, _ = detect_head_pose(rgb)
                required_pose = POSES[self.pose_index]

                if detected_pose == required_pose:
                    self.pose_progress += 1
                    color = (0, 255, 0)
                else:
                    self.pose_progress = 0
                    color = (0, 0, 255)

                cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
                cv2.putText(
                    display,
                    f"Required: {required_pose}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

                if self.pose_progress >= self.required_hold_frames:
                    self.reg_images[required_pose] = face_crop
                    self.pose_index += 1
                    self.pose_progress = 0

                    if self.pose_index >= len(POSES):
                        # prepare registration but do not commit to DB yet
                        self.prepare_registration()
                        return

        # Display frame
        qt_img = QImage(
            display.data,
            display.shape[1],
            display.shape[0],
            display.shape[1] * 3,
            QImage.Format.Format_BGR888,
        )

        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    # ================= REGISTRATION =================

    def start_registration(self):
        username = self.username_input.text()

        if not (username.isdigit() and len(username) == 3):
            QMessageBox.warning(self, "Error", "Username must be 3 digits")
            return

        if os.path.exists(os.path.join(DATASET_ROOT, username)):
            QMessageBox.warning(self, "Error", "User already exists")
            return

        self.registering = True
        self.pose_index = 0
        self.reg_images = {}
        # ensure input and preview visible regardless of panel
        self.username_input.setVisible(True)
        self.video_label.setVisible(True)
        self.button_row_widget.setVisible(True)
        self.status_label.setText("Follow pose instructions")
        # When admin triggered registration, show commit/cancel
        if self.admin_mode:
            self.complete_register_btn.setVisible(False)
            self.cancel_register_btn.setVisible(True)

    # ================= VERIFICATION =================

    def verify_user(self):
        """Enter continuous verification/monitoring mode"""
        if not self.user_mode:
            QMessageBox.warning(self, "Error", "Please select User panel first")
            return

        username = self.username_input.text()

        if not (username.isdigit() and len(username) == 3):
            QMessageBox.warning(self, "Error", "Username must be 3 digits")
            return

        # Lazy load ALL verification models on first verify (classifier + anti-spoof + phone detector)
        print("\n⏳ First-time model loading...")
        self.status_label.setText("⏳ Loading verification models... (first time only)")
        load_verification_models()
        self.load_phone_detector()  # Lazy load phone detector on first verify

        if not CLASSIFIER_LOADED:
            QMessageBox.warning(
                self, "Warning", "Face classifier could not load. Verification failed."
            )
            return

        if self.anti_spoof_enabled and not ANTI_SPOOF_AVAILABLE:
            QMessageBox.warning(
                self,
                "Warning",
                "Anti-spoofing models could not load. Continuing without anti-spoof check.",
            )
            self.anti_spoof_enabled = False

        # Display system status after model loading
        if print_system_status:
            print_system_status("Verification ready")

        # Check MongoDB first
        embeddings_dict = db_handler.get_user_embeddings(username)

        if not embeddings_dict:
            QMessageBox.critical(
                self,
                "Unauthorized Access Attempt",
                f"User '{username}' is not registered in the system.\n\n"
                "This access attempt has been logged and a security snapshot has been taken.\n\n"
                "Please contact an administrator if you believe this is an error.",
            )

            # Log unauthorized attempt
            ret, frame = self.cap.read()
            if ret:
                # frame is not flipped here, so flip it to match user view
                frame = cv2.flip(frame, 1)
                screenshot_data, _ = violation_logger.capture_violation(
                    frame,
                    "unauthorized_attempt",
                    username,
                    "User not registered in system",
                )
                db_handler.log_violation(
                    "unauthorized_attempt",
                    username,
                    "User not registered",
                    screenshot_data,
                )
            return

        # If anti-spoofing is OFF, go straight to verification
        if not self.anti_spoof_enabled:
            self.verify_username = username
            self.verifying = True
            self.status_label.setText(
                f"🔍 Monitoring User: {self.verify_username} (Anti-Spoof OFF)"
            )
            # Reset state
            self.bad_condition_frames = 0
            self.phone_detection_history = []
            self.phone_no_detection_frames = 0
            self.phone_warning_triggered = False
            self.phone_violation_triggered = False
            self.violation_popup_showing = False
            self.initial_auth_done = False
            self.reset_liveness()
            return

        # prepare for anti‑spoof check before actual verification
        self.verify_username = username
        self.pending_anti_spoof = True
        self.anti_spoof_filter = TemporalFilter(
            required_frames=ANTI_SPOOF_TEMPORAL_FRAMES,
            confidence_threshold=ANTI_SPOOF_CONF_THRESHOLD,
        )
        self.verifying = False  # will start once anti‑spoof passes

        # reset state used by verification session
        self.bad_condition_frames = 0
        self.phone_detection_history = []  # Reset phone detection history
        self.phone_no_detection_frames = 0  # Reset debounce counter
        self.phone_warning_triggered = False
        self.phone_violation_triggered = False
        self.violation_popup_showing = False
        self.initial_auth_done = False
        self.reset_liveness()
        self.status_label.setText(f"🔍 Present face for anti-spoof check ({username})")

    # ================= REGISTRATION (PREPARE / COMMIT) =================

    def prepare_registration(self):
        """Prepare registration embeddings but do not store to DB yet."""
        username = self.username_input.text()
        if not (username.isdigit() and len(username) == 3):
            QMessageBox.warning(self, "Error", "Username must be 3 digits")
            self.registering = False
            return

        # Generate embeddings from collected images
        embeddings_dict = {}
        for pose, img in self.reg_images.items():
            emb = run_arc(img)
            embeddings_dict[pose] = emb

        user_metadata = {
            "username": username,
            "registered_at": str(
                __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            ),
            "poses_registered": list(embeddings_dict.keys()),
        }

        self.pending_registration = {
            "username": username,
            "embeddings": embeddings_dict,
            "metadata": user_metadata,
        }

        self.registering = False
        # show Complete / Cancel buttons
        self.complete_register_btn.setVisible(True)
        self.cancel_register_btn.setVisible(True)
        self.status_label.setText(
            "Registration ready — click 'Complete Register' to save or 'Cancel Registration' to abort"
        )

    def commit_registration(self):
        """Commit pending registration to MongoDB and write local images."""
        if not self.pending_registration:
            return

        username = self.pending_registration["username"]
        embeddings_dict = self.pending_registration["embeddings"]
        metadata = self.pending_registration["metadata"]

        # Create local user folder and save pose images for records
        user_dir = os.path.join(DATASET_ROOT, username)
        os.makedirs(user_dir, exist_ok=True)
        for pose, img in self.reg_images.items():
            cv2.imwrite(
                os.path.join(user_dir, f"{pose}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

        encrypted_metadata = encryption_handler.encrypt_to_string(metadata)

        success = db_handler.store_user_embeddings(
            username, embeddings_dict, encrypted_metadata
        )

        if success:
            violation_logger.log_info(
                f"User {username} registered successfully",
                f"Embeddings stored with {len(embeddings_dict)} poses",
            )
            self.status_label.setText("✓ Registration Complete - Stored in MongoDB")
        else:
            violation_logger.log_warning(
                "registration_failed", f"Failed to store {username} in MongoDB"
            )
            self.status_label.setText("✗ Registration failed - MongoDB error")

        # cleanup pending state
        self.pending_registration = None
        self.reg_images = {}
        self.pose_index = 0
        self.complete_register_btn.setVisible(False)
        self.cancel_register_btn.setVisible(False)

        # if in admin flow, return to admin panel
        if self.admin_mode:
            self.show_admin_panel()

    def cancel_registration(self):
        """Abort pending registration — discard captured images/embeddings."""
        self.pending_registration = None
        self.reg_images = {}
        self.registering = False
        self.pose_index = 0
        self.complete_register_btn.setVisible(False)
        self.cancel_register_btn.setVisible(False)
        self.status_label.setText("Registration cancelled")

    def perform_verification_check(self, frame_rgb, username):
        """
        Perform verification check on a frame
        Returns: (score, is_valid)
        """
        # Ensure classifier is loaded before proceeding
        if not CLASSIFIER_LOADED or clf is None:
            print("⚠️ Classifier not loaded yet - may cause errors")
            return 0.0, False

        embeddings_dict = db_handler.get_user_embeddings(username)
        if not embeddings_dict:
            return 0.0, False

        face_crop, bbox = crop_face(frame_rgb)
        if face_crop is None:
            return 0.0, False

        emb_live = run_arc(face_crop)
        scores = []

        try:
            for pose, stored_emb in embeddings_dict.items():
                # Check for embedding corruption/mismatch
                if not isinstance(stored_emb, np.ndarray):
                    stored_emb = np.array(stored_emb)

                if stored_emb.shape != emb_live.shape:
                    # Likely encryption key mismatch
                    error_msg = (
                        f"✗ ENCRYPTION KEY MISMATCH\n"
                        f"User data corrupted: stored=({stored_emb.shape}), expected=({emb_live.shape})\n\n"
                        f"Solution:\n"
                        f"1. Check if both machines have the SAME ENCRYPTION_KEY in .env\n"
                        f"2. See ENCRYPTION_GUIDE.md for multi-machine setup\n"
                        f"3. Delete user '{username}' and re-register"
                    )
                    QMessageBox.critical(self, "Encryption Key Mismatch", error_msg)
                    violation_logger.log_warning(
                        "encryption_key_mismatch",
                        f"User {username}: embedding shape mismatch - likely different encryption key",
                    )
                    return 0.0, False

                feat = build_features(emb_live, stored_emb)
                scaled = (feat - sc_mean) / sc_scale
                prob = float(clf.predict(scaled.reshape(1, -1), verbose=0)[0][0])
                scores.append(prob)
        except Exception as e:
            violation_logger.log_warning(
                "verification_error",
                f"User {username}: {str(e)}",
            )
            return 0.0, False

        if not scores:
            return 0.0, False

        final = float(np.mean(np.sort(scores)[-TOP_K:]))
        return final, True

    # ---- liveness / anti-spoof helpers ----
    def reset_liveness(self):
        """Clear verification state."""
        pass

    def evaluate_liveness(self, frame_rgb, face_landmarks, face_crop):
        """Update blink/motion/texture statistics based on current frame."""
        if face_landmarks is None:
            return

        img_h, img_w, _ = frame_rgb.shape

        # helper to fetch 2d point
        def pt(idx):
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        # compute eye aspect ratio for left and right eyes
        left_h = np.linalg.norm(pt(159) - pt(145))
        left_w = np.linalg.norm(pt(33) - pt(133))
        right_h = np.linalg.norm(pt(386) - pt(374))
        right_w = np.linalg.norm(pt(263) - pt(362))
        left_ear = left_h / left_w if left_w > 0 else 0.0
        right_ear = right_h / right_w if right_w > 0 else 0.0
        ear = (left_ear + right_ear) / 2.0

        # blink detection
        if ear < BLINK_EAR_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
        else:
            if self.eye_closed:
                self.blink_count += 1
                self.eye_closed = False

        # motion detection (mean displacement of landmarks)
        curr = np.array(
            [[lm.x * img_w, lm.y * img_h] for lm in face_landmarks.landmark]
        )
        if self.prev_landmarks is not None and curr.shape == self.prev_landmarks.shape:
            dist = np.linalg.norm(curr - self.prev_landmarks, axis=1)
            mean_dist = float(np.mean(dist))
            alpha = 0.3
            self.motion_score = alpha * mean_dist + (1 - alpha) * self.motion_score
            self.motion_frames += 1
        self.prev_landmarks = curr

        # texture / sharpness check via Laplacian variance
        if face_crop is not None:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            self.texture_var = float(lap.var())

    def check_motion_sequence(self, current_pose):
        """Update and check if motion sequence is complete."""
        if current_pose and current_pose not in self.pose_sequence:
            self.pose_sequence.append(current_pose)
        # Check if sequence matches required
        if len(self.pose_sequence) >= len(MOTION_SEQUENCE):
            # Check if the last poses match the sequence
            recent = self.pose_sequence[-len(MOTION_SEQUENCE) :]
            if recent == MOTION_SEQUENCE:
                self.sequence_complete = True
                return True
        return False

    def liveness_conditions_met(self):
        """Return True if texture and motion checks pass (for continuous monitoring)."""
        return (
            self.texture_var >= LAPLACIAN_VAR_THRESH
            and self.motion_score >= MOTION_THRESH
        )

    def show_violation_popup(self, username, violation_type, frame):
        """Show violation popup to user"""

        # Customize message based on violation type
        if (
            violation_type == "verification_failed"
            or violation_type == "verification_violation"
        ):
            title = "⛔ UNAUTHORIZED ACCESS"
            message = (
                f"User {username}: Verification Failed.\n\n"
                "Face not recognized or unauthorized user.\n"
                "This security incident has been logged."
            )
            details_log = "Verification failed: Unauthorized user"
        elif violation_type == "phone_detected_violation":
            title = "⚠️ PHONE DETECTED"
            message = (
                f"User {username}: Phone usage detected.\n\n"
                "Using a phone during verification is prohibited."
            )
            details_log = "Phone detected during verification"
        else:
            # Default for bad conditions
            title = "⚠️ VIOLATION DETECTED"
            message = (
                f"User {username}: Bad conditions detected during verification.\n\n"
                "• Not looking forward\n"
                "• Multiple people detected\n"
                "• No face detected\n"
                "\nPlease maintain proper verification conditions."
            )
            details_log = "Continuous bad condition during verification"

        screenshot_data, filename = violation_logger.capture_violation(
            frame,
            violation_type,
            username,
            f"{violation_type}: {details_log}",
        )

        # Log to MongoDB with binary screenshot data
        db_handler.log_violation(
            violation_type,
            username,
            {
                "type": violation_type,
                "timestamp": str(
                    __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    )
                ),
                "details": details_log,
            },
            screenshot_data,
        )

        # Show popup with timeout
        self.violation_popup_showing = True
        self.last_violation_frame = frame.copy()

        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setDefaultButton(QMessageBox.StandardButton.Ok)

        # start timer to auto-close if no response
        def on_timeout():
            if dialog.isVisible():
                dialog.done(0)

        QTimer.singleShot(self.POPUP_DISMISS_TIMEOUT_MS, on_timeout)

        result = dialog.exec()

        self.violation_popup_showing = False
        # After popup (closed or timed out), restart verification or force re-auth on auto-dismiss
        self.bad_condition_frames = 0

        # If dialog was auto-closed (done(0) used), exec() returns 0 — require re-verification
        if result == 0:
            self.verifying = False
            self.initial_auth_done = False
            self.phone_detection_history = []  # Reset on re-verify
            self.phone_warning_triggered = False
            self.phone_violation_triggered = False
            self.verify_username = None
            self.status_label.setText("✗ Verification interrupted — please re-verify")
            violation_logger.log_info(
                f"User {username} auto-dismissed violation popup", "User must re-verify"
            )
            db_handler.log_violation(
                "popup_auto_dismiss",
                username,
                {
                    "details": "User did not respond to violation popup",
                    "timestamp": str(
                        __import__("datetime").datetime.now(
                            __import__("datetime").timezone.utc
                        )
                    ),
                },
                screenshot_data,
            )

    # ================= DELETE =================

    def delete_user(self):
        username = self.username_input.text()

        # Exit verification if active
        if self.verifying:
            self.verifying = False
            self.status_label.setText("Verification cancelled")

        user_dir = os.path.join(DATASET_ROOT, username)

        # Delete from MongoDB
        db_success = db_handler.delete_user(username)

        # Delete local files
        if os.path.isdir(user_dir):
            import shutil

            shutil.rmtree(user_dir)
            local_deleted = True
        else:
            local_deleted = False

        if db_success and local_deleted:
            violation_logger.log_info(
                f"User {username} deleted", "Removed from MongoDB and local storage"
            )
            self.status_label.setText("✓ User Deleted from System")
        elif db_success:
            violation_logger.log_info(
                f"User {username} deleted from MongoDB", "Local files not found"
            )
            self.status_label.setText("✓ User Deleted (MongoDB only)")
        else:
            self.status_label.setText("✗ User Not Found")

    def keyPressEvent(self, event):
        """Allow ESC to exit verification mode"""
        if event.key() == Qt.Key.Key_Escape and self.verifying:
            self.verifying = False
            self.bad_condition_frames = 0
            self.violation_popup_showing = False
            self.status_label.setText("✓ Verification cancelled (ESC pressed)")

    def style(self):
        return """
        QWidget { background-color: #121212; color: white; font-size: 14px; }
        QLineEdit { background-color: #1e1e1e; padding: 8px; border-radius: 6px; }
        QPushButton { background-color: #2979FF; padding: 10px; border-radius: 8px; }
        QPushButton:hover { background-color: #1565C0; }
        """

    def closeEvent(self, event):
        """Clean up when application closes"""
        print("\n🔄 Running cleanup tasks...")

        # Clean old violations from MongoDB
        retention_days = int(os.getenv("VIOLATION_RETENTION_DAYS", 30))
        db_handler.clean_old_violations(retention_days)

        # Clean old violation evidence screenshots
        violation_logger.cleanup_old_violations(retention_days)

        # Stop phone detector thread if running
        try:
            if getattr(self, "phone_detector", None):
                self.phone_detector.stop()
        except Exception:
            pass

        print("✓ Cleanup completed")
        self.cap.release()
        event.accept()


app = QApplication(sys.argv)
window = ContinuAuth()
window.show()
sys.exit(app.exec())
