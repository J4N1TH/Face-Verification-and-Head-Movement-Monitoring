import cv2
import os
from datetime import datetime
import logging
from pathlib import Path


class ViolationLogger:
    """Handles violation capture (screenshots and logs)"""

    def __init__(self, screenshot_folder="violations_evidence", log_folder="logs"):
        self.screenshot_folder = screenshot_folder
        self.log_folder = log_folder

        # Create folders
        os.makedirs(screenshot_folder, exist_ok=True)
        os.makedirs(log_folder, exist_ok=True)

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for violations and warnings"""
        log_file = os.path.join(self.log_folder, "violations.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def capture_violation(self, frame, violation_type, username, details):
        """
        Capture violation screenshot and return binary data + log details

        Returns: (screenshot_binary_data, filename) or (None, None) on error
        """
        try:
            timestamp = datetime.now()
            safe_user = str(username).replace(" ", "_")
            safe_type = str(violation_type).replace(" ", "_")
            filename = f"violation_{safe_type}_{safe_user}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"

            # Encode frame to JPEG bytes (no local save)
            retval, buffer = __import__("cv2").imencode(".jpg", frame)
            if not retval:
                raise Exception("Failed to encode frame")
            screenshot_data = buffer.tobytes()

            # Log violation details
            self.logger.warning(
                f"VIOLATION: {violation_type} | User: {username} | "
                f"Time: {timestamp.isoformat()} | Details: {details} | "
                f"Screenshot: {filename}"
            )

            return screenshot_data, filename
        except Exception as e:
            self.logger.error(f"Error capturing violation: {e}")
            return None, None

    def log_warning(self, warning_type, details):
        """
        Log warning message (no screenshot)

        Example: multiple_faces_detected, low_face_confidence
        """
        try:
            timestamp = datetime.now()
            self.logger.warning(
                f"WARNING: {warning_type} | Time: {timestamp.isoformat()} | "
                f"Details: {details}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error logging warning: {e}")
            return False

    def log_info(self, message, extra_details=None):
        """Log informational message"""
        try:
            if extra_details:
                message = f"{message} | {extra_details}"
            self.logger.info(message)
            return True
        except Exception as e:
            self.logger.error(f"Error logging info: {e}")
            return False

    def log_violation(self, violation_type, username, frame, details=""):
        """Convenience wrapper for capture_violation.

        Saves the screenshot to disk and logs the event. Returns the
        saved filename or None if something went wrong.
        """
        screenshot_data, filename = self.capture_violation(
            frame, violation_type, username, details
        )
        if screenshot_data and filename:
            # write binary file to screenshot folder
            try:
                path = os.path.join(self.screenshot_folder, filename)
                with open(path, "wb") as f:
                    f.write(screenshot_data)
            except Exception as e:
                self.logger.error(f"Failed to save violation image: {e}")
                return None
            return filename
        return None

    def get_violation_evidence(self, hours=24):
        """Get all violation screenshots from last N hours"""
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            violations = []

            for filename in os.listdir(self.screenshot_folder):
                filepath = os.path.join(self.screenshot_folder, filename)
                file_time = os.path.getmtime(filepath)

                if file_time > cutoff_time:
                    violations.append(
                        {
                            "filename": filename,
                            "path": filepath,
                            "timestamp": datetime.fromtimestamp(file_time),
                        }
                    )

            return sorted(violations, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            self.logger.error(f"Error retrieving evidence: {e}")
            return []

    def cleanup_old_violations(self, retention_days=30):
        """Delete violation evidence older than retention period"""
        try:
            cutoff_time = datetime.now().timestamp() - (retention_days * 86400)
            deleted_count = 0

            for filename in os.listdir(self.screenshot_folder):
                filepath = os.path.join(self.screenshot_folder, filename)
                file_time = os.path.getmtime(filepath)

                if file_time < cutoff_time:
                    os.remove(filepath)
                    deleted_count += 1
                    self.logger.info(f"Deleted old violation: {filename}")

            return deleted_count
        except Exception as e:
            self.logger.error(f"Error cleaning violations: {e}")
            return 0

    def get_violation_summary(self, days=7):
        """Get summary of violations from last N days"""
        try:
            logs_file = os.path.join(self.log_folder, "violations.log")
            cutoff_time = datetime.now().timestamp() - (days * 86400)

            violations = []
            if os.path.exists(logs_file):
                with open(logs_file, "r") as f:
                    for line in f:
                        if "VIOLATION" in line:
                            violations.append(line.strip())

            return violations
        except Exception as e:
            self.logger.error(f"Error getting summary: {e}")
            return []
