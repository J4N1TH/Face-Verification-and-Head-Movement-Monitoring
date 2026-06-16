import pymongo
from pymongo import MongoClient
from gridfs import GridFS
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
import pickle
import json

load_dotenv()


class MongoDBHandler:
    """Handles all MongoDB operations for face embeddings and violation logs"""

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME", "continuauth")
        self.client = None
        self.db = None
        self.gridfs = None
        self.local_storage_dir = os.path.join(
            os.path.dirname(__file__), "local_storage"
        )
        os.makedirs(self.local_storage_dir, exist_ok=True)
        self.connect()

    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.gridfs = GridFS(self.db)

            # Create indexes
            self.db.users.create_index("username", unique=True)
            self.db.violations.create_index("timestamp")
            self.db.logs.create_index("timestamp")

            print("✓ Connected to MongoDB")
            return True
        except Exception as e:
            print(f"✗ MongoDB Connection Error: {e}")
            self.db = None
            self.gridfs = None
            print("  Falling back to local storage")
            return False

    def _get_local_user_file(self, username):
        """Get path to local user storage file"""
        return os.path.join(self.local_storage_dir, f"user_{username}.pkl")

    def _store_user_locally(self, username, embeddings_dict, encrypted_data):
        """Store user data locally as fallback"""
        try:
            user_data = {
                "username": username,
                "embeddings": embeddings_dict,
                "encrypted_data": encrypted_data,
                "registered_at": datetime.utcnow(),
                "last_verified": None,
                "verification_count": 0,
            }

            filepath = self._get_local_user_file(username)
            with open(filepath, "wb") as f:
                pickle.dump(user_data, f)

            print(f"✓ Stored user {username} locally")
            return True
        except Exception as e:
            print(f"✗ Error storing user locally: {e}")
            return False

    def _get_user_locally(self, username):
        """Retrieve user data from local storage"""
        try:
            filepath = self._get_local_user_file(username)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"✗ Error loading user locally: {e}")
        return None

    def store_user_embeddings(self, username, embeddings_dict, encrypted_data):
        """
        Store user embeddings and encrypted data

        embeddings_dict: {"Forward": np.array, "Left": np.array, ...}
        encrypted_data: encrypted user information
        """
        if self.db is not None:
            # Use MongoDB if available
            try:
                # Validate embeddings before storing
                for pose, emb in embeddings_dict.items():
                    if not isinstance(emb, np.ndarray):
                        emb = np.array(emb)

                    if emb.shape != (512,):
                        print(
                            f"✗ ERROR: Invalid embedding shape for {pose}: {emb.shape} (expected (512,))"
                        )
                        return False

                    if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                        print(f"✗ ERROR: Embedding contains NaN/Inf for {pose}")
                        return False

                user_doc = {
                    "username": username,
                    "embeddings": {
                        pose: embeddings_dict[pose].tolist() for pose in embeddings_dict
                    },
                    "encrypted_data": encrypted_data,
                    "registered_at": datetime.utcnow(),
                    "last_verified": None,
                    "verification_count": 0,
                }

                result = self.db.users.update_one(
                    {"username": username}, {"$set": user_doc}, upsert=True
                )

                return result.upserted_id or result.matched_count > 0
            except Exception as e:
                print(f"✗ Error storing embeddings in MongoDB: {e}")
                return False
        else:
            # Fall back to local storage
            return self._store_user_locally(username, embeddings_dict, encrypted_data)

    def get_user_embeddings(self, username):
        """Retrieve user embeddings"""
        if self.db is not None:
            # Try MongoDB first
            try:
                user = self.db.users.find_one({"username": username})
                if user:
                    embeddings = {}
                    for pose, emb_data in user.get("embeddings", {}).items():
                        emb_array = np.array(emb_data)

                        # Validate shape
                        if emb_array.shape != (512,):
                            print(
                                f"✗ ERROR: Retrieved embedding has wrong shape: {emb_array.shape} (expected (512,))"
                            )
                            print(f"  Pose: {pose}, Username: {username}")
                            print(
                                f"  Raw data type: {type(emb_data)}, first 5 elements: {emb_data[:5] if hasattr(emb_data, '__getitem__') else 'N/A'}"
                            )
                            return None

                        embeddings[pose] = emb_array

                    return embeddings if embeddings else None
            except Exception as e:
                print(f"✗ Error retrieving embeddings from MongoDB: {e}")

        # Fall back to local storage
        user_data = self._get_user_locally(username)
        if user_data:
            embeddings = {}
            for pose, emb in user_data.get("embeddings", {}).items():
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings[pose] = emb
            return embeddings if embeddings else None

        return None

    def log_violation(self, violation_type, username, details, screenshot_data=None):
        """
        Log a violation event with optional screenshot stored in GridFS

        violation_type: "unauthorized_attempt", "phone_detected_violation", etc.
        screenshot_data: binary image data (bytes) from cv2.imencode() or None
        """
        if self.db is None:
            print("✗ Cannot log violation: MongoDB not connected")
            return None

        try:
            violation_doc = {
                "type": violation_type,
                "username": username,
                "timestamp": datetime.utcnow(),
                "details": details,
                "screenshot_id": None,
                "deleted": False,
            }

            # Store screenshot in GridFS if provided
            if screenshot_data is not None:
                try:
                    filename = f"violation_{violation_type}_{username}_{datetime.utcnow().isoformat()}.jpg"
                    file_id = self.gridfs.put(screenshot_data, filename=filename)
                    violation_doc["screenshot_id"] = file_id
                except Exception as e:
                    print(f"✗ Error storing screenshot to GridFS: {e}")

            result = self.db.violations.insert_one(violation_doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"✗ Error logging violation: {e}")
            return None

    def get_violation_screenshot(self, file_id):
        """Retrieve screenshot binary data from GridFS"""
        if self.db is None or self.gridfs is None:
            print("✗ Cannot retrieve screenshot: MongoDB not connected")
            return None

        try:
            return self.gridfs.get(file_id).read()
        except Exception as e:
            print(f"✗ Error retrieving screenshot: {e}")
            return None

    def log_warning(self, warning_type, details):
        """
        Log warning message (no screenshot)

        warning_type: "multiple_faces", "low_confidence", etc.
        """
        if self.db is None:
            print("✗ Cannot log warning: MongoDB not connected")
            return None

        try:
            log_doc = {
                "type": warning_type,
                "timestamp": datetime.utcnow(),
                "details": details,
            }

            result = self.db.logs.insert_one(log_doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"✗ Error logging warning: {e}")
            return None

    def update_verification(self, username, verification_successful, score):
        """Update user verification record"""
        if self.db is not None:
            try:
                self.db.users.update_one(
                    {"username": username},
                    {
                        "$set": {"last_verified": datetime.utcnow()},
                        "$inc": {"verification_count": 1},
                    },
                )
                return True
            except Exception as e:
                print(f"✗ Error updating verification in MongoDB: {e}")

        # Update local storage
        user_data = self._get_user_locally(username)
        if user_data:
            user_data["last_verified"] = datetime.utcnow()
            user_data["verification_count"] = user_data.get("verification_count", 0) + 1
            try:
                filepath = self._get_local_user_file(username)
                with open(filepath, "wb") as f:
                    pickle.dump(user_data, f)
                return True
            except Exception as e:
                print(f"✗ Error updating verification locally: {e}")

        return False

    def clean_old_violations(self, retention_days=None):
        """Delete violations older than retention period"""
        if self.db is None:
            print("✗ Cannot clean violations: MongoDB not connected")
            return 0

        if retention_days is None:
            retention_days = int(os.getenv("VIOLATION_RETENTION_DAYS", 30))

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            result = self.db.violations.delete_many({"timestamp": {"$lt": cutoff_date}})
            print(f"✓ Deleted {result.deleted_count} old violation records")
            return result.deleted_count
        except Exception as e:
            print(f"✗ Error cleaning violations: {e}")
            return 0

    def get_violations(self, username=None, days=30):
        """Retrieve violations from last N days"""
        if self.db is None:
            print("✗ Cannot retrieve violations: MongoDB not connected")
            return []

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = {"timestamp": {"$gte": cutoff_date}, "deleted": False}

            if username:
                query["username"] = username

            return list(self.db.violations.find(query).sort("timestamp", -1))
        except Exception as e:
            print(f"✗ Error retrieving violations: {e}")
            return []

    def get_logs(self, days=7, warning_type=None):
        """Retrieve warning logs from last N days"""
        if self.db is None:
            print("✗ Cannot retrieve logs: MongoDB not connected")
            return []

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = {"timestamp": {"$gte": cutoff_date}}

            if warning_type:
                query["type"] = warning_type

            return list(self.db.logs.find(query).sort("timestamp", -1))
        except Exception as e:
            print(f"✗ Error retrieving logs: {e}")
            return []

    def delete_user(self, username):
        """Delete user and associated records"""
        if self.db is None:
            print("✗ Cannot delete user: MongoDB not connected")
            return False

        try:
            self.db.users.delete_one({"username": username})
            self.db.violations.delete_many({"username": username})
            print(f"✓ Deleted user {username} from MongoDB")
            return True
        except Exception as e:
            print(f"✗ Error deleting user: {e}")
            return False
