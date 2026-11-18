import os
import cv2
import json
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.orm import Student, Attendance
from app.utils.geo import is_within_geofence
from app.models.student import StudentProfile, RecognitionResult, BoundingBox
from app.services.face_detector import HaarFaceDetector
from app.redis_client import redis_client
from app.es_client import es_client

class LBPHFaceRecognizerService:
    def __init__(self, dataset_dir: str = "dataset", trainer_dir: str = "trainer"):
        self.dataset_dir = dataset_dir
        self.trainer_dir = trainer_dir
        self.model_path = os.path.join(trainer_dir, "trainer.yml")
        self.id_map_path = os.path.join(trainer_dir, "id_map.json")
        
        self.detector = HaarFaceDetector()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._ensure_directories()
        self.load_model()

    def _ensure_directories(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.trainer_dir, exist_ok=True)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)

    def save_student_to_db(self, db: Session, profile: StudentProfile) -> Student:
        student = db.query(Student).filter(Student.nim == profile.nim).first()
        if not student:
            student = Student(
                nim=profile.nim,
                nama=profile.nama,
                prodi=profile.prodi,
                kelas=profile.kelas
            )
            db.add(student)
            db.commit()
            db.refresh(student)
        return student

    def save_dataset_and_train(self, db: Session, profile: StudentProfile, images_bytes: List[bytes]) -> int:
        student = self.save_student_to_db(db, profile)
        user_dir = os.path.join(self.dataset_dir, student.nim)
        os.makedirs(user_dir, exist_ok=True)

        saved_count = 0
        for idx, img_bytes in enumerate(images_bytes):
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect(gray)

            for (x, y, w, h) in faces:
                face_crop = gray[y:y+h, x:x+w]
                file_path = os.path.join(user_dir, f"{student.nim}_{idx+1}.jpg")
                cv2.imwrite(file_path, face_crop)
                saved_count += 1
                break

        if saved_count > 0:
            self.train_model(db)

        return saved_count

    def train_model(self, db: Session):
        students = db.query(Student).all()
        if not students:
            return

        nim_list = [s.nim for s in students]
        face_samples = []
        ids = []

        for nim in nim_list:
            label_id = nim_list.index(nim)
            user_dir = os.path.join(self.dataset_dir, nim)
            if not os.path.exists(user_dir):
                continue

            for img_name in os.listdir(user_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(user_dir, img_name)
                    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if gray_img is not None:
                        face_samples.append(gray_img)
                        ids.append(label_id)

        if face_samples:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(face_samples, np.array(ids))
            self.recognizer.write(self.model_path)

            id_map = {idx: nim for idx, nim in enumerate(nim_list)}
            with open(self.id_map_path, "w") as f:
                json.dump(id_map, f, indent=4)

    def recognize_frame(
            self, 
            db: Session, 
            image_bytes: bytes,
            attendance_type: str = "clock_in",
            latitude: Optional[float] = None,
            longitude: Optional[float] = None, 
            threshold: float = 85.0
            ) -> List[RecognitionResult]:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model belum di-train.")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect(gray)
        
        if not os.path.exists(self.id_map_path):
            return []

        with open(self.id_map_path, "r") as f:
            id_map = json.load(f)

        wib_tz = ZoneInfo("Asia/Jakarta")
        now_wib = datetime.now(wib_tz)
        today_date = now_wib.strftime("%Y-%m-%d")
        current_time = now_wib.strftime("%H:%M:%S")
        timestamp_naive = now_wib.replace(tzinfo=None)

        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label_id, confidence = self.recognizer.predict(face_roi)
            box = BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))

            if confidence < threshold:
                nim = id_map.get(str(label_id))

                # 1. Cek Data Mahasiswa dari Redis Cache
                cached_student = redis_client.get(f"student:{nim}")
                if cached_student:
                    student_data = json.loads(cached_student)
                else:
                    student_db = db.query(Student).filter(Student.nim == nim).first()
                    if student_db:
                        student_data = {
                            "id": student_db.id,
                            "nim": student_db.nim,
                            "nama": student_db.nama,
                            "prodi": student_db.prodi,
                            "kelas": student_db.kelas
                        }
                        redis_client.setex(f"student:{nim}", 3600, json.dumps(student_data))
                    else:
                        student_data = None

                if student_data:
                    
                    is_valid_loc = True
                    dist_meter = 0.0
                    if latitude is not None and longitude is not None:
                        is_valid_loc, dist_meter = is_within_geofence(latitude,longitude)

                    daily_lock_key = f"absensi: {student_data['id']}: {attendance_type}:{today_date}"
                    attendance_status = "recorded"

                    if not is_valid_loc:
                        attendance_status = "Absensi diluar Jangkauan"
                    elif redis_client.exists(daily_lock_key):
                        attendance_status = "Sudah melakukan absen hari ini"
                    else:
                        attendance = Attendance(
                            student_id = student_data["id"],
                            attendance_type = attendance_type,
                            confidence_score = round(float(confidence), 2),
                            latitude=latitude,
                            longitude=longitude,
                            is_valid_location=is_valid_loc,
                            detected_at = timestamp_naive
                        )
                        db.add(attendance)
                        db.commit()

                        redis_client.setex(daily_lock_key, 43200, current_time)

                        doc = {
                             "student_id": student_data["id"],
                             "nim": student_data["nim"],
                             "nama": student_data["nama"],
                             "prodi": student_data["prodi"],
                             "kelas": student_data["kelas"],
                             "attendance_type": attendance_type,
                             "confidence_score": round(float(confidence), 2),
                             "latitude": latitude,
                             "longitud": longitude,
                             "distance_meters": dist_meter,
                             "is_valid_location": is_valid_loc,
                             "date": today_date,
                             "time": current_time,
                             "detected_at": timestamp_naive.isoformat()
                        }
                        try:
                            es_client.index(index="attendace_logs", document=doc)
                        except Exception:
                            pass

                    results.append(
                        RecognitionResult(
                            bounding_box = box,
                            match = True,
                            nim = student_data["nim"],
                            nama = student_data["nama"],
                            prodi = student_data["prodi"],
                            kelas = student_data["kelas"],
                            confidence = round(float(confidence), 2),
                            attendance_type = attendance_type,
                            attendance_date = today_date,
                            attendance_time = current_time,
                            attendance_status = attendance_status
                        )
                    )

                    # lock_key = f"attendance_lock:{nim}"
                    # if not redis_client.get(lock_key):
                    #     redis_client.setex(lock_key, 10, "locked")

                    #     # Simpan Kehadiran ke MySQL
                    #     attendance = Attendance(
                    #         student_id=student_data["id"],
                    #         confidence_score=round(float(confidence), 2)
                    #     )
                    #     db.add(attendance)
                    #     db.commit()

                    #     # 3. Index Log ke Elasticsearch
                    #     doc = {
                    #         "student_id": student_data["id"],
                    #         "nim": student_data["nim"],
                    #         "nama": student_data["nama"],
                    #         "prodi": student_data["prodi"],
                    #         "kelas": student_data["kelas"],
                    #         "confidence_score": round(float(confidence), 2),
                    #         "detected_at": datetime.utcnow().isoformat()
                    #     }
                    #     try:
                    #         es_client.index(index="attendance_logs", document=doc)
                    #     except Exception:
                    #         pass

                    # results.append(
                    #     RecognitionResult(
                    #         bounding_box=box,
                    #         match=True,
                    #         nim=student_data["nim"],
                    #         nama=student_data["nama"],
                    #         prodi=student_data["prodi"],
                    #         kelas=student_data["kelas"],
                    #         confidence=round(float(confidence), 2)
                    #     )
                    # )
            else:
                results.append(
                    RecognitionResult(
                        bounding_box=box,
                        match=False,
                        nama="Unknown",
                        confidence=round(float(confidence), 2),
                        attendance_type = attendance_type,
                        attendance_date = today_date,
                        attendance_time = current_time,
                        attendance_status = "Waja Tidak Terdeteksi"
                    )
                )

        return results