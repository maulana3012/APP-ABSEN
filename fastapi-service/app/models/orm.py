from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from zoneinfo import ZoneInfo
from app.database import Base

def get_wib_now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).replace(tzinfo=None)

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    nim = Column(String(20), unique=True, nullable=False, index=True)
    nama = Column(String(100), nullable=False)
    prodi = Column(String(100), nullable=False)
    kelas = Column(String(50), nullable=False)
    
    attendances = relationship("Attendance", back_populates="student")

class Attendance(Base):
    __tablename__ = "attendances"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    attendance_type = Column(String(20), nullable=False, default="clock_in")
    confidence_score = Column(Float, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    is_valid_location = Column(Boolean, default=True)
    detected_at = Column(DateTime, default=get_wib_now, nullable=False)

    student = relationship("Student", back_populates="attendances")