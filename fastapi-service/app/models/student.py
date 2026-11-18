from pydantic import BaseModel
from typing import Optional

class StudentProfile(BaseModel):
    nim: str
    nama: str
    prodi: str
    kelas: str

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class RecognitionResult(BaseModel):
    bounding_box: BoundingBox
    match: bool
    nim: Optional[str] = None
    nama: str
    prodi: Optional[str] = None
    kelas: Optional[str] = None
    confidence: float
    attendance_type: Optional[str] = None
    attendance_date: Optional[str] = None
    attendance_time: Optional[str] = None
    attendance_status: Optional[str] = None