from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, List
import os
from pathlib import Path

app = FastAPI(title="Face Attendance System API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URL)
db = client.attendance_system

# Collections
students_collection = db.students
attendance_collection = db.attendance


# Models
class Student(BaseModel):
    enrollment_no: str
    name: str
    email: str
    department: str
    year: int


class AttendanceRecord(BaseModel):
    enrollment_no: str
    timestamp: datetime
    status: str = "present"


class AttendanceQuery(BaseModel):
    enrollment_no: Optional[str] = None
    date: Optional[str] = None


# Endpoints
@app.on_event("startup")
async def startup_db_client():
    try:
        await client.admin.command('ping')
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


@app.get("/")
async def root():
    return {"message": "Face Attendance System API"}


# Student Management
@app.post("/students/", response_model=dict)
async def add_student(student: Student):
    """Add a new student to the database"""
    existing = await students_collection.find_one({"enrollment_no": student.enrollment_no})
    if existing:
        raise HTTPException(status_code=400, detail="Student already exists")

    student_dict = student.dict()
    student_dict["created_at"] = datetime.now()
    result = await students_collection.insert_one(student_dict)

    return {"message": "Student added successfully", "id": str(result.inserted_id)}


@app.get("/students/{enrollment_no}")
async def get_student(enrollment_no: str):
    """Get student details by enrollment number"""
    student = await students_collection.find_one({"enrollment_no": enrollment_no})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    student["_id"] = str(student["_id"])
    return student


@app.get("/students/")
async def get_all_students():
    """Get all students"""
    students = []
    async for student in students_collection.find():
        student["_id"] = str(student["_id"])
        students.append(student)
    return students


@app.put("/students/{enrollment_no}")
async def update_student(enrollment_no: str, student: Student):
    """Update student details"""
    result = await students_collection.update_one(
        {"enrollment_no": enrollment_no},
        {"$set": student.dict()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")

    return {"message": "Student updated successfully"}


@app.delete("/students/{enrollment_no}")
async def delete_student(enrollment_no: str):
    """Delete a student"""
    # Delete student image
    image_path = f"images/{enrollment_no}.jpg"
    if os.path.exists(image_path):
        os.remove(image_path)

    result = await students_collection.delete_one({"enrollment_no": enrollment_no})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")

    return {"message": "Student deleted successfully"}


# Attendance Management
@app.post("/attendance/", response_model=dict)
async def mark_attendance(record: AttendanceRecord):
    """Mark attendance for a student"""
    # Check if student exists
    student = await students_collection.find_one({"enrollment_no": record.enrollment_no})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Check if already marked today
    today = datetime.now().date()
    existing = await attendance_collection.find_one({
        "enrollment_no": record.enrollment_no,
        "date": today.isoformat()
    })

    if existing:
        return {"message": "Attendance already marked for today", "duplicate": True}

    attendance_dict = {
        "enrollment_no": record.enrollment_no,
        "student_name": student["name"],
        "timestamp": record.timestamp,
        "date": record.timestamp.date().isoformat(),
        "time": record.timestamp.time().isoformat(),
        "status": record.status
    }

    result = await attendance_collection.insert_one(attendance_dict)
    return {"message": "Attendance marked successfully", "id": str(result.inserted_id), "duplicate": False}


@app.post("/attendance/query")
async def query_attendance(query: AttendanceQuery):
    """Query attendance records"""
    filter_dict = {}

    if query.enrollment_no:
        filter_dict["enrollment_no"] = query.enrollment_no

    if query.date:
        filter_dict["date"] = query.date

    records = []
    async for record in attendance_collection.find(filter_dict).sort("timestamp", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return records


@app.get("/attendance/today")
async def get_today_attendance():
    """Get today's attendance records"""
    today = datetime.now().date().isoformat()
    records = []
    async for record in attendance_collection.find({"date": today}).sort("timestamp", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return records


@app.get("/attendance/stats/{enrollment_no}")
async def get_attendance_stats(enrollment_no: str):
    """Get attendance statistics for a student"""
    total = await attendance_collection.count_documents({"enrollment_no": enrollment_no})
    present = await attendance_collection.count_documents({
        "enrollment_no": enrollment_no,
        "status": "present"
    })

    return {
        "enrollment_no": enrollment_no,
        "total_days": total,
        "present_days": present,
        "attendance_percentage": (present / total * 100) if total > 0 else 0
    }


@app.post("/upload-image/{enrollment_no}")
async def upload_student_image(enrollment_no: str, file: UploadFile = File(...)):
    """Upload student image"""
    # Create images directory if it doesn't exist
    Path("images").mkdir(exist_ok=True)

    # Save the image
    image_path = f"images/{enrollment_no}.jpg"
    with open(image_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {"message": "Image uploaded successfully", "path": image_path}


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI backend...")
    print("API will be available at: http://127.0.0.1:8000")
    print("API Documentation at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)