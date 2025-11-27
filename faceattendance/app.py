import streamlit as st
import requests
import pandas as pd
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime
from PIL import Image
import os

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- LIVENESS CONFIGURATION ---
SMILE_THRESH = 0.45  # Threshold for Smiling (Open Mouth)
NEUTRAL_THRESH = 0.32  # Threshold for Neutral (Closed Mouth)
FRAME_COUNT_REQ = 5  # Frames to hold the smile

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="📸",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .stSuccess { background-color: #d4edda; color: #155724; }
    .stError { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---

def get_mar(top_lip, bottom_lip):
    """Calculates Mouth Aspect Ratio to detect smiles/open mouth."""
    A = np.linalg.norm(np.array(top_lip[2]) - np.array(bottom_lip[10]))
    B = np.linalg.norm(np.array(top_lip[3]) - np.array(bottom_lip[9]))
    C = np.linalg.norm(np.array(top_lip[4]) - np.array(bottom_lip[8]))
    D = np.linalg.norm(np.array(top_lip[0]) - np.array(top_lip[6]))

    if D == 0: return 0
    mar = (A + B + C) / (2.0 * D)
    return mar


def add_student_api(student_data):
    try:
        response = requests.post(f"{API_URL}/students/", json=student_data)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend API is not running.")
        return {}


def get_all_students():
    try:
        response = requests.get(f"{API_URL}/students/")
        return response.json()
    except:
        return []


def get_student(enrollment_no):
    try:
        response = requests.get(f"{API_URL}/students/{enrollment_no}")
        return response.json()
    except:
        return {}


def mark_attendance_api(enrollment_no):
    try:
        data = {
            "enrollment_no": enrollment_no,
            "timestamp": datetime.now().isoformat(),
            "status": "present"
        }
        response = requests.post(f"{API_URL}/attendance/", json=data)
        return response.json()
    except:
        return {}


def get_today_attendance():
    try:
        response = requests.get(f"{API_URL}/attendance/today")
        return response.json()
    except:
        return []


def upload_image(enrollment_no, image_file):
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_URL}/upload-image/{enrollment_no}", files=files)
        return response.json()
    except:
        return {}


def get_attendance_stats(enrollment_no):
    try:
        response = requests.get(f"{API_URL}/attendance/stats/{enrollment_no}")
        return response.json()
    except:
        return {}


# --- SIDEBAR ---
st.sidebar.title("📸 Face Attendance System")
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "👤 Student Management",
    "📷 Mark Attendance",
    "📊 View Attendance",
    "📈 Statistics"
])

# --- PAGE: HOME ---
if page == "🏠 Home":
    st.title("Welcome to Face Attendance System")

    col1, col2, col3 = st.columns(3)

    students = get_all_students()
    today_attendance = get_today_attendance()

    with col1:
        st.metric("Total Students", len(students))

    with col2:
        st.metric("Today's Attendance", len(today_attendance))

    with col3:
        if students:
            percentage = (len(today_attendance) / len(students)) * 100
            st.metric("Attendance %", f"{percentage:.1f}%")
        else:
            st.metric("Attendance %", "0%")

    st.markdown("---")
    st.subheader("Today's Attendance Records")

    if today_attendance:
        df = pd.DataFrame(today_attendance)
        df = df[['enrollment_no', 'student_name', 'time', 'status']]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No attendance records for today")

# --- PAGE: STUDENT MANAGEMENT ---
elif page == "👤 Student Management":
    st.title("Student Management")

    tab1, tab2, tab3 = st.tabs(["Add Student", "View Students", "Update/Delete"])

    with tab1:
        st.subheader("Add New Student")
        with st.form("add_student_form"):
            enrollment_no = st.text_input("Enrollment Number*")
            name = st.text_input("Full Name*")
            email = st.text_input("Email*")
            department = st.selectbox("Department*",
                                      ["Computer Science", "Electronics", "Mechanical", "Civil", "Electrical"])
            year = st.selectbox("Year*", [1, 2, 3, 4])
            image_file = st.file_uploader("Upload Student Photo*", type=['jpg', 'jpeg', 'png'])

            submit = st.form_submit_button("Add Student")

            if submit:
                if not all([enrollment_no, name, email, image_file]):
                    st.error("Please fill all required fields")
                else:
                    try:
                        student_data = {
                            "enrollment_no": enrollment_no,
                            "name": name,
                            "email": email,
                            "department": department,
                            "year": year
                        }
                        add_student_api(student_data)
                        upload_image(enrollment_no, image_file)
                        st.success(f"Student {name} added successfully!")
                        st.info("⚠️ Please regenerate face encodings using encode_generator.py")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with tab2:
        st.subheader("All Students")
        students = get_all_students()
        if students:
            df = pd.DataFrame(students)
            df = df[['enrollment_no', 'name', 'email', 'department', 'year']]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No students found")

    with tab3:
        st.subheader("Delete Student")
        students = get_all_students()
        if students:
            enrollment_nos = [s['enrollment_no'] for s in students]
            selected_enrollment = st.selectbox("Select Student", enrollment_nos)

            if st.button("Delete Student", type="secondary"):
                try:
                    requests.delete(f"{API_URL}/students/{selected_enrollment}")
                    st.success("Student deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# --- PAGE: MARK ATTENDANCE (WITH LIVENESS) ---
elif page == "📷 Mark Attendance":
    st.title("Mark Attendance")
    st.info("Instructions: 1. Keep face neutral (Mouth Closed). 2. Then SMILE!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        run_camera = st.checkbox("Start Camera")

        if run_camera:
            try:
                # Load Encodings
                file = open('EncodeFile.p', 'rb')
                encodingListwithIds = pickle.load(file)
                file.close()
                encodingList, studentIdList = encodingListwithIds
                st.info("Encodings loaded.")

                cap = cv2.VideoCapture(0)
                stframe = st.empty()

                # --- DYNAMIC LIVENESS VARIABLES ---
                COUNTER = 0
                LIVENESS_CONFIRMED = False
                STATE = "WAITING_FOR_NEUTRAL"  # States: WAITING_FOR_NEUTRAL -> READY_FOR_SMILE -> VERIFIED

                while run_camera:
                    success, img = cap.read()
                    if not success:
                        st.error("Camera not found")
                        break

                    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                    faceCurrFrame = face_recognition.face_locations(imgS)
                    encodingCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

                    # --- LIVENESS LOGIC ---
                    face_locs_scaled = []
                    for (y1, x2, y2, x1) in faceCurrFrame:
                        face_locs_scaled.append((y1 * 4, x2 * 4, y2 * 4, x1 * 4))

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_landmarks_list = face_recognition.face_landmarks(img_rgb, face_locs_scaled)

                    if len(face_landmarks_list) > 0:
                        face_landmarks = face_landmarks_list[0]
                        top_lip = face_landmarks['top_lip']
                        bottom_lip = face_landmarks['bottom_lip']
                        mar = get_mar(top_lip, bottom_lip)

                        # --- STATE MACHINE LOGIC ---
                        if STATE == "WAITING_FOR_NEUTRAL":
                            if mar < NEUTRAL_THRESH:
                                STATE = "READY_FOR_SMILE"
                            status_msg = "Step 1: Keep Face Neutral"
                            status_color = (0, 255, 255)  # Yellow

                        elif STATE == "READY_FOR_SMILE":
                            if mar > SMILE_THRESH:
                                COUNTER += 1
                                if COUNTER >= FRAME_COUNT_REQ:
                                    LIVENESS_CONFIRMED = True
                                    STATE = "VERIFIED"
                            else:
                                COUNTER = 0
                                # If they open their mouth slightly but not enough, stay in READY state
                                # If they go back to full neutral, that's fine too.

                            status_msg = "Step 2: Now SMILE! 😁"
                            status_color = (0, 0, 255)  # Red (until verified)

                        elif STATE == "VERIFIED":
                            status_msg = "Liveness Verified ✅"
                            status_color = (0, 255, 0)  # Green

                        # Overlay Status
                        cv2.putText(img, f"MAR: {mar:.2f} | {status_msg}", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    else:
                        # Reset if face leaves frame
                        COUNTER = 0
                        LIVENESS_CONFIRMED = False
                        STATE = "WAITING_FOR_NEUTRAL"

                    # --- RECOGNITION LOGIC ---
                    for encodeFace, FaceLoc in zip(encodingCurrFrame, faceCurrFrame):
                        matches = face_recognition.compare_faces(encodingList, encodeFace)
                        face_dist = face_recognition.face_distance(encodingList, encodeFace)
                        matchIndex = np.argmin(face_dist)

                        if matches[matchIndex] and face_dist[matchIndex] < 0.50:
                            y1, x2, y2, x1 = FaceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                            # Box only Green if Verified
                            box_color = (0, 255, 0) if LIVENESS_CONFIRMED else (0, 0, 255)
                            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

                            enrollment_no = studentIdList[matchIndex]

                            if LIVENESS_CONFIRMED:
                                cv2.putText(img, f"ID: {enrollment_no}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                                try:
                                    result = mark_attendance_api(enrollment_no)
                                    if not result.get('duplicate', False):
                                        st.toast(f"✅ Attendance marked for {enrollment_no}!")
                                        # Reset State Machine for next person
                                        LIVENESS_CONFIRMED = False
                                        STATE = "WAITING_FOR_NEUTRAL"
                                    elif result.get('duplicate', False):
                                        pass
                                except:
                                    pass
                            else:
                                if STATE == "WAITING_FOR_NEUTRAL":
                                    msg = "NEUTRAL FACE FIRST"
                                else:
                                    msg = "PLEASE SMILE"
                                cv2.putText(img, msg, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                cap.release()

            except FileNotFoundError:
                st.error("Encoding file not found! Please run encode_generator.py first.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with col2:
        st.subheader("Manual Attendance")
        students = get_all_students()
        if students:
            enrollment_nos = [s['enrollment_no'] for s in students]
            selected = st.selectbox("Select Student", enrollment_nos)
            if st.button("Mark Present"):
                try:
                    result = mark_attendance_api(selected)
                    if result.get('duplicate', False):
                        st.warning("Already marked.")
                    else:
                        st.success("Marked successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- PAGE: VIEW ATTENDANCE ---
elif page == "📊 View Attendance":
    st.title("View Attendance")

    tab1, tab2 = st.tabs(["By Date", "By Student"])

    with tab1:
        selected_date = st.date_input("Select Date")
        if st.button("Fetch Records"):
            try:
                query = {"date": selected_date.isoformat()}
                response = requests.post(f"{API_URL}/attendance/query", json=query)
                records = response.json()
                if records:
                    st.dataframe(pd.DataFrame(records)[['enrollment_no', 'student_name', 'time', 'status']],
                                 use_container_width=True)
                else:
                    st.info("No records found.")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        students = get_all_students()
        if students:
            enrollment_nos = [s['enrollment_no'] for s in students]
            selected_enrollment = st.selectbox("Select Student for Logs", enrollment_nos)
            if st.button("Fetch Student Records"):
                try:
                    query = {"enrollment_no": selected_enrollment}
                    response = requests.post(f"{API_URL}/attendance/query", json=query)
                    records = response.json()
                    if records:
                        st.dataframe(pd.DataFrame(records)[['date', 'time', 'status']], use_container_width=True)
                    else:
                        st.info("No records found.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- PAGE: STATISTICS ---
elif page == "📈 Statistics":
    st.title("Statistics")
    students = get_all_students()

    if students:
        stats_data = []
        for student in students:
            stats = get_attendance_stats(student['enrollment_no'])
            if stats:
                stats_data.append({
                    'Name': student['name'],
                    'Total Days': stats.get('total_days', 0),
                    'Present Days': stats.get('present_days', 0),
                    'Percentage': f"{stats.get('attendance_percentage', 0):.1f}%"
                })

        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)