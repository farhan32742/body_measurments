# Updated app.py with fixes for NormalizedLandmarkList issue
import os
from flask import Flask, render_template, request, flash, redirect, url_for, session
import cv2
import numpy as np
import mediapipe as mp
from werkzeug.utils import secure_filename
import math
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = '54321'
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# MediaPipe setup with improved parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_distance(landmark1, landmark2, width, height):
    x1, y1 = landmark1.x * width, landmark1.y * height
    x2, y2 = landmark2.x * width, landmark2.y * height
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def pixel_to_inches(pixel_length, pixel_height, user_height):
    return (pixel_length / pixel_height) * user_height

def process_image_for_display(image, max_width=600, max_height=400):
    """Resize image maintaining aspect ratio for display"""
    height, width = image.shape[:2]
    scale = min(max_width/width, max_height/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')

def get_landmark(landmarks, landmark_type):
    """Helper function to safely get landmarks"""
    return landmarks[landmark_type.value]

def calculate_measurements(landmarks, width, height, user_height, pixel_height):
    """Calculate comprehensive body measurements with inch conversion"""
    measurements = {}
    
    def pixel_to_inches(pixel_length):
        return (pixel_length / pixel_height) * user_height
    
    def calculate_distance(l1, l2):
        return math.sqrt((l1.x*width - l2.x*width)**2 + (l1.y*height - l2.y*height)**2)
    
    # Improved circumference calculations
    def calculate_chest_circumference(width_inches):
        return width_inches * 2.0
    
    def calculate_waist_circumference(width_inches):
        return width_inches * 2.5
    
    try:
        # Upper body measurements
        left_shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        
        shoulder_width = pixel_to_inches(calculate_distance(left_shoulder, right_shoulder))
        measurements['Shoulder Width'] = shoulder_width * 1.2
        
        chest_width = shoulder_width
        measurements['Chest Width'] = chest_width
        measurements['Estimated Chest Circumference'] = calculate_chest_circumference(chest_width)
        
        # Improved waist measurement
        left_hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        
        waist_width = pixel_to_inches(
            math.sqrt(
                ((left_hip.x - right_hip.x) * width)**2 + 
                ((left_hip.y - right_hip.y) * height)**2
            )
        )
        measurements['Waist Width'] = waist_width
        measurements['Estimated Waist Circumference'] = calculate_waist_circumference(waist_width)
        
        # Arm measurements
        arm_adjustment = 1.08
        
        left_elbow = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        right_elbow = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
        
        left_upper_arm = pixel_to_inches(
            calculate_distance(left_shoulder, left_elbow)
        ) * arm_adjustment
        
        left_lower_arm = pixel_to_inches(
            calculate_distance(left_elbow, left_wrist)
        ) * arm_adjustment
        
        measurements['Left Upper Arm Length'] = left_upper_arm
        measurements['Left Lower Arm Length'] = left_lower_arm
        measurements['Left Total Arm Length'] = left_upper_arm + left_lower_arm
        
        right_upper_arm = pixel_to_inches(
            calculate_distance(right_shoulder, right_elbow)
        ) * arm_adjustment
        
        right_lower_arm = pixel_to_inches(
            calculate_distance(right_elbow, right_wrist)
        ) * arm_adjustment
        
        measurements['Right Upper Arm Length'] = right_upper_arm
        measurements['Right Lower Arm Length'] = right_lower_arm
        measurements['Right Total Arm Length'] = right_upper_arm + right_lower_arm
        
        # Torso measurements
        nose = get_landmark(landmarks, mp_pose.PoseLandmark.NOSE)
        measurements['Torso Length'] = pixel_to_inches(
            calculate_distance(nose, left_hip)
        )
        
        # Leg measurements
        left_knee = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_knee = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
        right_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        measurements['Left Upper Leg Length'] = pixel_to_inches(
            calculate_distance(left_hip, left_knee)
        )
        
        measurements['Left Lower Leg Length'] = pixel_to_inches(
            calculate_distance(left_knee, left_ankle)
        )
        
        measurements['Left Total Leg Length'] = (
            measurements['Left Upper Leg Length'] + measurements['Left Lower Leg Length']
        )
        
        measurements['Right Upper Leg Length'] = pixel_to_inches(
            calculate_distance(right_hip, right_knee)
        )
        
        measurements['Right Lower Leg Length'] = pixel_to_inches(
            calculate_distance(right_knee, right_ankle)
        )
        
        measurements['Right Total Leg Length'] = (
            measurements['Right Upper Leg Length'] + measurements['Right Lower Leg Length']
        )
        
        # Full body height
        measurements['Estimated Height'] = user_height
        
    except Exception as e:
        print(f"Error calculating measurements: {e}")
        return {}
    
    return measurements

def draw_measurement_lines(image, landmarks, measurements):
    """Draw measurement lines on the image with labels"""
    height, width, _ = image.shape
    
    def get_point(landmark):
        return (int(landmark.x * width), int(landmark.y * height))
    
    try:
        # Draw all pose landmarks and connections
        mp_drawing.draw_landmarks(
            image, 
            landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Define colors for different body parts
        torso_color = (0, 255, 0)  # Green
        arm_color = (255, 0, 0)    # Blue
        leg_color = (0, 0, 255)    # Red
        
        # Get landmarks
        left_shoulder = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.RIGHT_HIP)
        left_elbow = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.LEFT_WRIST)
        right_elbow = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.RIGHT_WRIST)
        left_knee = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_knee = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.RIGHT_KNEE)
        right_ankle = get_landmark(landmarks.landmark, mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # Draw torso lines
        cv2.line(image, get_point(left_shoulder), get_point(right_shoulder), torso_color, 2)
        cv2.line(image, get_point(left_shoulder), get_point(left_hip), torso_color, 2)
        cv2.line(image, get_point(right_shoulder), get_point(right_hip), torso_color, 2)
        cv2.line(image, get_point(left_hip), get_point(right_hip), torso_color, 2)
        
        # Draw arm lines
        cv2.line(image, get_point(left_shoulder), get_point(left_elbow), arm_color, 2)
        cv2.line(image, get_point(left_elbow), get_point(left_wrist), arm_color, 2)
        cv2.line(image, get_point(right_shoulder), get_point(right_elbow), arm_color, 2)
        cv2.line(image, get_point(right_elbow), get_point(right_wrist), arm_color, 2)
        
        # Draw leg lines
        cv2.line(image, get_point(left_hip), get_point(left_knee), leg_color, 2)
        cv2.line(image, get_point(left_knee), get_point(left_ankle), leg_color, 2)
        cv2.line(image, get_point(right_hip), get_point(right_knee), leg_color, 2)
        cv2.line(image, get_point(right_knee), get_point(right_ankle), leg_color, 2)
        
        # Add measurement labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Shoulder width label
        shoulder_mid = ((get_point(left_shoulder)[0] + get_point(right_shoulder)[0]) // 2, 
                       (get_point(left_shoulder)[1] + get_point(right_shoulder)[1]) // 2 - 10)
        cv2.putText(image, f"{measurements.get('Shoulder Width', 0):.1f}\"", 
                   shoulder_mid, font, font_scale, torso_color, font_thickness)
        
        # Waist width label
        waist_mid = ((get_point(left_hip)[0] + get_point(right_hip)[0]) // 2, 
                    (get_point(left_hip)[1] + get_point(right_hip)[1]) // 2 - 10)
        cv2.putText(image, f"{measurements.get('Waist Width', 0):.1f}\"", 
                   waist_mid, font, font_scale, torso_color, font_thickness)
        
    except Exception as e:
        print(f"Error drawing measurement lines: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    image_data = None
    measurements = {}
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file uploaded')
            return redirect(request.url)
            
        file = request.files['image']
        # Support feet + inches inputs while preserving backward compatibility with legacy 'height' (inches) field
        feet_str = request.form.get('feet', '').strip()
        inches_str = request.form.get('inches', '').strip()
        try:
            if feet_str != '' or inches_str != '':
                feet = float(feet_str) if feet_str != '' else 0.0
                inches = float(inches_str) if inches_str != '' else 0.0
                # Convert any extra inches into feet
                extra_feet = math.floor(inches / 12.0)
                if extra_feet:
                    feet += extra_feet
                    inches = inches - extra_feet * 12.0
                total_inches = feet * 12.0 + inches
            else:
                # Legacy single-field input (inches)
                total_inches = float(request.form.get('height', 0))
                feet = math.floor(total_inches / 12.0)
                inches = total_inches - feet * 12.0
        except ValueError:
            flash('Please enter numeric values for feet and inches')
            return redirect(request.url)
        
        if total_inches <= 0:
            flash('Please enter a valid height (feet and inches)')
            return redirect(request.url)
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a JPG, JPEG, or PNG file.')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Store original image path and provided height in session
            session['original_image'] = filepath
            session['user_height'] = total_inches
            # store split feet and inches (nice for pre-filling the form and display)
            session['user_feet'] = int(feet)
            session['user_inches'] = round(inches, 2)
            
            # Process image
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                height_px, width_px, _ = image.shape
                
                # Calculate pixel height from landmarks (nose to ankle)
                nose = get_landmark(results.pose_landmarks.landmark, mp_pose.PoseLandmark.NOSE)
                left_ankle = get_landmark(results.pose_landmarks.landmark, mp_pose.PoseLandmark.LEFT_ANKLE)
                right_ankle = get_landmark(results.pose_landmarks.landmark, mp_pose.PoseLandmark.RIGHT_ANKLE)
                
                # Use average of both ankles
                ankle_x = (left_ankle.x + right_ankle.x) / 2
                ankle_y = (left_ankle.y + right_ankle.y) / 2
                
                pixel_height = math.sqrt(
                    ((nose.x - ankle_x) * width_px) ** 2 + 
                    ((nose.y - ankle_y) * height_px) ** 2
                )
                
                # Account for head and feet
                pixel_height = pixel_height * 1.08
                session['pixel_height'] = pixel_height
                
                # Calculate all measurements
                measurements = calculate_measurements(
                    results.pose_landmarks.landmark, 
                    width_px, 
                    height_px, 
                    total_inches, 
                    pixel_height
                )
                
                # Create annotated image
                annotated_image = image.copy()
                draw_measurement_lines(annotated_image, results.pose_landmarks, measurements)
                
                # Process for display
                image_data = process_image_for_display(annotated_image)
                
                flash('Measurements calculated successfully!')
            else:
                flash('No body landmarks detected. Please upload a clearer full-body image.')
    
    return render_template('index.html', 
                         image_data=image_data, 
                         measurements=measurements,
                         user_height=session.get('user_height'),
                         user_height_feet=session.get('user_feet'),
                         user_height_inches=session.get('user_inches'))

if __name__ == '__main__':
    app.run(debug=True)