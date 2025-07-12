import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import math

class BodyMeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Measurement App")
        self.root.geometry("800x900")
        
        # Calibration variables
        self.user_height = None
        self.pixel_height = None
        self.scale_factor = 1.0  # Scale factor between display and original image
        
        # Setup UI
        self.create_widgets()
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Changed to True for better accuracy with still images
            model_complexity=2,      # Increased for better accuracy
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def create_widgets(self):
        # Title
        tk.Label(
            self.root, 
            text="Body Measurement App", 
            font=('Helvetica', 16, 'bold')
        ).pack(pady=10)
        
        # Image Frame
        self.image_frame = tk.Frame(
            self.root, 
            width=600, 
            height=400, 
            bg='lightgray'
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        # Image Label
        self.image_label = tk.Label(
            self.image_frame, 
            text="Upload an image to begin", 
            bg='lightgray'
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Buttons Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Upload Button
        tk.Button(
            button_frame, 
            text="Upload Body Photo", 
            command=self.upload_image
        ).pack(side=tk.LEFT, padx=5)
        
        # Calibrate with Height Button
        tk.Button(
            button_frame, 
            text="Enter Your Height", 
            command=self.enter_height
        ).pack(side=tk.LEFT, padx=5)
        
        # Measure Button
        tk.Button(
            button_frame, 
            text="Measure Body", 
            command=self.measure_body_from_current
        ).pack(side=tk.LEFT, padx=5)
        
        # Measurements Text Area
        self.measurements_text = tk.Text(
            self.root, 
            height=20, 
            width=70, 
            font=('Courier', 12)
        )
        self.measurements_text.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update status bar with message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def enter_height(self):
        """Allow user to enter their height for calibration"""
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("Calibration", "Please upload an image first")
            return
            
        # First detect if we can find landmarks
        self.update_status("Detecting body landmarks...")
        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            messagebox.showerror("Error", "No body landmarks detected in the image. Please upload a clearer full-body image.")
            return
            
        # Calculate pixel height from landmarks
        height, width, _ = self.original_image.shape
        landmarks = results.pose_landmarks.landmark
        
        # Estimate full body height in pixels
        # Using the distance from nose to ankle as an approximation
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Use the average of both ankles for more accuracy
        ankle_x = (left_ankle.x + right_ankle.x) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        pixel_height = math.sqrt(
            ((nose.x - ankle_x) * width) ** 2 + 
            ((nose.y - ankle_y) * height) ** 2
        )
        
        # Add a factor to account for the top of the head and feet
        pixel_height = pixel_height * 1.08
        
        # Ask user for their actual height
        user_height = simpledialog.askfloat(
            "Height Calibration", 
            "Enter your height in inches:",
            minvalue=24,  # Minimum reasonable height (2 feet)
            maxvalue=96   # Maximum reasonable height (8 feet)
        )
        
        if user_height and user_height > 0:
            # Store calibration
            self.user_height = user_height
            self.pixel_height = pixel_height
            
            self.update_status(
                f"Calibration complete: {pixel_height:.2f} pixels = {user_height} inches tall"
            )
            messagebox.showinfo(
                "Calibration", 
                f"Height calibration complete. {pixel_height:.2f} pixels = {user_height} inches"
            )
            
            # Show annotated image with height measurement
            annotated_image = self.original_image.copy()
            self.draw_height_line(annotated_image, landmarks, pixel_height)
            display_annotated = self.resize_image(annotated_image)
            self.show_image(display_annotated)
    
    def draw_height_line(self, image, landmarks, pixel_height):
        """Draw line showing height measurement"""
        height, width, _ = image.shape
        
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        
        # Get coordinates
        nose_point = (int(nose.x * width), int(nose.y * height))
        ankle_point = (int(left_ankle.x * width), int(left_ankle.y * height))
        
        # Draw vertical line
        x_line = nose_point[0] - 50  # Offset to the left
        top_y = int(nose_point[1] - 0.04 * height)  # Slightly above head
        bottom_y = int(ankle_point[1] + 0.04 * height)  # Slightly below feet
        
        cv2.line(image, (x_line, top_y), (x_line, bottom_y), (0, 255, 255), 2)
        
        # Draw top and bottom horizontal markers
        cv2.line(image, (x_line - 10, top_y), (x_line + 10, top_y), (0, 255, 255), 2)
        cv2.line(image, (x_line - 10, bottom_y), (x_line + 10, bottom_y), (0, 255, 255), 2)
        
        # Add text label
        cv2.putText(
            image, 
            f"Height: {self.user_height} inches", 
            (x_line - 40, (top_y + bottom_y) // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2
        )
    
    def upload_image(self):
        """Upload body measurement image"""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        self.update_status("Loading image...")
        
        # Read original image
        self.original_image = cv2.imread(file_path)
        
        if self.original_image is None:
            messagebox.showerror("Error", "Could not open the image file")
            return
        
        # Create a display copy
        self.display_image = self.resize_image(self.original_image.copy())
        
        # Calculate scale factor between original and display images
        self.scale_factor = self.display_image.shape[1] / self.original_image.shape[1]
        
        # Display image
        self.show_image(self.display_image)
        
        self.update_status("Image loaded. Please enter your height to calibrate.")
    
    def resize_image(self, image, max_width=600, max_height=400):
        """Resize image maintaining aspect ratio"""
        height, width = image.shape[:2]
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Store this scale factor for measurements
        self.scale_factor = scale
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def measure_body_from_current(self):
        """Process current image for measurements"""
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("Measurement", "Please upload an image first")
            return
        
        if not self.user_height or not self.pixel_height:
            messagebox.showwarning("Calibration", "Please enter your height first")
            return
            
        self.update_status("Processing image for body measurements...")
        
        # Process and measure body using original image for accuracy
        measurements, annotated_image = self.measure_body(self.original_image)
        
        # Resize for display while maintaining aspect ratio
        display_annotated = self.resize_image(annotated_image)
        
        # Display annotated image
        self.show_image(display_annotated)
        
        # Show measurements
        self.display_measurements(measurements)
        
        self.update_status("Measurements complete.")
    
    def measure_body(self, image):
        """Detect and measure body landmarks"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose.process(image_rgb)
        
        measurements = {}
        
        if results.pose_landmarks:
            # Create annotated image
            annotated_image = image.copy()
            
            # Draw landmarks with improved visibility
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Image dimensions
            height, width, _ = image.shape
            
            # Landmark references
            landmarks = results.pose_landmarks.landmark
            
            # Calculate measurements
            measurements = self.calculate_measurements(
                landmarks, width, height
            )
            
            # Draw measurement lines
            self.draw_measurement_lines(annotated_image, landmarks, measurements)
            
            return measurements, annotated_image
        else:
            messagebox.showwarning("Detection Error", "No body landmarks detected in the image.")
            return {}, image
    
    def draw_measurement_lines(self, image, landmarks, measurements):
        """Draw measurement lines on the image"""
        height, width, _ = image.shape
        
        # Helper function to get x,y coordinates
        def get_point(landmark):
            return (int(landmark.x * width), int(landmark.y * height))
        
        # Draw chest width line
        left_shoulder = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        cv2.line(image, left_shoulder, right_shoulder, (0, 255, 0), 2)
        
        # Draw waist width line
        left_hip = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        cv2.line(image, left_hip, right_hip, (0, 255, 0), 2)
        
        # Draw arm lines
        left_shoulder = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        left_elbow = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW])
        left_wrist = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST])
        right_shoulder = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        right_elbow = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW])
        right_wrist = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        
        cv2.line(image, left_shoulder, left_elbow, (255, 0, 0), 2)
        cv2.line(image, left_elbow, left_wrist, (255, 0, 0), 2)
        cv2.line(image, right_shoulder, right_elbow, (255, 0, 0), 2)
        cv2.line(image, right_elbow, right_wrist, (255, 0, 0), 2)
        
        # Draw leg lines
        left_hip = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
        left_knee = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE])
        left_ankle = get_point(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE])
        right_hip = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        right_knee = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE])
        right_ankle = get_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
        
        cv2.line(image, left_hip, left_knee, (0, 0, 255), 2)
        cv2.line(image, left_knee, left_ankle, (0, 0, 255), 2)
        cv2.line(image, right_hip, right_knee, (0, 0, 255), 2)
        cv2.line(image, right_knee, right_ankle, (0, 0, 255), 2)
        
        # Add measurement labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Shoulder width label
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                      (left_shoulder[1] + right_shoulder[1]) // 2 - 10)
        cv2.putText(image, f"{measurements.get('Shoulder Width', 0):.1f}\"", 
                  shoulder_mid, font, font_scale, (0, 255, 0), font_thickness)
        
        # Waist width label
        waist_mid = ((left_hip[0] + right_hip[0]) // 2, 
                   (left_hip[1] + right_hip[1]) // 2 - 10)
        cv2.putText(image, f"{measurements.get('Waist Width', 0):.1f}\"", 
                  waist_mid, font, font_scale, (0, 255, 0), font_thickness)
    
    def calculate_measurements(self, landmarks, width, height):
        """Calculate comprehensive body measurements with inch conversion"""
        # If no height calibration, return empty
        if not (self.pixel_height and self.user_height):
            return {}
        
        # Measurement calculations
        measurements = {}
        
        def pixel_to_inches(pixel_length):
            """Convert pixel length to inches based on height calibration"""
            return (pixel_length / self.pixel_height) * self.user_height
        
        def calculate_distance(landmark1, landmark2):
            """Calculate pixel distance between two landmarks"""
            x1 = landmark1.x * width
            y1 = landmark1.y * height
            x2 = landmark2.x * width
            y2 = landmark2.y * height
            
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Improved chest circumference calculation
        def calculate_chest_circumference(width_inches):
            """Estimate chest circumference based on shoulder width
            Using tailoring formulas that approximate the chest as an ellipse"""
            # This is a more accurate approximation for chest measurements
            # Based on common tailoring ratios
            return width_inches * 2.0  # Better approximation than using π
        
        # Improved waist circumference calculation  
        def calculate_waist_circumference(width_inches):
            """Estimate waist circumference based on waist width"""
            # Waist tends to be more circular than chest
            return width_inches * 2.5  # Adjusted factor based on typical body proportions
        
        # Upper body measurements
        shoulder_width = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            )
        )
        measurements['Shoulder Width'] = shoulder_width * 1.2  # Adjustment factor for accuracy
        
        # Use shoulder width to approximate chest width
        # In tailoring, chest is typically wider than shoulder measurement
        chest_width = shoulder_width
        measurements['Chest Width'] = chest_width
        measurements['Estimated Chest Circumference'] = calculate_chest_circumference(chest_width)
        
        # Improved waist measurement - using hip points but adjusted for typical waist position
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Move up slightly from hip to approximate waist
        waist_offset = 0.05  # Adjust based on typical proportions
        left_waist_x = left_hip.x
        left_waist_y = left_hip.y - waist_offset
        right_waist_x = right_hip.x
        right_waist_y = right_hip.y - waist_offset
        
        waist_width = pixel_to_inches(
            math.sqrt(
                ((left_waist_x - right_waist_x) * width)**2 + 
                ((left_waist_y - right_waist_y) * height)**2
            )
        )
        measurements['Waist Width'] = waist_width
        measurements['Estimated Waist Circumference'] = calculate_waist_circumference(waist_width)
        
        # Arm measurements - improved with adjustment factors
        arm_adjustment = 1.08  # Based on common measurement errors in 2D pose estimation
        
        left_upper_arm = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            )
        ) * arm_adjustment
        
        left_lower_arm = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            )
        ) * arm_adjustment
        
        measurements['Left Upper Arm Length'] = left_upper_arm
        measurements['Left Lower Arm Length'] = left_lower_arm
        measurements['Left Total Arm Length'] = left_upper_arm + left_lower_arm
        
        right_upper_arm = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            )
        ) * arm_adjustment
        
        right_lower_arm = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            )
        ) * arm_adjustment
        
        measurements['Right Upper Arm Length'] = right_upper_arm
        measurements['Right Lower Arm Length'] = right_lower_arm
        measurements['Right Total Arm Length'] = right_upper_arm + right_lower_arm
        
        # Rest of the function remains unchanged
        # Torso measurements
        measurements['Torso Length'] = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.NOSE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            )
        )
        
        # Leg measurements
        measurements['Left Upper Leg Length'] = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            )
        )
        
        measurements['Left Lower Leg Length'] = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            )
        )
        
        measurements['Left Total Leg Length'] = (
            measurements['Left Upper Leg Length'] + measurements['Left Lower Leg Length']
        )
        
        measurements['Right Upper Leg Length'] = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            )
        )
        
        measurements['Right Lower Leg Length'] = pixel_to_inches(
            calculate_distance(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            )
        )
        
        measurements['Right Total Leg Length'] = (
            measurements['Right Upper Leg Length'] + measurements['Right Lower Leg Length']
        )
        
        # Full body height - this should match the user's input height
        measurements['Estimated Height'] = self.user_height
        
        return measurements    
    def show_image(self, image):
        """Display image in the GUI"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update image label
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def display_measurements(self, measurements):
        """Display body measurements"""
        # Clear previous measurements
        self.measurements_text.delete(1.0, tk.END)
        
        # Display measurements
        if measurements:
            self.measurements_text.insert(
                tk.END, 
                "BODY MEASUREMENTS (in inches):\n"
                "================================\n\n"
            )
            
            # Group measurements by category
            categories = {
                "General": ["Estimated Height", "Shoulder Width"],
                "Torso": ["Chest Width", "Estimated Chest Circumference", 
                        "Waist Width", "Estimated Waist Circumference", "Torso Length"],
                "Arms": ["Left Upper Arm Length", "Left Lower Arm Length", "Left Total Arm Length",
                       "Right Upper Arm Length", "Right Lower Arm Length", "Right Total Arm Length"],
                "Legs": ["Left Upper Leg Length", "Left Lower Leg Length", "Left Total Leg Length",
                       "Right Upper Leg Length", "Right Lower Leg Length", "Right Total Leg Length"]
            }
            
            for category, measurement_keys in categories.items():
                self.measurements_text.insert(tk.END, f"{category}:\n")
                self.measurements_text.insert(tk.END, "-" * len(category) + "\n")
                
                for key in measurement_keys:
                    if key in measurements:
                        self.measurements_text.insert(
                            tk.END, 
                            f"  {key}: {measurements[key]:.2f} inches\n"
                        )
                
                self.measurements_text.insert(tk.END, "\n")
            
            # Calibration info
            self.measurements_text.insert(
                tk.END, 
                f"\nHeight Calibration: {self.user_height} inches\n"
            )
        else:
            self.measurements_text.insert(
                tk.END, 
                "No body landmarks detected or height calibration missing."
            )

def main():
    root = tk.Tk()
    app = BodyMeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()