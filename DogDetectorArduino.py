import numpy as np
import cv2
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import glob
import pickle
import time
import random
import serial


class DogDetectorArduino:
    """
    Dog detection using webcam with Arduino integration.
    Uses logistic regression to identify dogs in webcam frames.
    Sends command to Arduino only when a dog is detected for 2+ seconds.
    """

    def __init__(self):
        self.model_path = 'dog_classifier.pkl'
        self.img_size = (64, 64)  # Smaller size for faster processing
        self.confidence_threshold = 0.96  # Probability threshold for dog detection
        self.model = None

        # Arduino settings
        self.arduino_port = ''  # Change to your Arduino port
        self.baudrate = 9600
        self.arduino = None

        # Detection persistence tracking
        self.dog_detected_time = None  # Time when dog was first detected
        self.detection_duration = 2.0  # Required duration in seconds before sending command
        self.command_sent = False  # Flag to prevent multiple commands

    def connect_arduino(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(port=self.arduino_port, baudrate=self.baudrate, timeout=0.1)
            print(f"Connected to Arduino on {self.arduino_port}")
            # Allow time for connection to establish
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")
            print("Continuing without Arduino connection. Detection will still work but no signal will be sent.")
            return False

    def send_command(self, command):
        """Send command to Arduino"""
        if self.arduino:
            try:
                # Use the correct format as provided
                self.arduino.write((self.capture_image() + "\n").encode())
                print(f"Sent command '{self.capture_image()}' to Arduino")
                return True
            except Exception as e:
                print(f"Error sending to Arduino: {e}")
        return False

    def capture_image(self):
        """Return 'O' command for Arduino"""
        # This function always returns 'O' as specified
        return 'O'

    def extract_features(self, image):
        """Extract simple features from an image"""
        if image is None:
            return None

        # Resize to standard size
        img_resized = cv2.resize(image, self.img_size)

        # Convert to grayscale
        if len(img_resized.shape) == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

        # Flatten the image to a 1D array
        features = img_gray.flatten()

        # Normalize pixel values
        features = features / 255.0

        return features

    def train_model_with_folder_data(self):
        """Train a logistic regression model using images in the images folder"""
        # Path to dog images
        dog_folder = "images"

        # Look for files with the pattern n02099601_*.jpg
        dog_pattern = os.path.join(dog_folder, "n02099601_*.jpg")
        dog_images = glob.glob(dog_pattern)

        if not dog_images:
            print(f"Error: No dog images found with pattern {dog_pattern}")
            return None

        print(f"Found {len(dog_images)} dog images for training")

        # Prepare data
        X = []  # Features
        y = []  # Labels (1 for dog, 0 for non-dog)

        # Process dog images
        dog_count = 0
        for img_path in dog_images:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.extract_features(img)
                    if features is not None:
                        X.append(features)
                        y.append(1)  # 1 = dog
                        dog_count += 1

                        # Print progress every 10 images
                        if dog_count % 10 == 0:
                            print(f"Processed {dog_count} dog images...")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Generate synthetic non-dog images
        print("Generating synthetic non-dog images...")
        non_dog_count = 0

        # Select a subset of dog images to transform
        sample_dog_images = random.sample(dog_images, min(20, len(dog_images)))

        for img_path in sample_dog_images:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # 1. Heavily blurred version
                    blurred = cv2.GaussianBlur(img, (99, 99), 0)
                    blur_features = self.extract_features(blurred)
                    if blur_features is not None:
                        X.append(blur_features)
                        y.append(0)  # 0 = not dog
                        non_dog_count += 1

                    # 2. Inverted colors
                    inverted = cv2.bitwise_not(img)
                    invert_features = self.extract_features(inverted)
                    if invert_features is not None:
                        X.append(invert_features)
                        y.append(0)  # 0 = not dog
                        non_dog_count += 1

                    # 3. Random noise
                    noise = np.random.randint(0, 255, img.shape, dtype=np.uint8)
                    noise_features = self.extract_features(noise)
                    if noise_features is not None:
                        X.append(noise_features)
                        y.append(0)  # 0 = not dog
                        non_dog_count += 1

                    # 4. Extreme contrast
                    _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                              127, 255, cv2.THRESH_BINARY)
                    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                    thresh_features = self.extract_features(thresh_color)
                    if thresh_features is not None:
                        X.append(thresh_features)
                        y.append(0)  # 0 = not dog
                        non_dog_count += 1

                    # 5. Edge detection (definitely not a dog)
                    edges = cv2.Canny(img, 100, 200)
                    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    edges_features = self.extract_features(edges_color)
                    if edges_features is not None:
                        X.append(edges_features)
                        y.append(0)  # 0 = not dog
                        non_dog_count += 1

                    # Print progress every 5 images
                    if non_dog_count % 25 == 0:
                        print(f"Generated {non_dog_count} non-dog images...")
            except Exception as e:
                print(f"Error generating non-dog images from {img_path}: {e}")

        # Also create some purely random images
        print("Generating random noise images...")
        for i in range(20):
            # Random noise image
            noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            noise_features = self.extract_features(noise)
            if noise_features is not None:
                X.append(noise_features)
                y.append(0)  # 0 = not dog
                non_dog_count += 1

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"Training with {dog_count} dog images and {non_dog_count} non-dog images")

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train logistic regression model
        print("Training logistic regression model...")
        self.model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000)
        self.model.fit(X_train, y_train)

        # Evaluate model
        print("Evaluating model...")
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Not Dog', 'Dog']))

        # Calculate dog detection precision and recall specifically
        dog_indices = y_val == 1
        if sum(dog_indices) > 0:
            dog_correct = sum((y_pred == 1) & (y_val == 1))
            dog_precision = dog_correct / max(1, sum(y_pred == 1))
            dog_recall = dog_correct / sum(dog_indices)
            print(f"Dog Detection: Precision={dog_precision * 100:.2f}%, Recall={dog_recall * 100:.2f}%")

        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"Model saved to {self.model_path}")
        return self.model

    def load_model(self):
        """Load a trained model from disk"""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            return self.model
        else:
            print(f"Model file not found at {self.model_path}")
            return None

    def predict(self, image):
        """
        Predict whether an image contains a dog
        Returns: (is_dog, confidence)
        """
        if self.model is None:
            print("Error: Model not loaded")
            return False, 0.0

        if image is None:
            return False, 0.0

        # Extract features
        features = self.extract_features(image)

        if features is None:
            return False, 0.0

        # Reshape for single prediction
        features = features.reshape(1, -1)

        # Get prediction probability
        probabilities = self.model.predict_proba(features)[0]

        # Get dog probability (class 1)
        if len(probabilities) > 1:  # Make sure we have both classes
            dog_probability = probabilities[1]
            is_dog = dog_probability >= self.confidence_threshold
            return is_dog, dog_probability
        else:
            return False, 0.0

    def run_dog_detector(self):
        """Run dog detection on webcam feed with Arduino integration"""
        print("\n===================================")
        print("Dog Detector with Arduino Integration")
        print("===================================")

        # Make sure model is loaded
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return

        # Connect to Arduino
        arduino_connected = self.connect_arduino()
        if not arduino_connected:
            print("Warning: Arduino not connected. Detection will still work without servo control.")

        # Initialize camera with multiple attempts
        print("\nInitializing camera...")

        # Try different camera configurations
        cap = None
        camera_configs = [
            (0, None),  # Default
            (0, cv2.CAP_AVFOUNDATION),  # macOS specific
            (1, None),  # External camera
            (1, cv2.CAP_AVFOUNDATION),  # External with AVFoundation
        ]

        for idx, backend in camera_configs:
            print(f"Trying camera index {idx}{' with AVFoundation' if backend else ''}...")
            if backend:
                cap = cv2.VideoCapture(idx, backend)
            else:
                cap = cv2.VideoCapture(idx)

            # Set properties
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Test if we can actually read frames
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print(f"Successfully connected to camera {idx}")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                cap = None

        if cap is None or not cap.isOpened():
            print("Error: Could not open any camera.")
            print("\nTroubleshooting tips:")
            print("1. Try running this script from Terminal instead of PyCharm")
            print("2. Check camera permissions in System Settings")
            print("3. Try connecting an external webcam or use your iPhone as a camera")
            return

        # Wait for camera to initialize
        time.sleep(1)

        print("\nCamera initialized successfully!")
        print("Instructions:")
        print("- Hold a dog image in front of camera")
        print("- Press '+' or '-' to adjust confidence threshold")
        print("- Press 'q' to quit")
        print(f"- Current confidence threshold: {self.confidence_threshold * 100:.0f}%")
        print(f"- Detection duration requirement: {self.detection_duration} seconds")

        # Stats variables
        frame_count = 0
        start_time = time.time()

        # Tracking variables for displaying detection stats
        total_frames = 0
        dog_detections = 0

        try:
            while True:
                # Capture frame
                ret, frame = cap.read()

                if not ret or frame is None or frame.size == 0:
                    print("Warning: Failed to capture frame, retrying...")
                    time.sleep(0.5)
                    continue

                # Count frames for FPS calculation
                frame_count += 1
                total_frames += 1
                current_time = time.time()
                if frame_count % 30 == 0:  # Calculate FPS every 30 frames
                    fps = frame_count / (current_time - start_time)
                    frame_count = 0
                    start_time = current_time
                    print(f"FPS: {fps:.1f}")

                # Display the original frame
                cv2.imshow('Camera Feed', frame)

                # Process image for prediction
                is_dog, confidence = self.predict(frame)

                # Count dog detections for stats
                if is_dog:
                    dog_detections += 1

                # Create a copy for prediction display
                display_frame = frame.copy()

                # Handle dog detection timing and Arduino command logic
                if is_dog:
                    # Start timing if not already started
                    if self.dog_detected_time is None:
                        self.dog_detected_time = current_time
                        print(f"Dog detected! Starting verification timer...")

                    # Check if we've been detecting the dog long enough
                    detection_time = current_time - self.dog_detected_time

                    if detection_time >= self.detection_duration and not self.command_sent:
                        # Send command to Arduino
                        print(f"Dog detected continuously for {detection_time:.1f} seconds!")
                        if self.arduino:
                            self.send_command('O')  # This calls our updated send_command method
                            self.command_sent = True

                    label = "Dog Detected"
                    color = (0, 255, 0)  # Green

                    # Add detection time if in progress
                    if not self.command_sent:
                        time_left = max(0, self.detection_duration - detection_time)
                        verification_text = f"Verifying: {time_left:.1f}s remaining"
                        cv2.putText(display_frame, verification_text, (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        cv2.putText(display_frame, "Command Sent to Arduino!", (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Reset detection timer if we lose detection
                    self.dog_detected_time = None
                    self.command_sent = False

                    label = "No Dog Detected"
                    color = (0, 0, 255)  # Red

                # Add prediction label
                cv2.putText(display_frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Add confidence display
                confidence_text = f"Confidence: {confidence * 100:.2f}%"
                cv2.putText(display_frame, confidence_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Add threshold info
                threshold_text = f"Threshold: {self.confidence_threshold * 100:.0f}%"
                cv2.putText(display_frame, threshold_text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Add detection statistics
                if total_frames > 0:
                    detection_rate = (dog_detections / total_frames) * 100
                    stats_text = f"Detection Rate: {detection_rate:.1f}% ({dog_detections}/{total_frames})"
                    cv2.putText(display_frame, stats_text, (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show prediction window
                cv2.imshow('Prediction', display_frame)

                # Break loop on 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                # Adjust threshold with + and - keys
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold increased to {self.confidence_threshold * 100:.0f}%")
                elif key == ord('-') or key == ord('_'):
                    self.confidence_threshold = max(0.50, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold decreased to {self.confidence_threshold * 100:.0f}%")
                # Reset detection stats with 'r' key
                elif key == ord('r'):
                    total_frames = 0
                    dog_detections = 0
                    print("Detection statistics reset")
                # Trigger Arduino command manually with 'o' key
                elif key == ord('o'):
                    if self.arduino:
                        self.send_command('O')
                        print("Manually triggered Arduino command 'O'")
                # No need for 'c' key since we only have one command
                # according to your requirement to send 'O' when a dog is detected

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError during processing: {e}")
        finally:
            # Cleanup
            print("\nShutting down...")
            cap.release()
            cv2.destroyAllWindows()
            # Close Arduino connection
            if self.arduino:
                self.arduino.close()
                print("Arduino connection closed")
            print("Done!")


if __name__ == "__main__":
    print("\n===================================")
    print("Dog Detector with Arduino Integration")
    print("===================================")

    detector = DogDetectorArduino()

    # Check if model exists
    if os.path.exists(detector.model_path):
        choice = input("Existing model found. Do you want to (T)rain a new model or (L)oad existing? [L/T]: ")
        if choice.upper() == 'T':
            train_new = True
        else:
            train_new = False
            detector.load_model()
    else:
        print("No existing model found.")
        train_new = True

    # Train new model if needed
    if train_new:
        detector.train_model_with_folder_data()

    # Run dog detection with Arduino integration
    detector.run_dog_detector()