# Dog Detector Arduino Project

This project uses machine learning to detect specific dog images from your webcam and control an Arduino servo when a dog is detected. The system is designed to be reliable with a 2-second verification to avoid false positives.

## Features

- Detects dogs from printed images with high accuracy
- Requires 2-second continuous detection before triggering the Arduino
- Uses logistic regression for efficient, lightweight machine learning
- Customizable detection threshold
- Real-time visual feedback with confidence scores

## Requirements

- Python 3.6+
- OpenCV
- scikit-learn
- numpy
- pyserial
- Arduino board (connected to servo)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/DogDetectorArduino.git
   cd DogDetectorArduino
   ```

2. Install required packages:
   ```
   pip install numpy opencv-python scikit-learn pyserial
   ```

3. Connect your Arduino to your computer via USB

## Setup Instructions

### 1. Prepare Your Images

Before running the code, you need to set up folders with training images:

1. Create an `images` folder in the project directory
   ```
   mkdir -p images
   ```

2. Add dog images to the `images` folder
   - Images should be named using the pattern: `n02099601_*.jpg`
   - Example: `n02099601_1.jpg`, `n02099601_2.jpg`, etc.
   - You need at least 5-10 different dog images for good results
   - Use clear, well-lit images of the dogs you want to detect

3. Create a `non_dog` folder (for better accuracy)
   ```
   mkdir -p non_dog
   ```

4. Add non-dog images to the `non_dog` folder
   - Include images of anything that's NOT a dog
   - Especially include black boxes or objects that cause false positives
   - Any filename and format is accepted (jpg, png, etc.)

### 2. Update Arduino Settings

Before running, update the Arduino port in the code:

1. Open `DogDetectorArduino2.py` in a text editor
2. Find the line: `self.arduino_port = '/dev/cu.usbmodemC04E301371642'`
3. Change this to match your Arduino's port:
   - Windows: Usually `'COM3'` (or other COM port number)
   - macOS: Usually `/dev/cu.usbmodem*` or `/dev/cu.usbserial*`
   - Linux: Usually `/dev/ttyACM0` or `/dev/ttyUSB0`

### 3. Arduino Setup

1. Connect a servo motor to your Arduino:
   - Signal wire to a PWM pin (typically pin 9)
   - Power and ground to appropriate pins

2. Upload the appropriate servo control code to your Arduino that listens for the 'O' command.

## Using the Dog Detector

Run the program:
```
python DogDetectorArduino2.py
```

The first time you run it, the system will:
1. Train a model using your dog and non-dog images
2. Connect to your Arduino
3. Start the webcam for detection

If you already have a trained model, you'll be asked whether to load it or train a new one.

### Controls

- Hold a dog image in front of the camera
- After 2 seconds of continuous detection, the Arduino will receive the command
- Press '+' or '-' to adjust the confidence threshold
- Press 'o' to manually trigger the Arduino command
- Press 'r' to reset detection statistics
- Press 'q' to quit

## Troubleshooting

### Camera Issues
- The program tries multiple camera configurations automatically
- If it can't find your camera, try running from the terminal instead of an IDE
- Check camera permissions in your system settings

### Detection Issues
- If you get false positives, increase the confidence threshold with the '+' key
- Add more non-dog images to the training set, especially of objects causing false positives
- Make sure you have good lighting when showing images to the camera

### Arduino Issues
- Check that the port name is correct in the code
- Make sure your Arduino has the correct code uploaded
- Verify the connections to your servo

## Customization

- Detection Threshold: Adjust the `self.confidence_threshold` value in the code (default 0.96)
- Verification Time: Change the `self.detection_duration` value (default 2.0 seconds)
- Image Size: Modify the `self.img_size` tuple for different processing speed/accuracy balance

## How It Works

1. Training: The system extracts features from your dog and non-dog images to train a logistic regression model.

2. Detection: When running, it:
   - Captures frames from your webcam
   - Extracts the same features from each frame
   - Uses the model to predict whether it's a dog
   - Shows confidence score and decision

3. Verification: To avoid false triggers:
   - Requires a dog to be detected continuously for 2 seconds
   - Shows a countdown during verification
   - Only after verification completes is the command sent

4. Arduino Control: When a dog is verified:
   - Sends 'O' command to the Arduino
   - Arduino receives this and moves the servo
