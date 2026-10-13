import os
import sys
import time

import cv2
import numpy as np
import serial

# Loads the Teachable Machine model without Keras version problems
# (see model_loader.py - it rebuilds the model from keras_model.h5)
from model_loader import load_teachable_machine

# Find the model and labels next to this script, whatever folder you run it from
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Serial settings
# ----------------------------------------------------------------------

# Which message to send for each label number in labels.txt.
# 0 lighter   -> "fan_on"   (fan spins)
# 1 febreeze  -> "fan_off"  (fan stops)
# 2 remote    -> "led_on"   (LED turns on)
# 3 backround -> nothing is sent
MESSAGE_FOR_LABEL = {
    0: "fan_on",
    1: "fan_off",
    2: "led_on",
}

# Must match the Serial.begin() number in your Arduino sketch
BAUD_RATE = 9600

# Only act on a prediction the model is fairly sure about, so a blurry frame
# does not flip your device back and forth.
CONFIDENCE_THRESHOLD = 0.80

# The prediction must be the same for this many frames in a row before acting.
FRAMES_IN_A_ROW = 3

# The port your board is plugged into. Find it in the Arduino IDE under
# Tools > Port. On Windows it looks like "COM3".
# Leave as "PUT YOUR PORT HERE" to test with no hardware (messages just print).
SERIAL_PORT = "PUT YOUR PORT HERE"

# Open the serial port. If it cannot be opened the program still runs, so you
# can test the camera and the model without hardware plugged in.
board = None
try:
    board = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    # Opening the port resets most Arduino boards. Give it time to restart,
    # otherwise the first messages are sent into the void.
    time.sleep(2)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException as error:
    print(f"Could not open {SERIAL_PORT}: {error}")
    print("Running without hardware: messages will only be printed.")
    print("(To use the Arduino: set SERIAL_PORT, and close the Arduino Serial Monitor.)")


def send(message):
    """Send one message to the board, followed by a newline."""
    if board is not None:
        board.write((message + "\n").encode())
    print(f"Sent: {message}")


# ----------------------------------------------------------------------
# Model, labels, and camera
# ----------------------------------------------------------------------

model = load_teachable_machine(os.path.join(HERE, "keras_model.h5"))

# Load the labels. Each line looks like "0 lighter", so keep only the name.
with open(os.path.join(HERE, "labels.txt"), "r") as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f if line.strip()]


def open_camera(start=0):
    """Find a camera that gives real (not black) pictures. Tries several
    camera numbers and Windows camera drivers. Returns (camera, position)."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    options = [(i, b) for i in range(5) for b in backends]
    for step in range(len(options)):
        pos = (start + step) % len(options)
        index, backend = options[pos]
        cam = cv2.VideoCapture(index, backend)
        if cam.isOpened():
            for _ in range(40):  # cameras need a moment to warm up
                ok, img = cam.read()
                if ok and img is not None and img.mean() > 8:
                    print(f"Using camera {index}")
                    return cam, pos
                time.sleep(0.05)
        cam.release()
    sys.exit("No working camera found. Close other apps using the webcam, remove any "
             "lens cover, and check Windows Settings > Privacy & security > Camera.")


camera, camera_pos = open_camera(0)

print("Click the webcam window, then press Esc or q to quit.")
print("Press c to switch camera if the picture is black or wrong. Press l to turn the LED off.")

# Remember what we sent last, so we only send when the answer actually changes.
last_message = None
candidate = None   # the message we are currently counting frames for
count = 0

while True:
    # Grab the webcamera's image.
    ret, frame = camera.read()
    if not ret:
        time.sleep(0.05)
        continue

    # Crop the center square, like Teachable Machine does, so the image is not squashed
    h, w = frame.shape[:2]
    size = min(h, w)
    top, left = (h - size) // 2, (w - size) // 2
    frame = frame[top:top + size, left:left + size]

    # Resize the square image to (224-height,224-width) pixels
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # OpenCV uses BGR color order, but the model was trained on RGB images
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array to the range -1 to 1
    image = (image / 127.5) - 1

    # Predict. verbose=0 stops Keras from printing a progress bar every frame.
    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction[0])
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Show the image in a window, with what the model sees
    shown = cv2.resize(frame, (448, 448))
    cv2.putText(shown, f"{class_name} {confidence_score * 100:.0f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Webcam Image", shown)

    # Decide what should be sent for this frame. A label we have no message for,
    # or a guess we are not confident about, means send nothing.
    message = None
    if confidence_score >= CONFIDENCE_THRESHOLD:
        message = MESSAGE_FOR_LABEL.get(int(index))

    # Count how many frames in a row gave the same message
    if message is not None and message == candidate:
        count += 1
    else:
        candidate, count = message, 1

    # Send only when the message is steady and changed from the one before it
    if candidate is not None and count >= FRAMES_IN_A_ROW and candidate != last_message:
        send(candidate)
        last_message = candidate

    # Listen to the keyboard for presses. Esc key (27) or q to exit.
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break
    if key == ord("c"):
        camera.release()
        camera, camera_pos = open_camera(camera_pos + 1)
    if key == ord("l"):
        send("led_off")
        last_message = "led_off"

send("fan_off")
send("led_off")
camera.release()
cv2.destroyAllWindows()
if board is not None:
    board.close()
