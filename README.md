# FACE-RECOGNITION-WITH-SERIAL COMMUNICATION

## Project Title:
Interactive Robot with Python-based Face Recognition System

<img src="robo.png.png" alt="Image" style="width:200px;height:150px;">

## Objective:
To design and implement an interactive robot equipped with a face recognition capability using a Python-based model. The robot will recognize and greet identified individuals based on pre-trained facial data.

## Key Features:

### Webcam Integration:
The robot utilizes a webcam positioned within its visual interface (eyes) to capture real-time video for face recognition.

### Python-based Face Recognition:
- **Pre-trained Dataset:** A dataset of facial features is used as the recognition foundation.
- **Real-time Processing:** As the robot's webcam collects real-time video:
  - **Face Detection:** Using the Haar cascade classifier to identify faces in the real-time video.
  - **Feature Extraction:** Extract key facial features for recognition.
  - **Face Matching:** Compare extracted features against the pre-trained dataset for identification.

### Arduino Integration:
Serial communication bridges the Python model and Arduino, ensuring synchronized operations between face recognition and robotic responses.

### Display & Interaction Mechanism:
Upon successful recognition, the robot displays the person's name and subsequently greets the individual with a handshake.

## Technologies & Tools Used:
- **Python:** Core programming language for face recognition.
- **OpenCV:** Provides tools for facial feature extraction and utilizes the Haar cascade classifier for face detection.
- **Arduino:** Enables hardware interfacing for robotic responses.

## Outcome:
A fully functional robot capable of visually recognizing individuals from a pre-trained dataset, displaying their name, and offering a physical greeting through a handshake.

## Future Scope:
Enhancements could include integrating more complex deep learning models for improved recognition accuracy, expanding the pre-trained dataset, and implementing additional interactive features like voice recognition or personalized greetings based on individual profiles.


# REFERENCES
- https://github.com/ageitgey/face_recognition_models
