#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import ipinfo
from datetime import datetime
import json
import sqlite3
import numpy as np


# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define data transforms for the neural network input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # Change the final layer to match the number of gesture classes
model.load_state_dict(torch.load('C:/Users/91630/Downloads/2resnet_model.pth', map_location=torch.device('cpu')))
model.eval()


# Database setup
def initialize_db(db_path='gesture_images.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY, gesture_label INT, image BLOB, timestamp TEXT)''')
    conn.commit()
    conn.close()

def insert_image_to_db(gesture_label, image, timestamp):
    conn = sqlite3.connect('gesture_images.db')
    c = conn.cursor()
    # Convert image to binary format
    _, buffer = cv2.imencode('.png', image)
    image_bytes = buffer.tobytes()
    # Insert into the database
    c.execute("INSERT INTO images (gesture_label, image, timestamp) VALUES (?, ?, ?)",
              (gesture_label, image_bytes, timestamp))
    conn.commit()
    conn.close()

initialize_db()


# Define the output text and service type for each gesture label
gestures = {
    0: ('No Gesture Detected', 'None'),
    1: ('Gesture 100 Detected', 'Police'),
    2: ('Gesture 108 Detected', 'Ambulance'),
    3: ('Gesture 112 Detected', 'Emergency')
}

elastic_email_api_key = '41E26C631CB2D39A01355246745E0C2E033CBFCBD5D5A599644A36FEB523D56976DEB5CB816FB5C7A77B20A04A880862'
elastic_email_sender = 'sivasaisreeyakkala@gmail.com'
recipient_email = '22215a7202@bvrit.ac.in'

# Setup ipinfo
access_token = '589bee26e1b9b4'
handler = ipinfo.getHandler(access_token)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]

            padding = 20
            x_min = max(x_min - padding, 0)
            x_max = min(x_max + padding, image.shape[1])
            y_min = max(y_min - padding, 0)
            y_max = min(y_max + padding, image.shape[0])

            hand_region = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_region.size == 0:
                continue

            pil_image = Image.fromarray(cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB))
            preprocessed_image = transform(pil_image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(preprocessed_image)
                _, predicted = torch.max(outputs, 1)
                gesture_label = predicted.item()

            output_text, service_type = gestures.get(gesture_label, ('Unknown Gesture', 'None'))
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, output_text, (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Gesture Detection', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        ip_address = requests.get('https://api.ipify.org').text
        details = handler.getDetails(ip_address)
        latitude, longitude = details.loc.split(',')


        subject = 'Gesture Detected'
        body = f'Gesture detected: {output_text}. Service type: {service_type}. Location: {latitude}, {longitude}'
        response = requests.post(
            'https://api.elasticemail.com/v2/email/send',
            data={
                'from': elastic_email_sender,
                'to': recipient_email,
                'subject': subject,
                'body': body,
                'apikey': elastic_email_api_key,
                'isTransactional': 'true'
            }
        )
        print(response.text)
        
        if response.status_code == 200:
            print("Email sent successfully, saving image to database.")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'gesture_{gesture_label}_{timestamp}.png'
            cv2.imwrite(filename, image)  # Save the entire frame for context
            insert_image_to_db(gesture_label, image, timestamp)  # Save to DB
        else:
            print("Email not sent.")
        
        data1 = {
            "latitude": latitude,
            "longitude": longitude,
            "service_type": service_type
        }
        with open('latest_gesture_data.json', 'w') as f:
            json.dump(data1, f)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

