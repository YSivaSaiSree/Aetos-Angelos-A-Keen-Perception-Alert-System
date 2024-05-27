#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sqlite3
import cv2
import numpy as np

def fetch_images(db_path='gesture_images.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query all images and their labels from the database
    cursor.execute("SELECT gesture_label, image, timestamp From images")
    images = cursor.fetchall()

    # Close the database connection
    conn.close()

    return images

def display_images(images):
    for label, image_blob, timestamp in images:
        # Convert the binary blob to a numpy array
        image_array = np.frombuffer(image_blob, dtype=np.uint8)

        # Decode the numpy array into image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Display the image using OpenCV
        window_name = f"Label: {label}, Timestamp: {timestamp}"
        cv2.imshow(window_name, image)
        
        # Wait for a key press to close the image window
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

if __name__ == "__main__":
    images = fetch_images()
    if images:
        display_images(images)
    else:
        print("No images found in the database.")

