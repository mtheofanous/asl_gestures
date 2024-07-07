import streamlit as st
import os
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import cv2
import joblib
import mediapipe as mp
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from data import visualizaciones_datasets, show_random_image
import av
import subprocess
import requests
# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(page_title = "DSB01RT",
                   page_icon = ":panda_face:",
                   layout = "wide",
                   initial_sidebar_state = 'expanded')


def main():
    st.title("ASL Gesture Recognition App.")
    st.write("Use the Menu to the left to go to the page of your choice.")

    # st.sidebar.success("Navigation")
    menu = ["Home", "Data", "Real_time_Recognition"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the ASL Gesture Recognition App.")
        st.write("This web app recognizes American Sign Language (ASL) gestures using hand landmarks.")
        image_path = os.path.join("sources", "homepage_image.jpg")
        
        st.image(image_path)
        st.write("""
            ## Overview
            American Sign Language (ASL) is a complete language that uses gestures and visual signs to communicate with people who are deaf or hard of hearing. ASL has its own grammar and syntax, making it a fully-fledged language in its own right.

            ## Dataset
            The dataset used for this project is sourced from Kaggle’s ASL Alphabet dataset. It contains:
            - **Training Data**: 87,000 images of ASL gestures, each of size 200x200 pixels.
            - **Categories**: 29 labeled categories including letters A-Z and additional gestures like "del", "nothing", and "space".

            ## Approaches Considered
            We considered three potential approaches to solve this problem:
            1. **Using the Whole Image**: This approach involves training a model on the entire image of the gesture.
            2. **Cropping the Image to Only the Hand**: Here, we would preprocess the images to isolate and use only the hand region.
            3. **Extracting Hand Landmarks**: This approach focuses on extracting and using the hand landmarks for classification.

            ## Chosen Approach
            After evaluating the pros and cons of each method, we decided to pursue the third approach: extracting hand landmarks. This decision was based on several factors:
            - **Dimensionality Reduction**: By focusing on hand landmarks, we reduce the size of our input data significantly, which simplifies the model and speeds up the training process.
            - **Increased Robustness**: Hand landmarks provide a more invariant and robust representation of gestures compared to raw pixel data, which can be affected by variations in lighting, background, and other noise.
            - **Relevance**: Hand landmarks contain the essential information needed to distinguish between different ASL gestures, making them an efficient choice for classification.
        """)
    elif choice == "Data":
        st.header("Dataset Visualization")

        # Load the dataset
        csv_path = os.path.join("hand_landmarks_augment.csv")
        df = pd.read_csv(csv_path)
        df = pd.read_csv('/Users/DELL/Desktop/ASL_marios/hand_landmarks_augment.csv') 
        

        data_choice = st.sidebar.selectbox("Data Analysis", ["Show Dataset", "Label Distribution", "Show hand landmarks"])

        if data_choice == "Show Dataset":
            st.write("### Dataset")
            st.dataframe(df)

        elif data_choice == "Label Distribution":
            st.write("### Label Distribution")
            visualizaciones_datasets(df)

        elif data_choice == "Show hand landmarks":
            st.write("### Hand Gesture Image")
            st.sidebar.write("### Hand Gesture Image")
            letter = st.sidebar.text_input("Enter a letter (A-Z):", 'B')
            if letter:
                show_random_image(letter, data_dir='/Users/DELL/Desktop/Project_3/asl_alphabet_test')
    
    elif choice == "Real_time_Recognition":

        st.title("Live-Stream")
 
        st.sidebar.success("Use the image grid below to practice ASL gestures. The app will recognize your gestures in real-time.")
        image_path = os.path.join("sources", "image_grid.jpg")
        st.sidebar.image(image_path, caption="Image Grid", use_column_width=True)


        # filter = "none"
        # Load the pre-trained model and label encoder
        model_file = st.file_uploader("/Users/DELL/Desktop/ASL_marios/model_landmarks_augment.pkl", type=["pkl"], key="model")
        le_file = st.file_uploader("/Users/DELL/Desktop/ASL_marios/label_encoder_augment.pkl", type=["pkl"], key="le")
        # model_path = os.path.join("/Users/DELL/Desktop/ASL_marios/", "model_landmarks_augment.pkl")
        # le_path = os.path.join("/Users/DELL/Desktop/ASL_marios/", "label_encoder_augment.pkl")
        if model_file and le_file:
            try:
                model = joblib.load(model_file)
                le = joblib.load(le_file)
                st.write("Model and Label Encoder loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload the model and label encoder files.")

        # Initialize mediapipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        from mediapipe.python.solutions.drawing_utils import DrawingSpec

        # Customize the styles
        landmark_style = DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
        connection_style = DrawingSpec(color=(0, 0, 0), thickness=2)

        # Function to apply hand gesture recognition
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
                self.predictions = deque(maxlen=10)
                self.last_predicted_gesture = ""
                self.gesture_history = ""

            def transform(self, frame: av.VideoFrame):
                img = frame.to_ndarray(format="bgr24")

                # Process the frame to find hand landmarks
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                H, W, _ = img.shape

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw the hand landmarks
                        mp_drawing.draw_landmarks(
                            img,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_style,
                            connection_style
                        )

                        # Prepare data for prediction
                        data = []
                        x_ = []
                        y_ = []

                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)

                        for i in range(len(hand_landmarks.landmark)):
                            data.append(hand_landmarks.landmark[i].x - min(x_))
                            data.append(hand_landmarks.landmark[i].y - min(y_))

                        # Predict the gesture
                        prediction = model.predict([np.asarray(data)])
                        predicted_character = le.inverse_transform(prediction)[0]

                        # Smooth predictions
                        self.predictions.append(predicted_character)
                        smoothed_prediction = max(set(self.predictions), key=self.predictions.count)

                        # Update last predicted gesture
                        # self.last_predicted_gesture = f'Predicted Gesture: {smoothed_prediction}'

                        # Draw a bounding box around the hand
                        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)

                        # Put the predicted gesture on the frame
                        cv2.putText(img, smoothed_prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(key="streamer", video_frame_callback=VideoTransformer().transform, sendback_audio=False)
        # if ctx.video_transformer:
        #     st.write("### Predictions")
        #     st.write(ctx.video_transformer.last_predicted_gesture)
        #     st.write("### Gesture History")
        #     st.write(ctx.video_transformer.gesture_history)



if __name__ == "__main__":
    main()
    