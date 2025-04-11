import os
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import face_recognition

# Load the deepfake detection model
MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "deepfake_detection_model_v2.keras")
model = load_model(MODEL_PATH)

# Load ResNet50 for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_faces_from_video(video_path, max_faces=10, target_size=(128, 128)):
    """Extract faces from the video file along with their original frames."""
    video_capture = cv2.VideoCapture(video_path)
    faces_out = []
    
    while len(faces_out) < max_faces:
        ret, frame = video_capture.read()
        if not ret:
            break  # Stop if the video ends

        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            face = rgb_frame[top:bottom, left:right]
            face_resized = cv2.resize(face, target_size)
            faces_out.append(face_resized)

            if len(faces_out) >= max_faces:
                break

    video_capture.release()
    return np.array(faces_out) if faces_out else None

def extract_faces_from_image(image_path, target_size=(128, 128)):
    """Extract faces from a single image file."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    faces_out = []
    for top, right, bottom, left in face_locations:
        face = rgb_image[top:bottom, left:right]
        face_resized = cv2.resize(face, target_size)
        faces_out.append(face_resized)
    return np.array(faces_out) if faces_out else None

def extract_features_from_face(face_img):
    """Extract features from a detected face using ResNet50."""
    face_resized = cv2.resize(face_img, (224, 224))
    face_array = img_to_array(face_resized)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array)
    features = resnet_model.predict(face_array)
    return features.flatten()

def delete_old_images(image_paths, delay=300):
    """Delete images after a certain delay."""
    time.sleep(delay)
    for image_path in image_paths:
        if os.path.exists(image_path):
            os.remove(image_path)

@csrf_exempt
def upload_video_page(request):
    context = {}
    
    if request.method == "POST" and request.FILES.get("video"):
        upload_file = request.FILES["video"]
        file_type = upload_file.content_type
        
        # Determine if file is a video or an image
        if file_type.startswith("video/"):
            # Process video upload
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_videos")
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            # Save video with a timestamped filename
            video_filename = f"{int(time.time())}_{upload_file.name}"
            file_path = os.path.join(upload_dir, video_filename)
            with open(file_path, "wb") as f:
                for chunk in upload_file.chunks():
                    f.write(chunk)
            context["video_name"] = upload_file.name
            context["video_url"] = f"{settings.MEDIA_URL}uploaded_videos/{video_filename}"
            # Extract faces from video
            faces = extract_faces_from_video(file_path)
        elif file_type.startswith("image/"):
            # Process image upload
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_images")
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            # Save image with a timestamped filename
            image_filename = f"{int(time.time())}_{upload_file.name}"
            file_path = os.path.join(upload_dir, image_filename)
            with open(file_path, "wb") as f:
                for chunk in upload_file.chunks():
                    f.write(chunk)
            context["video_name"] = upload_file.name  # Using same key for simplicity
            context["video_url"] = f"{settings.MEDIA_URL}uploaded_images/{image_filename}"
            # Extract faces from image
            faces = extract_faces_from_image(file_path)
        else:
            context["error"] = "Unsupported file type."
            return render(request, "upload_video_page.html", context)
        
        # Process face extraction results
        if faces is None:
            context["error"] = "No faces detected in the file."
        else:
            # Compute a file-level confidence score (average over all detected faces)
            features_list = [extract_features_from_face(face) for face in faces]
            features_avg = np.mean(features_list, axis=0)
            features_avg = np.expand_dims(features_avg, axis=0)
            confidence_score = model.predict(features_avg)[0][0]
            prediction = "Manipulated (Deepfake)" if confidence_score > 0.5 else "Original (Real)"
            context["prediction"] = prediction
            context["confidence_score"] = float(confidence_score)
            
            # If manipulated, extract and save face images (frames)
            if confidence_score > 0.5:
                manipulated_faces_dir = os.path.join(settings.MEDIA_ROOT, "manipulated_faces")
                if not os.path.exists(manipulated_faces_dir):
                    os.makedirs(manipulated_faces_dir)
                image_urls = []
                for i, face in enumerate(faces):
                    face_features = extract_features_from_face(face)
                    face_features = np.expand_dims(face_features, axis=0)
                    face_prob = model.predict(face_features)[0][0]
                    if face_prob > 0.5:
                        face_filename = f"manipulated_face_{i}.jpg"
                        face_path = os.path.join(manipulated_faces_dir, face_filename)
                        print(f"Saving face at: {face_path}")
                        saved = cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                        if saved:
                            print(f"✅ Image saved successfully: {face_path}")
                        else:
                            print(f"❌ Failed to save image: {face_path}")
                        image_urls.append(f"{settings.MEDIA_URL}manipulated_faces/{face_filename}")
                context["image_urls"] = image_urls
                # Schedule deletion of these images after some time
                threading.Thread(
                    target=delete_old_images,
                    args=(
                        [os.path.join(settings.MEDIA_ROOT, url.replace(settings.MEDIA_URL, "")) for url in image_urls],
                    ),
                ).start()

    return render(request, "upload_video_page.html", context)

