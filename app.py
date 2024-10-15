import requests
import streamlit as st
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

# Load the Google API key from secrets
api_key = st.secrets["general"]["GOOGLE_API_KEY"]
print(f"Loaded API key: {api_key}")  # Debugging line to verify if the API key is loaded


# Load the YOLO model from Hugging Face
def load_model():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)
    return model

# Inference function for face detection
def detect_faces(image, model):
    output = model(image)
    results = Detections.from_ultralytics(output[0])
    return results

# Draw bounding boxes on the image
def draw_bounding_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]  # Get the bounding box coordinates
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)  # Draw the rectangle
    return image

# Function to call Google Gemini API for text generation
def generate_gemini_response(user_query, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "prompt": {
            "text": user_query
        },
        "temperature": 0.7,
        "maxOutputTokens": 256
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        return response_data["candidates"][0]["output"]
    else:
        st.error(f"Error {response.status_code}: Unable to generate response")
        return None

# Handle user query for text generation using Google Gemini API
def handle_user_query(user_query, api_key):
    if user_query:
        st.write("Received Query: ", user_query)

        # Generate content using the Google Gemini API
        generated_text = generate_gemini_response(user_query, api_key)
        if generated_text:
            st.write("Model Response:")
            st.write(generated_text)

# Load models only once
if 'model' not in st.session_state:
    st.session_state.model = load_model()



# User-provided chat input
if user_query := st.chat_input(placeholder="Enter your query here..."):
    st.write("Processing query: ", user_query)
    handle_user_query(user_query, api_key)

# Image upload for face detection
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Detect faces
    detected_faces = detect_faces(image, st.session_state.model)
    boxes = detected_faces.xyxy

    # Check if any faces are detected
    if boxes is not None and len(boxes) > 0:
        image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
        st.image(image_with_boxes, caption='Detected Faces', channels="RGB")
        st.write(f"Number of faces detected: {len(boxes)}")
    else:
        st.warning("No faces detected. Please try a different image.")
