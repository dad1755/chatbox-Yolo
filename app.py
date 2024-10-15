import streamlit as st
import random
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

# Load the Google API key from secrets
api_key = st.secrets["general"]["GOOGLE_API_KEY"]
print(f"Loaded API key: {api_key}")  # Debugging line to verify if the API key is loaded

# A simple document retrieval function
def retrieve_documents(query, documents):
    return random.choice(documents) if documents else "No documents available for retrieval."

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

# Centered title with responsive styling
st.markdown("""
    <style>
        @media (max-width: 600px) {
            h1 { font-size: 70px; line-height: 1.2; }
            h3 { font-size: 16px; line-height: 1.1; }
            .stTextInput > div > input {
                font-size: 16px !important;
                height: 48px !important;
                width: 100% !important;
            }
        }
        @media (min-width: 601px) {
            h1 { font-size: 36px; line-height: 1; }
            h3 { font-size: 24px; line-height: 0; }
            .stTextInput > div > input {
                font-size: 20px !important;
                height: 56px !important;
                width: 80% !important;
            }
        }
        .stButton > button { padding: 10px 20px; }
        .stFileUploader { margin-top: 20px; margin-bottom: 20px; }
        /* Mobile specific container for query form */
        .mobile-container {
            max-width: 100% !important;
        }
    </style>
    <h1 style='text-align: center; margin: 0;'>🦙💬 G10</h1>
    <h3 style='text-align: center; margin: 0;'>Face Counter Apps</h3>
""", unsafe_allow_html=True)

# Initialize the documents list
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Load the YOLO model only once
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Store chat messages for the user query in the chatbox
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you?"}]

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to handle document retrieval and model response
def handle_user_query(user_query):
    if user_query:
        if st.session_state.documents:
            retrieved_document = retrieve_documents(user_query, st.session_state.documents)
            st.write("Retrieved Document: Here are the extracted details:")
            st.write(retrieved_document)

        # Prepare to use the Gemini API
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Generate content using the Gemini model
            response = model.generate_content(user_query)

            # Handle the response
            if response:
                generated_text = response.text
                st.write("Model Response:")
                st.write(generated_text)
            else:
                st.warning("No response received from the Gemini model.")
        else:
            st.error("Google API key is missing. Check your secrets file.")

        # Clear the documents after submission
        st.session_state.documents.clear()
    else:
        st.error("Please enter a query before submitting.")

# User-provided chat input for queries
if user_query := st.chat_input(placeholder="Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing and generating response..."):
            handle_user_query(user_query)

# Add a file uploader for document and image
uploaded_file = st.file_uploader("Upload a document (text file) or image (jpg/png)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Process the uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.type
    # Handle text document upload
    if file_type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        st.session_state.documents.append(content)
        st.success("Document uploaded successfully!")

        # Analyze document
        if st.button("Analyze Document"):
            analysis_result = content  # For now, simply display the content
            st.write("Analysis Result: Here is the content of the uploaded document:")
            st.write(analysis_result)

    # Handle image file upload
    elif file_type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)

        # Automatically detect faces after uploading an image
        detected_faces = detect_faces(image, st.session_state.model)
        boxes = detected_faces.xyxy

        # Check if any boxes (faces) are detected
        if boxes is not None and len(boxes) > 0:
            # Draw bounding boxes on the image
            image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
            st.image(image_with_boxes, caption='Detected Faces', channels="RGB")

            # Display the number of detected faces
            st.write(f"Number of faces detected: {len(boxes)}")
        else:
            st.warning("No faces detected. Please try a different image.")
