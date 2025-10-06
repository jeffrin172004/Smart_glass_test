import cv2
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pyttsx3
from server_client import receive_image_from_pi,send_audio_to_pi

# Load environment variables from .env file
load_dotenv()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize LangChain LLM with API key from .env
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini", temperature=0)

# Define improved prompt template for blind users
prompt_template = PromptTemplate.from_template(
    "You are assisting a blind person navigating an environment. Based on these detected objects and their positions: {objects_with_positions}, provide a concise, natural-language summary. Describe the scene clearly, focusing on the spatial arrangement of objects to guide navigation or interaction. For example, note if an object is nearby or farther away, and on which side, to help the user understand their surroundings."
)

# Function to convert text to speech and save as MP3
def text_to_speech(text, output_file="summary.mp3"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)  # Speed of speech
    engine.save_to_file(text, output_file)
    engine.runAndWait()  # Waits until file is actually created
    if os.path.exists(output_file):
        send_audio_to_pi(output_file)
    else:
        print("Audio file not created properly.")
    

# Function to determine object position based on bounding box
def get_object_position(box, frame_width, frame_height):
    x_center = box.xywh[0][0].item()  # Center x-coordinate
    y_center = box.xywh[0][1].item()  # Center y-coordinate
    
    # X-axis: left, center, right
    if x_center < frame_width * 0.33:
        x_pos = "left"
    elif x_center > frame_width * 0.66:
        x_pos = "right"
    else:
        x_pos = "center"
    
    # Y-axis: near (bottom), far (top)
    if y_center < frame_height * 0.33:
        y_pos = "far"
    elif y_center > frame_height * 0.66:
        y_pos = "near"
    else:
        y_pos = ""
    
    # Combine positions (e.g., "far left", "near right", or just "center" if no y distinction)
    return f"{y_pos} {x_pos}".strip() if y_pos else x_pos

# Function to process a single frame from a video or an image
def get_environment_summary_from_frame(source):
    detected_objects_with_positions = []
    
    # Check if source is an image or video
    if source.endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(source)
        if frame is None:
            raise ValueError("Could not read the image file.")
    else:
        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Could not read the video file.")
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Detect objects in the frame
    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            obj_name = result.names[cls]
            position = get_object_position(box, frame_width, frame_height)
            detected_objects_with_positions.append(f"{obj_name} in the {position}")
    
    # Generate summary with LangChain
    object_list = ', '.join(detected_objects_with_positions) if detected_objects_with_positions else "No objects detected"
    chain = prompt_template | llm
    summary = chain.invoke({"objects_with_positions": object_list})
    
    # Convert summary to speech
    text_to_speech(summary.content)
    
    return summary.content

# Example usage
# For video (first frame):

img_path=receive_image_from_pi()

if img_path:  # Ensure image was received
    summary = get_environment_summary_from_frame(img_path)
    print(summary)
