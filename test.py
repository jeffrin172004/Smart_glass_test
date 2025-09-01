import cv2
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pyttsx3

# Load environment variables from .env file
load_dotenv()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize LangChain LLM with API key from .env
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini", temperature=0)

# Define prompt template
prompt_template = PromptTemplate.from_template(
    "Summarize the environment for a blind person based on these detected objects: {objects}."
)

# Function to convert text to speech and save as MP3
def text_to_speech(text, output_file="summary.mp3"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.save_to_file(text, output_file)
    engine.say(text)
    engine.runAndWait()

# Function to process a single frame from a video or an image
def get_environment_summary_from_frame(source):
    detected_objects = set()
    
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
    
    # Detect objects in the frame
    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            detected_objects.add(result.names[cls])
    
    # Generate summary with LangChain
    object_list = ', '.join(detected_objects) if detected_objects else "No objects detected"
    chain = prompt_template | llm
    summary = chain.invoke({"objects": object_list})
    
    # Convert summary to speech
    text_to_speech(summary.content)
    
    return summary.content

# Example usage
# For video (first frame):
summary = get_environment_summary_from_frame('eg.mp4')
print(summary)

