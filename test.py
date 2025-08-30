import cv2
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize LangChain LLM with API key from .env
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define prompt template
prompt_template = PromptTemplate.from_template(
    "Summarize the environment for a blind person based on these detected objects: {objects}."
)

def get_environment_summary_from_frame(source):
    detected_objects = set()
    
    # Check if source is an image or video
    if source.endswith(('.jpg', '.jpeg', '.png')):
        # Read image
        frame = cv2.imread(source)
        if frame is None:
            raise ValueError("Could not read the image file.")
    else:
        # Read first frame from video
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
    
    return summary.content

# Example usage
# For an image:
# summary = get_environment_summary_from_frame('path_to_your_image.jpg')
# For a video (uses first frame):
summary = get_environment_summary_from_frame('eg.mp4')
print(summary)