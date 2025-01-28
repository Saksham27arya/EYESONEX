import numpy as np
import cv2
from scipy.spatial import distance as dist
import pyttsx3
import speech_recognition as sr
import datetime
import time
# import pyaudio
import requests
import pytesseract
from pytesseract import Output
import wikipedia
# import pyjokes
# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Function to speak the provided text
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Function to wish the user based on the time of day
def wishMe():
    hour = int(datetime.datetime.now().hour)
    strTime = time.strftime("%H:%M")
    if hour >= 0 and hour < 12:
        speak("Good Morning! The time is")
        speak(strTime)
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon! The time is")
        speak(strTime)
    else:
        speak("Good Evening! The time is")
        speak(strTime)
    speak("Hello . Please tell me how may I help you boss ")

# Function to recognize voice commands
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
        return query.lower()
    except Exception as e:
        print("Say that again please...")
        return "None"
    return query

def get_weather(city_name):
    base_url = f"http://wttr.in/{city_name}?format=%C+%t"
    response = requests.get(base_url)
    if response.status_code == 200:
        weather_info = response.text
        speak(f"The weather in {city_name} is currently {weather_info}.")
    else:
        speak(f"Sorry, I couldn't retrieve the weather information for {city_name}.")

# def read_text_from_image():
#     # Open an image file for text recognition
#     img = cv2.imread('test_image.jpg')

#     # Convert the image to grayscale
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Use PyTesseract for text recognition
#     extracted_text = pytesseract.image_to_string(gray_img)

#     # Speak the extracted text
#     speak("The extracted text is:")
#     speak(extracted_text)


# Function to read text from live camera feed
def read_text_from_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            cap.release()
            return

        # Convert the frame to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use PyTesseract for text recognition
        text_data = pytesseract.image_to_data(gray_img, output_type=Output.DICT)

        for i in range(len(text_data['text'])):
            if text_data['conf'][i] != -1 and text_data['text'][i] != '':
                (x, y, w, h) = (text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, text_data['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                print(f"Recognized Text: {text_data['text'][i]}")

                # Speak the recognized text
                speak("Text reading complete")
                speak(text_data['text'][i])

                cv2.imshow('Text Recognition', frame)
                cv2.waitKey(5000)  # Display the frame for 5 seconds
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow('Text Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# Object detection and distance measurement function
def object_detection_and_distance():
    Known_distance = 30  # Inches
    Known_width = 5.7  # Inches
    thres = 0.5  # Threshold to detect object
    nms_threshold = 0.1  # (0.1 to 1) 1 means no suppress, 0.1 means high suppress

    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)
    font = cv2.FONT_HERSHEY_PLAIN

    cap = cv2.VideoCapture(0)
    Distance_level = 0
    classNames = []
    with open('coco.names', 'r') as f:
        classNames = f.read().splitlines()
    print(classNames)
    Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

    weightsPath = "frozen_inference_graph.pb"
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Focal length finder function
    def FocalLength(measured_distance, real_width, width_in_rf_image):
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    # Distance estimation function
    def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
        distance = (real_face_width * Focal_Length) / face_width_in_frame
        return distance

    # Face detection function
    def face_data(image, CallOut, Distance_level):
        face_width = 0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, h, w) in faces:
            line_thickness = 2
            LLV = int(h * 0.12)
            cv2.line(image, (x, y + LLV), (x + w, y + LLV), (GREEN), line_thickness)
            cv2.line(image, (x, y + h), (x + w, y + h), (GREEN), line_thickness)
            cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), (GREEN), line_thickness)
            cv2.line(image, (x + w, y + LLV), (x + w, y + LLV + LLV), (GREEN), line_thickness)
            cv2.line(image, (x, y + h), (x, y + h - LLV), (GREEN), line_thickness)
            cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), (GREEN), line_thickness)
            face_width = w
        return face_width, faces

    # Reading reference image from directory
    ref_image = cv2.imread("lena.png")
    ref_image_face_width, _ = face_data(ref_image, False, Distance_level)
    Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
    print(Focal_length_found)

    # **CHANGED**: Capture a single frame, process, and exit
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        cap.release()
        return

    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    face_width_in_frame, Faces = face_data(frame, True, Distance_level)

    for i in indices:
        box = bbox[i]
        confidence = str(round(confs[i], 2))
        color = Colors[classIds[i] - 1]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
        cv2.putText(frame, classNames[classIds[i]-1] + " " + confidence, (x + 10, y + 20),
                    font, 1, color, 2)

    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:
            distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
            distance = round(distance, 2)
            Distance_level = int(distance)
            cv2.putText(frame, f"Distance {distance} Inches", (face_x-6, face_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (BLACK), 2)

    # Text-to-speech
    objects = len(classIds)
    objects_description = ', '.join([classNames[classIds[i]-1] for i in indices])
    distances = [Distance_finder(Focal_length_found, Known_width, box[2]) for box in bbox]
    distances_description = ', '.join([f"{distance:.2f} inches" for distance in distances])
    speech_text = f"There are {objects} objects: {objects_description}. Their distances are: {distances_description}."
    speak(speech_text)

    cv2.imshow('Object Detection', frame)
    cv2.waitKey(5000)  # Display the frame for 5 seconds

    cap.release()
    cv2.destroyAllWindows()

# Main function to initiate voice command recognition and task execution loop
# 

def main():
    wishMe()
    while True:
        query = takeCommand()

        # Logic for executing tasks based on query
        try:
            if 'check ' in query:
                object_detection_and_distance()
                # **ADDED**: Acknowledge the completion and await next command
                speak("Finished checking the surroundings. Awaiting next command.")
            elif "weather" in query:
                speak("Boss, tell me the city name")
                city_name = takeCommand().lower()
                get_weather(city_name)
            elif "read the text" in query:
                read_text_from_camera()
                speak("Finished reading the text ... Awaiting for the next command")
            elif "wikipedia" in query:
                speak("Searching Wikipedia...")
                query = query.replace("wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)
            else:
                speak("I'm sorry, I didn't understand that. Please say again.")
        except Exception as e:
            speak(f"Boss, I'm sorry, there was an issue with the command: {e}. Please try again.")

if __name__ == "__main__":
    main()
