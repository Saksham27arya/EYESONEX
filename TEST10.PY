import cv2
import pytesseract
from pytesseract import Output
import pyttsx3

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def read_text_from_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the frame to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use PyTesseract for text recognition
        try:
            text_data = pytesseract.image_to_data(gray_img, output_type=Output.DICT)
        except pytesseract.pytesseract.TesseractNotFoundError:
            print("Tesseract not found. Ensure it's installed and the path is correctly set.")
            break

        for i in range(len(text_data['text'])):
            if int(text_data['conf'][i]) > 60 and text_data['text'][i].strip() != '':
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
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    read_text_from_camera()
