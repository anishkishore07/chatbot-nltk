import speech_recognition as sr
import pyttsx3

class ElectroDriveUI:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speech rate

    def display_message(self, message):
        print(message)

    def speak_message(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

class ElectroDriveVoiceRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_speech(self):
        with sr.Microphone() as source:
            print("Listening for command...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        
        try:
            print("Recognizing command...")
            command = self.recognizer.recognize_google(audio)
            print("Command:", command)
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print("Error fetching results; {0}".format(e))
            return None

def main():
    ui = ElectroDriveUI()
    voice_recognition = ElectroDriveVoiceRecognition()

    ui.display_message("Welcome to ElectroDrive!")

    while True:
        command = voice_recognition.recognize_speech()
        if command:
            ui.display_message("Command recognized: " + command)
            if "open window" in command:
                ui.display_message("Opening window...")
                # Code to control window mechanism
                ui.display_message("Window opened!")
            elif "nearest power station" in command:
                ui.display_message("Locating nearest power station...")
                # Code to find nearest power station using GPS
                ui.display_message("Nearest power station found at XYZ coordinates!")
            elif "assist" in command:
                ui.display_message("Assisting driver...")
                # Code to provide driver assistance based on context
                ui.display_message("Assistance provided!")
            elif "detect issue" in command:
                ui.display_message("Detecting vehicle issue...")
                # Code to detect vehicle issues using telemetry data
                ui.display_message("No issues detected!")
            elif "exit" in command:
                ui.display_message("Exiting ElectroDrive. Goodbye!")
                break
            else:
                ui.display_message("Command not recognized. Please try again.")

if __name__ == "__main__":
    main()
