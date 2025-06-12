import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
import cv2
import os

# Path to your model file
MODEL_PATH = r"..\guitar_chord_recognition_final.pth"

# Path to the fretboard images folder
IMAGES_PATH = r"..\images1"

# Define the chord classes
CHORD_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Load the trained model
def load_model():
    model = models.efficientnet_v2_s(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(CHORD_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# Predict the chord
def predict_chord(frame, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return CHORD_CLASSES[predicted.item()]

# Tkinter application class
class ChordPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Guitar Chord Predictor")
        self.running = True
        
        # Model
        self.model = load_model()
        
        # Load fretboard images
        self.chord_images = {chord: Image.open(os.path.join(IMAGES_PATH, f"{chord}.png")) for chord in CHORD_CLASSES}

        # Widgets
        self.label = tk.Label(root, text="Predicting chords from webcam...", font=("Arial", 14))
        self.label.pack(pady=10)

        self.frame = tk.Frame(root)
        self.frame.pack(pady=10)

        self.canvas = tk.Canvas(self.frame, width=640, height=480, bg="black")
        self.canvas.pack(side="left", padx=10)

        self.fretboard_canvas = tk.Canvas(self.frame, width=300, height=300, bg="white")
        self.fretboard_canvas.pack(side="right", padx=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16), fg="blue")
        self.result_label.pack(pady=10)

        # Start webcam
        self.capture = cv2.VideoCapture(1)
        self.update_frame()

    def update_frame(self):
        if self.running:
            ret, frame = self.capture.read()
            if ret:
                # Predict the chord
                chord = predict_chord(frame, self.model)

                # Display the prediction
                self.result_label.config(text=f"Predicted Chord: {chord}")

                # Convert the frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
                self.canvas.image = imgtk

                # Display the corresponding fretboard image
                fretboard_img = ImageTk.PhotoImage(self.chord_images[chord].resize((300, 300), Image.LANCZOS))

                self.fretboard_canvas.create_image(0, 0, anchor="nw", image=fretboard_img)
                self.fretboard_canvas.image = fretboard_img

            # Schedule the next frame update
            self.root.after(10, self.update_frame)

    def __del__(self):
        if self.running:
            self.capture.release()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x600")  # Set a window size that fits all elements
    app = ChordPredictorApp(root)
    root.mainloop()
