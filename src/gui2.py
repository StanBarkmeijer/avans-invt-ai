import tkinter as tk
from tkinter import Canvas, Button, messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import win32gui
import win32ui
import win32con

# Load the trained MNIST model
model = keras.models.load_model('mnist.h5')

# Create a canvas to draw on
canvas_width = 200
canvas_height = 200

def clear_canvas():
    canvas.delete("all")

def predict_number():
    # Get the handle of the canvas window
    hwnd = canvas.winfo_id()

    # Get the coordinates of the canvas window
    rect = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = rect

    # Calculate the width and height of the canvas window
    width = right - left
    height = bottom - top

    # Create a device context (DC) for the canvas window
    hdc_src = win32gui.GetWindowDC(hwnd)
    dc_src = win32ui.CreateDCFromHandle(hdc_src)

    # Create a bitmap object compatible with the DC
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(dc_src, width, height)

    # Select the bitmap object into the DC
    dc_dst = dc_src.CreateCompatibleDC()
    dc_dst.SelectObject(bmp)

    # Copy the content of the canvas window to the bitmap
    dc_dst.BitBlt((0, 0), (width, height), dc_src, (0, 0), win32con.SRCCOPY)

    # Convert the bitmap to a PIL Image
    pil_image = Image.frombuffer('RGB', (width, height), bmp.GetBitmapBits()[:width*height*4], 'raw', 'BGRX', 0, 1)

    # Convert the image to grayscale and resize it to 28x28
    pil_image_gray = pil_image.convert('L')
    resized_image = pil_image_gray.resize((28, 28))

    # Preprocess the image for prediction
    input_image = np.array(resized_image) / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.reshape(input_image, (1, 28, 28, 1))

    # Make the prediction
    prediction = model.predict(input_image)[0]
    messagebox.showinfo("Prediction", f"The drawn number is: {np.argmax(prediction)}")

root = tk.Tk()
root.title("Draw Number")

canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

canvas.bind("<B1-Motion>", lambda event: canvas.create_oval(event.x - 10, event.y - 10, event.x + 10, event.y + 10, fill='black'))

predict_button = Button(root, text="Predict", command=predict_number)
predict_button.pack(side="left")

clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side="right")

root.mainloop()
