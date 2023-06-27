from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
from PIL import ImageTk

model = load_model('mnist2.h5')

def predict_digit(img):
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0

    res = model.predict(img)
    return np.argmax(res), max(res[0])

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

        # Display the drawn image in a separate window
        self.display_image(im)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y

        r = 8

        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

    def display_image(self, img):
        top = tk.Toplevel()
        top.title("Drawn Image")

        # Resize and display the image
        img = img.resize((280, 280))
        img = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(top, width=img.width(), height=img.height())
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=img)

        top.mainloop()

app = App()
mainloop()
