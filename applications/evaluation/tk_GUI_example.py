"""Short example program to evaluate a dummy function representing a trained
neural network mapping on a set of scalar parameters using a GUI interface to
probe input parameter effects.

"""

import numpy as np

import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


# Dummy function to simulate neural network output as a 2D numpy array
def run_neural_network(param1, param2, param3):
    # Generate a 200x200 numpy array based on the parameters
    array = np.random.rand(200, 200) * param1 / 255  # Simple variation based on parameter
    if np.max(array) == 0.0:
        norm_array = 0.0 * np.random.rand(200, 200)
    else:
        norm_array = array * 255 / np.max(array)
    return norm_array.astype(np.uint8)  # Normalize and convert to 8-bit grayscale


def update_image(*args):
    # Get the current slider values
    param1 = slider1.get()
    param2 = slider2.get()
    param3 = slider3.get()

    # Run the neural network and get a 2D numpy array
    array = run_neural_network(param1, param2, param3)

    # Convert numpy array to PIL Image
    img = Image.fromarray(array, 'L')  # 'L' mode for grayscale
    # Anti-aliased resizing...
    current_width = max(image_label.winfo_width(), 400)
    current_height = max(image_label.winfo_height(), 400)
    img_resized = img.resize((int(0.7 * current_width), int(0.7 * current_height)),
                             Image.Resampling.LANCZOS)

    # Convert PIL Image to Tkinter PhotoImage
    photo = ImageTk.PhotoImage(img_resized)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference!


def update_sliders():
    current_width = image_label.winfo_width()
    current_height = image_label.winfo_height()
    slider1.configure(length=0.5 * current_height)
    slider2.configure(length=0.5 * current_height)
    # slider3.configure(length=0.3*current_width)


root = tk.Tk()
root.title("Neural Network GUI")

# Main frame
control_frame = Frame(root)
control_frame.pack(side=LEFT, fill='y', padx=10, pady=10)

# Create separate frame for vertical sliders.
vertical_sliders_frame = Frame(control_frame)
vertical_sliders_frame.pack(fill='both', expand=True)

# Create vertical sliders
slider1 = Scale(vertical_sliders_frame,
                from_=0,
                to=255,
                orient=VERTICAL,
                label="Parameter 1",
                command=lambda event: update_image())
slider1.pack(side=LEFT, fill='y', expand=True)

slider2 = Scale(vertical_sliders_frame,
                from_=0,
                to=255,
                orient=VERTICAL,
                label="Parameter 2",
                command=lambda event: update_image())
slider2.pack(side=LEFT, fill='y', expand=True)

# Create separate frame for horizontal sliders.
horizontal_sliders_frame = Frame(control_frame)
horizontal_sliders_frame.pack(fill='x', expand=True)

slider3 = Scale(horizontal_sliders_frame,
                from_=0,
                to=255,
                orient=HORIZONTAL,
                label="Parameter 3",
                command=lambda event: update_image())
slider3.pack(fill='x', expand=True)

# Set up a label for displaying the image
image_label = Label(root)
image_label.pack(side=RIGHT, fill='both', expand=True, padx=10)

# Update sliders to match window size
root.bind('<Configure>', lambda e: update_sliders())

# Update the image initially
update_image()

# Keep the mainloop running
root.mainloop()
