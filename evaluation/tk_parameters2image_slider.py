"""Program to evaluate a trained neural network mapping a set of scalar
parameters to an output image or set of images and plot the result. Eventually
this will become a GUI interface to probe trained image generation models.

"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.getenv('YOKE_DIR'))
from models.surrogateCNNmodules import jekelCNNsurrogate
import torch_training_utils as tr

import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


###################################################################
# Define command line argument parser
descr_str = ('GUI: Evaluate a trained model on a set of inputs.')
parser = argparse.ArgumentParser(prog='Image Prediction Slider.',
                                 description=descr_str,
                                 fromfile_prefix_chars='@')

parser.add_argument('--checkpoint',
                    action='store',
                    type=str,
                    default='./study001_modelState_epoch0070.hdf5',
                    help='Name of HDF5 model checkpoint to evaluate output for.')

args = parser.parse_args()

# YOKE env variables
YOKE_DIR = os.getenv('YOKE_DIR')

checkpoint = args.checkpoint

# Hardcode model hyperparameters for now.
kernel = [3, 3]
featureList = [512,
               512,
               512,
               512,
               256,
               128,
               64,
               32]
linearFeatures = [4, 4]
initial_learningrate = 0.0007

model = jekelCNNsurrogate(input_size=29,
                          linear_features=linearFeatures,
                          kernel=kernel,
                          nfeature_list=featureList,
                          output_image_size=(1120, 800),
                          act_layer=nn.GELU)

#############################################
## Initialize Optimizer
#############################################
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=initial_learningrate,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=0.01)

##############
## Load Model
##############
checkpoint_epoch = tr.load_model_and_optimizer_hdf5(model,
                                                    optimizer,
                                                    checkpoint)


# def run_jCNN(sa1, sa2, sa3, sa4, sa5, sa6, sa7,
#              st1, st2, st3, st4, st5, st6, st7,
#              tt1, tt2, tt3, tt4, tt5, tt6, tt7,
#              ct1, ct2, ct3, ct4, ct5, ct6, ct7, time):
def run_jCNN(ct6, ct7, time):
    """Function to evaluate loaded model on a set of inputs.

    """
    input_params = np.array([[4.5, 5.0, 5.5, 7.0, 8.5, 10.0, 10.5,
                              0.5, 0.25, 0.25, 0.25, 0.25, 0.3, 0.3,
                              0.4, 0.35, 0.25, 0.25, 0.2, 0.1, 0.1,
                              0.1, 0.1, 0.1, 0.1, 0.15, ct6, ct7, time]],
                            dtype=np.float32)
    # input_params = np.array([[sa1, sa2, sa3, sa4, sa5, sa6, sa7,
    #                           st1, st2, st3, st4, st5, st6, st7,
    #                           tt1, tt2, tt3, tt4, tt5, tt6, tt7,
    #                           ct1, ct2, ct3, ct4, ct5, ct6, ct7, time]],
    #                         dtype=np.float32)
    input_params = torch.from_numpy(input_params)

    # Evaluate model
    model.eval()
    pred_image = model(input_params)

    #print('Shape of inputs:', input_params.shape)
    #print('Prediction shape:', pred_image.shape)

    # Reshape for plotting
    input_params = np.squeeze(input_params.numpy())
    # Predictions from network must be detached from gradients in order to be
    # written to numpy arrays.
    pred_image = np.squeeze(pred_image.detach().numpy())
    #print('Shape of image prediction:', pred_image.shape)

    return pred_image.astype(np.uint8)


def update_image(*args):
    # Get the current slider values
    ct6 = slider_ct6.get()
    ct7 = slider_ct7.get()
    time = slider_time.get()
    
    # Run the neural network and get a 2D numpy array
    array = run_jCNN(ct6, ct7, time)
    
    # Convert numpy array to PIL Image
    img = Image.fromarray(array, 'L')  # 'L' mode for grayscale
    # Anti-aliased resizing...
    current_width = max(image_label.winfo_width(), 400)
    current_height = max(image_label.winfo_height(), 400)
    img_resized = img.resize((int(0.7*current_width), int(0.7*current_height)),
                             Image.Resampling.LANCZOS)
    
    # Convert PIL Image to Tkinter PhotoImage
    photo = ImageTk.PhotoImage(img_resized)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference!


def update_sliders():
    current_width = image_label.winfo_width()
    current_height = image_label.winfo_height()
    slider_ct6.configure(length=0.5*current_height)
    slider_ct7.configure(length=0.5*current_height)
    #slider_time.configure(length=0.6*current_width)

root = tk.Tk()
root.title("Neural Network GUI")

# Main frame
control_frame = Frame(root)
control_frame.pack(side=LEFT, fill='y', padx=10, pady=10)

# Create separate frame for vertical sliders.
vertical_sliders_frame = Frame(control_frame)
vertical_sliders_frame.pack(fill='both', expand=True)

# Create vertical sliders
slider_ct6 = Scale(vertical_sliders_frame,
                   from_=0.0,
                   to=1.0,
                   orient=VERTICAL,
                   label='ct6',
                   command=lambda event: update_image())
slider_ct6.pack(side=LEFT, fill='y', expand=True)

slider_ct7 = Scale(vertical_sliders_frame,
                   from_=0.0,
                   to=1.0,
                   orient=VERTICAL,
                   label='ct7',
                   command=lambda event: update_image())
slider_ct7.pack(side=LEFT, fill='y', expand=True)

# Create separate frame for horizontal sliders.
horizontal_sliders_frame = Frame(control_frame)
horizontal_sliders_frame.pack(fill='x', expand=True)

slider_time = Scale(horizontal_sliders_frame,
                    from_=0.0,
                    to=25.0,
                    orient=HORIZONTAL,
                    label='Time',
                    command=lambda event: update_image())
slider_time.pack(fill='x', expand=True)

# Set up a label for displaying the image
image_label = Label(root)
image_label.pack(side=RIGHT, fill='both', expand=True, padx=10)

# Update sliders to match window size
root.bind('<Configure>', lambda e: update_sliders())

# Update the image initially
update_image()

# Keep the mainloop running
root.mainloop()
