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

import wx
from PIL import Image

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib
#matplotlib.use('MacOSX')
#matplotlib.use('pdf')
# Get rid of type 3 fonts in figures
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
# Ensure LaTeX font
font = {'family': 'serif'}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (6, 6)
from mpl_toolkits.axes_grid1 import make_axes_locatable


###################################################################
# Define command line argument parser
descr_str = ('Evaluate a trained model on a set of inputs.')
parser = argparse.ArgumentParser(prog='Image Prediction Slider.',
                                 description=descr_str,
                                 fromfile_prefix_chars='@')

parser.add_argument('--checkpoint',
                    action='store',
                    type=str,
                    default='./study001_modelState_epoch0070.hdf5',
                    help='Name of HDF5 model checkpoint to evaluate output for.')

parser.add_argument('--savedir',
                    action='store',
                    type=str,
                    default='./',
                    help='Directory for saving images.')

parser.add_argument('--savefig', '-S',
                    action='store_true',
                    help='Flag to save figures.')

args = parser.parse_args()

# YOKE env variables
YOKE_DIR = os.getenv('YOKE_DIR')

checkpoint = args.checkpoint
    
# Additional input variables
savedir = args.savedir
SAVEFIG = args.savefig

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

###################################
## Input parameters and evaluation
###################################


# # Plot normalized radiograph and density field for diagnostics.
# fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
# fig1.suptitle('Time={:.3f}us'.format(input_params[-1]), fontsize=18)
# img1 = ax1.imshow(pred_image,
#                   aspect='equal',
#                   origin='lower',
#                   cmap='jet',
#                   vmin=pred_image.min(),
#                   vmax=pred_image.max())
# ax1.set_ylabel("Z-axis", fontsize=16)                 
# ax1.set_xlabel("R-axis", fontsize=16)
# ax1.set_title('Prediction', fontsize=18)

# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes('right', size='10%', pad=0.1)
# fig1.colorbar(img1,
#               cax=cax1).set_label('Density (g/cc)',
#                                   fontsize=14)

# # Save or plot images
# if SAVEFIG:
#     if not os.path.exists(savedir):
#         os.makedirs(savedir)

#     plt.figure(fig1.number)
#     filenameA = f'{savedir}/jCNN_pred_image.png'
#     plt.savefig(filenameA, bbox_inches='tight')
# else:
#     plt.show()


def run_jCNN(sa1, sa2, sa3, sa4, sa5, sa6, sa7,
             st1, st2, st3, st4, st5, st6, st7,
             tt1, tt2, tt3, tt4, tt5, tt6, tt7,
             ct1, ct2, ct3, ct4, ct5, ct6, ct7, time):
    """Function to evaluate loaded model on a set of inputs.

    """
    # input_params = np.array([[4.5, 5.0, 5.5, 7.0, 8.5, 10.0, 10.5,
    #                           0.5, 0.25, 0.25, 0.25, 0.25, 0.3, 0.3,
    #                           0.4, 0.35, 0.25, 0.25, 0.2, 0.1, 0.1,
    #                           0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2, 25.0]],
    #                         dtype=np.float32)
    input_params = np.array([[sa1, sa2, sa3, sa4, sa5, sa6, sa7,
                              st1, st2, st3, st4, st5, st6, st7,
                              tt1, tt2, tt3, tt4, tt5, tt6, tt7,
                              ct1, ct2, ct3, ct4, ct5, ct6, ct7, time]],
                            dtype=np.float32)
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


class MyFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, title='Jekel-tCNN GUI', size=(350, 400))
        
        self.panel = wx.Panel(self)
        self.init_ui()

    def init_ui(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Create sliders
        #sa1, sa2, sa3, sa4, sa5, sa6, sa7,
        #st1, st2, st3, st4, st5, st6, st7,
        #tt1, tt2, tt3, tt4, tt5, tt6, tt7,
        #ct1, ct2, ct3, ct4, ct5, ct6, ct7, time
        self.slider_sa1 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_sa1.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_sa1, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_sa2 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_sa2.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_sa2, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_sa3 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_sa3.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_sa3, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_sa4 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_sa4.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_sa4, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_sa5 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_sa5.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_sa5, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_sa6 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_sa6.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_sa6, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_sa7 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_sa7.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_sa7, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_st1 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_st1.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_st1, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_st2 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_st2.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_st2, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_st3 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_st3.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_st3, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_st4 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_st4.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_st4, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_st5 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_st5.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_st5, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_st6 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_st6.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_st6, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_st7 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_st7.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_st7, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_tt1 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_tt1.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_tt1, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_tt2 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_tt2.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_tt2, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_tt3 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_tt3.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_tt3, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_tt4 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_tt4.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_tt4, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_tt5 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_tt5.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_tt5, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_tt6 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_tt6.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_tt6, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_tt7 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_tt7.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_tt7, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_ct1 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_ct1.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_ct1, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_ct2 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_ct2.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_ct2, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_ct3 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_ct3.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_ct3, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_ct4 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_ct4.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_ct4, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_ct5 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_ct5.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_ct5, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_ct6 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_ct6.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_ct6, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_ct7 = wx.Slider(self.panel,
                                    value=125,
                                    minValue=0,
                                    maxValue=255,
                                    style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_ct7.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_ct7, flag=wx.EXPAND|wx.ALL, border=10)

        self.slider_time = wx.Slider(self.panel,
                                     value=125,
                                     minValue=0,
                                     maxValue=255,
                                     style=wx.SL_HORIZONTAL|wx.SL_LABELS)
        self.slider_time.Bind(wx.EVT_SLIDER, self.on_slide)
        vbox.Add(self.slider_time, flag=wx.EXPAND|wx.ALL, border=10)

        # Set up an area for displaying the image
        self.image_panel = wx.StaticBitmap(self.panel)
        vbox.Add(self.image_panel, proportion=1, flag=wx.EXPAND|wx.ALL, border=10)
        
        self.panel.SetSizer(vbox)
        self.update_image()

    def on_slide(self, event):
        self.update_image()
        
    def update_image(self):
        # Get the current slider values
        sa1 = self.slider_sa1.GetValue()
        sa2 = self.slider_sa2.GetValue()
        sa3 = self.slider_sa3.GetValue()
        sa4 = self.slider_sa4.GetValue()
        sa5 = self.slider_sa5.GetValue()
        sa6 = self.slider_sa6.GetValue()
        sa7 = self.slider_sa7.GetValue()
        
        st1 = self.slider_st1.GetValue()
        st2 = self.slider_st2.GetValue()
        st3 = self.slider_st3.GetValue()
        st4 = self.slider_st4.GetValue()
        st5 = self.slider_st5.GetValue()
        st6 = self.slider_st6.GetValue()
        st7 = self.slider_st7.GetValue()
        
        tt1 = self.slider_tt1.GetValue()
        tt2 = self.slider_tt2.GetValue()
        tt3 = self.slider_tt3.GetValue()
        tt4 = self.slider_tt4.GetValue()
        tt5 = self.slider_tt5.GetValue()
        tt6 = self.slider_tt6.GetValue()
        tt7 = self.slider_tt7.GetValue()
        
        ct1 = self.slider_ct1.GetValue()
        ct2 = self.slider_ct2.GetValue()
        ct3 = self.slider_ct3.GetValue()
        ct4 = self.slider_ct4.GetValue()
        ct5 = self.slider_ct5.GetValue()
        ct6 = self.slider_ct6.GetValue()
        ct7 = self.slider_ct7.GetValue()
        
        time = self.slider_time.GetValue()
        
        # Run the neural network and get a 2D numpy array
        array = run_jCNN(sa1, sa2, sa3, sa4, sa5, sa6, sa7,
                         st1, st2, st3, st4, st5, st6, st7,
                         tt1, tt2, tt3, tt4, tt5, tt6, tt7,
                         ct1, ct2, ct3, ct4, ct5, ct6, ct7, time)
        
        # Convert numpy array to PIL Image and then to wx.Image
        img = Image.fromarray(array, 'L')  # 'L' mode for grayscale
        wx_img = wx.Image(img.size[0], img.size[1])
        wx_img.SetData(img.convert("RGB").tobytes())
        
        # Update the displayed image
        self.image_panel.SetBitmap(wx.Bitmap(wx_img))
        self.Refresh()

app = wx.App(False)
frame = MyFrame(None)
frame.Show(True)
app.MainLoop()
