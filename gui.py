import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
import random, cv2,PIL,os
from glob import glob
import PIL.Image, PIL.ImageTk
from keras_segmentation.predict import predict_multiple
from keras_segmentation.models.model_utils import load_model
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.models.unet import vgg_unet

class App:
    def __init__(self):
        self.dir_images = ''
        self.model_file = ''
        self.image=''
        self.mask = ''
        self.window = tk.Tk()
        self.window.title("Semantic segmentation demo")
        self.window.geometry("500x700")

        self.dir_button = ttk.Button(
            self.window,
            text='Select a folder with images',
            command=self.select_dir
        )
        self.dir_button.pack(anchor=CENTER, expand=False)

        self.open_button = ttk.Button(
            self.window,
            text='Select a File of model',
            command=self.open_file
        )
        self.open_button.pack(anchor=CENTER, expand=False)

        self.predict_button = ttk.Button(
            self.window,
            text='Get predictions in output folder',
            command=self.predict_folder)
        self.predict_button.pack(anchor=CENTER, expand=False)

        self.window.mainloop()

    def select_dir(self):

        self.dir_images = filedialog.askdirectory()
        print(self.dir_images)

        images = glob(self.dir_images + "/*.*")
        self.image = random.choice(images)
        print(self.image)
        im = PIL.Image.open(self.image)
        im.thumbnail((600, 400), PIL.Image.ANTIALIAS)
        ph = PIL.ImageTk.PhotoImage(im)
        self.canvas = Canvas(self.window, width=ph.width(), height=ph.height())
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=W, image=ph)
        self.canvas.image = ph


    def open_file(self):
        # file type
        filetypes = (
            ('model h5 files', '*.h5'),
            ('All files', '*.*')
        )
        # show the open file dialog
        f = filedialog.askopenfile(filetypes=filetypes)
        # read the text file and show its content on the Text

        self.model_file = f.name
        print(self.model_file)

    def predict_folder(self):
        self.model = vgg_unet(n_classes=256, input_height=1024, input_width=1024)
        old_model = load_model(self.model_file)
        transfer_weights(old_model, self.model)
        self.model.predict_multiple(
                inp_dir = self.dir_images,
                out_dir = "outputs")

        self.mask = 'outputs/'+os.path.basename(os.path.normpath(self.image))
        im = PIL.Image.open(self.mask)
        im.thumbnail((600, 400), PIL.Image.ANTIALIAS)
        ph = PIL.ImageTk.PhotoImage(im)
        self.canvas = Canvas(self.window, width=ph.width(), height=ph.height())
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=W, image=ph)
        self.canvas.image = ph




App()
