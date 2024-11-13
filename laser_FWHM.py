import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageChops 
import math
from scipy.optimize import curve_fit
from scipy.signal import chirp, find_peaks, peak_widths
from pathlib import Path


from mpl_toolkits.mplot3d import Axes3D

from pylab import *

def gaus(x, a, x0, sigma):
    '''
     1D gaussian
    '''
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))



def gaussian_image(x, y, a=0.0, x0=0.0, y0=0.0, sx=1.0, sy=1.0):
    '''
    2d gaussian image
    '''
    return a * (np.exp(-0.5 * ((x - x0) / sx) ** 2) *
                np.exp(-0.5 * ((y - y0) / sy) ** 2))


def get_image(image_name):
    
    string_image_name = str(image_name)
    signal_img = Image.open(string_image_name).convert("L")

    return signal_img

# Function to make sure that users input the right values for image
def input_number(message):
    while True:
        try:
            userinput = int(input(message))
        except ValueError:
            print("Not an integer! Try again.")
            continue
        if userinput > 8:
            print("number to high must be between 1-8")
        else:
            return userinput
        
def  one_D_slice(numpydata_cropped, size_of_slice):
     y_slice = numpydata_cropped[size_of_slice, :]
   
     x_slice = numpydata_cropped[:, size_of_slice]
            
     return  y_slice , x_slice



# image Parameter
thershold_pixcel_value = 20

#CCD size Photek 1.4um and TORCH is 5.5um 
CCD_pix_size = 5.5

#change this to "TORCH" and "png" to use TORCH data  ("bmp" for photek)  some are now tif

image_name = "image_3.34sm_0.1_from_focus"
image_path = Path("C:/Users/lexda/Desktop/")
images = list(image_path.rglob("*from_focus*.png"))
print(images)
for image_name in images:
    image_type = "png"
    image_number = 1


    # size of the image
    img_size_of_Image = int(15)
    print(image_name)
    # halving  as this is adding in the y either side of the brightest pixcel
    img_pix_size = int(img_size_of_Image / 2)

    size_of_slice = img_pix_size



    # Gets the image and converts it to grayscale
    signal_img = get_image(image_name)

    numpydata = np.asarray(signal_img)

    #if hot pixcel is seen run this code change number if image number is not one 
    if image_number == 1: 
        ymax = []
        xmax = []
        
        # finds the pixels that are bigger then a thershold
        for i in range(len(numpydata)):
            for j in range(len(numpydata[i])):
                if numpydata[i][j] > thershold_pixcel_value:
                    ymax.append(j)
                    xmax.append(i)
        
        top_x = xmax[2] - img_pix_size
        top_y = ymax[2] - img_pix_size

        bottom_x = xmax[2] + img_pix_size
        bottom_y = ymax[2] + img_pix_size
        
    else:  
        x_max, y_max = np.unravel_index(np.argmax(numpydata, axis=None), numpydata.shape)

        top_x = x_max - img_pix_size
        top_y = y_max - img_pix_size

        bottom_x = x_max + img_pix_size
        bottom_y = y_max + img_pix_size
        

    #print(y_max)


    # crops the image to highlight spot in the middle
    cropped_signal_img = signal_img.crop((top_y, top_x, bottom_y, bottom_x))
    numpydata_cropped = np.array(cropped_signal_img)

    plot1 = plt.subplot()
    plot2 = plt.subplot()


    plot1.imshow(numpydata_cropped)
    plot1.set_xlabel("x_coordinate(pixels)")
    plot1.set_ylabel("Y_coordinate(pixels)")

        #1D arry traveling arrcoss the y and x axis 
    y_slice , x_slice = one_D_slice(numpydata_cropped, size_of_slice)

    # range of y goes across 
    x = range(len(y_slice))
   

    # popt returns the best fit values for parameters of the given model (func)
    # pcov is used to make popt in range
    popt, pcov = curve_fit(gaus, x, x_slice, p0=[thershold_pixcel_value, (len(y_slice)/2)+1, 2])

    # size of dis Gauss when starting from 0 hence -1
    Gauss_x_array_size = len(y_slice)

    # X range the Gaussian
    x_gauss = np.linspace(0, Gauss_x_array_size, 200)
    
    # The Gaussian Y values calculated from function.
    y_gauss = gaus(x_gauss, popt[0], popt[1], popt[2])


    popt2, pcov2 = curve_fit(gaus, x, x_slice)
    y_gauss_x_slice = gaus(x_gauss, popt2[0], popt2[1], popt2[2])
    x_c = np.linspace(0, y_slice.size, y_slice.size)
    #plot2.scatter(x_c,x_slice)
    #plot2.plot(x_gauss,y_gauss)
    #plot2.set_xlabel("x_coordinate(pixels)")
    #plot2.set_ylabel("greyscale")
    plt.show()