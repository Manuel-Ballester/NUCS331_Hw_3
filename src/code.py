import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib
import cv2

savedir = 'output/'

def save_fig_as_png(figtitle):
    '''
    Saves the current figure into the output folder
    The figtitle should not contain the ".png".
    This helper function shoudl be easy to use and should help you create the figures 
    needed for the report
    
    The directory where images are saved are taken from savedir in "Code.py" 
    and should be included in this function.
    
    Hint: The plt.gcf() might come in handy
    Hint 2: read about this to crop white borders
    https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    
    '''
    fig = plt.gcf()
    path = savedir+figtitle+".png"
    plt.savefig(path,bbox_inches='tight', pad_inches=0)
    plt.show()

    return 0

def load_imgs(flash,noflash):
    """
    This function loads in your images into a numpy array, it takes two arguments
    flash and noflash, which should be paths to your images; these will be the images
    you will be working with during the assignment. 
    
    args:
        path (str): the path to the folder of the images, flash and no flash.
        
    outputs:
        img_noisy (np.ndarray) dtype = float32 : this is a numpy array containing both images
        img_flash (np.ndarray) dtype = float32 : this is a numpy array containing both images

    """
    img_noisy=np.asarray(Image.open(noflash))
    img_noisy=img_noisy.astype(np.float32)
    img_noisy=(img_noisy-np.min(img_noisy))/(np.max(img_noisy)-np.min(img_noisy))
    
    img_flash=np.asarray(Image.open(flash))
    img_flash=img_flash.astype(np.float32)
    img_flash=(img_flash-np.min(img_flash))/(np.max(img_flash)-np.min(img_flash))
    
    return img_noisy,img_flash



def plot_imgs(img_noisy,img_flash):
    """
    plots the images in a subplot (left and right)
    
    make sure to have a large fontsize
    
    you can use plt.figure(figsize=(X,Y)) to make a larger figure
    so that the image doesn't appear too small
    
    
    args:
        img_noisy (np.ndarray): the noisy image
        img_flash (np.ndarray): the flash image

    outputs:
        void: plots the images!
    """
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
    fig.suptitle("Images")
    ax = axes.ravel()

    ax[0].imshow(img_noisy, cmap='gray')
    ax[0].set_title("Flash")

    ax[1].imshow(img_flash, cmap='gray')
    ax[1].set_title("Noflash")

    plt.tight_layout()
    plt.show()


def imshow_single_bilateral_filter(img_noisy,img_bilateral,xmin=600,xmax=700,ymin=400,ymax=500):
    """
    
    Visualizes the effect of the bilateral filter.
    
    The top row will show the full image (left img_noisy, right the bilateral)
    The bottom row will show a cropped/zoomed version where the input dimensions
    are defined by the function argument
    
    To visualize which part you are cropping you should overlay a rectangle
    over the region that you are corrping.
    
    You can e.g. use this resource to learn how to create a rectangle over a plot:
    https://www.delftstack.com/howto/matplotlib/how-to-draw-rectangle-on-image-in-matplotlib/
    
    
    args:
        img_noisy(np.array): noisy image
        img_bilateral(np.array): the bilateral filtered image
        xmin(int): the left start index for crop
        xmax(int): the right end index for crop
        ymin(int): the top index for crop
        ymax(int): the bottom index for crop
    
    """
    img_noisy_crop=img_noisy[xmin:xmax,ymin:ymax] # We select the interest area 
    img_bilateral_crop=img_bilateral[xmin:xmax,ymin:ymax] # We select the interest area 
    
    plot_imgs(img_noisy_crop,img_bilateral_crop)

    
def bilateral_filter(img, sigma_r, sigma_s):
    """
    computes a bilateral filter on a rgb image.
    
    Bilateral filtering is a edge-detecting, noise-reducing, smoothing filter 
    for images. It replaces the pixels with an average of the nearby pixels, 
    which is dependent on a Gaussian kernel in the spatial domain (σ_s) and 
    also the range (intensity of pixels) domain (σ_r)
    
    this can be implemented by yourself or an external party code (easier, recommended)
    
    Note: Not every bilateral function works on RGB images. You might need to calculate
    it seperately for each channel
    
    args:
        img_color (np.ndarray): rgb image
                
        sigma_r (float): the standard deviation of the Guassian kernel in the range (intensity of pixel) 
            domain.
            
        sigma_s (float): the standard deviation of the Guassian kernel in the spatial domain. 
        
    output:
        bilteral_filter (np.ndarray): the filtered image channel, with edges more pronounced,
        noise-reduced, and smoothed.
    """
    bilateral = cv2.bilateralFilter(img, 15, sigma_r, sigma_s)
    bilateral=(bilateral-np.min(bilateral))/(np.max(bilateral)-np.min(bilateral))
    bilateral=bilateral.astype(np.float32)
    
    return bilateral


def get_range_bilateral_filter():
    """
    Returns a few values for both the spatial and range parameter
    for the bilateral filter.
    
    This function doesn't do any computing. 
    You can e.g. use the following parameter, but they might not be the best
    
    range = 0.05,0.1,0.2,0.3,0.4,0.5
    spatial = 1,2,4,8,16
    
    output:
        range_of_sigma_r(1d-array)
        range_of_sigma_s(1d-array)
    """
    
    range_of_sigma_r=np.array([0.1,0.2,0.4,0.8])
    range_of_sigma_s=np.array([2,4,8,16])
    
    return range_of_sigma_r,range_of_sigma_s


def filter_all_images_with_bilateral(img, range_of_sigma_r, range_of_sigma_s):
    """
    
    Applies the bilateral filter function (that you've implemented) 
    for each combination in range_of_sigma_r and range_of_sigma_s
    
    E.g. if len(range_of_sigma_r) = 5 and len(range_of_sigma_s) = 6
    you have to compute 30 bilateral filtered images
    
    Return the result in large 5 dimensional array.
    
    args:
        img(np.array): image to be filtered
        range_of_sigma_r(1d-array): range sigma array for bilateral filter
        range_of_sigma_s(1d-array): spatial sigma array for bilateral filter
    output: 
        filtered_imgages(np.array): filtered output with shape = (numX,numY,3,len(range),len(spatial))
    """
    # this sets up a empty numpy array to store the newly bilateral filtered images
    output=np.zeros((int(img.shape[0]),int(img.shape[1]),3,len(range_of_sigma_r),len(range_of_sigma_s)))

    for i in range(len(range_of_sigma_r)):
        for j in range(len(range_of_sigma_s)):
            output[:,:,:,i,j]=bilateral_filter(img, range_of_sigma_r[i], range_of_sigma_s[j])
            name="filter_all_images_with_bilateral"+str(i)+"_"+str(j)
            #save_fig_as_png(name)
    
    output=output.astype(np.float32)
    
    return output
            
            
def plot_bilateral_filter(filtered_images, range_of_sigma_r, range_of_sigma_s, xMin = None, xMax = None, yMin = None, yMax = None):
    """
    plots the bilateral filtered images with different ranges of sigma_r and sigma_s.
    
    args:
        img (np.ndarray): image to put bilateral filter on.
        range_of_sigma_r (lst) dtype = float: list of the desired sigma_r to be plotted.
        range_of_sigma_s (lst) dtype = float: list of the desired sigma_s to be plotted.
        xMin (int): Min x you want to plot (used to zoom in and look at a specific region of the image)
        xMax (int): Max x you want to plot (used to zoom in and look at a specific region of the image)
        yMin (int): Min y you want to plot (used to zoom in and look at a specific region of the image)
        yMax (int): Max y you want to plot (used to zoom in and look at a specific region of the image)
    
    output:
        void, but should be subplots of the different results due to sigma_r and sigma_s.
    """

    #makes plot, and sets up the number of steps to change for sigma, and max(A), which is the max pixel value of the image.
    fig, axes = plt.subplots(nrows=len(range_of_sigma_r), ncols=len(range_of_sigma_s),figsize=(20,20))
    fig.suptitle("Images")
    ax = axes.ravel()    
    for i in range(len(range_of_sigma_r)):
        for j in range(len(range_of_sigma_s)):
            ax[j+i*len(range_of_sigma_s)].imshow(filtered_images[xMin:xMax,yMin:yMax,:,i,j], cmap='gray')
            title="r="+str(range_of_sigma_r[i])+"s="+str(range_of_sigma_s[j])
            ax[j+i*len(range_of_sigma_s)].set_title(title)
            
    plt.tight_layout()
    plt.show()
    

def visualize_detail_layer_single(img_flash,filtered,detail,eps,xmin=300,xmax=400,ymin=400,ymax=500):
    '''
    
    Visualizes the detail layer.
    
    This is very similair to the function "imshow_single_bilateral_filter",
    however now we have 3 images on the top and bottom.
    
    HINT: The detail image might not be scaled between 0 and 1.
    In order to see the full detail layer image you need to normalize 
    it with the maximum value to be between 0 and 1.

    
    To visualize which part you are cropping you should overlay a rectangle
    over the region that you are corrping.
    
    You can e.g. use this resource to learn how to create a rectangle over a plot:
    https://www.delftstack.com/howto/matplotlib/how-to-draw-rectangle-on-image-in-matplotlib/
    
    
    args:
        img_flash (np.ndarray): input_image
        filtered (np.ndarray): bilateral filtered image
        detail (np.array): the detail image
        eps (float): The epsilon used in the formula
        xMin (int): Min x you want to plot (used to zoom in and look at a specific region of the image)
        xMax (int): Max x you want to plot (used to zoom in and look at a specific region of the image)
        yMin (int): Min y you want to plot (used to zoom in and look at a specific region of the image)
        yMax (int): Max y you want to plot (used to zoom in and look at a specific region of the image)

    '''
    detail=(detail-np.min(detail))/(np.max(detail)-np.min(detail))
    
    plt.imshow(img_flash)
    ax = plt.gca()

    rect = patches.Rectangle((xmin,ymin),
                     xmax-xmin,
                     ymax-ymin,
                     linewidth=2,
                     edgecolor='cyan',
                     fill = False)

    ax.add_patch(rect)
    plt.show()
    
    plt.imshow(detail[ymin:ymax,xmin:xmax])
    plt.show()
    

    
def plot_details(imgs, range_of_sigma_r, range_of_sigma_s, xMin = None, xMax = None, yMin = None, yMax = None):
    """
    plots the bilateral filtered images with different ranges of sigma_r and sigma_s.
    
    args:
        img (np.ndarray): image to put bilateral filter on.
        range_of_sigma_r (lst) dtype = float: list of the desired sigma_r to be plotted.
        range_of_sigma_s (lst) dtype = float: list of the desired sigma_s to be plotted.
        xMin (int): Min x you want to plot (used to zoom in and look at a specific region of the image)
        xMax (int): Max x you want to plot (used to zoom in and look at a specific region of the image)
        yMin (int): Min y you want to plot (used to zoom in and look at a specific region of the image)
        yMax (int): Max y you want to plot (used to zoom in and look at a specific region of the image)
    
    output:
        void, but should be subplots of the different results due to sigma_r and sigma_s.
    """

    #makes plot, and sets up the number of steps to change for sigma, and max(A), which is the max pixel value of the image.
    raise NotImplementedError


    
    
def calc_detail_layer(img,img_filtered,eps):
    """
    Take a flash and denoised flash image and calculates the detail layer 
    acoording to this function:
    
    img_detail = \frac{F+\epsilon}{F_{d}+\epsilon}
    
    
    args:
        img(np.array): RGB flash image
        img_filtered(np.array): Bilateral filtered RGB flash image
        eps: epsilon offset used in formula
    output:
        detail_layer(np.array) detail layer in RGB format
    """
    
    return (img+eps)/(img_filtered+eps)


def calc_detail_layer_from_scratch(img,sigma_r,sigma_s,eps):
    """
    Calculate the detail layer from a flash_image
    using the following formula:
    
    img_detail = \frac{F+\epsilon}{F_{d}+\epsilon}
    where $\epsilon$ is a small number (e.g. 0.1 or 0.2)

    F : input flash image
    F_d: bilateral filtered image
    
    HINT: Implement the function calc_detail_layer(img,img_filtered,eps
    and call this function inside calc_detail_layer_from_scratch
    
    args:
        img(np.array): RGB image flash
        sigma_r(float): range sigma for bilateral filter
        sigma_s(float): spatial sigma for bilateral filter
        eps(float): cut-off value for filtering
    
    output:
        detail(nd.array): the filtered image
        filtered(nd.array): the bilateral filtered image
    """
    
    img_filtered=bilateral_filter(img, sigma_r, sigma_s)
    img_detail=calc_detail_layer(img,img_filtered,eps)
    
    return img_detail, img_filtered
    

    
def visualize_detail_layer(img,xmin=None,xmax=None,ymin=None,ymax=None):
    """
    
    Takes in a flash image and calcualates the detail layer for those parameters
    
    You'll have to define a set of parameter for eps, sigma_r and sigma_s
    
    It probably makes sense to fix one parameter (e.g. sigma_s) to a reasonable value
    and then vary only the other parameters
    
    I suggest calculating about 16 (4x4 images), otherwise it's hard to see the effect.
    
    You'll have to go through each combination in your parameters, calculate
    the detail layer using "calc_detail_layer_from_scratch" and then display it
    
    NOTE:
    1. Give a good title of each subfigure (large fontsize and descriptive)
    2. Use for loops to go through the parameters. Don't write each subplot manually
    3. Make the figure size large enough so that you can see something
    
    The xmin,xmax,ymin,ymax are either None or integer value.
    For easy implementation checkout the following command:
    img[xmin:xmax,ymin:ymax] - What happens if xmin is None ?! Numpy is quite smart,
    you don't even need an if-statement if you do it correctly!
    
    args:
        img(np.array): The image you use to calculate the detail layer
        xmin,xmax,ymin,ymax(int): Same definitions as before
        if None show the whole image, if they are not Non you have to crop them
    
    """
    range_of_sigma_r,range_of_sigma_s=get_range_bilateral_filter()
    eps=0.1
    
    fig, axes = plt.subplots(nrows=len(range_of_sigma_r), ncols=len(range_of_sigma_s),figsize=(20,20))
    fig.suptitle("Details")
    ax = axes.ravel()    
    for i in range(len(range_of_sigma_r)):
        for j in range(len(range_of_sigma_s)):
            
            img_detail, img_filtered=calc_detail_layer_from_scratch(img,range_of_sigma_r[i],range_of_sigma_s[j],eps)
            img_detail=(img_detail-np.min(img_detail))/(np.max(img_detail)-np.min(img_detail))
            img_filtered=(img_filtered-np.min(img_filtered))/(np.max(img_filtered)-np.min(img_filtered))
    
            ax[j+i*len(range_of_sigma_s)].imshow(img_detail[xmin:xmax,ymin:ymax,:], cmap='gray')
            title="r="+str(range_of_sigma_r[i])+"s="+str(range_of_sigma_s[j])
            ax[j+i*len(range_of_sigma_s)].set_title(title)
            
    plt.tight_layout()
    plt.show()


def visualize_fusion(img_flash,img_flash_filtered,img_noisy,img_noisy_filtered,
                     fused,detail,eps,xmin=None,xmax=None,ymin=None,ymax=None):
    """
    Visualizes the complete pipeline in a large subfigure
    
    Function arguments are the same as in the functions above
    
    xmin,xmax,... have the same purpose as before.
    
    """
    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(20,20))
    fig.suptitle("complete pipeline")
    ax = axes.ravel()    
 
    img_flash_filtered=(img_flash_filtered-np.min(img_flash_filtered))/(np.max(img_flash_filtered)-np.min(img_flash_filtered))
    ax[0].imshow(img_flash_filtered[xmin:xmax,ymin:ymax,:], cmap='gray')        
    ax[0].set_title("Flash filtered")
    
    img_flash=(img_flash-np.min(img_flash))/(np.max(img_flash)-np.min(img_flash))
    ax[1].imshow(img_flash[xmin:xmax,ymin:ymax,:], cmap='gray')        
    ax[1].set_title("Flash")
    
    img_noisy=(img_noisy-np.min(img_noisy))/(np.max(img_noisy)-np.min(img_noisy))
    ax[2].imshow(img_noisy[xmin:xmax,ymin:ymax,:], cmap='gray')        
    ax[2].set_title("Noflash")
    
    img_noisy_filtered=(img_noisy-np.min(img_noisy_filtered))/(np.max(img_noisy_filtered)-np.min(img_noisy_filtered))
    ax[3].imshow(img_noisy_filtered[xmin:xmax,ymin:ymax,:], cmap='gray')        
    ax[3].set_title("NoFlash filtered")
    
    detail=(detail-np.min(detail))/(np.max(detail)-np.min(detail))
    ax[4].imshow(detail[xmin:xmax,ymin:ymax,:], cmap='gray')        
    ax[4].set_title("detail")
    
    fused=(fused-np.min(fused))/(np.max(fused)-np.min(fused))
    ax[5].imshow(fused[xmin:xmax,ymin:ymax,:], cmap='gray')        
    ax[5].set_title("fused")
            
    plt.tight_layout()
    plt.show()

    
    

    
    
def fuse_flash_no_flash(img_noisy,img_flash,sigma_r,sigma_s,eps):
    """
    
    Calculates the pipeline to do the flash-no-flash photography
    
    Hint: Use the subfunction that you've already implemented in this assignment
    
    args:
        img_nosy(np.ndarray): noisy image
        
        img_flash(np.ndarray): flash image
        
        sigma_r(float): range sigma for bilateral
        sigma_s(float): spatial sigma for bilateral
        eps(float): cut off value for detail layer correction
        
    outputs:
        fused_image (np.ndarray) fused image
        detail (np.ndarray) the detail layer
        img_flash_filtered (np.ndarray)  the filtered flash image
        img_noisy_filtered (np.ndarray) the noisy image filtered
    """
    
    detail, img_noisy_filtered=calc_detail_layer_from_scratch(img_noisy,sigma_r,sigma_s,eps)
    detail=(detail-np.min(detail))/(np.max(detail)-np.min(detail))
    img_flash_filtered=bilateral_filter(img_flash, sigma_r, sigma_s)
    
    fused=detail*img_flash_filtered
    
    return fused, detail, img_flash_filtered, img_noisy_filtered


def complete_pipeline(foldername,sigma_r,sigma_s,eps):
    """
    basically does the same as fuse_flash_no_flash, so call this function inside.
    
    However, now we're not giving the images as input, but we're loading
    the image directly from the foldername.
    
    You can use your load_imgs function for this.
    
    
    args:
        foldername (str): foldername
        sigma_r(float)
        sigma_s(float)
        eps(float)
    
    outputs:
        img_nosy
        img_flash
        fused_image
        detail
        img_flash_filtered
        img_noisy_filtered
    """
    name_flash=foldername+"_00_flash.jpg"
    name_noflash=foldername+"_01_noflash.jpg"
    img_noisy,img_flash=load_imgs(name_flash,name_noflash)
    
    fused, detail, img_flash_filtered, img_noisy_filtered=fuse_flash_no_flash(img_noisy,img_flash,sigma_r,sigma_s,eps)

    return img_noisy,img_flash, fused, detail, img_flash_filtered, img_noisy_filtered
    
    