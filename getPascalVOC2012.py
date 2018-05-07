import os
import numpy as np
import os, cv2
import pickle

def normalize_x(image): return image/127.5 - 1
def normalize_y(image): return image/255
def denormalize_y(image): return (image*255).astype(int)

def load_X_files(image_files,IMAGE_SIZE):
    """
    imagefiles [string] filenames
    IMAGE_SIZE tuple  width,height  
    """
    images = np.zeros((len(image_files), IMAGE_SIZE[0], IMAGE_SIZE[1], 3), np.float32)
    for i, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
            image = cv2.resize(image, IMAGE_SIZE)
            images[i] = normalize_x(image)
    return images, image_files
    
def load_X(fdir,IMAGE_SIZE):

    image_files=os.listdir(fdir).sort()
    return load_X_flies(image_files,IMAGE_SIZE)    
    
def load_Y_files(image_files,IMAGE_SIZE):

    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, IMAGE_SIZE)
        image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)
    return images
    
def load_Y(fdir,IMAGE_SIZE):
    image_files = os.listdir(fdir)
    image_files.sort()
    return load_Y_files(image_files,IMAGE_SIZE)

def genVOC2012filelist(ddir,istrain=True,rate=0.9):
        yfiles=os.listdir(ddir+os.sep+'SegmentationObject' + os.sep)
        yfiles.sort()
        xfiles=[ ddir+os.sep+'JPEGImages'+os.sep+os.path.splitext(os.path.basename(y))[0]+".jpg" for y in yfiles]
        yfiles=[ ddir+os.sep+'SegmentationObject'+os.sep+y for y in yfiles]
        
        if(istrain):
            xfiles=xfiles[:int(rate*len(yfiles))]
            yfiles=yfiles[:int(rate*len(yfiles))]
        else:
            xfiles=xfiles[int((1-rate)*len(yfiles)):]
            yfiles=yfiles[int((1-rate)*len(yfiles)):]
            
        print("yfiles=%d"%len(yfiles))
        print("xfiles=%d"%len(xfiles))            
        return xfiles,yfiles
