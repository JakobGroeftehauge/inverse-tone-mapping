import numpy as np

def preprocess_image(img):
    img = np.maximum(1.0, img)
    img = sRGB2linear(img)
    img = img * 2 - 1
    return img.astype(np.float32)

def postprocess_image(img, sc, max_luminance=1000):
    img = np.exp(img)
    img = transformPQ(img * sc, MAX_LUM=max_luminance)
    img = img * 65535
    return img

def sRGB2linear(img):
    img = img / 255
    return np.where(img <= 0.04045, img / 12.92, np.power((img+0.055) / 1.055, 2.4))

def transformPQ(arr, MAX_LUM=1000.0): 
    L = MAX_LUM #max Luminance
    m = 78.8438
    n = 0.1593
    c1 = 0.8359
    c2 = 18.8516
    c3 = 18.6875
    Lp = np.power(arr/L, n)
    return np.power((c1 + c2*  Lp) / (1 + c3*Lp), m)    
  