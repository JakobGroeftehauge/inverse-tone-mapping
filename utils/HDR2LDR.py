import argparse
import cv2
import imutils

tonemap_params = {
    'method': None,
    'gamma': 1.0,
    'saturation': 1.0,
    'bias':0.85,
    'intensity':0.0,
    'light_adapt':1.0,
    'color_adapt':0.0,
    'scale':0.7,
    'scaling_factor':1.0}

hdr_global = None




def HDR2LDR(hdr, method, gamma, saturation, bias, intensity, light_adapt, color_adapt, scale, scaling_factor):
    # tone  map
    if method == 'drago':
        #print("Tonemaping using Drago's method ... ")
        tonemapDrago = cv2.createTonemapDrago(gamma, saturation, bias)
        ldrDrago = tonemapDrago.process(hdr)
        ldr = scaling_factor * ldrDrago

    elif method == 'reinhard':
        #print("Tonemaping using Reinhard's method ... ")
        tonemapReinhard = cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
        ldr = tonemapReinhard.process(hdr)

    elif method == 'mantiuk':
        #print("Tonemaping using Mantiuk's method ... ")
        tonemapMantiuk = cv2.createTonemapMantiuk(gamma, scale, saturation)
        ldrMantiuk = tonemapMantiuk.process(hdr)
        ldr = scaling_factor * ldrMantiuk

    else:
        #print("Tonemaping using default method... ")
        tonemap = cv2.createTonemap(gamma)
        ldr = tonemap.process(hdr)
        ldr = scaling_factor * ldr
    
    return ldr

def main(opt):
    global hdr_global
    #print(opt.HDR_input)
    # load Image 
    hdr_global = cv2.imread(opt.HDR_input, -1) # correct element size should be CV_32FC3
    method = 'drago' if opt.drago else 'reinhard' if opt.reinhard else 'mantiuk' if opt.mantiuk else None

    #ldr = HDR2LDR(hdr, method, opt.gamma, opt.saturation, opt.bias, opt.intensity, opt.light_adapt, opt.color_adapt, opt.scale, opt.scaling_factor)
    cv2.namedWindow('Tonemapping')
    cv2.createTrackbar('Gamma         ', 'Tonemapping', 0, 100, change_gamma)
    cv2.createTrackbar('Scaling Factor', 'Tonemapping', 0, 100, change_scaling_factor)

    if method == 'drago':
        cv2.createTrackbar('Saturation', 'Tonemapping', 0, 100, change_saturation)
        cv2.createTrackbar('Bias', 'Tonemapping', 0, 100, change_bias)

    if method == 'mantiuk':
        cv2.createTrackbar('Saturation', 'Tonemapping', 0, 100, change_saturation)
        cv2.createTrackbar('Scale', 'Tonemapping', 0, 100, change_scale)

    if method == 'reinhard':
        cv2.createTrackbar('Intensity', 'Tonemapping', 0, 100, change_saturation)
        cv2.createTrackbar('Light Adapt', 'Tonemapping', 0, 100, change_light_adapt)
        cv2.createTrackbar('Color Adapt', 'Tonemapping', 0, 100, change_color_adapt)


    tonemap_image()

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()

    cv2.imwrite(opt.save_name + ".jpg" , ldr * 255)
    return 


def change_gamma(new_val):
    change_params('gamma', new_val/10) 

def change_scaling_factor(new_val):
    change_params('scaling_factor', new_val/10) 

def change_saturation(new_val):
    change_params('saturation', new_val/10) 

def change_intensity(new_val):
    change_params('intensity', new_val/10) 

def change_scale(new_val):
    change_params('scale', new_val/10) 

def change_bias(new_val):
    change_params('bias', new_val/10) 

def change_color_adapt(new_val):
    change_params('color_adapt', new_val/10) 

def change_light_adapt(new_val):
    change_params('light_adapt', new_val/10) 


def change_params(name, value):
    global tonemap_params
    tonemap_params[name] = value
    tonemap_image()

def tonemap_image(): 
    global hdr_global
    ldr = HDR2LDR(hdr_global, tonemap_params['method'], tonemap_params['gamma'], tonemap_params['saturation'], tonemap_params['bias'], tonemap_params['intensity'], tonemap_params['light_adapt'], tonemap_params['color_adapt'], tonemap_params['scale'], tonemap_params['scaling_factor'])
    cv2.imshow('Tonemapping', imutils.resize(ldr, height=480))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--HDR_input', type=str, help='Path to HDR image to transform')
    parser.add_argument('--save_name', type=str, help='Path to HDR image to transform')
    parser.add_argument('--drago', action='store_true')
    parser.add_argument('--mantiuk', action='store_true')
    parser.add_argument('--reinhard', action='store_true')
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--saturation', default=1.0, type=float)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--bias', type=float)
    parser.add_argument('--intensity', type=float, default=0.0)
    parser.add_argument('--color_adapt', type=float, default=0.0)
    parser.add_argument('--light_adapt', type=float, default=1.0)
    parser.add_argument('--scaling_factor', type=float, default=1.0, help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)