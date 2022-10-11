import argparse
import cv2
import imutils
import os.path


def video2frames(videoCapture, output_dir): 
    counter = 0
    while videoCapture.isOpened(): 
        ret, frame = videoCapture.read()
    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        path = os.path.join(output_dir, str(counter) + '.jpg')
        print(path)
        cv2.imwrite(output_dir + '/' + str(counter) + '.jpg', frame)
        counter = counter + 1

    videoCapture.release()
    return 

def main(opt):
    video = cv2.VideoCapture(opt.input)
    video2frames(video, opt.output_dir)
    return 

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to HDR image to transform')
    parser.add_argument('--output_dir', type=str, help='Path to store frames')

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