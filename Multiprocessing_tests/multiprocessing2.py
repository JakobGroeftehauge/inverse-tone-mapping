#%%writefile test.py
import multiprocessing
import queue
import time
import ffmpeg 
from progressbar import progressbar
import numpy as np
import cv2
import ctypes
from time import sleep
import logging
import multiprocessing_logging
import argparse

def create_logger(name): 
    logging.basicConfig(filename=name, format='%(asctime)s:%(msecs)03d: %(levelname)s: %(name)s: %(message)s',level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    return logger


def setup_decoder(input_pth): 
    decoder = (ffmpeg.input(input_pth).output('pipe:', format='rawvideo', loglevel='quiet', pix_fmt='rgb24').run_async(pipe_stdout=True))
    return decoder

def setup_encoder(output_pth, width, height, fps): 
    encoder = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb48le', s='{}x{}'.format(width, height), framerate=fps, color_primaries='bt709', color_trc='smpte2084', colorspace='bt709') 
        .output(output_pth, pix_fmt='yuv420p10le', loglevel='quiet', **{'crf':0, 'c:v': 'libx265', 'x265-params': 'keyint=25:bframes=2:vbv-bufsize=50000:vbv-maxrate=50000:hdr-opt=1:no-open-gop=1:hrd=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,100):max-cll=1000,300'})
        .overwrite_output()
        .run_async(pipe_stdin=True))
    return encoder

def get_index_indicator(ind):
  while True:
    tmp_arr = np.frombuffer(ind, dtype=ctypes.c_long)
    for idx, i in enumerate(tmp_arr): 
      if i == 0: 
        ind[idx] = 1
        return idx
    #sleep(0.01)
  return 

def sRGB2linear(img):
    img = img / 255
    return np.where(img <= 0.04045, img / 12.92, np.power((img+0.055) / 1.055, 2.4))

def transformPQ(arr): 
    L = 1000.0 #max Luminance
    m = 78.8438
    n = 0.1593
    c1 = 0.8359
    c2 = 18.8516
    c3 = 18.6875
    Lp = np.power(arr/L, n)
    return np.power((c1 + c2*  Lp) / (1 + c3*Lp), m)    
  

def video_data(filename): 
    print("filename: ", filename)
    probe = ffmpeg.probe(filename)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = int(video_stream['r_frame_rate'].split('/')[0])
    num_frames= int(video_stream['nb_frames'])
    return [width, height], fps, num_frames
    

def run(d, ind, num, decQ, encQ):  
    dec = multiprocessing.Process(target=pre_process, args=(d,ind, num, decQ, ))
    dec.start()
    net = multiprocessing.Process(target=LANet, args=(d,num, encQ, decQ, ))
    net.start()
    enc = multiprocessing.Process(target=post_process, args=(d, num, ind, encQ, ))
    enc.start()
    
    dec.join()
    net.join()
    enc.join()

    return 


def pre_process(d, ind, num, decQ): 
    logger = create_logger(d["logger_name"])
    logger.info('Pre-Process/decode process started')
    decoder = setup_decoder(d["input_pth"])
    for i in progressbar(range(d["n_frames"])):
      logger.debug("Frame decoding initiated")
      in_bytes = decoder.stdout.read(d["size"])
      logger.debug('Frame decoded')
      img = np.frombuffer(in_bytes, np.uint8).reshape(d["arr_shape"])
      img = np.maximum(1.0, img)
      img = sRGB2linear(img)
      img = img * 2 - 1
      img = img.astype(np.float32)
      idx = get_index_indicator(ind)
      #print("max preprocess: ", img.max())

      np.copyto(num[idx], img)
      decQ.put(idx)
      logger.debug("Index added to decodeQueue")
    
    decQ.put(None)
    decoder.wait()
    return 

def post_process(d, num, ind, encQ):
    logger = create_logger(d["logger_name"])
    logger.info('Post-Process/encode process started')
    encoder = setup_encoder(d["output_pth"], d["width"], d["height"], d["fps"])
    while True:
      idx = encQ.get() 
      logger.debug("Item retrieved from encodeQueue")
      if idx is None: 
        break
      img = np.frombuffer(num[idx], dtype=np.float32).reshape(d["arr_shape"])

      #img = np.asarray(num[idx*d["size"]:idx*d["size"] + d["size"]]).reshape(d["arr_shape"])
      img = np.exp(img)
      img = transformPQ(img * d["sc"])
      img = img * 65535
      logger.debug('Write to encoder initiated')
      encoder.stdin.write(img.astype(np.uint16).tobytes())
      logger.debug('Write to encoder finshed')
      ind[idx] = 0

    logger.debug("Initiated closing of encoder")
    encoder.stdin.close()
    encoder.wait()
    logger.debug("Encoder closed")
    return 


def LANet(d, num, encQ, decQ): 
    #Input: RGB-image [h, w, 3]
    #Output: PQ encoded RGB-image
    import time
    logger = create_logger(d["logger_name"])
    logger.info('LANet process started')
    if not d["disable_onnx"]:
      import onnxruntime as onnxrt
      session = onnxrt.InferenceSession(d["model_pth"], None, providers=d["providers"])
    while True: 
      logger.debug("decQ size: {} encQ size: {}".format(decQ.qsize(), encQ.qsize()))
      idx = decQ.get()
      if idx is None: 
        break
      frame = np.frombuffer(num[idx], dtype=np.float32).reshape([1] + d["arr_shape"])
      #print("max lanet: ", frame.max())

      logger.debug('Inference started')
      if d["disable_onnx"]:
        output = frame.reshape(d["arr_shape"])
        time.sleep(0.5)
      else:
        onnx_inputs = {session.get_inputs()[0].name: frame}
        onnx_output = session.run(None, onnx_inputs)
        output = onnx_output[0].astype(np.float32)

      logger.debug('Inference Stopped')
      #output = frame.astype(np.float32)
      np.copyto(num[idx], output)
      #num[idx] = output.ravel()
      #num[idx*d["size"]: idx*d["size"] + d["size"]] = output.ravel()
      encQ.put(idx)

    encQ.put(None)
    return  

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-file', dest="input", type=str, help='Path to HDR image to transform')
    parser.add_argument('-output-filename', dest="output", type=str, help='Path to store frames')
    parser.add_argument('-model', type=str, help="path to inference ONNX-model")
    parser.add_argument('-logging-file', dest="log", default="debug_process.log", type=str, help="name of logging file" )
    parser.add_argument('--disable-onnx', dest='disable_onnx', type=bool, default=False, help="disable onnx for debugging on computer wit limited resources")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':

  #multiprocessing.set_start_method('spawn')
  opt = parse_opt(True)

  multiprocessing_logging.install_mp_handler()
  logger = create_logger(opt.log)
  logger.debug("Start of program2")
  
  data = multiprocessing.Manager().dict()
  print("Model: ", opt.model, " input: ", opt.input, " output: ", opt.output)
  data["disable_onnx"] = opt.disable_onnx
  data["logger_name"] = opt.log
  data["model_pth"] = opt.model
  data["sc"] = 20 #pre-scaling
  data["output_pth"] = opt.output
  data["input_pth"] = opt.input

  data["providers"] = ["TensorrtExecutionProvider", "CUDAExecutionProvider", 'CPUExecutionProvider']
  [data["width"], data["height"]], data["fps"], data["n_frames"] = video_data(opt.input)

  decodeQueue = multiprocessing.Queue(maxsize=3)
  encodeQueue = multiprocessing.Queue(maxsize=3)

  data["size"] = int(data["width"] * data["height"]*3) # size of array
  data["N_numbers"] = int(6)
  data["arr_shape"] = [data["height"], data["width"], 3]

  data_arrays = []
  data_arrays_np = []
  for i in range(data["N_numbers"]): 
      arr = multiprocessing.RawArray(ctypes.c_float, int(data["size"]))
      data_arrays.append(arr)
      data_arrays_np.append(np.frombuffer(arr, dtype=np.float32).reshape(data["arr_shape"]))

  
  indicator= multiprocessing.Array(ctypes.c_long,[0] * data["N_numbers"], lock=False)

  t1 = time.time()
  run(data, indicator, data_arrays_np, decodeQueue, encodeQueue)
  t2 = time.time()
  t = t2-t1
  print("Avg. Per Image: ", str(t/data["n_frames"]))
  print("Complete video: ", str(t))
