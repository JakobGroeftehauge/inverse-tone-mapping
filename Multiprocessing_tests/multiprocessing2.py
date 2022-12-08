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
from dataclasses import dataclass




def create_logger(name): 
    logging.basicConfig(filename=name, format='%(asctime)s:%(msecs)03d: %(levelname)s: %(name)s: %(message)s',level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    return logger

def setup_decoder(input_pth): 
    decoder = (ffmpeg.input(input_pth).output('pipe:', format='rawvideo', loglevel='quiet', pix_fmt='rgb24').run_async(pipe_stdout=True))
    return decoder

def setup_encoder(output_pth, width, height, fps, MAX_LUM=1000): 
    encoder = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb48le', s='{}x{}'.format(width, height), framerate=fps, color_primaries='bt709', color_trc='smpte2084', colorspace='bt709') 
        .output(output_pth, pix_fmt='yuv420p10le', loglevel='quiet', **{'crf':0, 'c:v': 'libx265', 'x265-params': 'keyint=25:bframes=2:vbv-bufsize=50000:vbv-maxrate=50000:hdr-opt=1:no-open-gop=1:hrd=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,100):max-cll={},300'.format(MAX_LUM)})
        .overwrite_output()
        .run_async(pipe_stdin=True))
    return encoder

def get_index_indicator(idx_indicator):
    """
    get_index_indicator - finds unoccopied index in shared memery 

    :param idx_indicator: list of bool indicating occupied spaces 
    :return: free index
    """ 
    while True:
      tmp_ind = np.frombuffer(idx_indicator, dtype=ctypes.c_long)
      for idx, i in enumerate(tmp_ind): 
        if i == 0: 
          idx_indicator[idx] = 1
          return idx
    return 

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
  

def run(params, ind, num, decQ, encQ):  
    dec = multiprocessing.Process(target=pre_process, args=(params, ind, num, decQ, ))
    dec.start()
    net = multiprocessing.Process(target=LANet, args=(params, num, encQ, decQ, ))
    net.start()
    enc = multiprocessing.Process(target=post_process, args=(params, num, ind, encQ, ))
    enc.start()
    
    dec.join()
    net.join()
    enc.join()
    return 


def pre_process(params, ind, num, decQ): 
    logger = create_logger(params.logger_name)
    logger.info('Pre-Process/decode process started')
    decoder = setup_decoder(params.input_pth)
    for i in progressbar(range(params.n_frames)):
      logger.debug("Frame decoding initiated")
      in_bytes = decoder.stdout.read(params.size)
      logger.debug('Frame decoded')
      img = np.frombuffer(in_bytes, np.uint8).reshape(params.arr_shape)
      img = np.maximum(1.0, img)
      img = sRGB2linear(img)
      img = img * 2 - 1
      img = img.astype(np.float32)
      idx = get_index_indicator(ind)

      np.copyto(num[idx], img)
      decQ.put(idx)
      logger.debug("Index added to decodeQueue")
    
    decQ.put(None)
    decoder.wait()
    return 

def post_process(params, num, ind, encQ):
    logger = create_logger(params.logger_name)
    logger.info('Post-Process/encode process started')
    encoder = setup_encoder(params.output_pth, params.width, params.height, params.fps, params.max_luminance)
    while True:
      idx = encQ.get() 
      logger.debug("Item retrieved from encodeQueue")
      if idx is None: 
        break
      img = np.frombuffer(num[idx], dtype=np.float32).reshape(params.arr_shape)

      img = np.exp(img)
      img = transformPQ(img * params.sc, MAX_LUM=params.max_luminance)
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


def LANet(params, num, encQ, decQ): 
    #Input: RGB-image [h, w, 3]
    #Output: PQ encoded RGB-image
    import time
    logger = create_logger(params.logger_name)
    logger.info('LANet process started')
    if not params.disable_onnx:
      import onnxruntime as onnxrt
      session = onnxrt.InferenceSession(params.model_pth, None, providers=params.providers)
    while True: 
      logger.debug("decQ size: {} encQ size: {}".format(decQ.qsize(), encQ.qsize()))
      idx = decQ.get()
      if idx is None: 
        break
      frame = np.frombuffer(num[idx], dtype=np.float32).reshape([1] + params.arr_shape)

      logger.debug('Inference started')
      if params.disable_onnx:
        output = frame.reshape(params.arr_shape)
        time.sleep(0.5)
      else:
        onnx_inputs = {session.get_inputs()[0].name: frame}
        onnx_output = session.run(None, onnx_inputs)
        output = onnx_output[0].astype(np.float32)

      logger.debug('Inference Stopped')
      np.copyto(num[idx], output)

      encQ.put(idx)

    encQ.put(None)
    return  

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-file', dest="input", type=str, help='Path to HDR image to transform')
    parser.add_argument('-output-filename', dest="output", type=str, help='Path to store frames')
    parser.add_argument('-model', type=str, help="path to inference ONNX-model")
    parser.add_argument('-logging-file', dest="log", default="debug_process.log", type=str, help="name of logging file" )
    parser.add_argument('--disable-onnx', dest='disable_onnx', action='store_true', help="disable onnx for debugging on computer wit limited resources")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


@dataclass
class pipelineParams:
    model_pth: str
    input_pth: str
    output_pth: str

    providers: list = None # Execution providers for ONNX inference
    N_numbers: int = 6  # Number of 
    disable_onnx: bool = False      # disable onnx calls, used for debugging on hardware without GPU. 
    logger_name: str = "debug.log"  # Name of the debug log
    sc: float = 20 #
    max_luminance: int = 1000

    # Automatically initalised variables:
    width: int = None
    height: int = None
    fps: float = None
    n_frames: int = None
    size: int = None # Number of data elements in a frame
    arr_shape: list = None # Shape of a frame

    def __post_init__(self):
      self.width, self.height, self.fps, self.n_frames = self.extract_video_data()
      self.size = self.width * self.height * 3
      self.arr_shape = [self.height, self.width, 3]
      if self.providers is None:
        self.providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", 'CPUExecutionProvider']
      

    def extract_video_data(self): 
      probe = ffmpeg.probe(self.input_pth)
      video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
      width = int(video_stream['width'])
      height = int(video_stream['height'])
      fps = int(video_stream['r_frame_rate'].split('/')[0])
      num_frames= int(video_stream['nb_frames'])
      return width, height, fps, num_frames


if __name__ == '__main__':

  #multiprocessing.set_start_method('spawn')
  opt = parse_opt(True)
  params = pipelineParams(opt.model, opt.input, opt.output, logger_name=opt.log, disable_onnx=opt.disable_onnx)

  multiprocessing_logging.install_mp_handler()
  logger = create_logger(params.logger_name)
  logger.debug("Start of program2")
  
  decodeQueue = multiprocessing.Queue(maxsize=3)
  encodeQueue = multiprocessing.Queue(maxsize=3)

  indicator= multiprocessing.Array(ctypes.c_long,[0] * params.N_numbers, lock=False)

  data_arrays = []
  data_arrays_np = []

  for i in range(params.N_numbers): 
      arr = multiprocessing.RawArray(ctypes.c_float, int(params.size))
      data_arrays.append(arr)
      data_arrays_np.append(np.frombuffer(arr, dtype=np.float32).reshape(params.arr_shape))

  t1 = time.time()
  run(params, indicator, data_arrays_np, decodeQueue, encodeQueue)
  t2 = time.time()
  t = t2-t1
  print("Avg. Per Image: ", str(t/params.n_frames))
  print("Complete video: ", str(t))
