#%%writefile run_thread.py
import threading
import queue
import time
import ffmpeg 
from progressbar import progressbar
import numpy as np
import cv2
import logging
import multiprocessing_logging
from importlib import reload  # Not needed in Python 2
import argparse
import time 


data = {}

def create_logger(name): 
    logging.basicConfig(filename=name, format='%(asctime)s:%(msecs)03d: %(levelname)s: %(name)s: %(message)s',level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    return logger

def video_data(filename): 
    probe = ffmpeg.probe(filename)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = int(video_stream['r_frame_rate'].split('/')[0])
    num_frames= int(video_stream['nb_frames'])
    return [width, height], fps, num_frames

class onnx_LANet_thread(): 
  def __init__(self, model_pth, input_pth, output_pth, sc=20, providers=None): 
      if not data["disable_onnx"]:
        import onnxruntime as onnxrt
        self.providers = ['TensorrtExecutionProvider', "CUDAExecutionProvider"] if providers is None else providers
        self.session = onnxrt.InferenceSession(model_pth, None, providers=self.providers)
      self.sc = sc #pre-scaling

      self.output_pth = output_pth
      self.input_pth = input_pth
      [self.width, self.height], self.fps, self.n_frames = video_data(self.input_pth)

      self.setup_decoder()
      self.setup_encoder() 

      self.decodeQueue = queue.Queue(maxsize=1)
      self.encodeQueue = queue.Queue(maxsize=1)


  def setup_decoder(self): 
      self.decoder = (ffmpeg.input(self.input_pth).output('pipe:', format='rawvideo', pix_fmt='rgb24').run_async(pipe_stdout=True))
      return 

  def setup_encoder(self): 
      self.encoder = (ffmpeg
          .input('pipe:', format='rawvideo', pix_fmt='rgb48le', s='{}x{}'.format(self.width, self.height), framerate=self.fps, color_primaries='bt709', color_trc='smpte2084', colorspace='bt709') 
          .output(self.output_pth, pix_fmt='yuv420p10le', **{'crf':0, 'c:v': 'libx265', 'x265-params': 'keyint=25:bframes=2:vbv-bufsize=50000:vbv-maxrate=50000:hdr-opt=1:no-open-gop=1:hrd=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,100):max-cll=1000,300'})
          .overwrite_output()
          .run_async(pipe_stdin=True))
      return 

  def run(self):  
      logger = create_logger(data["logger_name"])
      
      dec = threading.Thread(target=self.pre_process)
      dec.start()
      net = threading.Thread(target=self.LANet)
      net.start()
      enc = threading.Thread(target=self.post_process)
      enc.start()

      dec.join()
      net.join()
      enc.join()

      self.decoder.wait()
      logger.debug("Initiated closing of encoder")
      self.encoder.stdin.close()
      self.encoder.wait()
      logger.debug("Encoder closed")

      
      return 
      

  def pre_process(self): 
      #img.astype(np.float32)
      logger = create_logger(data["logger_name"])
      logger.info('Pre-Process/decode process started')
      for i in progressbar(range(self.n_frames)):
        logger.debug("Frame decoding initiated")
        in_bytes = self.decoder.stdout.read(self.width * self.height * 3)
        logger.debug('Frame decoded')

        img = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        img = np.maximum(1.0, img)
        img = self.sRGB2linear(img)
        img = img * 2 - 1

        self.decodeQueue.put(np.expand_dims(img, axis=0).astype(np.float32))
        logger.debug("Index added to decodeQueue")

      
      self.decodeQueue.put(None)
      return 

  def post_process(self): 
      logger = create_logger(data["logger_name"])
      logger.info('Post-Process/encode process started')
      while True:
        img = self.encodeQueue.get() 
        logging.debug("Item retrieved from encodeQueue")
        if img is None: 
          break
      
        img = np.squeeze(img, axis=0) 
        img = np.exp(img)
        img = self.transformPQ(img * self.sc)
        img = img * 65535
        logger.debug('Write to encoder initiated')
        self.encoder.stdin.write(img.astype(np.uint16).tobytes())
        logger.debug('Write to encoder finshed')

      return 


  def LANet(self): 
      #Input: RGB-image [h, w, 3]
      #Output: PQ encoded RGB-image
      logger = create_logger(data["logger_name"])
      logger.info('LANet process started')
      while True: 
        logging.debug("decQ size: {} encQ size: {}".format(self.decodeQueue.qsize(), self.encodeQueue.qsize()))
        input = self.decodeQueue.get()

        if input is None: 
          break
        #print("shape input: ", input.shape)
        logger.debug('Inference started')
        if data["disable_onnx"]:
            onnx_output = [input.astype(np.float32)]
            time.sleep(0.5)
        else:
            onnx_inputs = {self.session.get_inputs()[0].name: input}
            onnx_output = self.session.run(None, onnx_inputs)
        logger.debug('Inference Stopped')
        #print("shape", onnx_output[0].shape)
        self.encodeQueue.put(onnx_output[0])

      self.encodeQueue.put(None)
      return  

  def sRGB2linear(self, img):
      img = img / 255
      return np.where(img <= 0.04045, img / 12.92, np.power((img+0.055) / 1.055, 2.4))

  def transformPQ(self, arr): 
      L = 1000.0 #max Luminance
      m = 78.8438
      n = 0.1593
      c1 = 0.8359
      c2 = 18.8516
      c3 = 18.6875
      Lp = np.power(arr/L, n)
      return np.power((c1 + c2*  Lp) / (1 + c3*Lp), m)    
  

def video_data(filename): 
    probe = ffmpeg.probe(filename)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = int(video_stream['r_frame_rate'].split('/')[0])
    num_frames= int(video_stream['nb_frames'])
    return [width, height], fps, num_frames

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-file', dest="input", type=str, help='Path to HDR image to transform')
    parser.add_argument('-output-filename', dest="output", type=str, help='Path to store frames')
    parser.add_argument('-model', type=str, help="path to inference ONNX-model")
    parser.add_argument('-logging-file', dest="log", default="debug_process_thread.log", type=str, help="name of logging file" )
    parser.add_argument('--disable-onnx', dest='disable_onnx', type=bool, default=False, help="disable onnx for debugging on computer wit limited resources")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt



if __name__ == '__main__':
    opt = parse_opt(True)

    multiprocessing_logging.install_mp_handler()
    logger = create_logger(opt.log)
    logger.debug("Start of program - threaded version")
    
    filename = opt.input #"eagle-reference.mp4"
    filename_out = opt.output #"video-inference1.mkv"

    model_pth = opt.model #"/content/model2.onnx"
    data["logger_name"] = opt.log
    data["disable_onnx"] = opt.disable_onnx

    model = onnx_LANet_thread(model_pth, filename, filename_out)

    t1 = time.time()
    model.run()
    t2 = time.time()
    t = t2-t1
    print("Avg. Per Image: ", str(t/model.n_frames))
    print("Complete video: ", str(t))
