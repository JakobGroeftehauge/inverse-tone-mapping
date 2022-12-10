#%%writefile test.py
import multiprocessing
import time
from progressbar import progressbar
import numpy as np
import ctypes
import multiprocessing_logging
import argparse
from image_processing import preprocess_image, postprocess_image
from common import create_logger, setup_decoder, setup_encoder, PipelineParams
from tensorRT_model import TensorRT_model


def get_index_indicator(idx_indicator):
    """
    get_index_indicator - finds unoccopied index of list of shared memory 

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

def launch_inference(params, ind, frames_buffer, decQ, encQ):  
    dec = multiprocessing.Process(target=preprocess, args=(params, ind, frames_buffer, decQ, ))
    dec.start()
    net = multiprocessing.Process(target=LANet, args=(params, frames_buffer, encQ, decQ, ))
    net.start()
    enc = multiprocessing.Process(target=postprocess, args=(params, frames_buffer, ind, encQ, ))
    enc.start()
    
    dec.join()
    net.join()
    enc.join()
    return 


def preprocess(params, ind, frames_buffer, decQ): 
    logger = create_logger(params.logger_name)
    logger.info('Pre-Process/decode process started')
    decoder = setup_decoder(params.input_pth)

    for i in progressbar(range(params.n_frames)):
      logger.debug("Frame decoding initiated")
      in_bytes = decoder.stdout.read(params.size)
      logger.debug('Frame decoded')
      
      img = np.frombuffer(in_bytes, np.uint8).reshape(params.arr_shape)
      img = preprocess_image(img)

      idx = get_index_indicator(ind)
      np.copyto(frames_buffer[idx], img)
      decQ.put(idx)
      logger.debug("Index added to decodeQueue")
    
    decQ.put(None) # Indicate that there is no more data to process. 
    decoder.wait()
    return 

def postprocess(params, frames_buffer, ind, encQ):
    logger = create_logger(params.logger_name)
    logger.info('Post-Process/encode process started')
    encoder = setup_encoder(params.output_pth, params.width, params.height, params.fps, params.max_luminance)
    
    while True:
      idx = encQ.get() 
      logger.debug("Index retrieved from encodeQueue")
      
      if idx is None: 
        break
      
      img = np.frombuffer(frames_buffer[idx], dtype=np.float32).reshape(params.arr_shape)
      img = postprocess_image(img, sc=params.sc,  max_luminance=params.max_luminance )

      logger.debug('Write to encoder initiated')
      encoder.stdin.write(img.astype(np.uint16).tobytes())
      logger.debug('Write to encoder finshed')
      ind[idx] = 0

    logger.debug("Initiated closing of encoder")
    encoder.stdin.close()
    encoder.wait()
    logger.debug("Encoder closed")
    return 


def LANet(params, frames_buffer, encQ, decQ): 
    import time
    logger = create_logger(params.logger_name)
    logger.info('LANet process started')

    if not params.disable_onnx:
      #import onnxruntime as onnxrt
      #session = onnxrt.InferenceSession(params.model_pth, None, providers=params.providers)
      model = TensorRT_model(params.model_pth, [1] + params.arr_shape)

    while True: 
      logger.debug("decQ size: {} encQ size: {}".format(decQ.qsize(), encQ.qsize()))
      idx = decQ.get()

      if idx is None: 
        break
      
      frame = np.frombuffer(frames_buffer[idx], dtype=np.float32).reshape([1] + params.arr_shape)

      logger.debug('Inference started')
      if params.disable_onnx:
        output = frame.reshape(params.arr_shape)
        time.sleep(0.5)
      else:
        #onnx_inputs = {session.get_inputs()[0].name: frame}
        #onnx_output = session.run(None, onnx_inputs)
        #output = onnx_output[0].astype(np.float32)
        output = model(frame)

      logger.debug('Inference Stopped')
      np.copyto(frames_buffer[idx], output)

      encQ.put(idx)

    encQ.put(None) # Indicate that there is no more data to process. 
    return  

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-file', dest="input", type=str, help='Path to HDR image to transform')
    parser.add_argument('-output-filename', dest="output", type=str, help='Path to store frames')
    parser.add_argument('-model', type=str, help="path to inference ONNX-model")
    parser.add_argument('-logging-file', dest="log", default="debug_process.log", type=str, help="name of logging file" )
    parser.add_argument('--disable-model', dest='disable_model', action='store_true', help="disable onnx for debugging on computer wit limited resources")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == '__main__':

  #multiprocessing.set_start_method('spawn')
  opt = parse_opt(True)
  params = PipelineParams(opt.model, opt.input, opt.output, logger_name=opt.log, disable_model=opt.disable_model)

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
  launch_inference(params, indicator, data_arrays_np, decodeQueue, encodeQueue)
  t2 = time.time()
  t = t2-t1

  print("----- Inference Completed -----")
  print("Avg. Per Image: ", str(t/params.n_frames))
  print("Complete video: ", str(t))
  print("-------------------------------")