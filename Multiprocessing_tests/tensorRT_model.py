import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class TensorRT_model:
    def __init__(self, plan_pth, shape):

        with open(model_pth, "rb") as fp:
          plan_model = fp.read()

        # initialize the TensorRT objects
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan_model)
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("input", shape)
        
        self.n_bytes = int(np.dtype(np.float32).itemsize * np.prod(shape))

        # create device buffers and TensorRT bindings
        self.output = np.zeros(shape, dtype=np.float32)
        self.d_input  = cuda.mem_alloc(self.n_bytes)
        self.d_output = cuda.mem_alloc(self.n_bytes)
        self.bindings = [self.d_input, self.d_output]


    def run_model(self, frame):
        cuda.memcpy_htod(self.d_input, frame)
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh(self.output, self.d_output)
        return self.output

