import numpy as np
import onnxruntime as onnxrt


class ONNX_model:
    def __init__(self, onnx_pth, providers, dtype=np.float32):

        self.dtype = dtype
        self.session = onnxrt.InferenceSession(onnx_pth, None, providers=providers)

    def run_model(self, frame):
        onnx_inputs = {self.session.get_inputs()[0].name: frame}
        onnx_output = self.session.run(None, onnx_inputs)
        output = onnx_output[0].astype(self.dtype)
        return output


