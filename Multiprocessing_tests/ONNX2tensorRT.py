import tensorrt as trt

def onnx_2_trt(onnx_model_name, trt_model_name, fp16=False):
   G_LOGGER = trt.Logger(trt.Logger.WARNING)
   explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)    
   with trt.Builder(G_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
        #config.max_workspace_size = 1 << 30
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if fp16:
            config.flags = 1 << (int)(trt.BuilderFlag.FP16)
        #builder.max_batch_size = batch_size        
        print('Loading ONNX file from path {}...'.format(onnx_model_name))

        with open(onnx_model_name, 'rb') as model:
            print('Beginning ONNX file parsing')
            print(parser.parse(model.read()))

        print('Completed parsing of ONNX file')        
        print(parser.get_error(0))        
        assert(network.num_layers > 0)        
        network.get_input(0).shape = (1, -1, -1, 3)
        network.get_input(0).name = "input"        
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 16, 16, 3), (1, 608, 720, 3), (1, 1080, 2048, 3))
        config.add_optimization_profile(profile)        
        last_layer = network.get_layer(network.num_layers - 1)

        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))

        print('Building an engine from file {}; this may take a while...'.format(onnx_model_name))
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")        
        with open(trt_model_name, "wb") as f:
            f.write(engine.serialize())        
        print("Engine saved at {}".format(trt_model_name))



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest="input", type=str, help='Path to input ONNX model')
    parser.add_argument('-output', dest="output", type=str, help='Path to out .plan model')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help="converts model to fp16")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt(True)
    onnx_2_trt(opt.input, opt.output, opt.fp16)
