# Inverse Tone Mapping - With LANet

## Contents
### Workflow 
Fine tune model
```
!OPENCV_IO_ENABLE_OPENEXR=1 python ./main.py --phase train --gpu 0 --epoch 10 --batch_size 1 --checkpoint_dir ./checkpoint_LANet/ --train_dir "/data/training_data" --continue_train True
```

Convert training checkpoint to test checkpoint

```
!OPENCV_IO_ENABLE_OPENEXR=1 python ./main.py --phase convert --checkpoint_dir ../checkpoint_LANet/ 
```

Convert tensorflow checkpoint to ONNX model can be achieved with the command:  
```
!python3 -m tf2onnx.convert --checkpoint /content/LANet.ckpt-0.meta --output model.onnx --inputs test_L:0 --outputs generator/out/Conv/Conv2D:0 --opset 12
```
Opset should be greater than 12 for conversion to tensorRT to work. 

Convert ONNX to 




### Utils
* HDR2LDR - Tool for converting HDR images to LDR images for display on lDR displays. Allows testing various methods and tunning the parameters. 




