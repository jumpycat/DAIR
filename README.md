# DAIR: Dynamic-Architecture Image Reconstructor

DAIR is an image autoencoder framework designed for dynamic architectural experimentation. Unlike static models, the DAIR decoder can adapt its structure—switching between different upsampling methods, activation functions, and normalization layers either randomly or via a JSON specification.

## Key Features

* **Dynamic Decoder Architecture**: The decoder utilizes `SubBlocks` that can shuffle and select operations (Conv, Act, Norm) on the fly.
* **Weighted Convolutional Pooling**: Features a custom `WeightedConvPool3` layer that dynamically weights multiple convolution outputs based on the input features.
* **Flexible Upsampling**: Supports multiple upsampling modes: Bilinear, Nearest, Bicubic, and PixelShuffle.
* **Architecture Export**: Capability to save and load specific architecture configurations via JSON.


## Usage

To start training with the default settings (using COCO dataset as an example):

```bash
python train.py --train_dir /path/to/your/dataset --batch_size 4 --img_loss both
