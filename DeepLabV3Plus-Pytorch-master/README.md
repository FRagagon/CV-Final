# DeepLabv3Plus-Pytorch

DeepLabv3, DeepLabv3+ with pretrained models for Pascal VOC & Cityscapes.

## Quick Start 


The pre-trained model with mobile net can be obtained from google drive.
Just put it under this directory.

The file model.py is used to generate picture folder from video and generate video from picture folder.


### 1. Load the pretrained model:
```python
model.load_state_dict( torch.load( CKPT_PATH )['model_state']  )
```
### 2. Visualize segmentation outputs:
```python
outputs = model(images)
preds = outputs.max(1)[1].detach().cpu().numpy()
colorized_preds = val_dst.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# Do whatever you like here with the colorized segmentation maps
colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
```

### 3. Prediction
Single image:
```bash
python predict.py --input image_path  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```

Image folder:
```bash
python predict.py --input image_folder_path  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```

