# super-colmap
SuperPoint replace the sift in colmap framework

## Get Started:
-------
- install the colmap: https://colmap.github.io/install.html
- clone the super colmap: git clone https://github.com/Xbbei/super-colmap.git
- Download [SuperPoint](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf) to ./
```
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superpoint_v1.pth?raw=true -O superpoint_v1.pth 
```

## Dependencies:
--------
- Python 3.7+
- PyTorch and TorchVision
- OpenCV
- imageio and imageio-ffmpeg

## Run:
--------
```
python super_colmap.py \
      --projpath projpath \
      --cameraModel SIMPLE_RADIAL \
      --images_path rgb \
      --single_camera
```
projpath is your projpath, it must have the "rgb" or "images" dir that contains the images; cameraModel can be "SIMPLE_RADIAL, SIMPLE_PINHOLE, PINHOLE, RADIAL, OPENCV, FULL_OPENCV, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, OPENCV_FISHEYE, FOV, THIN_PRISM_FISHEYE"; images_path is rgb and it denotes that the images is in projpath/rgb; single_camera controls the images are in the same model.
