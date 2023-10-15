This code was uploaded to Jetson Nano 2 (2 CSI ports for camera).
The experiments were performed on a laptop with Intel Core i5-11th CPU with 16 GB RAM, RTX 3050Ti 4GB Laptop GPU.
IMPORTANT: Due to the version of Jetpack on Jetson Nano. The model must be trained with Tensorflow 2.4.0 and CUDA 11.1, Cudnn 8.0 (Consider create a virtual environment with Python venv).
Steps:
1. Change training data directory
2. Train transfer learning model
3. Test performance on test images
4. Test the camera on Jetson Nano. Run code to test reading camera frames using Python's Open CV.
   (require installing v4l2-ctl for camera in Jetson nano so that OpenCV on Python can be used)
   Using CSI camera will require more installation (This repo used CSI camera - Pi V2 camera).
5. Upload the model to Jetson Nano and run the camera classification code.
6. Code to control the moving belt  will be updated later.
Video demonstration:
https://www.youtube.com/shorts/dkxyju6WXfg
