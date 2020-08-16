# YoloV3-Object-Detection-and-Classification-OpenCV
This project was built for a school assignment.
This project implements an image and video object detection classifier using pretrained yolov3 models. The yolov3 models are taken from the official yolov3 paper which was released in 2018. The yolov3 implementation is from darknet. Also, this project implements an option to perform classification real-time using the webcam.

# Stack
- Nvidia CUDA / cuDNN
- Python3
- Flask
- OpenCV
# How to use?
1. To infer on an image that is stored on your local machine
```
python3 yolo.py --image-path='/path/to/image/'
```

2. To infer on a video that is stored on your local machine
```
python3 yolo.py --video-path='/path/to/video/'
```

3. To infer real-time on webcam
```
python3 yolo.py
```

# Reference
[PyImageSearch blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
