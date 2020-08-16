import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import datetime
from yolo_utils import *
from flask import Flask, render_template, request, Response
import threading
import requests
import os

FLAGS = []

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
	

@app.route('/timestamps')
def content():
	with open('./templates/Timestamp.txt', 'r') as f:
		return render_template('timestamp.html', text=f.read())
		
@app.route('/suggestions')
def suggestions():

    return render_template('suggestions.html')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(webcam_inference(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

def image_inference():
# Read the image
		try:
			img = cv.imread(FLAGS.image_path)
			height, width = img.shape[:2]
		except:
			raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			img, _, _, _, _ = infer_image(
			    net, layer_names, height, width, img, colors, labels, FLAGS)
			show_image(img)
			counter()
			
def video_inference():
# Read the video
		count = 0

		vid = cv.VideoCapture(FLAGS.video_path)
		starting_time = time.time()
		frame_id = 0
		writer = None
		while True:
			_, frame = vid.read()
			frame_id += 1
			height, width = frame.shape[:2]
			timestamp = datetime.datetime.now()
			cv.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),cv.FONT_HERSHEY_DUPLEX, 0.50, (0, 0, 255), 2)
			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
		    						height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
		    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6
				print(count)

			if FLAGS.video_output_path is not None: 
				if writer is None:
						# Initialize the video writer
					fourcc = cv.VideoWriter_fourcc(*"H264")
					writer = cv.VideoWriter(FLAGS.video_output_path, 0x00000021, 20.0, (frame.shape[1], frame.shape[0]), True)	
		    
				writer.write(frame)
			elapsed_time = time.time() - starting_time
			fps = frame_id / elapsed_time
			print("Avg FPS: " + str(round(fps,2)))
			cv.imshow('webcam', frame)
			
			
			
			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		print("[INFO] Cleaning up...")
		writer.release()
		vid.release()
		counter()
		
def webcam_inference():	
# Infer real-time on webcam
		count = 0
		lock = threading.Lock()
		outputFrame = None
		vid = cv.VideoCapture(FLAGS.webcam)
		writer = None
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]
			timestamp = datetime.datetime.now()
			cv.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),cv.FONT_HERSHEY_DUPLEX, 0.50, (0, 0, 255), 2)
			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
		    						height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
		    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6
				

			if FLAGS.video_output_path is not None: 
				if writer is None:
						# Initialize the video writer
					fourcc = cv.VideoWriter_fourcc(*"MJPG")
					writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]), True)	
		    
				writer.write(frame)
			cv.imshow('webcam', frame)
			with lock:
				outputFrame = frame.copy()
				if outputFrame is None:
					continue
				(flag, encodedImage) = cv.imencode(".jpg", outputFrame)
				if not flag:
					continue
			yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
			
			
			if cv.waitKey(1) & 0xFF == ord('q'):
				break
			if frame is None:
				break
				
		print("[INFO] Cleaning up...")
		counter()
		
		writer.release()
		
		vid.release()

	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')

	parser.add_argument('-vo', '--video-output-path',
		type=str,
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	parser.add_argument("-u", "--use-gpu", type=bool, default=True,
		help="boolean indicating if CUDA GPU should be used")
		
	parser.add_argument("-wb", "--webcam", type=str, default=0,
		help="Choose which webcam you want to use")

	FLAGS, unparsed = parser.parse_known_args()

	# drone = tellopy.Tello()
		  # drone.connect()
		  # drone.start_video()
		  # drone.wait_for_connection(60.0)
		  # drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
		  # container = av.open(ns.drone.get_video_stream())
		  # container = av.open(drone.get_video_stream())
	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(100, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	if FLAGS.use_gpu:
		# set CUDA as the preferable backend and target
		print("[INFO] setting preferable backend and target to CUDA...")
		net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
		print("[INFO] successful")
	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# If both image and video files are given then raise error
	if FLAGS.image_path is None and FLAGS.video_path is None:
	    print('Neither path to an image or path to video provided')
	    print('Starting Inference on Webcam')

	

	# Do inference with given image
	if FLAGS.image_path:
		image_inference()
		
	elif FLAGS.video_path:
		video_inference()

	else:
		app.run(host="0.0.0.0", port="8000", debug=True,
		threaded=True, use_reloader=False)
		# webcam_inference()
		counter()
		cv.destroyAllWindows()
