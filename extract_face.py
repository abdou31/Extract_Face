import os
import cv2
import numpy as np
from pascal_voc_writer import Writer
from PIL import Image



# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Create directory 'faces' if it does not exist
if not os.path.exists('faces'):
	print("New directory created")
	os.makedirs('faces')

# Loop through all images and strip out faces
count = 0
for file in os.listdir('images'):
	file_name, file_extension = os.path.splitext(file)
	if (file_extension in ['.png','.jpg']):
		image = cv2.imread('images\\'+file)
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		model.setInput(blob)
		detections = model.forward()

		# Identify each face
		for i in range(0, detections.shape[1]):
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			print(str(file))
			print(startX, startY, endX, endY)

			confidence = detections[0, 0, i, 2]

			# If confidence > 0.5, save it as a separate file
			# Write XML files afer get the face extracted from every image.
			if (confidence > 0.5):
				count += 1
				cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
				frame = image[startY:endY, startX:endX]
				#cv2.imwrite(base_dir + 'faces/' + str(i) + '_' + file, image)
				cv2.imwrite(base_dir + 'faces/' + file, image)
				im = Image.open('images\\'+str(file))
				width,height = im.size
				writer= Writer(str(file),width,height)
#
				writer.addObject('Face_no_mask',startX, startY, endX, endY)
				img_filename= file.split('.')
				writer.save('faces\\'+ img_filename[0]+'.xml')

print("Extracted " + str(count) + " faces from all images")
