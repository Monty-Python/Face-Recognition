import cv2,os,msgpack,sys
import numpy as np



def detect_face(img, cascadePath):
	cascade = cv2.CascadeClassifier(cascadePath)
	face = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
	return face






def get_image_from_cam():
	ret_val, img = cam.read()
	if type(img) != np.ndarray:
		os.system('fuser -k /dev/video0')
		ret_val, img = cam.read()
	return cv2.flip(img, 1)





def train_recognizer():
	images = []
	image_label = []
	for label in labels:
		folder = os.path.join(database_path, labels[label])
		print 'getting images from -> ', folder
		for item in os.listdir(folder):
			tmp = cv2.imread(os.path.join(folder, item), 0)
			images.append(tmp)
			image_label.append(label)
			cv2.imshow('Adding to Database', tmp)
			cv2.waitKey(25)
	cv2.destroyAllWindows()
	recognizer.train(images, np.array(image_label))
	recognizer.save(recog_path)
	print 'trained model saved'






def store_faces(img, face):
	global face_ext

	folder = os.path.join(database_path, sys.argv[2])
	if not os.path.isdir(folder):
		os.mkdir(folder)
		if labels == {}:
			labels[0] = sys.argv[2]
		else:
			k = labels.keys()
			k.sort()
			labels[k[-1]+1] = sys.argv[2]
		msgpack.pack(labels, open(label_file,'wb'))

	start = [int(item.split(face_ext)[0]) for item in os.listdir(folder)]
	start.sort()
	if start != []:
		face_counter = start[-1] + 1
	else:
		face_counter = 0
	face_name = '{}/{}{}'.format(folder, face_counter, face_ext)
	x,y,w,h = face[0]
	cv2.imwrite(face_name, img[y:y+h, x:x+w])
	print 'Stored entry -> ', face_name
	





def recog_face(img, face):
	ident = {}
	for (x,y,w,h) in face:
		im_id, confidence = recognizer.predict(img[y:y+h, x:x+w])
		ident[(x,y,w,h)] = (im_id, confidence)
	return ident





if __name__ == '__main__':
	database_path = 'database'
	recog_path = os.path.join(database_path, 'trained_model.yml')
	label_file = os.path.join(database_path, 'labels.mgp')
	face_ext = '.jpg'
	store_counter = 0
	max_store = 50

	cascades = ['opencv-2.4.9/data/lbpcascades/lbpcascade_frontalface.xml',
	'opencv-2.4.9/data/lbpcascades/lbpcascade_frontalface_improved.xml',
	'opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml',
	'opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt2.xml',
	'opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt_tree.xml',
	'opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml',
	'opencv-2.4.9/data/haarcascades/haarcascade_profileface.xml'
	]

	try:
		recognizer = cv2.createLBPHFaceRecognizer()
		if os.path.isfile(label_file):
			labels = msgpack.unpack(open(label_file, 'rb'))
		else:
			labels = {}

		if sys.argv[1] == 'train':
			train_recognizer()
			sys.exit()
		else:
			if os.path.isfile(recog_path):
				recognizer.load(recog_path)
			cam = cv2.VideoCapture(0)

			while True:
				img = get_image_from_cam()
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				face = detect_face(gray, cascades[1])

				if sys.argv[1] == 'store' and len(face) == 1:
					store_faces(gray, face)
					store_counter += 1
					if store_counter >= max_store:
						sys.exit()

				if sys.argv[1] == 'recog' and len(face) >= 1:
					ident = recog_face(gray, face)

				for (x,y,w,h) in face:
					if 'ident' in locals():
						if (x,y,w,h) in ident:
							name = labels[ident[(x,y,w,h)][0]]
							confidence = str(ident[(x,y,w,h)][1])[:5]

							cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
							cv2.putText(img, confidence, (x+(w/2), y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )

					cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
				cv2.imshow('Camera Feed', img)
				cv2.waitKey(25)

	except KeyboardInterrupt:
		print 'Exiting'
		cv2.destroyAllWindows()
