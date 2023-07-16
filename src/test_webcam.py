import align.detect_face
import cv2
import facenet
import numpy as np
import pickle
import tensorflow as tf

from scipy import misc

MODEL_PRETRAINED = "./model/20180402-114759.pb"
CLASSIFIER_FILENAME_EXP = "./model/facemodel.pkl"

MIN_SIZE = 20
DETECT_THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 160
CLASSIFIER_THRESHOLD = 0.75

with open(CLASSIFIER_FILENAME_EXP, "rb") as infile:
    model, class_names = pickle.load(infile)

print("Loaded classifier model from file {}".format(CLASSIFIER_FILENAME_EXP))

with tf.Graph().as_default():
    with tf.Session() as sess:
        print("Loading feature extraction model")
        facenet.load_model(MODEL_PRETRAINED)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "./src/align")

        webcam = cv2.VideoCapture(0)

        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while webcam.isOpened():
            ret, frame = webcam.read()

            bounding_boxes, _ = align.detect_face.detect_face(frame, MIN_SIZE, pnet, rnet, onet, DETECT_THRESHOLD, FACTOR)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(frame.shape)[0:2]

                if nrof_faces > 1:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    det_arr.append(np.squeeze(det))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0], 0)
                    bb[1] = np.maximum(det[1], 0)
                    bb[2] = np.minimum(det[2], img_size[1])
                    bb[3] = np.minimum(det[3], img_size[0])
                    cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interp="bilinear")
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    best_name = class_names[best_class_indices[0]]

                    print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                    if best_class_probabilities > CLASSIFIER_THRESHOLD:
                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
                        cv2.putText(frame, best_name, (bb[0], bb[3] + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (bb[0], bb[3] + 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Face Detection Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        webcam.release()
        cv2.destroyAllWindows()