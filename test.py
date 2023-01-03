import cv2
import matplotlib.pyplot as plt

from pyvino.model import build_object_detection_model
model = build_object_detection_model(name='face_detector_0100', draw=True)

frame = cv2.imread('data/test/multi_person.jpg')
frames = [frame]
results = model.compute(frames)
plt.figure(figsize=(9, 16))
plt.imshow(results[0]['image'][:,:,::-1])
#plt.show()

frame = cv2.imread('data/test/multi_person.jpg')
face_images = []
for det in results[0]['output']:
    _, class_id, conf, xmin, ymin, xmax, ymax = det
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    face_image = frame[ymin:ymax, xmin:xmax]
    face_images.append(face_image)

from pyvino.model import build_face_recognition_model
face_age_gender_model = build_face_recognition_model(name='face_age_gender', draw=True)
emotion_model = build_face_recognition_model(name='emotion')
facial_landmark_model = build_face_recognition_model(name='facial_landmark', draw=True)
head_pose_model = build_face_recognition_model(name='head_pose', draw=True)

for plot_num, face_image in enumerate(face_images, 1):
    plt.subplot(1, 3, 1)
    plt.imshow(face_image[:,:,::-1])
    
    face_age_gender_results = face_age_gender_model.compute(face_image)
    emotion_results = emotion_model.compute(face_image)
    facial_landmark_results = facial_landmark_model.compute(face_image.copy())
    head_pose_results = head_pose_model.compute(face_image.copy())
    
    plt.subplot(1, 3, 2)
    plt.imshow(facial_landmark_results[0]['image'][:,:,::-1])
    plt.subplot(1, 3, 3)
    plt.imshow(head_pose_results[0]['image'][:,:,::-1])
#    plt.show()
    
    
    print(face_age_gender_results[0])
    print(emotion_results[0])

#from pyvino.model import build_instance_segmentation_model

#model = build_instance_segmentation_model(name='instance_segmentation_0010', draw=True)
#frame = cv2.imread('data/test/person1.jpg')
#frames = [frame]
#results = model.compute(frames)
#plt.figure(figsize=(9, 16))
#plt.imshow(results[0]['image'][:,:,::-1])
#plt.show()

from pyvino.model import build_object_detection_model
model = build_object_detection_model(name='person_detector', draw=True)

frame = cv2.imread('data/test/person1.jpg')
frames=[frame]
results = model.compute(frames)
plt.figure(figsize=(9, 16))
plt.imshow(results[0]['image'][:,:,::-1])
plt.show()


from pyvino.model import build_pose_estimation_model

model = build_pose_estimation_model(draw=True)
frame = cv2.imread('data/test/person1.jpg')
frames = [frame]
results = model.compute(frames)
plt.figure(figsize=(9, 16))
plt.imshow(results[0]['image'][:,:,::-1])
plt.show()