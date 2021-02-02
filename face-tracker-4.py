import face_recognition
import cv2
import numpy as np

def face_detection(video_path):
    prototxt = 'results/face_detection_model/deploy.prototxt'
    model = 'results/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    unique_face_encodings = []
    unique_faces = []
    video_capture = cv2.VideoCapture(video_path)
    c = 0
    d = 0
    create = None
    out_name = 'results/street.mp4'
    while True:
        r, frame = video_capture.read()
        if r:
            c += 1
            video_capture.set(1, c)
        else:
            video_capture.release()
            cv2.destroyAllWindows()
            break
        try:
            (h, w) = frame.shape[:2]
        except:
            break
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        face_locations = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            face_locations.append((startY, endX, endY, startX))
        if create is None:
        	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        	create = cv2.VideoWriter(out_name, fourcc, 30, (frame.shape[1], frame.shape[0]), True) 
        cv2.imshow("Output", frame)
        create.write(frame)
        if len(face_locations) != 0:
            encodings = face_recognition.face_encodings(frame, face_locations)
            if len(unique_faces) == 0:
                for i in range(len(face_locations)):
                    top, right, bottom, left = face_locations[i]
                    crop_img = frame[top:bottom, left:right]
                    try:
                        crop_img = cv2.resize(crop_img, (100, 100))
                    except:
                        x = 0
                    try:
                        cv2.imwrite('results/people/' + str(d) + '.png', crop_img)
                        print('Unique Face: ', d)
                        print('File saved at ' + '/content/drive/My Drive/mask-detector/results/people/' + str(
                            d) + '.png')
                        print()
                        unique_faces.append(crop_img)
                        unique_face_encodings.append(encodings[d])
                        d += 1
                    except:
                        continue
                    #image_upload('/content/drive/My Drive/mask-detector/results/people/'+str(d)+'.png')
            else:
                for i in range(len(encodings)):
                    match = face_recognition.compare_faces(unique_face_encodings, encodings[i])
                    false_count = 0
                    for j in match:
                        if j == False:
                            false_count += 1
                    if false_count == len(match):
                        top, right, bottom, left = face_locations[i]
                        crop_img = frame[top:bottom, left:right]
                        try:
                            crop_img = cv2.resize(crop_img, (100, 100))
                        except:
                            x = 0
                        try:
                            cv2.imwrite('results/people/' + str(d) + '.png', crop_img)
                            print('Unique Face: ', d)
                            print('File saved at ' + '/content/drive/My Drive/mask-detector/results/people/' + str(
                                d) + '.png')
                            print()
                            unique_faces.append(crop_img)
                            unique_face_encodings.append(encodings[i])
                            d += 1
                        except:
                            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    video_path = 'data/street.mp4'
    face_detection(video_path)

main()
