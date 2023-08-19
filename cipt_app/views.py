from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from .forms import UserRegisterForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required

#FOR SHOWING VIDEO O/P IN BROWSER
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
# from cipt_app.camera import VideoCamera

# Specifically for FER
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
# import base64

#IMPORTS FOR BODY POSTURE
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#IMPORTS FOR eye_detection
import cv2 as cv
import mediapipe as mp
import time , math
import cipt_app.utils
import numpy as np


def home(request):
    return render(request, 'cipt_app/home.html')


def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Hi {username}, your account was created successfully')
            return redirect('home')
    else:
        form = UserRegisterForm()

    return render(request, 'cipt_app/register.html', {'form': form})

def main(request):
    return render(request, 'cipt_app/main.html')

def phase1_main(request):
    return render(request, 'cipt_app/phase1_main.html')

# def tech_int(request):
#     return render(request, 'cipt_app/tech_int.html')


#FOR SHOWING VIDEO O/P IN BROWSER
# def gen(camera):
# 	while True:
# 		frame = camera.get_frame()
# 		yield (b'--frame\r\n'
# 				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
#
# def video_feed(request):
# 	return StreamingHttpResponse(gen(VideoCamera()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

#FOR FER
def fer(request):
    # return render(request, 'cipt_app/fer.html')
    emotion_report = {'Angry':0, 'Disgust': 0, 'Fear':0, 'Happy':0, 'Neutral':0, 'Sad':0, 'Surprise':0}
    face_classifier = cv2.CascadeClassifier(r'cipt_app/haarcascade_frontalface_default.xml')
    classifier = load_model(r'cipt_app/model.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                emotion_report[label] += 1
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            #cv2.resizeWindow('image', 1000, 1000)
            cv2.imshow('Emotion Detector',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'cipt_app/fer.html')


#BDDY POSTURE RECOGNITION
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle


def body_posture(request):
    cap = cv2.VideoCapture(0)

    # Illegal move counter
    counter = 0
    movement = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                lt_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                rt_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                # Calculate angle
                angle = calculate_angle(elbow, lt_shoulder, rt_shoulder)

                # Visualize angle
                cv2.putText(image, str(angle),
                               tuple(np.multiply(lt_shoulder, [640, 480]).astype(int)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                #Status box
                cv2.rectangle(image, (0,0), (1000,55), (0,0,0), -1)
                # Illegal move counter logic
                if angle > 96 and angle <130:
                    movement = "Straight :D"
                if angle>110:
                    movement = "Over Extended :("
                    cv2.putText(image, "Seems that you've over extended your arms", (260,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                    counter +=1
                if angle < 96:
                    movement = "Cross :("
                    cv2.putText(image, "Seems that you've crossed your arms", (260,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                    counter +=1
                    #print(counter)

            except:
                pass


            # Setup status box
            #cv2.rectangle(image, (0,0), (1000,55), (0,0,0), -1)

            # Rep data
            cv2.putText(image, 'ILLEGAL_MOVES', (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'MOVEMENT-->', (145,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, movement,
                        (145,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)


            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                                     )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    return render(request, 'cipt_app/body_posture.html')

#EYE_DETECTION
def eye_detection(request):
    # variables
    frame_counter =0
    CEF_COUNTER =0
    TOTAL_BLINKS =0
    illegal_movements_count = 0
    # constants
    CLOSED_EYES_FRAME =3
    FONTS =cv.FONT_HERSHEY_COMPLEX
    # face bounder indices
    FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
    # lips indices for Landmarks
    LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
    LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    # Left eyes indices
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
    # right eyes indices
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
    RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

    map_face_mesh = mp.solutions.face_mesh
    # camera object
    #camera = cv.VideoCapture(1)
    camera = cv.VideoCapture(0)

    # landmark detection function
    def landmarksDetection(img, results, draw=False):
        img_height, img_width= img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw :
            [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
        # returning the list of tuples for each landmarks
        return mesh_coord
    # Euclaidean distance
    def euclaideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance
    # Blinking Ratio
    def blinkRatio(img, landmarks, right_indices, left_indices):
        # Right eyes
        # horizontal line
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]
        # draw lines on right eyes
        # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)
        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]
        # vertical line
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]
        rhDistance = euclaideanDistance(rh_right, rh_left)
        rvDistance = euclaideanDistance(rv_top, rv_bottom)
        lvDistance = euclaideanDistance(lv_top, lv_bottom)
        lhDistance = euclaideanDistance(lh_right, lh_left)
        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance
        ratio = (reRatio+leRatio)/2
        return ratio
    # Eyes Extrctor function,
    def eyesExtractor(img, right_eye_coords, left_eye_coords):
        # converting color image to  scale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # getting the dimension of image
        dim = gray.shape
        # creating mask from gray scale dim
        mask = np.zeros(dim, dtype=np.uint8)
        # drawing Eyes Shape on mask with white color
        cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
        # showing the mask
        # cv.imshow('mask', mask)

        # draw eyes image on mask, where white shape is
        eyes = cv.bitwise_and(gray, gray, mask=mask)
        # change black color to gray other than eys
        # cv.imshow('eyes draw', eyes)
        eyes[mask==0]=155

        # getting minium and maximum x and y  for right and left eyes
        # For Right Eye
        r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
        r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
        r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
        r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]
        # For LEFT Eye
        l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
        l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
        l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
        l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]
        # croping the eyes from mask
        cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]
        # returning the cropped eyes
        return cropped_right, cropped_left
    # Eyes Postion Estimator
    def positionEstimator(cropped_eye):
        # getting height and width of eye
        h, w =cropped_eye.shape

        # remove the noise from images
        gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)

        # applying thresholding to convert binary_image
        ret, threshed_eye = cv.threshold(gaussain_blur, 130, 255, cv.THRESH_BINARY)
        # create fixed part for eye with
        piece = int(w/3)
        # slicing the eyes into three parts
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece+piece]
        left_piece = threshed_eye[0:h, piece +piece:w]

        # calling pixel counter function
        eye_position, color = pixelCounter(right_piece, center_piece, left_piece)
        return eye_position, color
    # creating pixel counter function
    def pixelCounter(first_piece, second_piece, third_piece):
        # counting black pixel in each part
        right_part = np.sum(first_piece==0)
        center_part = np.sum(second_piece==0)
        left_part = np.sum(third_piece==0)
        # creating list of these values
        eye_parts = [right_part, center_part, left_part]
        # getting the index of max values in the list
        max_index = eye_parts.index(max(eye_parts))
        pos_eye =''
        if max_index==0:
            pos_eye="RIGHT"
            color=[cipt_app.utils.BLACK, cipt_app.utils.WHITE]
        elif max_index==1:
            pos_eye = 'CENTER'
            color = [cipt_app.utils.BLACK, cipt_app.utils.WHITE]
        elif max_index ==2:
            pos_eye = 'LEFT'
            color = [cipt_app.utils.BLACK, cipt_app.utils.WHITE]
        else:
            pos_eye="Closed"
            color = [cipt_app.utils.BLACK, cipt_app.utils.WHITE]
        return pos_eye, color


    with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

        # starting time here
        start_time = time.time()
        # starting Video loop here.
        while True:
            frame_counter +=1 # frame counter
            ret, frame = camera.read() # getting frame from camera
            if not ret:
                break # no more frames break
                #  resizing frame

            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                cipt_app.utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, cipt_app.utils.PINK, cipt_app.utils.YELLOW)
                if ratio >4.1:
                    CEF_COUNTER +=1
                    # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                    cipt_app.utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, cipt_app.utils.YELLOW, pad_x=6, pad_y=6, )
                else:
                    if CEF_COUNTER>CLOSED_EYES_FRAME:
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
                        # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                cipt_app.utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)

                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, cipt_app.utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, cipt_app.utils.GREEN, 1, cv.LINE_AA)
                # Blink Detector Counter Completed
                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                # cv.imshow('right', crop_right)
                # cv.imshow('left', crop_left)
                eye_position, color = positionEstimator(crop_right)
                cipt_app.utils.colorBackgroundText(frame, eye_position, FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                cipt_app.utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                eye_position_left, color = positionEstimator(crop_left)
                cipt_app.utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)


            # calculating  frame per seconds FPS
            end_time = time.time()-start_time
            fps = frame_counter/end_time
            frame = cipt_app.utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
            # writing image for thumbnail drawing shape
            # cv.imwrite(f'img/frame_{frame_counter}.png', frame)


            #counting illegal eye movements
            if eye_position == "RIGHT" or eye_position == "LEFT" or eye_position_left == "RIGHT" or eye_position_left == "LEFT":
                illegal_movements_count +=1


            cv.imshow('frame', frame)
            key = cv.waitKey(2)
            if key==ord('q') or key ==ord('Q'):
                break
        cv.destroyAllWindows()
        camera.release()
    return render(request, 'cipt_app/eye_detection.html')

@login_required()
def profile(request):
    return render(request, 'cipt_app/profile.html')
