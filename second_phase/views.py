from django.shortcuts import render
from .models import QuestionsPython, Record, Solution, SolutionFace, SolutionBodyPosture, SolutionEyeContact
import random
import ast
#FOR RECORDING
import speech_recognition as sr
import pyaudio

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

# IMPORTS FOR EYE DETECTION********************************
import cv2 as cv
import mediapipe as mp
import time
import second_phase.utils
import math
import numpy as np

#IMPORTS FOR BODY POSTURE DETECTION*************************
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#IMPORTS FOR FACE EMOTION DETECTION*************************
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

#FOR RECORDING
import sounddevice
from scipy.io.wavfile import write
import tkinter
from tkinter import messagebox

# imports for text processing
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
import string
# Create your views here.

def tech_int_home(request):
    '''
    Here we inform user what is going to happen : Info page!!
    '''
    return render(request, 'phase2/tech_int_home.html')

def tech_int_select(request):
    '''
    Here candidate will select their topic of interest
    '''
    return render(request, 'phase2/tech_int_select.html')

def tech_int_python(request):
    '''
    Interview for Python
    '''
    ran_lst = [1,2,3,4,5,6]
    ran_int = random.choice(ran_lst)
    ques = QuestionsPython.objects.get(question_number=1)
    context = {
    'question' : ques
    }
    return render(request, 'phase2/tech_int_python.html',context)

def tech_int_ds(request):
    '''
    Interview for Python
    '''
    ran_lst = [1,2,3,4,5,6]
    ran_int = random.choice(ran_lst)
    ques = QuestionsDS.objects.get(question_number=1)
    context = {
    'question' : ques
    }
    return render(request, 'phase2/tech_int_python.html',context)

def tech_int_dbms(request):
    '''
    Interview for Python
    '''
    ran_lst = [1,2,3,4,5,6]
    ran_int = random.choice(ran_lst)
    ques = QuestionsDBMS.objects.get(question_number=1)
    context = {
    'question' : ques
    }
    return render(request, 'phase2/tech_int_python.html',context)

def tech_int_os(request):
    '''
    Interview for Python
    '''
    ran_lst = [1,2,3,4,5,6]
    ran_int = random.choice(ran_lst)
    ques = QuestionsOS.objects.get(question_number=1)
    context = {
    'question' : ques
    }
    return render(request, 'phase2/tech_int_python.html',context)

def tech_int_cn(request):
    '''
    Interview for Python
    '''
    ran_lst = [1,2,3,4,5,6]
    ran_int = random.choice(ran_lst)
    ques = QuestionsCN.objects.get(question_number=1)
    context = {
    'question' : ques
    }
    return render(request, 'phase2/tech_int_python.html',context)

# def ans_python(request):
#     return render(request, 'phase2/ans_python.html')


def record(request):
    #For recording the answers
    # VARIABLES FOR FER
    emotion_report = {'Angry':0, 'Disgust': 0, 'Fear':0, 'Happy':0, 'Neutral':0, 'Sad':0, 'Surprise':0}
    face_classifier = cv2.CascadeClassifier(r'second_phase/haarcascade_frontalface_default.xml')
    classifier =load_model(r'second_phase/model.h5')
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    illegal_movements_count_face=0
    cap = cv2.VideoCapture(0)
    # VARIABLES FOR BODY POSTURE
    illegal_movements_count_body = 0
    movement = None
    # VARIABLES FOR EYE CONTACT
    frame_counter =0
    CEF_COUNTER =0
    TOTAL_BLINKS =0
    illegal_movements_count_eye = 0
    start_time = time.time()
    # constants
    CLOSED_EYES_FRAME =3
    FONTS =cv2.FONT_HERSHEY_COMPLEX
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
    #VARIAVLES FOR RECORDING
    count_text = ['x']


    # FUNCTIONS FOR BODY POSTURE
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle

        return angle
    # FUNCTIONS FOR EYE CONTACT
    def landmarksDetection(img, results, draw=False):
        img_height, img_width= img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw :
            [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
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
        gray = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)

        # getting the dimension of image
        dim = gray.shape
        # creating mask from gray scale dim
        mask = np.zeros(dim, dtype=np.uint8)
        # drawing Eyes Shape on mask with white color
        cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
        # showing the mask
        # cv.imshow('mask', mask)

        # draw eyes image on mask, where white shape is
        eyes = cv2.bitwise_and(gray, gray, mask=mask)
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
        gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9),0)

        # applying thresholding to convert binary_image
        ret, threshed_eye = cv2.threshold(gaussain_blur, 130, 255, cv.THRESH_BINARY)
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
            color=[utils.BLACK, utils.WHITE]
        elif max_index==1:
            pos_eye = 'CENTER'
            color = [utils.BLACK, utils.WHITE]
        elif max_index ==2:
            pos_eye = 'LEFT'
            color = [utils.BLACK, utils.WHITE]
        else:
            pos_eye="Closed"
            color = [utils.BLACK, utils.WHITE]
        return pos_eye, color
    #FUNCTION FOR RECORDING
    def record_audio():

        #It takes microphone input from the user and returns string output

        global count_text

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            r.pause_threshold = 1
            r.energy_threshold = 100  # minimum audio energy to consider for recording
            audio = r.listen(source)

        try:
            print("Recognizing...")
            text = r.recognize_google(audio, language='en-in')
            count_text.append(text)
            print(f"Your Command: {text}\n")

        except Exception as e:
            print("Say that again please...")
            return "None"
        return count_text

    #FUNCTION FOR TEXT PROCESSING
    def text_process(mess):
        ps = PorterStemmer()
        stemming = []
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        4. Returns stemming words from sentences
        """
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation]
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        # Now just remove any stopwords
        # And stemming words from sentences
        nosw = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
        for w in nosw:
            stemming.append(ps.stem(w))
        return stemming

    def comparison(lst1, lst2):
        lst3 = [word for word in lst1 if word in lst2]
        return len(lst3)

    #MAIN PROGRAM
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
            while True:
                _, frame = cap.read()


                #FER
                labels = []
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray,1.2,4)
                # BODY POSTURE
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# Recolor image to RGB
                image.flags.writeable = False
                results = pose.process(image) # Make detection
                image.flags.writeable = True # Recolor back to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #EYE CONTACT
                frame_eye = frame
                frame_eye = cv.resize(frame_eye, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
                frame_height, frame_width= frame_eye.shape[:2]
                rgb_frame = cv.cvtColor(frame_eye, cv.COLOR_RGB2BGR)
                results_eye  = face_mesh.process(rgb_frame)


                #FER
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
                        if label=='Angry' or label=='Disgust' or  label == 'Fear' or label == 'Sad' or label == 'Surprise':
                                    illegal_movements_count_face+=1
                        label_position = (x,y)
                    else:
                        pass

                # BODY POSTURE
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    lt_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    rt_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    # Calculate angle
                    angle = calculate_angle(elbow, lt_shoulder, rt_shoulder)
                    # illegal movement counter logic
                    if angle > 96 and angle <130:
                        movement = "Straight :D"
                    if angle>110:
                        illegal_movements_count_body +=1
                    if angle < 96:
                        illegal_movements_count_body +=1

                    # EYE CONTACT
                    if results_eye.multi_face_landmarks:
                        mesh_coords = landmarksDetection(frame_eye, results_eye, False)
                        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                        if ratio >4.1:
                            CEF_COUNTER +=1
                        else:
                            if CEF_COUNTER>CLOSED_EYES_FRAME:
                                TOTAL_BLINKS +=1
                                CEF_COUNTER =0
                        # Blink Detector Counter Completed
                        right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                        left_coords = [mesh_coords[p] for p in LEFT_EYE]
                        crop_right, crop_left = eyesExtractor(frame_eye, right_coords, left_coords)
                        eye_position, color = positionEstimator(crop_right)
                        eye_position_left, color = positionEstimator(crop_left)
                        #counting illegal eye movements
                        if eye_position == "RIGHT" or eye_position == "LEFT" or eye_position_left == "RIGHT" or eye_position_left == "LEFT":
                            illegal_movements_count_eye +=1
                    # calculating  frame per seconds FPS
                    end_time = time.time()-start_time
                    fps = frame_counter/end_time



                except:
                    pass

                cv2.imshow("PROCTURING (Please press 'q' or say 'stop' to quit",frame)
                # RECORD
                record_audio()
                if (cv2.waitKey(1) & 0xFF == ord('q')) or count_text[-1] == 'stop':
                    break
            cap.release()
            cv2.destroyAllWindows()

            df = pd.read_csv("second_phase/Interview_questions.csv", index_col='Index')
            df['Keywords'] = df['Answers'].apply(text_process)
            text = text_process(count_text)
            text_match_count = comparison(df['Keywords'][1], text)

            record_obj = Record(record_audio_text=count_text, illegal_face = illegal_movements_count_face,
                                illegal_body = illegal_movements_count_body,
                                illegal_eye = illegal_movements_count_eye,
                                eye_blink_count = TOTAL_BLINKS,
                                text_match = text_match_count,
                                emotion_report = emotion_report)
            record_obj.save()
            cv.destroyAllWindows()
            cap.release()

        context = {
        'recording' : record_obj
        }
    return render(request, 'phase2/ans_python.html',context)

def ans_python(request):
    return render(request,'phase2/ans_python.html' )

def report(request):

    #report = Record.objects.latest('pk')
    report = Record.objects.get(id=10)
    #Variables
    ief = 0
    ieb = 0
    iee = 0
    tmp = 0
    tm = 0
    # for face
    if report.illegal_face <= 10:
        ief =100
    elif report.illegal_face > 10 and report.illegal_face <= 30:
        ief = 80
    elif report.illegal_face > 30 and report.illegal_face <= 50:
        ief = 60
    elif report.illegal_face > 50 and report.illegal_face <= 80:
        ief = 40
    else:
        ief = 20

    # for body
    if report.illegal_body <= 10:
        ieb = 100
    elif report.illegal_body > 10 and report.illegal_body <= 30:
        ieb = 80
    elif report.illegal_body > 30 and report.illegal_body <= 50:
        ieb = 60
    elif report.illegal_body > 50 and report.illegal_body <= 80:
        ieb = 40
    else:
        ieb = 20

    # for body
    if report.illegal_eye <= 10:
        iee = 100
    elif report.illegal_eye > 10 and report.illegal_eye <= 30:
        iee = 80
    elif report.illegal_eye > 30 and report.illegal_eye <= 50:
        iee = 60
    elif report.illegal_eye > 50 and report.illegal_eye <= 80:
        iee = 40
    else:
        iee = 20

    #FOR TECHNICAL INTERVIEW TEXT MATCH
    text_match = report.text_match

    if text_match < 2 :
        tmp = 80
    elif text_match >= 10:
        tm = 10
    else:
        tm = text_match
        tmp = 20

    tmu = tm * 10
    context = {
    'recording' : report,
    'ief' : ief,
    'ieb' : ieb,
    'iee' : iee,
    'tmu' : tmu,
    'tmp' : tmp
    }


    return render(request, 'phase2/report.html',context)


def illegal_face(request):
    videos = SolutionFace.objects.all()
    #report = Record.objects.latest('pk')
    report = Record.objects.get(id=10)
    #Variables
    ief = 0
    Angry = 0
    Disgust = 0
    Fear = 0
    Happy = 0
    Neutral = 0
    Sad = 0
    Surprise =0

    # for face
    if report.illegal_face <= 10:
        ief =100
    elif report.illegal_face > 10 and report.illegal_face <= 30:
        ief = 80
    elif report.illegal_face > 30 and report.illegal_face <= 50:
        ief = 60
    elif report.illegal_face > 50 and report.illegal_face <= 80:
        ief = 40
    else:
        ief = 20

    # d = dict(str(report.emotion_report))
    # Angry = d['Angry']
    # Disgust = d['Disgust ']
    # Fear = d['Fear']
    # Happy = d['Happy']
    # Neutral = d['Neutral']
    # Sad = d['Sad']
    # Surprise = d['Surprise']

    context = {
    'recording' : report,
    'ief' : ief,
    'videos' : videos,
    # 'd' : d,
    # 'Angry' : Angry,
    # 'Disgust' : Disgust,
    # 'Fear' : Fear,
    # 'Happy' : Happy,
    # 'Neutral' : Neutral,
    # 'Sad ':Sad ,
    # 'Surprise' : Surprise
    }

    return render(request, 'phase2/illegal_face.html', context)

def illegal_body(request):
    videos = SolutionBodyPosture.objects.all()
    report = Record.objects.latest('pk')
    #Variables
    ieb = 0

    # for body
    if report.illegal_body <= 10:
        ieb = 100
    elif report.illegal_body > 10 and report.illegal_body <= 30:
        ieb = 80
    elif report.illegal_body > 30 and report.illegal_body <= 50:
        ieb = 60
    elif report.illegal_body > 50 and report.illegal_body <= 80:
        ieb = 40
    else:
        ieb = 20

    context = {
    'recording' : report,
    'ieb' : ieb,
    'videos' : videos
    }
    return render(request, 'phase2/illegal_body.html', context)

def illegal_eye(request):
    videos = SolutionEyeContact.objects.all()
    report = Record.objects.latest('pk')
    #Variables
    iee = 0

    # for eye
    if report.illegal_eye <= 10:
        iee = 100
    elif report.illegal_eye > 10 and report.illegal_eye <= 30:
        iee = 80
    elif report.illegal_eye > 30 and report.illegal_eye <= 50:
        iee = 60
    elif report.illegal_eye > 50 and report.illegal_eye <= 80:
        iee = 40
    else:
        iee = 20

    context = {
    'recording' : report,
    'iee' : iee,
    'videos' : videos
    }
    return render(request, 'phase2/illegal_eye.html', context)

def tech_int_issue(request):
    videos = Solution.objects.all()
    report = Record.objects.latest('pk')
    #Variables
    tmp = 0
    tm = 0
    tech_int_score = 0

    #FOR TECHNICAL INTERVIEW TEXT MATCH
    text_match = report.text_match

    if text_match < 2 :
        tmp = 80
    elif text_match >= 10:
        tm = 10
    else:
        tm = text_match
        tmp = 20

    tmu = tm * 10

    if tmu <= 20:
        tech_int_score = 20
    elif tmu > 20 and tmu <= 40:
        tech_int_score = 40
    elif tmu > 40 and tmu <= 60:
        tech_int_score = 60
    elif tmu > 60 and tmu <= 80:
        tech_int_score = 80
    else:
        tech_int_score = 100

    context = {
    'recording' : report,
    'tmu' : tmu,
    'tmp' : tmp,
    'tech_int_score' : tech_int_score,
    'videos' : videos
    }
    return render(request, 'phase2/tech_int_issue.html', context)
