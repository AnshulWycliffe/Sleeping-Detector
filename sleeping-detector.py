from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import pyfiglet
import pyttsx3

# text to speech
engine = pyttsx3.init()
voice = engine.getProperty('voices')
# initialize mixer from pygame
mixer.init()
sound = mixer.Sound('alarm.wav')
print(pyfiglet.figlet_format("Sleeping Detector"))
print('''Created & Modified by- Anshul Wycliffe
    Following libraries used in creating this program

    1.) OpenCV
    2.) Dlib
    3.) pyGame
    4.) Scipy
    5.) Imutils
    6.) Pyttsx3
    ''')


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("/>>Error: Camera not found or could not be opened.")
else:
    print("/>>Camera is working.")
while True:
    ret, frame=cap.read()
    if not ret:
        print("Error: Failed to grab a frame from the camera.")
        break
    frame = imutils.resize(frame, width=1000)
    face_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    if(subjects):
        sound.stop()
    else:
        print("/>>Face not in range")
        sound.play()
        engine.say("Face  not  in  range")
        engine.runAndWait()
        sound.stop()
    for subject in subjects:
        x1 = subject.left()
        y1 = subject.top()
        x2 = subject.right()
        y2 = subject.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (200, 255, 0), 1)
        if(ear>0.25):
            cv2.putText(frame, "Status : Awake :)", (100,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 255, 2), 2)
            sound.stop()
            cv2.putText(frame,f"EAR(Eye Aspect Ratio) : {round(ear,3)} ", (100,500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 255, 2), 2)
            
        elif(ear<=0.25):
            cv2.putText(frame, "Status : Sleeping!!", (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            sound.play()
            cv2.putText(frame,f"EAR(Eye Aspect Ratio) : {round(ear,3)} ", (100,500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)
    
        for n in range(0, 68):
        	(x,y) = shape[n]
        	cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
        cv2.imshow("Landmarks", face_frame) 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


cv2.destroyAllWindows()
cap.release()    


         
        
    
    
    
    
 

