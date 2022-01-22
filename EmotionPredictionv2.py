from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator, load_img
import cv2
import os
import sys
import json
import spotipy
import webbrowser
import spotipy.util as util
from json.decoder import JSONDecodeError
import pandas as pd
import time
from random import shuffle
import joblib

classes = 4
username = "" #Enter your Username 
SPOTIPY_CLIENT_ID="" # Client ID
SPOTIPY_CLIENT_SECRET="" # Client Secret 
SPOTIPY_REDIRECT_URI="https://www.google.com"
scope = "user-read-private user-read-playback-state user-modify-playback-state app-remote-control"



songsdb = pd.read_csv("Music/MusicData.csv")

happyAngryDb = songsdb[songsdb["mood"].isin(["Happy","Angry"])].reset_index(drop = True)

try:
    token = util.prompt_for_user_token(username,
                                       scope = scope,
                                      client_id = SPOTIPY_CLIENT_ID,
                                      client_secret = SPOTIPY_CLIENT_SECRET,
                                      redirect_uri = SPOTIPY_REDIRECT_URI)
except:
    #os.remove(f".cache={username}")
    token = util.prompt_for_user_token(username,
                                       scope = scope,
                                       client_id = SPOTIPY_CLIENT_ID,
                                      client_secret = SPOTIPY_CLIENT_SECRET,
                                      redirect_uri = SPOTIPY_REDIRECT_URI)

spotifyObject = spotipy.Spotify(auth=token)    

def model_1():
    # load model without classifier layers
    model = VGG16(include_top=False, weights = "imagenet", input_shape=(48, 48, 3))
    # add new classifier layers
    # flat1 = Flatten()(pretrained.layers[-1].output)
    # denseout1 = Dense(1024, activation='relu')(flat1)
    # denseout2 = Dense(512, activation='relu')(denseout1)
    # output = Dense(classes, activation='softmax')(denseout2)
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(classes, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    
    # summarize
    return model
    
    
def model_2():
    # load model without classifier layers
    pretrained = VGG16(include_top=False, weights = "imagenet", input_shape=(48, 48, 3))
    
    # add new classifier layers
    flat1 = Flatten()(pretrained.layers[-1].output)
    denseout1 = Dense(1024, activation='relu')(flat1)
    denseout2 = Dense(512, activation='relu')(denseout1)
    output = Dense(1, activation='sigmoid')(denseout2)
    # define new model
    model = Model(inputs=pretrained.inputs, outputs=output)
    
    # summarize
    return model

model = model_1()
model.load_weights("model_1.h5") #model_1.h5

model2 = model_2()
model2.load_weights("Sad Neutral.h5")

nn = joblib.load("NearestNeighberSadCalm.pkl")
nn2 = joblib.load("NearestNeighberHappyAngry.pkl")
sadparams = joblib.load("sadParams.wt")
calmparams = joblib.load("calmParams.wt")
happyparams = joblib.load("happyParams.wt")
angryparams = joblib.load("angryParams.wt")


def emotion_prediction(img_array):
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    emotion_class = {0:'Angry', 1:'Happy', 2:'Calm', 3:'Sad'}
    pred = model.predict(img_preprocessed)   
    if emotion_class[pred.argmax()] == "Calm":
        pred = model2.predict(img_preprocessed)
        if pred > 0.4:
            return 'Sad'
        else:
            return 'Calm'
    return emotion_class[pred.argmax()]
# define a video capture object
vid = cv2.VideoCapture(0)

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

face_cascade = cv2.CascadeClassifier('/Users/aman/Project/MRFEA-code/HaarCascaseFace/Face_Default.xml') #Update the Path for HaarCascaseFace Filter

def adjusted_detect_face(img):
     
    face_img = img.copy()

    face_rect = face_cascade.detectMultiScale(face_img,
                                              scaleFactor = 1.1,
                                              minNeighbors = 10)
    
    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y),
                      (x + w, y + h), (255, 255, 255), 10)
    try:
       x,y,w,h = face_rect[0]
       return face_img[y:y+h,x:x+w],face_rect
    except:
       return face_img,face_rect

def get_trackIds(emotion):
    if emotion == "Sad":
        distance, index = nn.kneighbors(sadparams)
        return list(songsdb.loc[index[0]]["id"].values)
    elif emotion == "Calm":
        distance, index = nn.kneighbors(calmparams)
        return list(songsdb.loc[index[0]]["id"].values)
    elif emotion == "Happy":
        distance, index = nn2.kneighbors(happyparams)
        return list(happyAngryDb.loc[index[0]]["id"].values)
    elif emotion == "Angry":
        distance, index = nn2.kneighbors(angryparams)
        return list(happyAngryDb.loc[index[0]]["id"].values)
    

progress = 0.1
musictime = 1
trackIds = []

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read() 
    
    try:
    
        progress = spotifyObject.current_playback()["progress_ms"]
        musictime = spotifyObject.current_playback()["item"]["duration_ms"]
    
    except:
        print("Music is not yet playing")
        
    finally:
    
        adjusted,face_rect = adjusted_detect_face(frame)
        emotion = emotion_prediction(crop_square(adjusted, 48))
        
        
        # Display the resulting frame
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (200,400)
        fontScale              = 3
        fontColor              = (255,255,255)
        lineType               = 2
        
        for (x, y, w, h) in face_rect:
                        cv2.rectangle(frame, (x, y),
                                      (x + w, y + h), (255, 255, 255), 10)
                                      
        cv2.putText(frame,emotion, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.imshow('frame', frame)
        
        if progress/musictime < 0.98:
            trackIds = get_trackIds(emotion)
            trackIds = ["spotify:track:{}".format(idx) for idx in trackIds] 
            shuffle(trackIds)
            
        else:
            print("Trigger spotify playlist with {}".format(emotion))
        
            deviceId = spotifyObject.current_playback()["device"]["id"]
            
            shuffle(trackIds)
            
            spotifyObject.start_playback(deviceId, None, uris = trackIds[:20])

            time.sleep(2)
            
    
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

