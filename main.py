#!/usr/bin/python3
# -*- coding: utf-8 -*-
# coding=utf-8
### General imports ###
from __future__ import division
import imp
import numpy as np
import pandas as pd
import time
import re
import os
import sys
from collections import Counter
import altair as alt
import pickle
import pyaudio
import wave
import librosa
import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks

from flask import send_file
import pdfkit

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response

### Audio imports ###
#from library.speech_emotion_recognition import *

### Text imports ###

from nltk import *
from tika import parser
from werkzeug.utils import secure_filename
import tempfile

from tensorflow.keras.models import load_model


from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'

#New additions (OMKAR)
from sklearn.feature_extraction.text import CountVectorizer
#bow_vectorizer = CountVectorizer()

# Model saved with Keras model.save()
#MODEL_PATH1 ='Personality_traits_NN.h5'
MODEL_PATH1 ='cEXT.p'
MODEL_PATH2 ='cNEU.p'
MODEL_PATH3 ='cAGR.p'
MODEL_PATH4 ='cCON.p'
MODEL_PATH5 ='cOPN.p'
DL_MODEL_PATH = 'DL_model.h5'
RFC_MODEL_PATH = 'RFC_model.pkl'

# Load your trained model
'''
model1 = load_model(MODEL_PATH1)
model2 = load_model(MODEL_PATH2)
model3 = load_model(MODEL_PATH3)
model4 = load_model(MODEL_PATH4)
model5 = load_model(MODEL_PATH5)
'''

model1 = pickle.load(open('cEXT.p','rb'))
model2 = pickle.load(open('cNEU.p','rb'))
model3 = pickle.load(open('cAGR.p','rb'))
model4 = pickle.load(open('cCON.p','rb'))
model5 = pickle.load(open('cOPN.p','rb'))
model7 = pickle.load(open('RFC_model.pkl', 'rb'))
bow_vectorizer = pickle.load(open('bow_vectorizer.p','rb'))
model6 = load_model(DL_MODEL_PATH) #DL model for audio

################################################################################
################################## INDEX #######################################
################################################################################

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

################################################################################
################################## RULES #######################################
################################################################################

# Rules of the game
@app.route('/rules')
def rules():
    return render_template('rules.html')


################################################################################
############################### TEXT INTERVIEW #################################
################################################################################

global df_text

tempdirectory = tempfile.gettempdir()

@app.route('/text', methods=['POST'])
def text() :
    return render_template('text_analysis.html')

def predict_personality(text):
    try:
        scentences=[]
        scentences.append(text)
        print(scentences)
        #bow_vectorizer = CountVectorizer()
        text_vector_31 = bow_vectorizer.transform(scentences)  # Else just: text_vector_f=bow_vectorizer.transform(scentences) and EXT = cEXT.predict(text_vector_f) likewise others
        text_vector_30 = bow_vectorizer.transform(scentences)
        EXT = model1.predict_proba(text_vector_31)
        NEU = model2.predict_proba(text_vector_30)
        AGR = model3.predict_proba(text_vector_31)
        CON = model4.predict_proba(text_vector_31)
        OPN = model5.predict_proba(text_vector_31)

        EXT[0][1] = round(EXT[0][1]*100)
        NEU[0][1] = round(NEU[0][1]*100)
        AGR[0][1] = round(AGR[0][1]*100)
        CON[0][1] = round(CON[0][1]*100)
        OPN[0][1] = round(OPN[0][1]*100)

        return [EXT[0][1], NEU[0][1], AGR[0][1], CON[0][1], OPN[0][1]]
    except KeyError:
        return None
#New added---
def predict_proba(text):
    try:
        scentences=[]
        scentences.append(text)
        print(scentences)
        #bow_vectorizer = CountVectorizer()
        text_vector_31 = bow_vectorizer.transform(scentences)  # Else just: text_vector_f=bow_vectorizer.transform(scentences) and EXT = cEXT.predict(text_vector_f) likewise others
        text_vector_30 = bow_vectorizer.transform(scentences)
        EXT = model1.predict_proba(text_vector_31)
        NEU = model2.predict_proba(text_vector_30)
        AGR = model3.predict_proba(text_vector_31)
        CON = model4.predict_proba(text_vector_31)
        OPN = model5.predict_proba(text_vector_31)


        return [EXT[0][1], NEU[0][1], AGR[0][1], CON[0][1], OPN[0][1]]
    except KeyError:
        return None
#New ended---

@app.route('/oi_1',methods=['POST'])
def oi_1():
    predictions = predict_personality(text1)
    traitsT = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    trait = traitsT[predictions.index(max(predictions))]
    if trait=="Openness":
        c = make_pipeline(bow_vectorizer, model5)
        explainer = LimeTextExplainer(class_names=[0,1])
        exp = explainer.explain_instance(text1, c.predict_proba)
        exp = exp.as_html()

    return render_template('hello.html',exp=exp)

text1=""

@app.route('/text_1', methods=['POST'])
def text_1():
    global text1
    text1 = request.form.get('text')
    predictions = predict_personality(text1)
    
    #New Added---
    #probas=predict_proba(text)
    perso = {}
    perso['Extraversion'] = predictions[0]
    perso['Neuroticism'] = predictions[1]
    perso['Agreeableness'] = predictions[2]
    perso['Conscientiousness'] = predictions[3]
    perso['Openness'] = predictions[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['TRAIT', 'VALUE']
    
    df_text_perso.to_csv('static/js/db/text_perso.csv', sep=',', index=False)
    traitsT = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    trait = traitsT[predictions.index(max(predictions))]
    if trait=="Openness":
        c = make_pipeline(bow_vectorizer, model5)
        explainer = LimeTextExplainer(class_names=[0,1])
        exp = explainer.explain_instance(text1, c.predict_proba)
        exp = exp.as_html()
        #exp.save_to_file('tmp/oi.html')
    
    #New Ended---
    #Added 4March
    df_text = pd.read_csv('static/js/db/textIndividual.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([predictions], columns=traitsT))
    df_new.to_csv('static/js/db/textIndividual.txt', sep=",", index=False)

    df_text_cummulative=pd.read_csv('static/js/db/textCummulative.txt', sep=",")
    df_new2 = df_text_cummulative.append(pd.DataFrame([predictions], columns=traitsT))
    df_new2.to_csv('static/js/db/textCummulative.txt', sep=",", index=False)


    return render_template('new_text_dash.html', traits = predictions,trait=trait)

@app.route('/oi_2',methods=['POST'])
def oi_2():
    df_textIndividual = pd.read_csv('static/js/db/textIndividual.txt', sep=",")
    meansIndividual = {}
    meansIndividual['Extraversion'] = np.mean(df_textIndividual['Extraversion'])
    meansIndividual['Neuroticism'] = np.mean(df_textIndividual['Neuroticism'])
    meansIndividual['Agreeableness'] = np.mean(df_textIndividual['Agreeableness'])
    meansIndividual['Conscientiousness'] = np.mean(df_textIndividual['Conscientiousness'])
    meansIndividual['Openness'] = np.mean(df_textIndividual['Openness'])
    

    
    df_textCummulative = pd.read_csv('static/js/db/textCummulative.txt', sep=",")
    meansCummulative = []
    meansCummulative.append(np.mean(df_textCummulative['Extraversion']))
    meansCummulative.append(np.mean(df_textCummulative['Neuroticism'])) 
    meansCummulative.append(np.mean(df_textCummulative['Agreeableness']))  
    meansCummulative.append(np.mean(df_textCummulative['Conscientiousness'])) 
    meansCummulative.append(np.mean(df_textCummulative['Openness'])) 


    df_mean = pd.DataFrame.from_dict(meansIndividual, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    df_mean['Value_Others']=meansCummulative
    df_mean.to_csv('static/js/db/text_mean.csv', sep=',', index=False)

    df_mean_list=df_mean['Value'].tolist()
    traitsT = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    trait = traitsT[df_mean_list.index(max(df_mean_list))]
    df_mean['Value']=df_mean['Value'].round(decimals=2)

    return render_template('textreport.html',traits=df_mean['Value'],trait=trait)

@app.route('/returnHome', methods=['POST'])
def returnHome() :
    with open("static/js/db/textIndividual.txt", 'r+') as fp:
        # read an store all lines into list
        lines = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()

        # start writing lines except the last line
        # lines[:-1] from line 0 to the second last line
        fp.writelines(lines[0:1])
    
    return render_template('index.html')


################################################################################
############################### AUDIO INTERVIEW #################################
################################################################################

@app.route('/audio', methods=['POST'])
def audio() :
    return render_template('audio_analysis.html')

@app.route('/audio_recording', methods=['POST'])
def audio_recording():
        rec_duration = 16 # in sec
        rec_sub_dir = os.path.join('tmp','voice_recording.wav')
        duration=5
        sample_rate=22050
        chunk=1024
        channels=1
    # Start the audio recording stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

        # Create an empty list to store audio recording
        frames = []

        # Determine the timestamp of the start of the response interval
        print('* Start Recording *')
        stream.start_stream()
        start_time = time.time()
        current_time = time.time()

        # Record audio until timeout
        while (current_time - start_time) < rec_duration:

            # Record data audio data
            data = stream.read(chunk)

            # Add the data to a buffer (a list of chunks)
            frames.append(data)

            # Get new timestamp
            current_time = time.time()

        # Close the audio recording stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording * ')

        # Export audio recording to wav format
        wf = wave.open(rec_sub_dir, 'w')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return render_template('audio_analysis.html')

@app.route('/audio_dash', methods=['POST'])
def audio_dash():
    sampling_rate = 22050
    lst = []
    DL_pred_array = []
    avg_value = 0
    avg_array = []
    
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    _emotion = {1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'fearful', 7:'Disgust', 8:'Surprise'}

    
    #Preprocessing
    temp_result = []
    X, sample_rate = librosa.load(rec_sub_dir, res_type='kaiser_fast')

    #Framing
    # Number of frames
    myframes = librosa.util.frame(X,  int(len(X)/5),  int(len(X)/5), axis = 0)

    for i in range(0, len(myframes)):
        #Conversion to features
        stft = np.abs(librosa.stft(myframes[i]))  #Use only for contrast, chroma

        mfcc = np.mean(librosa.feature.mfcc(y=myframes[i], sr=sampling_rate, n_mfcc=40).T, axis=0)

        mel = np.mean(librosa.feature.melspectrogram(myframes[i], sr=sample_rate).T,axis=0)

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

        #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

        temp_result = np.hstack((mfcc, mel, chroma))
    
        arr = temp_result
        lst.append(arr)

    lst = np.asarray(lst)
    #lst_temp = np.expand_dims(lst[0], axis = 1)
    #print(lst_temp.shape)

    for i in range(0, len(lst)):
        empty_array = []
        empty_array.append(lst[i])
        #lst_temp = np.expand_dims(empty_array, axis = -1)
        #lst_temp = np.newaxis
        #DL_pred = model6.predict(lst_temp)
        DL_pred = model7.predict_proba(empty_array)
        DL_pred = DL_pred.tolist()
        DL_pred_array.append(DL_pred)
    print("PREDICTION: ",DL_pred_array)

    for i in range(0, 8):
        for j in range(0, len(DL_pred_array)):
            avg_value += DL_pred_array[j][0][i]
        avg_value = avg_value/5
        avg_array.append(avg_value)
        avg_value = 0
    print("Average array: ",avg_array)

    emotion_dist = []
    for i in range(0, len(avg_array)):
        emotion_dist.append(round(avg_array[i]*100))
    print("emotion dist: ",emotion_dist)
        
    maximum_value = max(emotion_dist)
    maximum_index = emotion_dist.index(maximum_value)
    print("Maximum value",maximum_value)
    emotion_detected = _emotion[maximum_index+1]

    df = pd.DataFrame(emotion_dist, index=_emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','new_audio_emotions_dist.csv'), sep=',')

    time.sleep(0.5)

    traitsAudio = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry','Fearful','Disgust','Surprise']
    df_audio = pd.read_csv('static/js/db/AudioIndividual.txt', sep=",")
    df_new_audio = df_audio.append(pd.DataFrame([emotion_dist], columns=traitsAudio))
    df_new_audio.to_csv('static/js/db/AudioIndividual.txt', sep=",", index=False)

    df_audio_cummulative=pd.read_csv('static/js/db/AudioCummulative.txt', sep=",")
    df_new_audio2 = df_audio_cummulative.append(pd.DataFrame([emotion_dist], columns=traitsAudio))
    df_new_audio2.to_csv('static/js/db/AudioCummulative.txt', sep=",", index=False)

    return render_template('new_audio_dash.html', traits=emotion_dist, trait=emotion_detected)

@app.route('/oi_3',methods=['POST'])
def oi_3():
    df_AudioIndividual = pd.read_csv('static/js/db/AudioIndividual.txt', sep=",")
    meansIndividual = {}
    meansIndividual['Neutral'] = np.mean(df_AudioIndividual['Neutral'])
    meansIndividual['Calm'] = np.mean(df_AudioIndividual['Calm'])
    meansIndividual['Happy'] = np.mean(df_AudioIndividual['Happy'])
    meansIndividual['Sad'] = np.mean(df_AudioIndividual['Sad'])
    meansIndividual['Angry'] = np.mean(df_AudioIndividual['Angry'])
    meansIndividual['Fearful'] = np.mean(df_AudioIndividual['Fearful'])
    meansIndividual['Disgust'] = np.mean(df_AudioIndividual['Disgust'])
    meansIndividual['Surprise'] = np.mean(df_AudioIndividual['Surprise'])
    

    
    df_AudioCummulative = pd.read_csv('static/js/db/AudioCummulative.txt', sep=",")
    meansCummulative = []
    meansCummulative.append(np.mean(df_AudioCummulative['Neutral']))
    meansCummulative.append(np.mean(df_AudioCummulative['Calm']))
    meansCummulative.append(np.mean(df_AudioCummulative['Happy']))
    meansCummulative.append(np.mean(df_AudioCummulative['Sad']))
    meansCummulative.append(np.mean(df_AudioCummulative['Angry']))
    meansCummulative.append(np.mean(df_AudioCummulative['Fearful']))
    meansCummulative.append(np.mean(df_AudioCummulative['Disgust']))
    meansCummulative.append(np.mean(df_AudioCummulative['Surprise']))


    df_mean = pd.DataFrame.from_dict(meansIndividual, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    df_mean['Value_Others']=meansCummulative
    df_mean.to_csv('static/js/db/audio_mean.csv', sep=',', index=False)

    df_mean_list=df_mean['Value'].tolist()
    traitsAudio = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry','Fearful','Disgust','Surprise']
    trait = traitsAudio[df_mean_list.index(max(df_mean_list))]
    df_mean['Value']=df_mean['Value'].round(decimals=2)
    return render_template('audioreport.html',traits=df_mean['Value'],trait=trait)

@app.route('/returnHomeAudio', methods=['POST'])
def returnHomeAudio() :
    with open("static/js/db/AudioIndividual.txt", 'r+') as fp:
        # read an store all lines into list
        lines = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()

        # start writing lines except the last line
        # lines[:-1] from line 0 to the second last line
        fp.writelines(lines[0:1])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
