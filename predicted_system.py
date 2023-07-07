# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

## loading the model
loaded_model=pickle.load(open('D:/aiml/Trained_model.sav','rb'))

## model predict
t1=np.array([[12,25,43,43.004459,82.320763,7.840207,263.964248]])
t1

pre = loaded_model.predict(t1)
class_labels = ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", "grapes", "jute", "kidneybeans", 
                "lentil", "maize", "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
                "rice", "watermelon"]

for prediction in pre:
    if prediction >= 0 and prediction < len(class_labels):
        print(class_labels[prediction])