# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:57:03 2023

@author: Tathagata
"""

import numpy as np
import pickle 
import streamlit as st
from streamlit_option_menu import option_menu


## loading the model
loaded_model=pickle.load(open('D:/aiml/Trained_model.sav','rb'))

## creating a function for  prediction

def recommend(t1):
    
    ## model predict
    x=np.asarray(t1)
    x_reshape=x.reshape(1, -1)
    pre = loaded_model.predict(x_reshape)
    class_labels = ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", "grapes", "jute", "kidneybeans", 
                    "lentil", "maize", "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
                    "rice", "watermelon"]

    for prediction in pre:
        if prediction >= 0 and prediction < len(class_labels):
            return class_labels[prediction]
        

def main():
    
    #as sidebar
    with st.sidebar:
        selected=option_menu( menu_title="Main Menu",
                             options=["Home","Project","Contact Us"])
    if selected=="Home":
        #giving a title
        st.title('Crop Recommendation Web App')
        
        #getting the onput data from the user
        ## N	P	K	temperature	humidity	ph	rainfall
        N=st.text_input('Enter the soil nitrogen contain')
        P=st.text_input('Enter the soil Phosphorus contain')
        K=st.text_input('Enter the soil Potasium contain')
        temperature=st.text_input('Enter the temperature')
        humidity=st.text_input('Enter the humidity contain')
        ph=st.text_input('Enter the soil ph contain')
        rainfall=st.text_input('Enter the amount of rainfall')
        
        ## code for prediction
        recommending=''
        
        #button creating
        
        if st.button('Crop Recommendation'):
            recommending=recommend([N,P,K,temperature,humidity,ph,rainfall])
            
        st.success(recommending)
    
    
    if selected=="Project":
        st.title("Coming soon")
        text = st.text_area("Enter your text", value="", height=150)
        st.write("You entered:")
        st.write(text)
    
    if selected=="Contact Us":
        st.title("contact us on email")
        st.text("tathagatadasgupta1234@gmail.com")
    
    
if __name__=='__main__':
    main()

    
    
    
    
    
    
    
    
    
    