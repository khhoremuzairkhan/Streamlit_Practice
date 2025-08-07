import streamlit as st
from PIL import Image

st.write('''
         # Add media files in streamlit
         ''')

image1 = Image.open('leo.jpg')
st.image(image1)


# adding video
video1 = open('leo.mp4','rb')
st.video(video1)

# add audio
st.write('''**Audio**''')

audio1=open('leo.mp3','rb')
st.audio(audio1)