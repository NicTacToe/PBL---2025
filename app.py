import streamlit as st
from audiorecorder import audiorecorder
from speechbrain.inference import EncoderDecoderASR

st.title("SpeechxF: The Speech to Text Converter")

def SpeechToText():
  asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-conformer-transformerlm-librispeech", 
    savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
  text = asr_model.transcribe_file("audio.wav")
  return text

audio = audiorecorder("🔴 Click to Record", "🔵 Recording… Click to Stop")

if len(audio) > 0:
  st.audio(audio.export().read(), autoplay=True)  
  audio.export("audio.wav", format="wav")
  
  transcript = SpeechToText()
  st.markdown(transcript)