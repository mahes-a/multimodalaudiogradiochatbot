import gradio as gr
import requests
import base64
import io
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
import os

#Reference
#https://learn.microsoft.com/en-us/azure/ai-services/openai/audio-completions-quickstart?tabs=keyless%2Cwindows%2Ctypescript-keyless&pivots=rest-api

load_dotenv()  

# Set environment variables or edit the corresponding values here.
endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

# Keyless authentication
credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")



# Initialize a list to store the conversation history
conversation_history = []
gradio_history = []

def chatbot_response(text_input=None, audio_input=None):
    api_version = '2025-01-01-preview'
    #if your deploymnet name is gpt-4o-audio-preview update below
    url = f"{endpoint}/openai/deployments/gpt-4o-audio-preview/chat/completions?api-version={api_version}"
    #for api key version -- not recommended for security reasons
    #headers = { "api-key": api_key, "Content-Type": "application/json" }
    headers= { "Authorization": f"Bearer {token.token}", "Content-Type": "application/json" }
    
    if text_input:
        # Append the new text input to the conversation history
        conversation_history.append({ "role": "user", "content": [{ "type": "text", "text": text_input }] })
        # Append the new text input to the gradio chatbot ui history
        gradio_history.append(("User", text_input))
        
        body = {
            "modalities": ["audio","text"],
            "model": "gpt-4o-audio-preview",
            "audio": {
                "format": "wav",
                "voice": "alloy"
            },
            "messages": conversation_history
        }
        
        # assumes happy path no error handling done here
        response = requests.post(url, headers=headers, json=body)
        response_data = response.json()
        transcript = response_data['choices'][0]['message']['audio']['transcript']
        
        audio_data = response_data['choices'][0]['message']['audio']['data']
        wav_bytes = base64.b64decode(audio_data)
        
        # Use an in-memory buffer instead of saving to disk only works for the latest audio input and output
        audio_buffer = io.BytesIO(wav_bytes)
        
        # Convert the in-memory buffer to a numpy array
        audio_array, sample_rate = sf.read(audio_buffer)
        
        # Convert the numpy array back to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_array, sample_rate, format='wav')
        audio_bytes.seek(0)
        
        # Append the assistant's response to the conversation history
        conversation_history.append({ 
            "role": "assistant", 
            "audio": { "id": response_data['choices'][0]['message']['audio']['id'] } 
        }) 
        
        # Append the assistant's response to the gradio ui conversation history
        gradio_history.append(("assistant", gr.Audio(audio_bytes.read(), format="wav")))
        
        return gradio_history, transcript
    
    elif audio_input:
        with open(audio_input, 'rb') as wav_reader:
            encoded_string = base64.b64encode(wav_reader.read()).decode('utf-8')
        
        # Append the new audio input to the conversation history
        gradio_history.append(("user", gr.Audio(audio_input)))
        
        conversation_history.append({ 
            "role": "user", 
            "content": [
                { "type": "text", "text": "provide response for audio input" },
                { "type": "input_audio", "input_audio": { "data": encoded_string, "format": "wav" } }
            ] 
        })
        
        body = {
            "modalities": ["audio","text"],
            "model": "gpt-4o-audio-preview",
            "audio": {
                "format": "wav",
                "voice": "alloy"
            },
            "messages": conversation_history
        }
        # assumes happy path no error handling done here
        response = requests.post(url, headers=headers, json=body)
        response_data = response.json()
        transcript = response_data['choices'][0]['message']['audio']['transcript']
        
        audio_data = response_data['choices'][0]['message']['audio']['data']
        wav_bytes = base64.b64decode(audio_data)
        
        # Use an in-memory buffer instead of saving to disk
        audio_buffer = io.BytesIO(wav_bytes)
        
        # Convert the in-memory buffer to a numpy array
        audio_array, sample_rate = sf.read(audio_buffer)
        
        # Convert the numpy array back to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_array, sample_rate, format='wav')
        audio_bytes.seek(0)
        
        # Append the assistant's response to the conversation history
        conversation_history.append({ 
            "role": "assistant", 
            "audio": { "id": response_data['choices'][0]['message']['audio']['id'] } 
        }) 
        
        gradio_history.append(("assistant", gr.Audio(audio_bytes.read(), format="wav")))
        
        return gradio_history, transcript

# Define the Gradio interface
# Chatbot displays audio input and output only for the recent conversation older conversations doesnot play audio 
iface = gr.Interface(
    fn=chatbot_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text here..."),
        gr.Audio(sources=["microphone"], type="filepath")
    ],
    outputs=[
        gr.Chatbot(label="Chat History"),
         gr.Textbox(label="Audio Response Transcription")
    ]
    #,live=True
)

# Launch the interface
iface.launch()