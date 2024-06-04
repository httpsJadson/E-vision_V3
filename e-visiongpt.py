import base64
import requests
import pyttsx3
import openai


engine = pyttsx3.init()
def speak(speech):
    engine.say(speech)
    engine.runAndWait()

# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "cap_frame.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Qual o texto presente na imagem?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)


#CORREÇÃO DO TEXTO#
# Define o texto a ser corrigido
texto = response.json()

# Configuração da API do OpenAI
openai.api_key = ''

# Chama a função de completude para corrigir o texto
response = openai.Completion.create(
  model='gpt-3.5-turbo-0125',
  prompt=texto,
  max_tokens=300
)

# Obtém o texto corrigido
texto_corrigido = response.choices[0].text.strip()

print(texto_corrigido)
speak(texto_corrigido)
