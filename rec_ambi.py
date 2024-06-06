# BIBLIOTECAS #
import cv2 as cv
import mediapipe as mp
import math
import pytesseract
import pyttsx3
from time import time
import os
import base64
import requests

# DIRETÓRIO DO EXECUTÁVEL #
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# KEYPOINTS DAS MÃOS # 
POLEGAR_TOPO = 4
MINDINHO_TOPO = 20

# VARIÁVEIS DE CAPTURA DA CÂMERA E HANDMARKS #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands() 
cap = cv.VideoCapture(1)

# VARIÁVEIS DE CONTROLE #
modo_leitura = False
controle_leitura = False

# Inicializar o motor de fala
engine = pyttsx3.init()

print(pytesseract.get_languages())
tempo = 0
# FUNCTIONS #

def speak(speech):
    engine.say(speech)
    engine.runAndWait()

def teste():
    print(tempo)
    print(f"Tempo de execução do programa: {(time() - tempo)}")

speak(
    "Olá, eu sou a hivi, sou uma assistente de leitura virtual, sou capaz de realizar leitura de textos impressos"
)

while cap.isOpened():

    status, frame = cap.read()

    if status == False:
        break

    frame_to_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    if not modo_leitura:  # Verifica se não está no modo de captura de texto

        hand_results = hands.process(frame_to_rgb)

        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                for _id, landmark in enumerate(landmarks.landmark):
                    h, w, c = frame.shape
                    # POLEGAR
                    polegar = landmarks.landmark[POLEGAR_TOPO]
                    polegar_x, polegar_y = int(polegar.x * w), int(polegar.y * h)
                    # MINDINHO
                    mindinho = landmarks.landmark[MINDINHO_TOPO]
                    mindinho_x, mindinho_y = int(mindinho.x * w), int(mindinho.y * h)
                    # Distância euclidiana -> d = √ ( X2 - X1 )^2 + ( Y2 - Y1 )^2
                    distancia = math.sqrt( ( polegar_x - mindinho_x ) ** 2 + ( polegar_y - mindinho_y) ** 2 ) 
                    max_distance = 20
                    if distancia < max_distance and not controle_leitura:
                        controle_leitura = True
                        modo_leitura = not modo_leitura
                        if modo_leitura:
                            tempo = time()
                            print(tempo)
                            print("Captando imagem, aguarde e evite movimentar-se")
                            speak("Captando imagem, aguarde e evite movimentar-se")
                    elif distancia > max_distance and controle_leitura:
                        controle_leitura = False
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv.circle(frame, (cx, cy), 5, (0, 255, 0), 1)


    if modo_leitura and ( time() - tempo ) >= 10:
        cv.imwrite('cap_frame.png', frame)
        modo_leitura = False

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
                "text": "Descreva o que está presente no ambiente da imagem."
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

        # Extrair o texto identificado da resposta
        texto_identificado = response.json()['choices'][0]['message']['content']

        # Encontrar o índice onde começa o texto identificado
        inicio_texto = texto_identificado.find('O texto presente na imagem é:') + len('O texto presente na imagem é:')

        # Extrair apenas o texto identificado
        text = texto_identificado[inicio_texto:]
        text = text.replace('\n', ' ').strip() # REMOÇÃO DE QUEBRAS DE LINHA E ESPAÇOS EXTRAS
        text = text.replace('- ', '') # REMOÇÃO DE HÍFENS QUE INDICAM PALAVRAS DIVIDIDAS ENTRE LINHAS

        if len(text):
            print(text, end=' ')
            speak(text)
        else:
            feedback = 'Não foi possível detectar nenhum texto.'
            print(feedback)
            speak(feedback)


    cv.imshow('E-Vision', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

teste()
cap.release()
cv.destroyAllWindows()