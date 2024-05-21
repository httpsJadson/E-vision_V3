from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import cv2
import mediapipe as mp
import math
import pytesseract
from time import time
import os

app = Flask(__name__)

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

POLEGAR_TOPO = 4
MINDINHO_TOPO = 20

modo_leitura = False
controle_leitura = False

tempo = 0
texto_extraido = ""

def gen_frames():
    global cap, modo_leitura, controle_leitura, tempo, texto_extraido
    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not modo_leitura:
            hand_results = hands.process(frame_to_rgb)

            if hand_results.multi_hand_landmarks:
                for landmarks in hand_results.multi_hand_landmarks:
                    for _id, landmark in enumerate(landmarks.landmark):
                        h, w, c = frame.shape
                        polegar = landmarks.landmark[POLEGAR_TOPO]
                        polegar_x, polegar_y = int(polegar.x * w), int(polegar.y * h)
                        mindinho = landmarks.landmark[MINDINHO_TOPO]
                        mindinho_x, mindinho_y = int(mindinho.x * w), int(mindinho.y * h)
                        distancia = math.sqrt((polegar_x - mindinho_x) ** 2 + (polegar_y - mindinho_y) ** 2)
                        max_distance = 20

                        if distancia < max_distance and not controle_leitura:
                            controle_leitura = True
                            modo_leitura = not modo_leitura
                            if modo_leitura:
                                tempo = time()

                        elif distancia > max_distance and controle_leitura:
                            controle_leitura = False

                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), 1)

        if modo_leitura and (time() - tempo) >= 10:
            cv2.imwrite('cap_frame.png', frame)
            modo_leitura = False
            image = cv.imread('cap_frame.png') # CARREGAR IMAGEM
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # CONVERTER PARA ESCALA DE CINZA
            denoised = cv.fastNlMeansDenoising(gray, None, 10, 7, 21) # REDUÇÃO DE RUIDO
            binary = cv.adaptiveThreshold(denoised, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2) # BINARIZAÇÃO ADAPTATIVA
            texto_extraido = pytesseract.image_to_string(binary, lang='por')
            texto_extraido = texto_extraido.replace('\n', ' ').strip() # REMOÇÃO DE QUEBRAS DE LINHA E ESPAÇOS EXTRAS
            texto_extraido = texto_extraido.replace('- ', '') # REMOÇÃO DE HÍFENS QUE INDICAM PALAVRAS DIVIDIDAS ENTRE LINHAS
            print("texto_extraido", texto_extraido, end=' ')
            

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', texto_extraido=texto_extraido)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/texto_extraido', methods=['GET'])
def get_texto_extraido():
    return jsonify(texto_extraido)

if __name__ == '__main__':
    app.run(debug=True)