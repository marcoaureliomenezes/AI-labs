import cv2
import mediapipe as mp


webcam = cv2.VideoCapture(0)

mpHands = mp.solutions.face_detection
hands = mpHands.FaceDetection()

desenho = mp.solutions.drawing_utils

while True:

    sucesso, imagem = webcam.read()

    if not sucesso:
        print("Não foi possível obter a imagem")
        break

    imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imagemRGB)

    if resultado.detections:
        for id, deteccao in enumerate(resultado.detections):
            desenho.draw_detection(imagem, deteccao)

    cv2.imshow("Imagem", imagem)

    if cv2.waitKey(5) == 27:
        break

webcam.release()