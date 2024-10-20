import cv2 as cv
import mediapipe as mp
import time
import mouse
import tkinter as tk



class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComp = modelComp

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        # Dibujar los puntos de las mano y sus conexiones
        if self.results.multi_hand_landmarks:
            for handLmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLmks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10,(255,0,255), cv.FILLED)

        return lmList

class Gestures():
    def __init__(self) -> None:
       pass
        
    def move_mouse(self,x,y,width,height):
        #x,y son cooordenadas destino
        #width y height son medidas de la pantalla

        #transforma coordenadas de camara a pantalla
        movX=width-(x*width)/610
        movY=(y*height)/450
        #obtiene posicion del mouse
        position=mouse.get_position()
        #mueve el mouse a las coordenadas de X y Y 
        mouse.drag(position[0],position[1],movX,movY,True,0.5)

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()

    # Crear una ventana oculta
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana

    # Obtener el tamaño de la pantalla
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    # Cerrar la ventana
    root.destroy()
    

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition (img)
        movement = Gestures()
        if len(lmlist) != 0:
            print(lmlist[8])
            #el nodo 8 corresponde al dedo indice, la posicion 1 a X, la 2 a Y
            movement.move_mouse(lmlist[8][1],lmlist[8][2],width,height)
            #time.sleep(0.5)



        # Calcular FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Modo Espejo
        img = cv.flip(img, 1)

        # Ejecutar la camara
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow('Image', img)
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()