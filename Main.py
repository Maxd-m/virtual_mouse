import cv2 as cv
import time
import mouse
import tkinter as tk
import numpy as np
from hand_detector import HandDetector


class Gestures:
    def __init__(self) -> None:
        self.screen_width = 0
        self.screen_height = 0
        self.cam_width = 1920
        self.cam_height = 1080

        self.bound_x = int(self.cam_width * 0.2)
        self.bound_y = int(self.cam_height * 0.1)
        self.bound_w = int(self.cam_width * 0.6)
        self.bound_h = int(self.cam_height * 0.7)

        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 0.5

        # Variables para control de click
        self.clicking = False
        self.click_delay = 0.3
        self.last_click_time = 0

    def init_screen_size(self):
        root = tk.Tk()
        root.withdraw()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()

    def calculate_finger_distance(self, p1, p2):
        # Obtenido de ChatGPT
        return np.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def click_mouse(self, img, landmark_list):
        if len(landmark_list) >= 9:  # Aseguramos que tenemos los landmarks necesarios (0-8)
            # Obtener posiciones de la punta del índice y su articulación
            index_tip = landmark_list[8]  # Punta del dedo índice
            index_joint = landmark_list[6]  # Articulación del dedo índice

            # Calcular distancia entre la punta y la articulación
            distance = self.calculate_finger_distance(index_tip, index_joint)

            # Dibujar línea entre los puntos de referencia
            cv.line(img, (index_tip[1], index_tip[2]),
                    (index_joint[1], index_joint[2]), (255, 0, 0), 2)

            current_time = time.time()
            # Si el dedo está doblado  y ha pasado suficiente tiempo
            if distance < 40 and not self.clicking and current_time - self.last_click_time > self.click_delay:
                mouse.click()
                self.clicking = True
                self.last_click_time = current_time
                return True
            elif distance >= 40:
                self.clicking = False

        return False

    def move_mouse(self, img, landmark_list):
        if len(landmark_list) == 0:
            return

        # Obtener la posición del dedo índice (landmark 8)
        index_finger = landmark_list[8]
        x = self.cam_width - index_finger[1]
        y = index_finger[2]

        # Verificar si el punto está dentro del área de detección
        if (self.bound_x <= x <= self.bound_x + self.bound_w and
                self.bound_y <= y <= self.bound_y + self.bound_h):
            # Mapear coordenadas a la pantalla
            screen_x = np.interp(x,
                                 (self.bound_x, self.bound_x + self.bound_w),
                                 (0, self.screen_width))
            screen_y = np.interp(y,
                                 (self.bound_y, self.bound_y + self.bound_h),
                                 (0, self.screen_height))

            # Aplicar suavizado
            smooth_x = self.prev_x * self.smoothing + screen_x * (1 - self.smoothing)
            smooth_y = self.prev_y * self.smoothing + screen_y * (1 - self.smoothing)

            self.prev_x, self.prev_y = smooth_x, smooth_y

            # Mover el mouse
            mouse.move(smooth_x, smooth_y)
            return True

        return False

    def process_hand_gestures(self, img, landmark_list):
        if len(landmark_list) == 0:
            return

        # Procesar movimiento y click
        is_in_bounds = self.move_mouse(img, landmark_list)
        is_clicking = self.click_mouse(img, landmark_list)

        # Visualización
        if len(landmark_list) >= 9:  # Asegurarse de que existe el landmark del índice
            index_finger = landmark_list[8]
            if is_in_bounds:
                # Verde si está haciendo click, magenta si no
                color = (0, 255, 0) if is_clicking else (255, 0, 255)
                cv.circle(img, (index_finger[1], index_finger[2]), 15, color, cv.FILLED)
            else:
                # Rojo si está fuera del área
                cv.circle(img, (index_finger[1], index_finger[2]), 15, (0, 0, 255), cv.FILLED)


def main():
    detector = HandDetector(maxHands=1, detectionCon=0.7)
    gestures = Gestures()
    gestures.init_screen_size()

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, gestures.cam_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, gestures.cam_height)

    actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolución de cámara: {actual_width}x{actual_height}")

    if actual_width != gestures.cam_width or actual_height != gestures.cam_height:
        gestures.cam_width = int(actual_width)
        gestures.cam_height = int(actual_height)
        gestures.bound_x = int(gestures.cam_width * 0.2)
        gestures.bound_y = int(gestures.cam_height * 0.1)
        gestures.bound_w = int(gestures.cam_width * 0.6)
        gestures.bound_h = int(gestures.cam_height * 0.7)

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmlist = detector.findPosition(img, draw=False)

        gestures.process_hand_gestures(img, lmlist)

        cv.rectangle(img,
                     (gestures.bound_x, gestures.bound_y),
                     (gestures.bound_x + gestures.bound_w, gestures.bound_y + gestures.bound_h),
                     (255, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        img = cv.flip(img, 1)

        cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        display_width = 1280
        if img.shape[1] > display_width:
            scale = display_width / img.shape[1]
            display_height = int(img.shape[0] * scale)
            img_display = cv.resize(img, (display_width, display_height))
            cv.imshow('Image', img_display)
        else:
            cv.imshow('Image', img)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()