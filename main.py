import cv2
import mediapipe as mp
import numpy as np


def main() -> None:
    # Initialisation de MediaPipe
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Configuration des modèles
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # 1. Ouverture de la webcam
    cap = cv2.VideoCapture(0)

    print("Webcam ouverte avec succès. Appuyer sur 'Echap' pour quitter.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Échec de la capture du frame.")
            break

        # Inversion miroir pour que ce soit plus intuitif
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convertir en RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Détection du visage
        face_results = face_detection.process(rgb_frame)

        box_center = None
        if face_results.detections:
            for detection in face_results.detections:
                # Récupérer la boîte englobante du visage
                bbox = detection.location_data.relative_bounding_box
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                
                # Dessiner le visage (Bleu)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

                # Définir la zone des gestes en dessous du visage
                # On commence à y + bh (bas du visage) + un petit décalage (40 px)
                padding = 40
                roi_w, roi_h = 250, 180
                
                # Centrer le rectangle horizontalement par rapport au visage
                roi_x = x + (bw // 2) - (roi_w // 2)
                roi_y = y + bh + padding
                
                # S'assurer que le rectangle ne sort pas de l'image (évite les crashs)
                roi_x = max(0, min(roi_x, w - roi_w))
                roi_y = max(0, min(roi_y, h - roi_h))

                # Dessiner la zone de détection des gestes (Vert)
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                cv2.putText(frame, "Zone Interaction", (roi_x, roi_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Affichage du flux
        cv2.imshow("Webcam", frame)

        # Fermer l'affichage avec ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'): # 's' pour screenshot
            cv2.imwrite("screen.png", frame)
            print("Capture sauvegardée.")

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()