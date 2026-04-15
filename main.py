import cv2
import mediapipe as mp
import numpy as np
import math

# Fonction pour calculer la distance entre deux points
def get_distance(p1, p2, w, h):
    return math.sqrt(((p1.x - p2.x) * w)**2 + ((p1.y - p2.y) * h)**2)


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

        # 4. et 5. Détection de gestes et mapping gestes -> sorts
        hand_results = hands.process(rgb_frame)
        geste = "Aucun"

        if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
            # On récupère les deux mains
            h1 = hand_results.multi_hand_landmarks[0].landmark
            h2 = hand_results.multi_hand_landmarks[1].landmark

            # Points d'intérêt (bouts des doigts)
            idx1, maj1, thumb1 = h1[8], h1[12], h1[4]
            idx2, maj2, thumb2 = h2[8], h2[12], h2[4]

            # Sort 1 : CLONAGE (index et majeurs des deux mains se touchent)
            dist_idx = get_distance(idx1, idx2, w, h)
            dist_maj = get_distance(maj1, maj2, w, h)
            if dist_idx < 30 and dist_maj < 30:
                geste = "CLONAGE"

            # Sort 2 : FLOU (Geste triangle où les deux index se touchent ET les deux pouces se touchent)
            dist_thumb = get_distance(thumb1, thumb2, w, h)
            if dist_idx < 40 and dist_thumb < 40 and geste == "Aucun":
                geste = "FLOU"

            # Sort 3 : Transformations (Mains serrées, poignets proches)
            dist_wrist = get_distance(h1[5], h2[5], w, h)
            if dist_wrist < 50 and geste == "Aucun":
                geste = "TRANSFORMATION"

            # Vérification si le centre entre les mains est dans la zone
            cx, cy = int(((idx1.x + idx2.x)/2)*w), int(((idx1.y + idx2.y)/2)*h)
            if roi_x < cx < roi_x + roi_w and roi_y < cy < roi_y + roi_h:
                color = (0, 0, 255) # Rouge si actif
                cv2.putText(frame, f"SORT : {geste}", (roi_x, roi_y - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            else:
                color = (0, 255, 0) # Vert si zone non atteinte

            for hl in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # --- Implémentation des sorts ---
        # 6. Effet Clonage
        if geste == "CLONAGE":
            pass
        # 7. Effet flou
        if geste == "FLOU":
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Calcul des dimensions du visage
                fx, fy, f_w, f_h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                
                # Agrandissement pour englober toute la tête
                # On ajoute 40% en largeur et 60% en hauteur (vers le haut pour les cheveux)
                head_x = max(0, int(fx - f_w * 0.2))
                head_y = max(0, int(fy - f_h * 0.5))
                head_w = min(w - head_x, int(f_w * 1.4))
                head_h = min(h - head_y, int(f_h * 1.8))
                
                # Extraction de la zone de la tête entière
                head_roi = frame[head_y:head_y+head_h, head_x:head_x+head_w]
                # Application du flou Gaussien
                blurred_head = cv2.GaussianBlur(head_roi, (99, 99), 30)
                # Réinsertion dans l'image
                frame[head_y:head_y+head_h, head_x:head_x+head_w] = blurred_head

        # 8. Effet Transformation
        if geste == "TRANSFORMATION":
            pass


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