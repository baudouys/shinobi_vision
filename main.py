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
    # Module de segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # Configuration des modèles
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    # Création du segmenteur (0 pour modèle général, 1 pour paysage)
    segment_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

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

        # --- Génération du Masque de Segmentation ---
        # Cela détecte la personne entière sur l'image
        seg_results = segment_model.process(rgb_frame)
        
        # Le masque est une image en niveaux de gris (0 à 255)
        # On applique un seuil pour avoir un masque binaire pur (0 ou 255)
        _, binary_mask = cv2.threshold(seg_results.segmentation_mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8) # Conversion en uint8 pour OpenCV

        # 2. Détection du visage
        face_results = face_detection.process(rgb_frame)

        # Coordonnées par défaut de la tête+épaules
        person_roi = None
        mask_roi = None
        hs_x, hs_y, hs_w, hs_h = 0, 0, 0, 0

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

                # --- CALCUL ZONE TÊTE + ÉPAULES ---
                # On agrandit : 60% plus large (pour les épaules), 50% plus haut (pour les cheveux), 200% plus bas (pour le buste)
                hs_x = max(0, int(x - bw * 0.4)) # point de départ en x
                hs_y = max(0, int(y - bh * 0.6)) # point de départ en y
                hs_w = min(w - hs_x, int(bw * 1.8)) # hauteur totale
                hs_h = min(h - hs_y, int(bh * 3.2)) # largeur totale
                
                # Extraction de la zone image originale
                person_roi = frame[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w]
                # Extraction de la zone masque correspondante
                mask_roi = binary_mask[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w]

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
                cv2.putText(frame, f"{geste}", (roi_x, roi_y - 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
            else:
                color = (0, 255, 0) # Vert si zone non atteinte

            for hl in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # --- Implémentation des sorts ---
        # 6. Effet Clonage
        if geste == "CLONAGE" and person_roi is not None and mask_roi is not None:
            # Vérification de sécurité des tailles
            if person_roi.shape[:2] == mask_roi.shape[:2] and person_roi.size > 0:
                
                # Paramètres de l'effet
                nb_clones_par_cote = 3
                offset_x_base = hs_w * 0.8
                offset_y_base = -70 # Décalage vertical pour le V (négatif = monte)
                
                for i in range(1, nb_clones_par_cote + 1):
                    # Calcul des décalages cumulatifs
                    current_offset_x = int(offset_x_base * i)
                    current_offset_y = int(offset_y_base * i)
                    dest_y = hs_y + current_offset_y
                    
                    # Opacité dégressive (plus transparent quand éloigné)
                    opacity = 0.8 / i 

                    # Fonction interne pour appliquer le clone détouré
                    def apply_cloned_person(dest_x, dest_y, current_frame, roi_color, roi_mask, alpha):
                        # Vérifier les limites de l'image
                        if dest_x < 0 or dest_y < 0 or dest_x + hs_w >= w or dest_y + hs_h >= h:
                            return current_frame
                        
                        # Isoler la zone de destination en arrière-plan
                        bg_target = current_frame[dest_y:dest_y+hs_h, dest_x:dest_x+hs_w]
                        
                        # Vérifier la correspondance des tailles (sécurité)
                        if bg_target.shape[:2] != roi_color.shape[:2]:
                            return current_frame

                        # --- LA MAGIE EST ICI : Le "Poisson Blending" manuel ---
                        # 1. Créer un masque inverse (le fond de la destination)
                        mask_inv = cv2.bitwise_not(roi_mask)
                        
                        # 2. Noirci la zone de la personne dans l'arrière-plan cible
                        # (On ne garde que le décor là où la personne va aller)
                        bg_cleaned = cv2.bitwise_and(bg_target, bg_target, mask=mask_inv)
                        
                        # 3. Noirci le fond dans la ROI de la personne
                        # (On ne garde que la personne, le fond devient noir)
                        person_cleaned = cv2.bitwise_and(roi_color, roi_color, mask=roi_mask)
                        
                        # 4. Combiner les deux (Personne détourée + Décor propre)
                        # C'est comme coller un autocollant parfaitement découpé
                        cloned_combined = cv2.add(bg_cleaned, person_cleaned)
                        
                        # 5. Appliquer l'opacité (addWeighted entre le décor original et le clone combiné)
                        result_zone = cv2.addWeighted(bg_target, 1 - alpha, cloned_combined, alpha, 0)
                        
                        # Réinsérer dans l'image finale
                        current_frame[dest_y:dest_y+hs_h, dest_x:dest_x+hs_w] = result_zone
                        return current_frame

                    # --- Clone GAUCHE ---
                    left_x = hs_x - current_offset_x
                    frame = apply_cloned_person(left_x, dest_y, frame, person_roi, mask_roi, opacity)

                    # --- Clone DROIT ---
                    right_x = hs_x + current_offset_x
                    frame = apply_cloned_person(right_x, dest_y, frame, person_roi, mask_roi, opacity)

            #cv2.putText(frame, "SORT : TRINITÉ SUPRÊME (V)", (50, 50), 
                        #cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        # 7. Effet flou
        if geste == "FLOU":
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Calcul des dimensions du visage
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                
                # Agrandissement pour englober toute la tête
                # On ajoute 40% en largeur et 60% en hauteur (vers le haut pour les cheveux)
                head_x = max(0, int(x - bw * 0.2))
                head_y = max(0, int(y - bh * 0.5))
                head_w = min(w - head_x, int(bw * 1.4))
                head_h = min(h - head_y, int(bh * 1.8))
                
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
    segment_model.close()
    hands.close()
    face_detection.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()