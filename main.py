import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import pygame


# --- Initialisation Audio ---
# Initialise le module audio de Pygame pour jouer des sons
pygame.mixer.init()

def load_sound(name):
    path = os.path.join("sounds_src", name)
    if os.path.exists(path):
        return pygame.mixer.Sound(path)
    return None

# --- Fonctions utilitaires ---
def get_distance(p1, p2, w, h):
    """Calcule la distance entre deux points en pixels, en tenant compte de la résolution"""
    return math.sqrt(((p1.x - p2.x) * w)**2 + ((p1.y - p2.y) * h)**2)

def load_transformation_images():
    """Charge les images de transformation (Messi, Saitama, ballon) et leurs points de référence"""
    images = {}
    folder = "img_src"
    files = {"messi": "messi.png", "saitama": "saitama.png", "ballon": "ballon.png"}
    
    # # Coordonnées de référence pour le warping [oreille droite, oreille gauche, Menton]
    refs = {
        "messi": np.array([[1236, 867], [499, 836], [812, 1388]], dtype=np.float32),
        "saitama": np.array([[1467, 904], [558, 904], [941, 1213]], dtype=np.float32),
        "ballon": np.array([[4120, 2525], [380, 2525], [2250, 4123]], dtype=np.float32)
    }

    # Chargement des images
    for key, filename in files.items():
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED) # Charge avec transparence si PNG
            if img is not None:
                images[key] = {"img": img, "ref": refs[key]} # Stocke image + points de référence
        else:
            print(f"Fichier introuvable : {filepath}")
    return images


def apply_cloned_person(dest_x, dest_y, current_frame, roi_color, roi_mask, hs_w, hs_h):
    """
    Applique un clone de la personne à une position donnée dans le frame.
    Gère les bords de l'écran et la transparence.
    """
    h_f, w_f = current_frame.shape[:2]
    
    # Calcul des limites pour ne pas dépasser de l'écran
    x1, y1 = max(0, int(dest_x)), max(0, int(dest_y)) # Coin supérieur gauche visible
    x2, y2 = min(w_f, x1 + hs_w), min(h_f, y1 + hs_h) # Coin inférieur droit visible
    
    # Calcul des dimensions réelles à copier
    copy_w, copy_h = x2 - x1, y2 - y1
    
    if copy_w <= 0 or copy_h <= 0:
        return current_frame # Si la zone est invalide, retourne le frame inchangé

    # Découpage de la ROI (Region Of Interest) et du masque
    roi_sub = roi_color[0:copy_h, 0:copy_w] # Partie de l'image à coller
    mask_sub = roi_mask[0:copy_h, 0:copy_w] # Masque correspondant
    
    bg_target = current_frame[y1:y2, x1:x2] # Zone cible dans le frame
    
    if bg_target.shape[:2] != roi_sub.shape[:2]:
        return current_frame # Si les tailles ne correspondent pas

    # Création des masques inverses pour la fusion
    mask_inv = cv2.bitwise_not(mask_sub) # Masque inversé (fond)
    bg_cleaned = cv2.bitwise_and(bg_target, bg_target, mask=mask_inv) # Fond sans la personne
    person_cleaned = cv2.bitwise_and(roi_sub, roi_sub, mask=mask_sub) # Personne sans fond
    
    # Fusion des deux couches
    current_frame[y1:y2, x1:x2] = cv2.add(bg_cleaned, person_cleaned)
    return current_frame


def load_substitution_background():
    """Charge une image de fond pour la disparition."""
    path = os.path.join("img_src", "disparition_bg.jpeg")
    if os.path.exists(path):
        return cv2.imread(path)
    else:
        # Crée un fond gris neutre si aucune image n'est trouvée
        return np.full((720, 1280, 3), (150, 150, 150), dtype=np.uint8)


# --- MAIN ---
def main():
    # Initialisation des modèles MediaPipe
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_seg = mp.solutions.selfie_segmentation
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialisation des détecteurs
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    seg_model = mp_selfie_seg.SelfieSegmentation(model_selection=0)

    # Chargement des assets (images de transformation)
    assets = load_transformation_images()
    choix_transfo = ["messi", "saitama", "ballon"]
    transfo_idx = 1 # Saitama par défaut

    # Variables d'état du sort
    sort_actif = False # Indique si un sort est actif
    type_sort_actif = "Aucun" # Type de sort actuel
    temps_activation = 0 # Temps de début du sort
    DUREE_MAX = 30 # Durée maximale d'un sort en secondes

    # Chargement des sons
    son_clonage = load_sound("clonage.mp3")
    son_transfo = load_sound("transformation.mp3")
    son_flou = load_sound("flou.mp3")
    son_cancel = load_sound("cancel.mp3")
    son_territoire = load_sound("territoire.mp3")
    son_disparition = load_sound("disparition.mp3") 
    img_territory = cv2.imread("img_src/territoire.png") # Image du territoire
    img_substitution = load_substitution_background() # Fond pour la disparition
    last_valid_pts = None # Mémorise les derniers points valides du visage (pour la Disparition)
    
    # Initialisation de la capture vidéo
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame_brut = cap.read()
        if not ret: break

        frame = cv2.flip(frame_brut, 1) # Retourne horizontalement (effet miroir) pour un rendu plus naturel
        h, w, _ = frame.shape # Dimensions du frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Conversion pour MediaPipe
        current_time = time.time()

        # Redimensionnement de l'image du territoire (pour pouvoir bouger dedans)
        img_territory_large = cv2.resize(img_territory, (int(w*1.5), int(h*1.5)))

        # Copie du frame pour l'affichage (les effets iront dessus)
        # On garde la frame brute pour détecter visage et gestes en cas de flou ou disparition
        frame_affichage = frame.copy()

        # Initialisation des variables par frame
        geste_instantane = "Aucun"
        face_detected = False
        dst_warping_points = None
        person_roi, mask_roi = None, None
        hs_x, hs_y, hs_w, hs_h = 0, 0, 0, 0
        roi_x, roi_y, roi_w, roi_h = 0, 0, 250, 180

        # --- Segmentation (pour extraire la personne du fond) ---
        seg_res = seg_model.process(rgb_frame)
        _, binary_mask = cv2.threshold(seg_res.segmentation_mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # --- Détection du visage ---
        mesh_res = face_mesh.process(rgb_frame)
        if mesh_res.multi_face_landmarks:
            face_detected = True
            lm = mesh_res.multi_face_landmarks[0].landmark # Landmarks du visage
            
            # Points de référence : Oreille G (234), Oreille D (454), Menton (152)
            pts = []
            for i in [234, 454, 152]:
                pts.append([lm[i].x * w, lm[i].y * h]) # Conversion en coordonnées pixels
            dst_warping_points = np.array(pts, dtype=np.float32)

            # Mémorisation des points pour la disparition
            last_valid_pts = dst_warping_points.copy()

            # Zone d'interaction (boîte autour de la zone de geste)
            roi_x, roi_y = int(lm[152].x * w) - 125, int(lm[152].y * h) + 50
            roi_x, roi_y = max(0, min(roi_x, w-roi_w)), max(0, min(roi_y, h-roi_h)) # Limites de l'écran

            # Zone de clonage (tête + épaules)
            fw = int((lm[454].x - lm[234].x) * w) # Largeur du visage
            fh = int((lm[152].y - lm[10].y) * h) # Hauteur du visage
            hs_w, hs_h = int(fw * 2.4), int(fh * 3.5) # Dimensions élargies pour le corps
            hs_x, hs_y = int(lm[1].x * w) - hs_w//2, int(lm[1].y * h) - hs_h//3 # Position
            
            # Capture sécurisée de la ROI (si dans les limites)
            if hs_x >= 0 and hs_y >= 0 and hs_x+hs_w < w and hs_y+hs_h < h:
                person_roi = frame[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w].copy() # Image de la personne
                mask_roi = binary_mask[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w].copy() # Masque correspondant

        # --- Détection des mains et gestes ---
        hand_res = hands.process(rgb_frame)
        if hand_res.multi_hand_landmarks and len(hand_res.multi_hand_landmarks) == 2:
            h1 = hand_res.multi_hand_landmarks[0].landmark # Points de la main 1
            h2 = hand_res.multi_hand_landmarks[1].landmark # Points de la main 2
            
            # GESTE DE LA CROIX (Annulation du sort)
            idx1, idx2 = h1[6], h2[6] # Index des deux mains
            dist_indices_cancel = get_distance(idx1, idx2, w, h)
            
            # Vérifie si les index se croisent (pour annuler le sort)
            is_crossed = (idx1.x > h2[0].x and idx2.x < h1[0].x) or (idx1.x < h2[0].x and idx2.x > h1[0].x)
            
            if dist_indices_cancel < 20 and is_crossed:
                if sort_actif: 
                    if son_cancel: son_cancel.play() # Joue le son d'annulation
                sort_actif = False
                type_sort_actif = "Aucun"

            # DÉTECTION DES GESTES DANS LA ZONE D'INTERACTION
            cx_hand, cy_hand = int(((idx1.x + idx2.x)/2)*w), int(((idx1.y + idx2.y)/2)*h)
            if roi_x < cx_hand < roi_x + roi_w and roi_y < cy_hand < roi_y + roi_h:
                
                # Calcul des distances entre différents points des mains
                d_index = get_distance(h1[8], h2[8], w, h)
                d_pouces = get_distance(h1[4], h2[4], w, h)
                d_majeurs = get_distance(h1[12], h2[12], w, h)
                d_base_maj = get_distance(h1[9], h2[9], w, h)
                d_paumes = get_distance(h1[0], h2[0], w, h)
                d_auriculaires = get_distance(h1[20], h2[20], w, h)

                # LOGIQUE DE RECONNAISSANCE DES GESTES
                if d_paumes < 30: geste_instantane = "EXTENSION_TERRITOIRE" # Mains jointes
                elif d_base_maj < 20: geste_instantane = "TRANSFORMATION" # Poings serrées
                elif d_index < 20 and d_pouces < 20: geste_instantane = "FLOU" # Index et pouces proches
                elif d_index < 20 and d_majeurs < 20: geste_instantane = "CLONAGE" # Index proches
                elif d_auriculaires < 20 and d_index > 60: geste_instantane = "DISPARITION" # Auriculaires proches

            # Activation du sort si un geste est détecté
            if geste_instantane != "Aucun" and (not sort_actif or type_sort_actif != geste_instantane):
                # Joue le son correspondant au geste
                if geste_instantane == "CLONAGE" and son_clonage: son_clonage.play()
                if geste_instantane == "TRANSFORMATION" and son_transfo: son_transfo.play()
                if geste_instantane == "FLOU" and son_flou: son_flou.play()
                if geste_instantane == "EXTENSION_TERRITOIRE" and son_territoire: son_territoire.play()
                if geste_instantane == "DISPARITION" and son_disparition: son_disparition.play()

                sort_actif = True
                type_sort_actif = geste_instantane
                temps_activation = current_time    

            # Dessin des landmarks de la main
            #for hl in hand_res.multi_hand_landmarks:
                #mp_drawing.draw_landmarks(frame_affichage, hl, mp_hands.HAND_CONNECTIONS)

        # --- Gestion du Timer et Affichage ---
        if sort_actif:
            dt_sort = current_time - temps_activation
            if dt_sort > DUREE_MAX:
                sort_actif = False
                type_sort_actif = "Aucun"
            else:
                temps_restant = int(DUREE_MAX - dt_sort)
                
                # Affichage du nom du sort (haut gauche) et du timer (haut droite)
                cv2.putText(frame_affichage, f"SORT: {type_sort_actif}", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                txt_timer = f"TEMPS: {temps_restant}s"
                size_timer = cv2.getTextSize(txt_timer, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                pos_x_timer = w - size_timer[0] - 20 # Position à droite
                cv2.putText(frame_affichage, txt_timer, (pos_x_timer, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        # --- Application des effets ---
        if sort_actif:
            if type_sort_actif == "CLONAGE" and person_roi is not None:
                # Création de clones autour de la personne
                gx, gy = int(hs_w * 0.9), int(hs_h * 0.4)
                # Positions relatives des clones (x_offset, y_offset)
                pos = [(0, -gy*1.5), (-gx*2, -gy), (gx*2, -gy), (-gx, 0), (gx, 0), (0, gy*1.2)]
                
                # Séparation des clones en deux groupes (derrière et devant)
                clones_derriere = []
                clones_devant = []
                
                for ox, oy in pos:
                    if abs(ox) < 10 and abs(oy) < 10: continue # Ignore la position centrale pour ne pas doubler la personne   
                    if oy <= 0: clones_derriere.append((ox, oy)) # Clones derrière
                    else: clones_devant.append((ox, oy)) # Clones devant
                
                # Tri des clones pour un rendu correct (du fond vers l'avant)
                clones_derriere.sort(key=lambda p: p[1])
                clones_devant.sort(key=lambda p: p[1])

                 # Application des clones (derrière -> personne -> devant)
                for ox, oy in clones_derriere:
                    frame_affichage = apply_cloned_person(hs_x+ox, hs_y+oy, frame_affichage, person_roi, mask_roi, hs_w, hs_h)

                frame_affichage = apply_cloned_person(hs_x, hs_y, frame_affichage, person_roi, mask_roi, hs_w, hs_h)

                for ox, oy in clones_devant:
                    frame_affichage = apply_cloned_person(hs_x+ox, hs_y+oy, frame_affichage, person_roi, mask_roi, hs_w, hs_h)

            elif type_sort_actif == "TRANSFORMATION" and face_detected and dst_warping_points is not None:
                # Application d'une transformation (Messi, Saitama, ballon)
                asset = assets.get(choix_transfo[transfo_idx])
                if asset:
                    src_pts = asset["ref"] # Points de référence de l'image à transformer
                    
                    # Calcul de la matrice de transformation affine
                    mat = cv2.getAffineTransform(src_pts, dst_warping_points)
                    
                    # Application du warping
                    warped = cv2.warpAffine(asset["img"], mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                    
                    if warped.shape[2] == 4: # Si l'image a un canal alpha (transparence)
                        bgr_w = warped[:, :, :3] # Canaux RGB
                        alpha_w = warped[:, :, 3] # Canal alpha
                        
                        # Fusion avec le fond
                        mask_inv = cv2.bitwise_not(alpha_w)
                        img_bg = cv2.bitwise_and(frame_affichage, frame_affichage, mask=mask_inv)
                        img_fg = cv2.bitwise_and(bgr_w, bgr_w, mask=alpha_w)
                        frame_affichage = cv2.add(img_bg, img_fg)

            elif type_sort_actif == "FLOU" and face_detected:
                # Application d'un flou gaussien sur la zone du visage
                f_zone = frame_affichage[max(0,hs_y):hs_y+hs_h, max(0,hs_x):hs_x+hs_w]
                if f_zone.size > 0:
                    frame_affichage[max(0,hs_y):hs_y+hs_h, max(0,hs_x):hs_x+hs_w] = cv2.GaussianBlur(f_zone, (99,99), 30)

            if type_sort_actif == "EXTENSION_TERRITOIRE":
                dt = current_time - temps_activation
                duree_son = 6.2 # Durée du son en secondes
                
                if dt > duree_son: # Après la fin du son
                    if img_territory_large is not None and face_detected:
                        
                        # Calcul du parallaxe (effet de profondeur)
                        face_center_x = dst_warping_points[0][0] # Centre du visage
                        
                        # Déplacement du fond en fonction de la position du visage
                        ratio_x = (face_center_x / w) - 0.5
                        center_bg_x = int(img_territory_large.shape[1] / 2)
                        center_bg_y = int(img_territory_large.shape[0] / 2)
                        move_x = int(ratio_x * 300)  # Amplitude du mouvement
                        
                         # Extraction de la zone de fond
                        start_x = (center_bg_x - w//2) + move_x
                        start_y = (center_bg_y - h//2)
                        bg_window = img_territory_large[start_y:start_y+h, start_x:start_x+w]

                        # Fusion avec chroma key
                        mask_inv = cv2.bitwise_not(binary_mask)
                        fg_cleaned = cv2.bitwise_and(frame, frame, mask=binary_mask)
                        bg_cleaned = cv2.bitwise_and(bg_window, bg_window, mask=mask_inv)
                        
                        frame_affichage = cv2.add(fg_cleaned, bg_cleaned)
                        
                else:
                    # Effet de flash pendant le son
                    if int(dt * 10) % 2 == 0:
                        frame_affichage = cv2.convertScaleAbs(frame_affichage, alpha=1.2, beta=30)

            if type_sort_actif == "DISPARITION":
                dt = current_time - temps_activation

                if last_valid_pts is not None:
                    # Remplacement du corps par un fond de substitution
                    p1, p2 = last_valid_pts[0], last_valid_pts[1] # Oreilles
                    center_face = np.mean(last_valid_pts[:2], axis=0) # Centre du visage
                    
                    # 1. Redimensionnement du fond de substitution
                    bg_sub_resized = cv2.resize(img_substitution, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # 2. Masque inversé (tout sauf la personne)
                    mask_inv_person = cv2.bitwise_not(binary_mask)
                    
                    # 3. Extraction du décor réel (sans la personne)
                    frame_bg_real = cv2.bitwise_and(frame, frame, mask=mask_inv_person)
                    
                    # 4. Extraction du fond de substitution (forme de la personne)
                    frame_fg_sub = cv2.bitwise_and(bg_sub_resized, bg_sub_resized, mask=binary_mask)
                    
                    # 5. Fusion pour boucher le trou
                    frame_recomposed = cv2.add(frame_bg_real, frame_fg_sub)
                    frame_affichage = frame_recomposed.copy()

                    # Effet de fumée pendant la première seconde
                    if dt < 1.0:
                        for r in range(10, int(dt*300), 15):
                            overlay = frame_affichage.copy()
                            cv2.circle(overlay, (int(center_face[0]), int(center_face[1])), r, (180, 180, 180), -1)
                            cv2.addWeighted(overlay, 0.4, frame_affichage, 0.6, 0, frame_affichage)


        # Affichage boite tête
        #if face_detected:
            #cv2.rectangle(frame_affichage, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 1)

        # Affichage du frame final
        cv2.imshow("Shinobi Vision", frame_affichage)

        # Gestion des touches clavier
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break # Quitter avec ESC
        elif key == ord('s'): # 's' pour screenshot
            cv2.imwrite(f"screen_{int(time.time)}.png", frame_affichage)
            print("Capture sauvegardée.")
        elif key == ord('1'): transfo_idx = 0 # Sélectionne Messi
        elif key == ord('2'): transfo_idx = 1 # Sélectionne Saitama
        elif key == ord('3'): transfo_idx = 2 # Sélectionne Ballon

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()