import cv2
import mediapipe as mp
import numpy as np
import math
import os

# --- FONCTIONS UTILITAIRES ---

def get_distance(p1, p2, w, h):
    return math.sqrt(((p1.x - p2.x) * w)**2 + ((p1.y - p2.y) * h)**2)

def load_transformation_images():
    images = {}
    folder = "img_src"
    files = {"messi": "messi.png", "saitama": "saitama.png", "vase": "vase.png"}
    
    # /!\ AJUSTE CES POINTS selon tes images (x, y en pixels dans le PNG source)
    # Format: [Oeil Gauche, Oeil Droit, Menton]
    refs = {
        "messi": np.array([[693, 705], [1012, 742], [812, 1388]], dtype=np.float32),
        "saitama": np.array([[779, 726], [1172, 753], [941, 1213]], dtype=np.float32),
        "vase": np.array([[1620, 2006], [2455, 2006], [2008, 3157]], dtype=np.float32)
    }

    for key, filename in files.items():
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images[key] = {"img": img, "ref": refs[key]}
        else:
            print(f"Fichier introuvable : {filepath}")
    return images

def apply_cloned_person(dest_x, dest_y, current_frame, roi_color, roi_mask, hs_w, hs_h):
    h_f, w_f = current_frame.shape[:2]
    
    # Calcul des zones d'intersection (pour ne pas dépasser de l'écran)
    x1, y1 = max(0, int(dest_x)), max(0, int(dest_y))
    x2, y2 = min(w_f, x1 + hs_w), min(h_f, y1 + hs_h)
    
    # Calcul des dimensions réelles à copier
    copy_w, copy_h = x2 - x1, y2 - y1
    
    if copy_w <= 0 or copy_h <= 0:
        return current_frame

    # On découpe la ROI et le masque pour qu'ils correspondent à la zone visible
    roi_sub = roi_color[0:copy_h, 0:copy_w]
    mask_sub = roi_mask[0:copy_h, 0:copy_w]
    
    bg_target = current_frame[y1:y2, x1:x2]
    
    if bg_target.shape[:2] != roi_sub.shape[:2]:
        return current_frame

    mask_inv = cv2.bitwise_not(mask_sub)
    bg_cleaned = cv2.bitwise_and(bg_target, bg_target, mask=mask_inv)
    person_cleaned = cv2.bitwise_and(roi_sub, roi_sub, mask=mask_sub)
    
    current_frame[y1:y2, x1:x2] = cv2.add(bg_cleaned, person_cleaned)
    return current_frame

# --- MAIN ---

def main():
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_seg = mp.solutions.selfie_segmentation
    
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    seg_model = mp_selfie_seg.SelfieSegmentation(model_selection=0)

    assets = load_transformation_images()
    choix_transfo = ["messi", "saitama", "vase"]
    transfo_idx = 1 # Saitama par défaut
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialisation sécurisée (Fail-Safe)
        geste = "Aucun"
        face_detected = False
        dst_warping_points = None
        person_roi, mask_roi = None, None
        hs_x, hs_y, hs_w, hs_h = 0, 0, 0, 0
        roi_x, roi_y, roi_w, roi_h = 0, 0, 250, 180

        # 1. Segmentation
        seg_res = seg_model.process(rgb_frame)
        _, binary_mask = cv2.threshold(seg_res.segmentation_mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 2. Face Mesh
        mesh_res = face_mesh.process(rgb_frame)
        if mesh_res.multi_face_landmarks:
            face_detected = True
            lm = mesh_res.multi_face_landmarks[0].landmark
            
            # Points Warping (Oeil G: 130, Oeil D: 359, Menton: 152)
            pts = []
            for i in [130, 359, 152]:# Oeil G, Oeil D, Menton
                lm_point = mesh_res.multi_face_landmarks[0].landmark[i]
                pts.append([lm_point.x * w, lm_point.y * h])
            dst_warping_points = np.array(pts, dtype=np.float32)

            # Zone Interaction
            roi_x, roi_y = int(lm[152].x * w) - 125, int(lm[152].y * h) + 50
            roi_x, roi_y = max(0, min(roi_x, w-roi_w)), max(0, min(roi_y, h-roi_h))

            # Zone Tête+Épaules (Clonage)
            # On définit une boîte autour du visage (Landmarks 10 haut, 152 bas, 234 gauche, 454 droite)
            fw = int((lm[454].x - lm[234].x) * w)
            fh = int((lm[152].y - lm[10].y) * h)
            hs_w, hs_h = int(fw * 2.4), int(fh * 3.5)
            hs_x, hs_y = int(lm[1].x * w) - hs_w//2, int(lm[1].y * h) - hs_h//3
            
            # Capture sécurisée
            if hs_x >= 0 and hs_y >= 0 and hs_x+hs_w < w and hs_y+hs_h < h:
                person_roi = frame[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w].copy()
                mask_roi = binary_mask[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w].copy()

        # 3. Mains
        hand_res = hands.process(rgb_frame)
        if hand_res.multi_hand_landmarks and len(hand_res.multi_hand_landmarks) == 2:
            h1, h2 = hand_res.multi_hand_landmarks[0].landmark, hand_res.multi_hand_landmarks[1].landmark
            d_idx = get_distance(h1[8], h2[8], w, h)
            d_th = get_distance(h1[4], h2[4], w, h)
            d_base = get_distance(h1[5], h2[5], w, h)
            
            if d_idx < 35 and d_th < 35: geste = "FLOU"
            elif d_idx < 30: geste = "CLONAGE"
            elif d_base < 60: geste = "TRANSFORMATION"

        # --- APPLICATION DES EFFETS ---

        if geste == "CLONAGE" and person_roi is not None:
            gx, gy = int(hs_w * 0.9), int(hs_h * 0.4)
            pos = [(0,-gy*1.5), (-gx*2,-gy), (gx*2,-gy), (-gx,0), (gx,0), (0,gy*1.2)]
            pos.sort(key=lambda p: p[1])
            for ox, oy in pos:
                if abs(ox)<10 and abs(oy)<10: continue
                frame = apply_cloned_person(hs_x+ox, hs_y+oy, frame, person_roi, mask_roi, hs_w, hs_h)
            frame = apply_cloned_person(hs_x, hs_y, frame, person_roi, mask_roi, hs_w, hs_h)

        elif geste == "TRANSFORMATION" and face_detected and dst_warping_points is not None:
            asset = assets.get(choix_transfo[transfo_idx])
            if asset:
                # 1. On récupère les points de l'image PNG
                src_pts = asset["ref"]
                
                # 2. On calcule la matrice
                mat = cv2.getAffineTransform(src_pts, dst_warping_points)
                
                # 3. On applique le warping
                # IMPORTANT : On crée une image vide de la taille du FRAME
                warped = cv2.warpAffine(asset["img"], mat, (w, h), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(0,0,0,0))
                
                if warped.shape[2] == 4:
                    # Extraction BGR et Alpha
                    bgr_w = warped[:, :, :3]
                    alpha_w = warped[:, :, 3]
                    
                    # Fusion
                    mask_inv = cv2.bitwise_not(alpha_w)
                    img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                    img_fg = cv2.bitwise_and(bgr_w, bgr_w, mask=alpha_w)
                    frame = cv2.add(img_bg, img_fg)

        elif geste == "FLOU" and face_detected:
            # Simple flou de zone
            f_zone = frame[max(0,hs_y):hs_y+hs_h, max(0,hs_x):hs_x+hs_w]
            if f_zone.size > 0:
                frame[max(0,hs_y):hs_y+hs_h, max(0,hs_x):hs_x+hs_w] = cv2.GaussianBlur(f_zone, (99,99), 30)

        # UI
        if face_detected:
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 1)
            cv2.putText(frame, f"Geste: {geste}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Shinobi Vision", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('s'): # 's' pour screenshot
            cv2.imwrite("screen.png", frame)
            print("Capture sauvegardée.")
        elif key == ord('1'): transfo_idx = 0
        elif key == ord('2'): transfo_idx = 1
        elif key == ord('3'): transfo_idx = 2

    # Nettoyage
    cap.release()
    segment_model.close()
    hands.close()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()