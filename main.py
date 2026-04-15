import cv2

def main() -> None:
    # Ouverture de la webcam
    cap = cv2.VideoCapture(0)

    print("Webcam ouverte avec succès. Appuyez sur 'Echap' pour quitter.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Échec de la capture du frame.")
            break

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