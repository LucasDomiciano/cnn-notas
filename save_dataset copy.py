import cv2
import mediapipe as mp

# Inicializar o módulo de detecção de mãos da Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializar o módulo de desenho das landmarks
mp_drawing = mp.solutions.drawing_utils

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Flag para indicar se deve salvar a imagem da mão
save_image = False
image_count = 1

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Converter a imagem para o formato RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Detectar as mãos na imagem
    results = hands.process(image)

    # Desenhar as landmarks das mãos na imagem
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obter as coordenadas da bounding box da mão
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                x_min = min(x_min, x) - 5
                y_min = min(y_min, y) - 5
                x_max = max(x_max, x) + 5
                y_max = max(y_max, y) + 5

            # Recortar a região da mão da imagem
            hand_image = image[y_min:y_max, x_min:x_max]

            # Salvar a imagem da mão quando a tecla 's' for pressionada
            if save_image:
                cv2.imwrite("./dataset/C/" + str(image_count) + ".jpg", cv2.cvtColor(hand_image, cv2.COLOR_RGB2BGR))
                print("Imagem da mão salva!")

                image_count += 1
                # Resetar a flag após salvar a imagem
                save_image = False

    # Exibir a imagem
    cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Sair do loop quando a tecla 'q' for pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Definir a flag para salvar a imagem da mão quando a tecla 's' for pressionada
        save_image = True

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
