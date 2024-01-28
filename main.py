from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import mediapipe as mp

# Carregando o modelo
classifier = load_model('./models/model.h5')

classes = 7
letras = {'0' : 'A', '1' : 'B', '2' : 'C' , '3': 'D', '4': 'E', '5':'F', '6':'G'}

def predictor():          
       # Carregando uma imagem para teste (substitua 'caminho_da_imagem' pelo caminho real da sua imagem)
        img_path = './temp/img.jpg'
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalizando os valores dos pixels para estar no intervalo [0, 1]

        # Fazendo a previsão
        prediction = classifier.predict(img_array)

        # Obtendo a classe prevista
        class_index = np.argmax(prediction)

        return [letras[str(class_index)]]

#img_text = predictor()
#print(img_text)

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

            # Pré-processar a imagem para fazer previsões
            if hand_image is not None and hand_image.size != 0:
                hand_image = cv2.resize(hand_image, (64, 64))  # Ajustar ao tamanho esperado pelo modelo
                hand_image = cv2.cvtColor(hand_image, cv2.COLOR_RGB2BGR)
                hand_image = np.expand_dims(hand_image, axis=0)
                hand_image = hand_image / 255.0  # Normalizar valores dos pixels para [0, 1]

                # Fazer a previsão usando o modelo
                #img_text = predictor()
                
                prediction = classifier.predict(hand_image)
                class_index = np.argmax(prediction)

                # Exibir a classe prevista
                #print("Classe prevista:", letras[str(class_index)])
                # Exibir a classe prevista na imagem
                cv2.putText(image, f'Nota: {letras[str(class_index)]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
    # Exibir a imagem
    cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Sair do loop quando a tecla 'q' for pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
# Liberar os recursos
cap.release()
cv2.destroyAllWindows()

