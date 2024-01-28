import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# Exemplo: Carregar imagens de um diretório
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './dataset',
    target_size = (128,128), # Defina o tamanho desejado das imagens
    batch_size=32,
    class_mode='categorical' # ou 'binary' dependendo do seu problema
)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax')) # 7 classes para 7 notas musicais


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # ou 'categorical_crossentropy' para classificação multiclasse
              metrics=['accuracy'])

model.fit(train_generator, epochs=30)  # ajuste o número de épocas conforme necessário

# Salvando o modelo
model.save('modelo.h5')