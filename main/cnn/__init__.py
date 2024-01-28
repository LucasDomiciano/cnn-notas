'''
@author Lucas Lacerda 
@date 05/2019
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

class Convolucao(object):
    
    @staticmethod
    def build(width, height, channels, classes):
        inputShape = (height, width, channels)
                  
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=inputShape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(filters=32, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Flatten())  # Adiciona uma camada Flatten antes da camada densa
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax')) 
        
        return model


    '''@staticmethod
    def build(width, height, channels, classes):
        """
        :param width: Largura em pixel da imagem.
        :param height: Altura em pixel da imagem.
        :param channels: Quantidade de canais da imagem.
        :param classes: Quantidade de classes para o output.
        
        :return: CNN com a arquitetura: 
            INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT
        """
        inputShape = (height, width, channels)
                  
        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = inputShape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Conv2D(filters = 32, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Conv2D(filters = 64, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation = 'softmax')) 
        
        return model'''