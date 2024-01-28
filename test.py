import os
import shutil
from sklearn.model_selection import train_test_split

# Caminho para a pasta principal do conjunto de dados
dataset_path = './dataset'

# Lista de todas as pastas de classes dentro do conjunto de dados
class_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

# Criar diretórios para conjuntos de treinamento e teste
train_path = './dataset/train'
test_path = './dataset/test'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Loop sobre as pastas de classes e dividir os arquivos em treinamento e teste
for class_folder in class_folders:
    class_path = os.path.join(dataset_path, class_folder)
    
    # Lista de arquivos na pasta da classe
    class_files = [file for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))]
    
    # Dividir os arquivos em treinamento e teste
    train_files, test_files = train_test_split(class_files, test_size=0.2, random_state=42)
    
    # Mover os arquivos para os diretórios correspondentes
    for file in train_files:
        src_path = os.path.join(class_path, file)
        dest_path = os.path.join(train_path, class_folder, file)
        os.makedirs(os.path.join(train_path, class_folder), exist_ok=True)
        shutil.move(src_path, dest_path)
    
    for file in test_files:
        src_path = os.path.join(class_path, file)
        dest_path = os.path.join(test_path, class_folder, file)
        os.makedirs(os.path.join(test_path, class_folder), exist_ok=True)
        shutil.move(src_path, dest_path)

print("Divisão entre treinamento e teste concluída.")
