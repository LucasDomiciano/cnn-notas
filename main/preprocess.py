import os
import pandas as pd

base_path = './dataset'  # Substitua pelo caminho real do seu conjunto de dados
classes = ['A', 'C', 'D', 'E', 'F', 'G', 'B']

annotations = []

for class_name in classes:
    class_path = os.path.join(base_path, class_name)
    for filename in os.listdir(class_path):
        annotations.append({
            'filename': os.path.join(class_path, filename),
            'class': class_name,
            'xmin': None,  # Preencha com as coordenadas corretas
            'ymin': None,
            'xmax': None,
            'ymax': None
        })

# Converta para DataFrame para facilitar manipulação
annotations_df = pd.DataFrame(annotations)
