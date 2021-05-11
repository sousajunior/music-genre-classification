import json
import numpy as np

def load_data(dataset_path):
    print(f"Carregando o arquivo {dataset_path}")

    # abre o arquivo com os dados
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

        # converte as listas em um array numpy
        inputs = np.array(data['mfcc'])
        targets = np.array(data['labels'])

        print("Arquivo carregado com sucesso!")

        return inputs, targets
