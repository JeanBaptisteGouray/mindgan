import os
import pandas as pd

folder_tgt = '../Nb_lancement/'

os.chdir('../../Hyperparameters')

folders = os.listdir('.')

for file in folders:
    filename, ext = os.path.splitext(file)
    
    if ext =='.csv':
        nb_lancement = 1

        datas = pd.read_csv(file, sep=';')

        for key in datas.keys():
            nb_lancement *= datas[key].value_counts().sum()

        if not os.path.exists(folder_tgt):
            os.makedirs(folder_tgt)

        filename = filename.split('_')
        filename = filename[1]
        
        with open(folder_tgt + filename +'.txt', 'w') as fichier:
            fichier.write(str(nb_lancement))