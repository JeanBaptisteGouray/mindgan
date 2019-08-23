import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import os

root = '/Users/yaelfregier/Git/WIP/M2M_Dogs/'
programs = ['Classifier_AE','Classifier_MindGAN']
metrics = ['accuracy','train_loss','test_loss']

os.chdir(root)

def plot_fig(data_path, tgt_folder,name, x_key, y_keys,legend):
    if not os.path.isfile(data_path) or '.csv' not in data_path:
        print('Le fichier ' + data_path + 'n \' est pas valide, il doit exister et être au format csv')
        exit()

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    reader = pd.read_csv(data_path,sep=';')
    fig, axes = plt.subplots(nrows=1,ncols=len(y_keys),figsize=(5*len(y_keys),5))
    for ax, y_label in zip(axes, y_keys):
        ax.xaxis.set_label_text(x_key)
        ax.set(ylabel = y_label)
        reader.plot(x=x_key, y=y_label,ax=ax, legend = legend)
        if legend:
            ax.legend()
    
    fig.savefig(tgt_folder + '/'+ name + '_' + x_key + '.png',dpi = 200)
    plt.close('all')



def calculate_statistics(src_folder, tgt_folder,metrics):
    if not os.path.exists(src_folder):
        print(src_folder + ' not exists')
        exit()

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    liste = os.listdir(src_folder)
    liste = [ files for files in liste if '.csv' in files]
    if len(liste) == 0:
        print('There is no log file in csv format in ' + src_folder)
        exit()
    readers = [pd.read_csv(src_folder + '/' + files,sep=';') for files in liste]

    epoch = np.array(readers[0]['epoch'])
    time = np.array([np.array(reader['time'].tolist()) for reader in readers])
    mean_time = np.mean(np.transpose(time),axis = 1)
    

    for metric in metrics:
        dic = {}
        dic['epoch'] = epoch
        dic['time'] = mean_time
        local = np.array([np.array(reader[metric].tolist()) for reader in readers])
        mean_metric = np.mean(np.transpose(local),axis = 1)
        std_metric = np.std(np.transpose(local),axis = 1)
        mean_std_min = mean_metric - std_metric
        mean_std_plus = mean_metric + std_metric
        dic['mean_'+ metric] = mean_metric
        dic['mean - std'] = mean_std_min
        dic['mean + std'] = mean_std_plus
        df = pd.DataFrame(dic)
        df.to_csv(tgt_folder + '/'+metric+'.csv')

def plot_statistic(src_folder,tgt_folder,programs,x_key,metrics):
    for metric in metrics:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        ax.xaxis.set_label_text(x_key)
        couleurs = ['black','green','blue','orange','red','purple']
        for program in programs:
            reader = pd.read_csv(src_folder+ '/'+program+ '/Statistics/' + metric +'.csv')
            couleur = random.choice(couleurs)
            couleurs.remove(couleur)
            n = reader['epoch'].count()
            reader.plot(x=x_key, y='mean_' + metric ,ax=ax,label = program + ' mean '+ metric ,color = [couleur for _ in range(n)])
            reader.plot(x=x_key, y='mean - std' ,ax=ax,label = program + ' mean - std', color = [couleur for _ in range(n)],style=['--'])
            reader.plot(x=x_key, y='mean + std' ,ax=ax,label = program + ' mean + std', color = [couleur for _ in range(n)],style=['--'])

        fig.savefig(tgt_folder + '/' + metric + '_in_' + x_key + '.png',dpi = 200)
        plt.close('all')

for program in programs:
    for files in os.listdir('Results/'+program+'/Logs/'):
        plot_fig('Results/'+program+'/Logs/' + files, 'Results/' +program +'/Logs_graph/',files.replace('.csv',''),'epoch', metrics, False)


# for program in programs:
#     calculate_statistics('Results/'+program+'/Logs','Results/'+program+'/Statistics',metrics)


# plot_statistic('Results','Results',programs,'epoch',['accuracy','train_loss','test_loss'])