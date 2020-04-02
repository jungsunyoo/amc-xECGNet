# -*- coding: utf-8 -*-
import librosa
import librosa.display
import os
from scipy.io import loadmat
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers

#currdir= os.getcwd()

def main():
    rootdir = '/home/taejoon/PhysioNetChallenge'
    input_directory = os.path.join(rootdir, 'Training_WFDB')
    mel_name = 'Mel_data_20200402' 
    mel_directory = os.path.join(rootdir, mel_name)
    minimum_len = 215 # 

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    nepochs=1000

    #save_directory = os.path.join(currdir, '')
    if not os.path.isdir(input_directory):
            os.mkdir(input_directory)
    if not os.path.isdir(mel_directory):
            os.mkdir(mel_directory)       

            # Find files
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)


    input_file_names = sorted(input_files)
    
    
    # Find unique number of classes  
    def get_unique_classes(input_directory,files):

        unique_classes=set()
        for f in files:
            g = f.replace('.mat','.hea')
            input_file = os.path.join(input_directory,g)
            with open(input_file,'r') as f:
                for lines in f:
                    if lines.startswith('#Dx'):
                        tmp = lines.split(': ')[1].split(',')
                        for c in tmp:
                            unique_classes.add(c.strip())

        return sorted(unique_classes)

    unique_classes = get_unique_classes(input_directory, input_files)
    # Creating one-hot vector for Y
    # num = np.unique(classes, axis=0)
    class2index = {}
    for a, b in enumerate(unique_classes):
        class2index[b] = a
    #class2index

    def one_hot_encoding(y, class2index):
           one_hot_vector = [0]*(len(class2index))
           ind=class2index[y]
           one_hot_vector[ind]=1
           return one_hot_vector



    # Get classes of sorted file names
    def get_classes(input_directory,files):

    #     classes=set()
        classes = []
        for f in files:
            g = f.replace('.mat','.hea')
            input_file = os.path.join(input_directory,g)
            with open(input_file,'r') as f:
                for lines in f:
                    if lines.startswith('#Dx'):
                        tmp = lines.split(': ')[1].split(',')
                        for c in tmp:
    #                         curr_label = c.strip()
                            curr_label = one_hot_encoding(c.strip(), class2index)
                        classes.append(curr_label)

        return classes





    def block_feature(sequence_en): 
        new_en = []
        if len(sequence_en) > minimum_len:  
            start = random.randint(0,len(sequence_en)-minimum_len)    
            new_en = sequence_en[start:start+minimum_len]
        elif len(sequence_en) == minimum_len: 
            new_en = sequence_en
        else: 
            assert len(sequence_en) <= minimum_len
        return new_en


    mel_files = []
    for file in input_file_names:
        tmp_file = np.load(mel_directory + '/' + file.replace('.mat', '.npy'))
        clip_file = block_feature(tmp_file)
        mel_files.append(clip_file)



    classes= np.asarray(classes)    
    mel_files = np.asarray(mel_files)

    x, x_test, y, y_test  = train_test_split(mel_files, classes, test_size=0.2, train_size = 0.8)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, train_size = 0.75)




    ### Model design

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(minimum_len, minimum_len, 12)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(9, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=nepochs, validation_data=(x_val, y_val), verbose=2)

    model.save('ECG1.h5')


if __name__ == '__main__':
    main()