import librosa
import librosa.display
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import datetime as dt
from datetime import datetime
from keras import layers
from keras import models
from keras import optimizers
from keras.applications.densenet import DenseNet121, DenseNet169
from keras.layers import Input, GlobalAveragePooling2D, Dense, Activation, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model 
from sklearn.model_selection import train_test_split

tf.set_random_seed(1234)
random.seed(100)

os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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

def one_hot_encoding(one_hot_vector,y, class2index):
    ind=class2index[y]
    one_hot_vector[ind]=1
    return one_hot_vector

# Search for multi-label subjects
def searching_overlap(input_directory,class2index, input_file_names):
    multiclasses=[]
    multisubjects=[]
    number = []
    for file in input_file_names:
        f=file
        g = f.replace('.mat','.hea')
        input_file = os.path.join(input_directory,g)
        with open(input_file,'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    if len(tmp)>1:
                        one_hot_vector = [0]*(len(class2index))
                        for c in tmp:
                            one_hot_vector = one_hot_encoding(one_hot_vector, c.strip(), class2index)
                        multiclasses.append(one_hot_vector)
                        multisubjects.append(g)
                        number.append(len(tmp))
    return multisubjects, multiclasses, number

def block_feature(sequence_en, minimum_len): 
    new_en = []
    if len(sequence_en) > minimum_len:  
        start = random.randint(0,len(sequence_en)-minimum_len)
        #print(start)
        new_en = sequence_en[start:start+minimum_len]
    elif len(sequence_en) == minimum_len: 
        new_en = sequence_en
    else: 
        assert len(sequence_en) <= minimum_len
    return new_en

def exploratory_look(input_directory,file, class2index):
    classes = []
    f = file
    g = f.replace('.mat','.hea')
    input_file = os.path.join(input_directory,g)
    with open(input_file,'r') as f:
        for lines in f:
            if lines.startswith('#Dx'):
                tmp = lines.split(': ')[1].split(',')
                print(tmp, len(tmp))
    return tmp     

# Get classes of sorted file names
def get_labels(input_directory,file, class2index):
    f = file
    g = f.replace('.mat','.hea')
    input_file = os.path.join(input_directory,g)
    with open(input_file,'r') as f:
        for lines in f:
            if lines.startswith('#Dx'):
                tmp = lines.split(': ')[1].split(',')
                one_hot_vector = [0]*(len(class2index))
                for c in tmp:
                    one_hot_vector = one_hot_encoding(one_hot_vector, c.strip(), class2index)
                
    return one_hot_vector

def randextract_mels(curr_step, batch_size, data, mel_directory, class2index, minimum_len, x_mean_final, x_std_final):
    mel_files = []
    classes = []
    start = batch_size*curr_step
    end = batch_size*(curr_step+1)
    curr_file_indices = data[start:end]
    for file in curr_file_indices:
        tmp_file = np.load(mel_directory + '/' + file.replace('.mat', '.npy'))
        clip_file = block_feature(tmp_file, minimum_len)
        #print(clip_file.shape)
        #clip_file = tmp_file[:minimum_len]
        clip_file -= x_mean_final
        clip_file /= x_std_final
        mel_files.append(clip_file)
        label = get_labels(input_directory, file, class2index)
        classes.append(label)
    concat = list(zip(mel_files, classes))
    random.shuffle(concat)
    mel_files, classes = zip(*concat)
    return mel_files, classes

def train(data_train, mel_directory, batch_size, class2index, minimum_len, x_mean_final, x_std_final): 
    loss=[]
    acc = []

    total_steps = int(np.ceil(len(data_train)/batch_size))
    for curr_step in range(total_steps):
        batch_mels, batch_labels = randextract_mels(curr_step, batch_size, data_train, mel_directory, class2index, minimum_len, x_mean_final, x_std_final)
        batch_mels = np.asarray(batch_mels)
        batch_labels = np.asarray(np.squeeze(batch_labels))
        train_tmp = model.train_on_batch(batch_mels, batch_labels)
        loss.append(train_tmp[0])
        acc.append(train_tmp[1])

    loss = np.mean(np.array(loss))
    acc = np.mean(np.array(acc))
    return loss, acc

def test(data, mel_directory, input_directory, class2index, minimum_len, model, x_mean_final, x_std_final):
    scores = []
    predicted_labels=[]
    accuracy=np.zeros(len(data))
    #total_loss=[]
    total_acc = 0
    
    mel_files = []
    classes = []
    for i, file in enumerate(data):
        tmp_file = np.load(mel_directory + '/' + file.replace('.mat', '.npy'))
        steps = int(np.floor(tmp_file.shape[0]/minimum_len))
        mel_files = []
        for block in range(steps): 
            start = block*minimum_len
            end = (block+1)*minimum_len
            clip_file = tmp_file[start:end]
            clip_file -= x_mean_final
            clip_file /= x_std_final
            mel_files.append(clip_file)
        mel_files = np.asarray(mel_files)
        logit = model.predict(mel_files)
        logit = np.mean(logit, axis=0)
        #print(logit.shape)
        pred = np.argmax(logit)
        
        #clip_file = tmp_file[:minimum_len]
        #clip_file = np.expand_dims(clip_file,0)
        #clip_file -= x_mean_final
        #clip_file /= x_std_final
        #mel_files.append(clip_file)
        label = np.argmax(get_labels(input_directory, file, class2index))
        #print(pred, label)
        if pred == label:
            acc = 1
        else:
            acc = 0
        total_acc += acc
    final_acc = total_acc / i
    return final_acc

batch_size = 32
minimum_len = 128
epochs = 50
loss_function = 'categorical_crossentropy'
activation_function = 'softmax'
rootdir = '../'
date = datetime.today().strftime("%Y%m%d")
input_directory = os.path.join(rootdir, 'Training_WFDB')
mel_name = 'Mel_data_20200417_128' 
mel_directory = os.path.join(rootdir, mel_name)
results_directory = os.path.join(rootdir, 'results_'+date)
if not os.path.isdir(input_directory):
    os.mkdir(input_directory)
if not os.path.isdir(mel_directory):
    os.mkdir(mel_directory)
if not os.path.isdir(results_directory):
    os.mkdir(results_directory)
        
input_files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
        input_files.append(f)
input_file_names = sorted(input_files)

unique_classes = get_unique_classes(input_directory, input_files)
class2index = {}
for a, b in enumerate(unique_classes):
    class2index[b] = a
    
classes_orig= [x.replace('.mat', '.hea') for x in input_file_names] # total subjects
classes_multi, _, _ = searching_overlap(input_directory,class2index, input_file_names)
classes_single = [x for x in classes_orig if x not in classes_multi]
classes_single = [x.replace('.hea', '.mat') for x in classes_single]

# double-checking if classes_single have single-label
a, b, c  = searching_overlap(input_directory,class2index,classes_single)

# we can safely use classes_single as input_file_names
input_file_names = classes_single
random.shuffle(input_file_names)
np.shape(input_file_names)

x_mean_all = []
x_std_all = []
for file in input_file_names:
    x = np.load(mel_directory + '/' + file.replace('.mat', '.npy'))
    x_mean = [np.mean(x[:,:,0]), np.mean(x[:,:,1]), np.mean(x[:,:,2]), np.mean(x[:,:,3]), np.mean(x[:,:,4]), np.mean(x[:,:,5]),
             np.mean(x[:,:,6]), np.mean(x[:,:,7]), np.mean(x[:,:,8]), np.mean(x[:,:,9]), np.mean(x[:,:,10]), np.mean(x[:,:,11])]
    
    x_std = [np.std(x[:,:,0]), np.std(x[:,:,1]), np.std(x[:,:,2]), np.std(x[:,:,3]), np.std(x[:,:,4]), np.std(x[:,:,5]),
             np.std(x[:,:,6]), np.std(x[:,:,7]), np.std(x[:,:,8]), np.std(x[:,:,9]), np.std(x[:,:,10]), np.std(x[:,:,11])]
    #print(x_mean)
    x_mean_all.append(x_mean)
    x_std_all.append(x_mean)
x_mean_final = np.mean(x_mean_all, axis=0)
x_std_final = np.mean(x_std_all, axis=0)
print(x_mean_final)

data, data_test = train_test_split(input_file_names, test_size = 0.2, train_size = 0.8, shuffle=True)
data_train, data_val = train_test_split(data, test_size = 0.25, train_size = 0.75, shuffle=True)
print(np.shape(data_train), np.shape(data_val), np.shape(data_test))

input_tensor = Input(shape=(minimum_len, minimum_len, 12))
input_tensor = GaussianNoise(0.01)(input_tensor)
base_model = DenseNet121(input_tensor=input_tensor, weights=None, include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation=None)(x)
x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
pred = Dense(9, activation=activation_function)(x)
model = Model(inputs=base_model.input, outputs=pred)
model.compile(loss=loss_function,
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc'])

val_acc_sum=[]
train_loss_sum=[]
train_acc_sum=[]
val_loss_sum=[]
val_acc_min = 0
for num_epoch in range(epochs):
    random.shuffle(data_train)
    train_loss, train_acc = train(data_train, mel_directory, batch_size, class2index, minimum_len, x_mean_final, x_std_final)
    print('\nEpoch',num_epoch+1,'train_loss:',f'{train_loss:.3f}','train_acc:',f'{train_acc:.3f}',"\t")
    model_output = "ecg_mel_E%02dL%.2f" % (num_epoch, train_loss)
    save_name = os.path.join(results_directory, model_output)
    val_acc = test(data_val, mel_directory, input_directory, class2index, minimum_len, model, x_mean_final, x_std_final)
    #print('\nValidation', num_epoch+1,'valid_loss:',f'{val_loss:.3f}','valid_acc:',f'{val_acc:.3f}',"\t", dt.datetime.now())
    if val_acc > val_acc_min:
        val_acc_min = val_acc
        model.save(save_name)
        #print("Validation loss is improved")
    print('\nValidation', num_epoch+1, 'valid_acc:',f'{val_acc:.3f}', 'best_acc:',f'{val_acc_min:.3f}', "\t")
    #val_acc_sum.append(val_acc)
    #val_loss_sum.append(val_loss)
    #train_loss_sum.append(train_loss)
    #train_acc_sum.append(train_acc)