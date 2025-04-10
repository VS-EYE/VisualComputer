import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import time, pickle, glob, os
import random
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras import layers, models
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras.callbacks import *
from keras.models import Model
from tensorflow.keras.models import *
from keras.applications import *
from tensorflow.keras.layers import *
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score
from keras_flops import get_flops

Width_Imgs, Heigth_Imgs = 224,224
Channal_Imags = 3
BATCH_SIZE = 64
shapeImage = (Width_Imgs, Heigth_Imgs, Channal_Imags)
model_input = Input(shape=shapeImage )


TrainDS_Path = "/root/public/tf211/PDST/DS/Train"
ValidDS_Path = "/root/public/tf211/PDST/DS/Validate"
TestDS_Path  =  "/root/public/tf211/PDST/DS/Test"

trge = ImageDataGenerator(
   
    #width_shift_range=0.2, 
    #height_shift_range=0.2,
    horizontal_flip=True, 
    vertical_flip=True,
    rescale=1. / 255
  
)


validation_datage = ImageDataGenerator(
   
    #width_shift_range=0.2, 
    #height_shift_range=0.2,
    horizontal_flip=True, 
    vertical_flip=True,
    rescale=1. / 255
  
)




trgen = ImageDataGenerator(rescale=1. / 255)


validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = trgen.flow_from_directory(
    TrainDS_Path,
    #classes=['Glioma tumor', 'Meningioma tumor','Pituitary tumor'],
    #classes=['Non Diabetic', 'Diabetic TYPE1','Diabetic TYPE2'],
    #classes=['Benign','Malignant'],
    #classes=['Chickenpox', 'Cowpox','Healthy', 'HFMD','Measles','Monkeypox'],
    #classes = ['no','yes'],
    #classes = ['Colorectal cancer','Esophagitis','Pylorus'],
    #classes = ['autistic','non_autistic'],
    classes = ['Normal','Tumor'],
    #classes = ['Cyst','Normal','Stone','Tumor'],
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
    shuffle=True )

valid_gen = validation_datagen.flow_from_directory(
    ValidDS_Path,
    classes = ['Normal','Tumor'],
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
    shuffle=True )


test_gen = test_datagen.flow_from_directory(
    TestDS_Path,
    classes = ['Normal','Tumor'],
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
    shuffle=False )



# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath= '/root/public/tf211/PDST/FUSEDV19D169-F22.h5',
                                  save_best_only=True,
                                  monitor = "val_accuracy",
                                  verbose=1)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=3,
    verbose=0,
    mode="auto",
    min_delta=0.00001,
    cooldown=0,
    min_lr=0)


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


METRICS = [keras.metrics.CategoricalAccuracy(name='accuracy'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.AUC(name='auc')]



  # Define SE block function
def se_block(input_tensor, ratio=16):
    num_channels = input_tensor.shape[-1]
    squeeze = GlobalAveragePooling2D()(input_tensor)
    excitation = Dense(num_channels // ratio, activation='relu')(squeeze)
    excitation = Dense(num_channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, num_channels))(excitation)
    scaled_input = multiply([input_tensor, excitation])
    return scaled_input

def inception_module(x, filters):
    # 1x1 Convolution
    conv1x1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)

    # 3x3 Convolution
    conv3x3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)

    # 5x5 Convolution
    conv5x5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(x)

    # Max Pooling
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    max_pool_conv = Conv2D(filters, (1, 1), padding='same', activation='relu')(max_pool)

    # Concatenate the outputs
    inception_output = Concatenate()([conv1x1, conv3x3, conv5x5, max_pool_conv])

    return inception_output

def builder_a(model_input):
    builder_a = VGG19(weights='imagenet',
                                    include_top=False,
                                    input_tensor = model_input)

    for layer in builder_a.layers:
        layer.trainable = False
       
    for layer in builder_a.layers:
        layer._name = layer.name + '_a'
       
    for BatchNormalization in builder_a.layers:
        BatchNormalization.trainable = False
   
    x = builder_a.output
    x = inception_module(x, filters=16)
   
 

    dcnn_a = Model(inputs=builder_a.input, outputs=x)
    return dcnn_a

dcnn_a = builder_a(model_input)

dcnn_a.summary()


print("successfully built!") 


def builder_b(model_input):
    builder_b = ResNet50(weights='imagenet',
                                    include_top=False,
                                    input_tensor = model_input)

    for layer in builder_b.layers:
        layer.trainable = False
       
    for layer in builder_b.layers:
        layer._name = layer.name + '_b'
       
    for BatchNormalization in builder_b.layers:
        BatchNormalization.trainable = False
   
    x = builder_b.output
    x = inception_module(x, filters=16)


    dcnn_b = Model(inputs=builder_b.input, outputs=x)
    return dcnn_b

dcnn_b = builder_b(model_input)

dcnn_b.summary()


print("successfully built!") 


dcnn_a = builder_a(model_input)
dcnn_b = builder_b(model_input)




models = [dcnn_a,
          dcnn_b]

print(" successfully built!")


def fusion_builder(models, model_input):
    outputs = [m.output for m in models]
    y = Concatenate(name='InitialFusionLayer')(outputs)

  

    
    
    #input_ = tf.expand_dims(y,axis = 1)

    #x = ConvLSTM2D(filters=64, kernel_size=(1,1),padding = "same")(input_)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = Dropout(0.2)(x)

    #x = inception_module(x, filters=16)

    #x = GlobalAveragePooling2D()(x)

 
    
    
    
    x = Flatten()(y)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.2)(x)

    prediction = Dense(2, activation='softmax')(x)
    model = Model(model_input, prediction)
    return model

# Instantiate the model and report the summary
fused = fusion_builder(models, model_input)

fused.summary()

optim_1 = Adam(learning_rate=0.001)
n_epochs = 10
fused.compile(optimizer=optim_1, loss='categorical_crossentropy', metrics=METRICS)

start = time.time()
Train_history = fused.fit(train_gen, batch_size=BATCH_SIZE, epochs=n_epochs, validation_data=valid_gen, callbacks=[tl_checkpoint_1, reducelr], verbose=1)
Elapsed = time.time()-start
print (f'Training time: {hms_string(Elapsed)}')



model = keras.models.load_model('/root/public/tf211/PDST/FUSEDV19D169-F22.h5')
print (model.summary())


start = time.time()
prediction = np.argmax(model.predict(test_gen), axis=1)
print('Test Data accuracy: ',accuracy_score(test_gen.classes, prediction)*100)
Elapsed = time.time()-start
print (f'Training time: {hms_string(Elapsed)}')



print("F1-Score", f1_score(test_gen.classes, prediction, average='macro'))
print("Recall", recall_score(test_gen.classes, prediction, average='macro'))
print("Precision", precision_score(test_gen.classes, prediction, average='macro'))
print(classification_report(test_gen.classes, prediction))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_gen.classes, prediction))
print("Cohen", cohen_kappa_score(test_gen.classes, prediction))






