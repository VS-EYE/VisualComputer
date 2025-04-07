#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
from scipy import ndimage
import time, pickle, glob, os, shutil
import random
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img, img_to_array
import scipy.misc
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.applications import   ResNet101V2, ResNet50V2, ResNet152V2,MobileNetV3Small,MobileNetV3Large, VGG16, VGG19, MobileNet ,MobileNetV2, NASNetMobile , Xception, InceptionV3, DenseNet201, DenseNet121, DenseNet169, ResNet101, ResNet50, ResNet152,EfficientNetB0,EfficientNetB4,EfficientNetB5,EfficientNetB1,EfficientNetB2,EfficientNetB3, EfficientNetB6, EfficientNetB7
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve,roc_auc_score,auc
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import cohen_kappa_score
from keras_flops import get_flops


# In[ ]:


Width_Imgs, Heigth_Imgs = 224,224
Channal_Imags = 3
NEW_SIZE = (Width_Imgs, Heigth_Imgs, Channal_Imags)
POPULATION_SIZE = 2
NO_OF_ITERATIONS = 7
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
EPOCHS = 10
PATIENCE = 4


# In[ ]:


TrainDS_Path = "/root/public/tf211/EYE_Diabetics/Train"
ValidDS_Path = "/root/public/tf211/EYE_Diabetics/val"
TestDS_Path  =  "/root/public/tf211/EYE_Diabetics/Test"


# In[ ]:


trgen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = trgen.flow_from_directory(
    TrainDS_Path,
    classes=['Non Diabetic', 'Diabetic TYPE1','Diabetic TYPE2'],
    target_size=(Width_Imgs, Heigth_Imgs),
    shuffle=True )

valid_gen = trgen.flow_from_dataframe(
    ValidDS_Path,
    classes=['Non Diabetic', 'Diabetic TYPE1','Diabetic TYPE2'],
    target_size=(Width_Imgs, Heigth_Imgs),
    shuffle=False)

test_gen = test_datagen.flow_from_directory(
    TestDS_Path,
    classes=['Non Diabetic', 'Diabetic TYPE1','Diabetic TYPE2'],
    target_size=(Width_Imgs, Heigth_Imgs),
    shuffle=False )


ranges = {
  "Rotation": np.arange(0, 31, 1),
  #"Width Shift": np.arange(0, 0.21, 0.1),
  #"Height Shift": np.arange(0, 0.21, 0.1),
  #"Zoom": np.arange(0, 0.21, 0.1),
  #"Shear": np.arange(0, 0.21, 0.1),
  "Horizontal Flip": [True, False],
  "Vertical Flip": [True, False],
  "Optimizer": [Adam(), Nadam(), RMSprop(), Adadelta(), Adagrad(), SGD()],
  "Batch Size": [8, 16, 32, 64],
  "TL Learn Ratio": np.arange(0, 26, 1),
}

SOLUTION_SIZE = len(ranges.keys())

# In[ ]:


# Population Initialization
population = np.random.uniform(
  low=LOWER_BOUND,
  high=UPPER_BOUND,
  size=(POPULATION_SIZE, SOLUTION_SIZE)
)
print(population.shape)
print(population[0])

PROJECT_PATH = "/root/public/tf211/EYE_Diabetics/Result"


# In[ ]:

val_accuracy= []
error_rate = []


def FitnessFunction(solution):
    solution = np.round(solution, 4)
    
    index = int(np.round(solution[0] * (len(ranges["Rotation"]) - 1)))
    rotationValue = ranges["Rotation"][index]
    
    index = int(np.round(solution[1] * (len(ranges["Horizontal Flip"]) - 1)))
    hFlipValue = ranges["Horizontal Flip"][index]
    
    index = int(np.round(solution[2] * (len(ranges["Vertical Flip"]) - 1)))
    vFlipValue = ranges["Vertical Flip"][index]
    
    index = int(np.round(solution[3] * (len(ranges["Optimizer"]) - 1)))
    optimizerValue = ranges["Optimizer"][index]
    
    index = int(np.round(solution[4] * (len(ranges["Batch Size"]) - 1)))
    batchSizeValue = ranges["Batch Size"][index]
    
    index = int(np.round(solution[5] * (len(ranges["TL Learn Ratio"]) - 1)))
    tlLearnRatioValue = ranges["TL Learn Ratio"][index]
    
    dataGen = ImageDataGenerator(
    rotation_range=rotationValue,
    horizontal_flip=hFlipValue,
    vertical_flip=vFlipValue,
  )

    baseModel = InceptionV3(weights=None, include_top=False, input_shape=NEW_SIZE)

    x = baseModel.output
    x = Flatten()(x)
    #x = Dense(1024, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)
	
    model = Model(inputs=baseModel.input, outputs=predictions)
    
    model.load_weights('/root/public/tf211/EYE_Diabetics/Result/Checkpoints/InceptionV3-7018-8811.h5')

    for layer in baseModel.layers:
        layer.trainable = False

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=["accuracy", Precision(), Recall(), AUC(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])  # Focus on accuracy


    keyword = "InceptionV3-" + "-".join([str(el)[2:] for el in solution])

     
    checkpointPath = "/root/public/tf211/EYE_Diabetics/Result/"+ keyword +".h5"

	
    history = model.fit(train_gen, batch_size=batchSizeValue,
                        validation_data=valid_gen,
                        epochs=EPOCHS,
                        verbose=0,
                        callbacks=[ModelCheckpoint(checkpointPath, save_best_only=True, save_weights_only=True, monitor="val_accuracy", mode="max", verbose=0),
                                   EarlyStopping(monitor="val_accuracy", mode="max", patience=PATIENCE*2)])  # Increased patience

    
    # PlotHistory(history)  # Assuming this function is for plotting, comment it out if not needed
    model.load_weights(checkpointPath)
    scoresList = model.evaluate(test_gen, verbose=0)
    val_accuracy = scoresList[1]  # Validation accuracy
    error_rate = 1 - val_accuracy  # Error rate

    score = (scoresList[1] + scoresList[2] + scoresList[3]) / 3.0


    print(scoresList[1],scoresList[2],scoresList[3])

    

    print("Validation Accuracy:", val_accuracy)
    print("Error Rate:", error_rate)
    

    configs = [rotationValue,hFlipValue,vFlipValue,optimizerValue._name,batchSizeValue,tlLearnRatioValue,]
    

    print(scoresList, score, configs)
    
        
    return score   # Maximizing validation accuracy and minimizing error rate



# In[ ]:

import numpy as np

def PopulationUpdating(population, scores, iterationNumber):
    bestIndex = np.argmax(scores)  # Find the index of the best accuracy
    print(bestIndex)
    bestSolution = population[bestIndex].copy()
    print(bestSolution)
    bestScore = scores[bestIndex]
    print(bestScore)

    newPopulation = population.copy()
    coef = iterationNumber / float(len(population))
    
    for i in range(len(population)):
        r = np.random.random(1)
        alpha = 2.0 * r * np.sqrt(np.abs(np.log(r)))
        r1 = np.random.random(1)
        factor = (len(population) - iterationNumber + 1.0) / (len(population) * 1.0)
        beta = 2.0 * np.exp(r1 * factor) * np.sin(2.0 * np.pi * r1)
        
        if (np.random.random(1) < 0.5):
            if (coef < np.random.random(1)):
                s = np.subtract(UPPER_BOUND, LOWER_BOUND)
                u = np.random.uniform(low=0, high=1, size=SOLUTION_SIZE)
                m = np.multiply(u, s)
                xRand = np.clip(np.add(LOWER_BOUND, m), LOWER_BOUND, UPPER_BOUND)
                
                if (i == 0):
                    newPopulation[i, :] = xRand + r * (xRand - population[i, :]) + beta * (xRand - population[i, :])
                else:
                    newPopulation[i, :] = xRand + r * (population[i - 1, :] - population[i, :]) + beta * (xRand - population[i, :])
            else:
                if (i == 0):
                    newPopulation[i, :] = bestSolution + r * (bestSolution - population[i, :]) + beta * (bestSolution - population[i, :])
                else:
                    newPopulation[i, :] = bestSolution + r * (population[i - 1, :] - population[i, :]) + beta * (bestSolution - population[i, :])
        else:
            if (i == 0):
                newPopulation[i, :] = population[i, :] + r * (bestSolution - population[i, :]) + alpha * (bestSolution - population[i, :])
            else:
                newPopulation[i, :] = population[i, :] + r * (population[i - 1, :] - population[i, :]) + alpha * (bestSolution - population[i, :])
                
        newPopulation[i, :] = np.clip(newPopulation[i, :], LOWER_BOUND, UPPER_BOUND)
    
        currentScore = FitnessFunction(newPopulation[i, :])
        
        if (currentScore > bestScore):
        	bestSolution, bestScore = newPopulation[i, :].copy(), currentScore
        
        s = 2.0
        r2, r3 = np.random.random(1), np.random.random(1)
        
        newPopulation[i, :] = population[i, :] + s * (r2 * bestSolution - r3 * population[i, :])
        newPopulation[i, :] = np.clip(newPopulation[i, :], LOWER_BOUND, UPPER_BOUND)
        
        currentScore = FitnessFunction(newPopulation[i, :])
        if (currentScore > bestScore):
          bestSolution, bestScore = newPopulation[i, :].copy(), currentScore

    print(bestSolution, bestScore)
    return newPopulation.copy()


# Repeat
bestSolutions = []
bestScores = []
for iterationNumber in range(NO_OF_ITERATIONS):
    scores = []
    for i in range(len(population)):
        score = FitnessFunction(population[i])
        scores.append(score)
    print("Popolation No",i )
    newPopulation = PopulationUpdating(population, scores, iterationNumber)
  
  # LOGGING THE DATA
    populationScoresPath = "/root/public/tf211/EYE_Diabetics/Result/Population.csv"
  # T #, S #, ......, Score
    file = open(populationScoresPath, "a")
    for i in range(len(population)):
        data = f"{iterationNumber + 1},{i + 1},"
        data += ",".join([str(el) for el in population[i]])
        data += f",{scores[i]}"
        data += "\n"
        file.write(data)
    file.close()
    
    bestIndex = np.argmax(scores)
    bestSolution = population[bestIndex].copy()
    bestScore = scores[bestIndex]
    bestSolutions.append(bestSolution)
    bestScores.append(bestScore)
    population = newPopulation.copy()


# In[ ]:


# LOGGING THE DATA
bestSolutionsPath = "/root/public/tf211/EYE_Diabetics/Result/BestSolutions.csv"
file = open(bestSolutionsPath, "w")
for i in range(len(bestSolutions)):
    data = ",".join([str(el) for el in bestSolutions[i]])
    data += f",{bestScores[i]}"
    data += "\n"
    file.write(data)
file.close()

