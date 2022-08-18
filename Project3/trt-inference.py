# trt-inference
#!pip install sklearn -qqq

import time
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorrt as trt
from TensorRTUtils import *
import onnx
import tf2onnx
import numpy as np
from PIL import Image as im
import os
from onnxUtils import convertKerasToONNX
import wandb_helpers as wbh

import seaborn as sns
import matplotlib.pyplot as plt     

modelName = "FCNN"

'''
Stage 1: Load an existing model
===============================
In this part we load the model we created in the previous project
which is built to infer from FASHION-MNIST images.
It is not a sofisticated model, but the idea to use something we
know.
'''
dataset_path = '.\\artifacts\\fashion-mnist-v2'

if not os.path.exists(dataset_path):
    with wbh.start_wandb_run("FCNN-metrics", None) as run:
        train_set, validation_set, test_set = wbh.read_datasets(run)
        model = wbh.read_model(run, "FCNN", "latest")
else:
    test_set = wbh.read_dataset('.\\artifacts\\fashion-mnist-v2', 'test')
    model = tf.keras.models.load_model('.\\artifacts\\FCNN-v3')

'''
Stage 1.5: Run the TensorFlow model
========================
Run the TensorFlow model on the test_set, 
and check the running-time.
'''
startTimeCpu = time.time()
model.evaluate(test_set.images, test_set.labels, verbose=2)
endTimeCpu = time.time()

# total time taken
averageTime = (endTimeCpu - startTimeCpu) / 1e-3 / len(test_set)
print(f"Nir: TensorFlow inference average time is: {averageTime} milliseconds")
print(f"Nir: TensorFlow inference average FPS is: {1000 / averageTime}")

'''
Stage 2: Convert to ONNX
========================
Convert the model to ONNX and save it to a file. This will allow
us to load the model into a tensor-rt engine.
'''
modelFile, _, _ = convertKerasToONNX(modelName, model, True)

'''
Stage 3: Create the tensor-rt engine
====================================
Now that we a model file, we can load it into a 
tensor rt engine.
We use FP 32 precision.
'''
TrtModelParse(modelFile)
print("===================================")
print("Before TrtModelOptimizeAndSerialize")
print("===================================")
#TrtModelOptimizeAndSerialize(precision='fp32')
#TrtModelOptimizeAndSerialize(precision='fp16')
calibSet=MatrixIterator(validation_set.images)
TrtModelOptimizeAndSerialize(precision='int8', calibPath="/content", calibSet=calibSet)
print("===================================")
print("After TrtModelOptimizeAndSerialize")
print("===================================")
ModelInferSetup()

'''
Stage 4: Inference
==================
Now the model is ready for inference. The model is executed several
times on different images from the test set we've loaded on Stage 1
'''
inputs = []

startTimeCpu = time.time()
for i in range(len(test_set)):
    img = test_set.images[i]
    lbl = test_set.labels[i]
    inputs.append(img)
    outputsTrt = Inference(externalnputs=inputs)
    #print(' topClassIdx - ', np.argmax(outputsTrt[0]))
    inputs.clear()
    
    
endTimeCpu = time.time()

# total time taken
averageTime = (endTimeCpu - startTimeCpu) / 1e-3 / len(test_set)
print(f"TRT Keras inference average time is: {averageTime} milliseconds")
print(f"TRT Keras inference average FPS is: {1000 / averageTime}")

# Perform the DlewareAnalyzer inference with TRT & ORT

#np.testing.assert_allclose(kerasPredictions, onnxPredictions[0], rtol=0, atol=1e-05, err_msg='Keras Vs. Onnx Failure!!!')


#y_test = np.argmax(test_set.labels)
# predictions = model.predict(test_set.images)
# y_test = np.argmax(predictions, axis = 1)
# print (classification_report(test_set.labels, y_test))
# cm = confusion_matrix(test_set.labels, y_test)

# class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ax = plt.subplot()
# h = sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# # labels, title and ticks
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(class_names)
# ax.yaxis.set_ticklabels(class_names)

# plt.show()
