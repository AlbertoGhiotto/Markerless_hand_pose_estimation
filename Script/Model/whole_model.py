import keras
from keras.layers import Dense, Activation, Conv2DTranspose, Reshape, Input, Add, Conv2D, Cropping2D, UpSampling2D
from keras.regularizers import l2
from keras import backend as K; 
from keras.models import Model
from keras.activations import sigmoid
import numpy as np
from resnet50_model import ResNet50

NUM_JOINTS = 16


def whole_model(input_tensor):
    
    #================================== ResNet =======================================
    # Importing the ResNet architecture pretrained on ImageNet
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in resnet_model.layers: # In questo modo tutti i layer del modello importato non sono trainable
        layer.trainable = False
    #resnet_model.summary() 
    
    #============================= Part classification ================================
    x = Conv2DTranspose(NUM_JOINTS, (3,3), strides=(2,2), activation = None)(resnet_model.output)  
    y = Conv2D(NUM_JOINTS, (1,1), strides=(1, 1))(resnet_model.get_layer("res4a_branch2a").input)
    #x = UpSampling2D(size=(2, 2))(x)
    x = Cropping2D(cropping=((1, 0), (0, 1)))(x)
    x = Add()([x, y]) 
    scmap = Activation('sigmoid',  name='scmap')(x)
    
    #============================= Location refinement ================================
    #z = Conv2DTranspose(NUM_JOINTS*2, (3,3), strides=(2,2), activation = None)(resnet_model.output)
    #y = Conv2D(NUM_JOINTS*2, (1,1), strides=(1, 1))(resnet_model.get_layer("res4a_branch2a").input)
    #z = UpSampling2D(size=(2, 2))(z)
    #z = Cropping2D(cropping=((1, 0), (0, 1)))(z)
    #locref = Add( name='locref')([z, y]) 
    
    #============================= Regression to other joints ==========================
    #w = Conv2DTranspose(364, (3,3), strides=(2,2), activation = None)(resnet_model.output)        
    #y = Conv2D(364, (1,1), strides=(1, 1))(resnet_model.get_layer("res4a_branch2a").input)
    #w = UpSampling2D(size=(2, 2))(w)
    #w = Cropping2D(cropping=(1, 1))(w)
    #w = Add()([w, y]) 
    
    #============================ Building the model ===================================
    
    #model = Model(inputs=resnet_model.input, outputs= [scmap, locref]) # In case of using also location refinement
    model = Model(inputs=resnet_model.input, outputs=scmap)
    
    return model
