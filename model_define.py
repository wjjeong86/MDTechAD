import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    BatchNormalization as BN,
    GlobalAveragePooling2D as GAP2D,
    AveragePooling2D,    
    Activation,
)
from tensorflow_addons.activations import gelu

'''-----------------------------------------------------------------------------Custom Layer'''
class GELU(layers.Layer):

    def __init__(self):
        super(GELU, self).__init__()

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.gelu = gelu

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return self.gelu(inputs)

'''-----------------------------------------------------------------------------Model'''
def model_wj(ts_in:Input):
    
    ts_z = ts_in
    
    '''-----Embadding'''
    ### 256->128
    ts_z = Conv2D( filters=16, kernel_size=3, strides=2, padding='same', use_bias=False )(ts_z)
    ts_z = BN()(ts_z)
    ts_z = GELU()(ts_z)
    ts_z = Conv2D( filters=32, kernel_size=3, strides=1, padding='same', use_bias=False )(ts_z)
    
    '''----Body'''   
    def BRC(x,filters,down=False):
        s = 2 if down else 1
        Cin = x.shape[3]
        
        if   Cin==filters and down==False:
            P = x        
        else:       
            P = Conv2D( filters=filters, kernel_size=1, strides=s, padding='same', use_bias=False )(x)            
            P = BN()(P)
        
        if down :
            x = AveragePooling2D()(x)
        x = BN()(x)
        x = GELU()(x)
        x = Conv2D( filters=filters, kernel_size=5, strides=1, padding='same', use_bias=False )(x)
#         x = BN()(x)
#         x = GELU()(x)
#         x = Conv2D( filters=filters, kernel_size=3, padding='same', use_bias=False )(x)
        x = P+x
        return x
        
    ### 128->64
    ts_z = BRC(ts_z,filters=64,down=True)
    
    ### 64->32
    ts_z = BRC(ts_z,filters=128,down=True)
    
    ### 32->16
    ts_z = BRC(ts_z,filters=256,down=True)
    
    '''----Exit'''
#     ts_z = BRC(ts_z,filters=128)
    
    ### 16->8
    ts_z = BRC(ts_z,filters=512,down=True)
    
    '''----GAP'''
    ts_feature = GAP2D()(ts_z)
    ts_z = ts_feature
        
    '''----FC'''
    ts_z = Dense( units=26, activation=None, use_bias=False)(ts_z)
    ts_out = Activation('linear', dtype=tf.float32)(ts_z)
        
    model = keras.Model(inputs=ts_in,outputs=ts_out)
        
    return model


'''-----------------------------------------------------------------------------Main'''
if __name__ == '__main__' :
    ts_in = Input((256,256,1))
    model = model_wj(ts_in)

    model.summary()