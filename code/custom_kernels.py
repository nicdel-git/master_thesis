#Here are defined custom kernels for GPs

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_probability.python.math.psd_kernels.internal import util

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

class M52_custom(psd_kernels.PositiveSemidefiniteKernel):
    def __init__(self,feature_ndims=1, amplitude=None, length_scale=None,dtype=None,\
                 name='M52_custom', validate_args=False):
        parameters = dict(locals())
        super().__init__(feature_ndims, dtype, name, validate_args, parameters)
        self.amplitude = amplitude
        self.length_scale=length_scale
        
    def _apply(self,x1,x2,example_ndims=0):
        norm = tf.math.abs(x1-x2)/self.length_scale*np.sqrt(5)
        log_result = tf.math.log1p(norm + norm**2/3.0) - norm
        log_result = util.sum_rightmost_ndims_preserving_shape(log_result, ndims=self.feature_ndims)
        log_result += tf.math.log(self.amplitude)
        return tf.exp(log_result)
    
    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape

    def _batch_shape_tensor(self):
        return tf.convert_to_tensor(self.batch_shape, dtype=tf.int32, name='batch_shape')
    
class SE_custom(psd_kernels.PositiveSemidefiniteKernel):
    def __init__(self,feature_ndims=1, amplitude=None, length_scale=None,dtype=None,\
                 name='SE_custom', validate_args=False):
        parameters = dict(locals())
        super().__init__(feature_ndims, dtype, name, validate_args, parameters)
        self.amplitude = amplitude
        self.length_scale=length_scale
        
    def _apply(self,x1,x2,example_ndims=0):
        norm = (x1-x2)**2/(2.0*self.length_scale**2)
        log_result = -norm
        log_result = util.sum_rightmost_ndims_preserving_shape(log_result, ndims=self.feature_ndims)
        log_result += tf.math.log(self.amplitude)
        return tf.exp(log_result)
    
    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape

    def _batch_shape_tensor(self):
        return tf.convert_to_tensor(self.batch_shape, dtype=tf.int32, name='batch_shape')


@tf.function
def integral(y, x):
    y = tf.transpose(y)
    dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
    return ((y[0] + y[-1])/2 + tf.reduce_sum(y[1:-1],axis=0)) * dx

class integral_1D_kernel_custom(psd_kernels.PositiveSemidefiniteKernel):
    def __init__(self,feature_ndims=1, omega = None, amplitude=None, dtype=None,\
                 name='SE_custom', validate_args=False):
        parameters = dict(locals())
        super().__init__(feature_ndims, dtype, name, validate_args, parameters)
        self.amplitude = amplitude
        self.omega = omega
        
    def _apply(self,x1,x2,example_ndims=0):
        temp = self.amplitude*tf.transpose(integral(tf.math.multiply(x1, x2), self.omega))
        return temp
    
    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape

    def _batch_shape_tensor(self):
        return tf.convert_to_tensor(self.batch_shape, dtype=tf.int32, name='batch_shape')
    
class integral_2D_kernel_custom(psd_kernels.PositiveSemidefiniteKernel):
    def __init__(self,feature_ndims=1, omega = None, amplitude=None, length_scale=None, dtype=None,\
                 name='SE_custom', validate_args=False):
        parameters = dict(locals())
        super().__init__(feature_ndims, dtype, name, validate_args, parameters)
        self.amplitude = amplitude
        self.length_scale = length_scale
        self.omega = omega
        
    def _apply(self,x1,x2,example_ndims=0):
        y1 = tf.gather(x1, indices = tf.range(tf.shape(x1)[-1] - 1), axis = -1)
        y2 = tf.gather(x2, indices = tf.range(tf.shape(x1)[-1] - 1), axis = -1)
        z1 = tf.gather(x1, indices = [tf.shape(x1)[-1] - 1], axis = -1)
        z2 = tf.gather(x2, indices = [tf.shape(x1)[-1] - 1], axis = -1)
        K_temp = tf.squeeze(np.sqrt(5) * tf.math.sqrt(tf.math.squared_difference(z1, z2)) / self.length_scale, axis=-1)
        K = (1.0 + K_temp + (K_temp ** 2) / 3.0) * tf.exp(-K_temp)
        temp = self.amplitude*tf.transpose(integral(tf.math.multiply(y1, y2), self.omega))*K
        return temp
    
    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape

    def _batch_shape_tensor(self):
        return tf.convert_to_tensor(self.batch_shape, dtype=tf.int32, name='batch_shape')
