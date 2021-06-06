import tensorflow as tf
import numpy as np


#Posterior variance of call price
def K_analytic(gp, x_random, x_training):
    K_test = gp.kernel.tensor(x_random,x_random,1,1)
    K_training = gp.kernel.tensor(x_training,x_training,1,1)
    K_1 = gp.kernel.tensor(x_random,x_training,1,1)
    K_2 = tf.transpose(K_1)
    len_I = len(x_training)
    K_analytic = K_test - K_1@tf.linalg.inv(K_training + gp.observation_noise_variance*tf.eye(len_I, dtype = tf.float64))@K_2
    return K_analytic

#first derivative of prior Matern5/2 kernel
def first_derivative_M52(gp, x_random, x_training, theta_delta):
    x1 = tf.gather(x_random,[theta_delta],axis=1)
    x2 = tf.transpose(tf.gather(x_training,[theta_delta],axis=1))
    l = tf.gather(gp.kernel.length_scale, [theta_delta])
    diff = x1 - x2
    factor = -5./(3.0*l**2)*diff-np.sqrt(5.)**3/(3.0*l**3)*diff*tf.math.abs(diff)
    factor = factor/(1 + np.sqrt(5.)/l*tf.math.abs(diff) + 5./(3.0*l**2)*diff**2)
    factor = factor*gp.kernel.tensor(x_random, x_training,1,1)
    return factor

# first derivative of prior RBF kernel 
def first_derivative_SE(gp, x_random, x_training, theta_delta):
    x1 = tf.gather(x_random,[theta_delta],axis=1)
    x2 = tf.transpose(tf.gather(x_training,[theta_delta],axis=1))
    l = tf.gather(gp.kernel.length_scale, [theta_delta])
    diff = x2-x1
    factor = (diff)/l**2
    factor = factor*gp.kernel.tensor(x_random, x_training,1,1)
    return factor

#second derivative of prior Matern5/2 kernel
def second_derivative_M52(gp, x_random, x_random2, theta_delta):
    l = tf.gather(gp.kernel.length_scale, [theta_delta])
    factor = 5.0/(3.0*l**2)*gp.kernel.amplitude
    return factor

#second derivative of prior RBF kernel
def second_derivative_SE(gp, x_random, x_random2, theta_delta):
    x1 = tf.gather(x_random,[theta_delta],axis=1)
    x2 = tf.transpose(tf.gather(x_random2,[theta_delta],axis=1))
    l = tf.gather(gp.kernel.length_scale, [theta_delta])
    diff = x1 - x2
    factor = (1. - 1./(l**2)*diff**2)/l**2
    factor = factor*gp.kernel.tensor(x_random, x_random2,1,1)
    return factor

#posterior variance of delta or theta
def V_g_analytic(gp, x_random, x_training, name, theta_delta):
    K_training = gp.kernel.tensor(x_training,x_training,1,1)
    if(name == 'M52'):
        K_1 = -first_derivative_M52(gp, x_random, x_training, theta_delta)
        K_2 = first_derivative_M52(gp, x_training, x_random, theta_delta)
        K_0 = second_derivative_M52(gp, x_random, x_random, theta_delta)
    elif(name == 'SE'):
        K_1 = first_derivative_SE(gp, x_random, x_training, theta_delta)
        K_2 = tf.transpose(K_1)
        K_0 = second_derivative_SE(gp, x_random, x_random, theta_delta)
    len_I = len(x_training)
    V_g = tf.linalg.inv(K_training + gp.observation_noise_variance*tf.eye(len_I, dtype = tf.float64))
    V_g = K_0 - K_1@V_g@K_2
    return tf.linalg.diag_part(V_g)
