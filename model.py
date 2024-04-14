import tensorflow as tf
from tensorflow.keras import layers
from tools import fft2c_mri, ifft2c_mri
import scipy.io as scio

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res


class TLR_Net(tf.keras.Model):
    def __init__(self, niter):
        super(TLR_Net, self).__init__(name='TLR_Net')
        self.niter = niter
        self.celllist = []

    def build(self, input_shape):
        for i in range(self.niter - 1):
            self.celllist.append(TLRCell(input_shape, i))
        self.celllist.append(TLRCell(input_shape, self.niter - 1, is_last=True))

    def call(self, d, mask):
        """
        d: undersampled k-space
        mask: sampling mask
        """
        # nb, nc, nt, nx, ny = d.shape
        x_rec = ifft2c_mri(d)
        A = tf.zeros_like(x_rec)
        L = tf.zeros_like(x_rec)
        data = [x_rec, L, A, d, mask]

        for i in range(self.niter):
            data = self.celllist[i](data)

        x_rec = data[0]

        return x_rec


class TLRCell(layers.Layer):
    def __init__(self, input_shape, i, is_last=False):
        super(TLRCell, self).__init__()
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape

        if is_last:
            self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef %d' % i)
            self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)
            self.eta = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=False, name='eta %d' % i)
        else:
            self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef %d' % i)
            self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)
            self.eta = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=True, name='eta %d' % i)

        self.conv_1 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=2, ifactivate=False)

        self.conv_4 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_6 = CONV_OP(n_f=2, ifactivate=False)

    def call(self, data, **kwargs):
        x_rec, L, A, d, mask = data

        A = self.lowrank_step(x_rec + L)
        x_rec = self.x_step(L, A, d, mask)
        L = self.L_step(L, x_rec, A)

        data[0] = x_rec
        data[1] = L
        data[2] = A

        return data

    def x_step(self, L, A, d, mask):
        temp = A - L
        k_rec = fft2c_mri(temp)
        k_rec = tf.math.scalar_mul(tf.cast(tf.nn.relu(self.mu), tf.complex64), d) + k_rec
        k_rec = tf.math.divide_no_nan(k_rec, tf.math.scalar_mul(tf.cast(tf.nn.relu(self.mu), tf.complex64), mask) + 1)
        x_rec = ifft2c_mri(k_rec)
        return x_rec

    def lowrank_step(self, x):
        [batch, Nt, Nx, Ny] = x.get_shape()
        x_in = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        x_1 = self.conv_1(x_in)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        # SVT
        x_3_c = tf.complex(x_3[:, :, :, :, 0], x_3[:, :, :, :, 1])

        St, Ut, Vt = tf.linalg.svd(x_3_c)
        thres = tf.sigmoid(self.thres_coef) * St[..., 0]
        thres = tf.expand_dims(thres, -1)
        St = tf.nn.relu(St - thres)
        St = tf.linalg.diag(St)

        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 1, 3, 2])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        x_soft = tf.linalg.matmul(US, Vt_conj)

        x_soft = tf.stack([tf.math.real(x_soft), tf.math.imag(x_soft)], axis=-1)
        
        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        A = x_6 + x_in

        A = tf.complex(A[:, :, :, :, 0], A[:, :, :, :, 1])

        return A

    def L_step(self, L, x_rec, A):
        eta = tf.cast(tf.nn.relu(self.eta), tf.complex64)
        return L + tf.math.scalar_mul(eta, x_rec - A)

